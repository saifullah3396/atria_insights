import inspect
import time
from functools import partial
from typing import Callable, Optional, cast

import ignite.distributed as idist
import torch
from atria._core.utilities.logging import get_logger
from ignite.exceptions import NotComputableError
from ignite.handlers import ProgressBar
from ignite.metrics import Metric
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.utils import apply_to_tensor
from insightx.utilities.containers import ExplanationModelOutput
from torchxai.explainers.explainer import Explainer

logger = get_logger(__name__)


def default_output_transform(output):
    assert isinstance(
        output, ExplanationModelOutput
    ), "The output of the model must be an instance of ExplanationModelOutput, "
    return output


class TorchXAIMetric(Metric):
    def __init__(
        self,
        metric_func: partial[Callable],
        forward_func: torch.nn.Module,
        explainer: partial[Explainer],
        output_transform=default_output_transform,
        device="cpu",
        progress_bar: Optional[ProgressBar] = None,
        attached_name: Optional[str] = None,
    ):
        self._attached_name = attached_name
        self._metric_func = metric_func
        self._metric_name = self._metric_func.func.__name__
        self._forward_func = forward_func
        self._explainer = explainer
        self._metric_outputs = None
        self._progress_bar = progress_bar
        super().__init__(output_transform=output_transform, device=device)

    @property
    def metric_name(self):
        return self._metric_name

    @reinit__is_reduced
    def reset(self):
        self._metric_outputs = []
        self._time_taken = 0.0
        self._num_examples = 0
        self._result: Optional[float] = None
        super().reset()

    def _prepare_metric_kwargs(
        self,
        output: ExplanationModelOutput,
    ):
        explainer = self._explainer(self._forward_func)
        metric_kwargs = dict(
            forward_func=self._forward_func,
            inputs=tuple(x.detach() for x in output.explainer_inputs.inputs.values()),
            attributions=tuple(x.detach() for x in output.explanations.values()),
            baselines=tuple(
                x.detach() for x in output.explainer_inputs.baselines.values()
            ),
            feature_mask=tuple(
                x.detach() for x in output.explainer_inputs.feature_masks.values()
            ),
            additional_forward_args=tuple(
                x.detach()
                for x in output.explainer_inputs.additional_forward_kwargs.values()
            ),
            target=output.target.detach(),
            explainer=explainer,
            constant_shifts=(
                tuple(  # these are only for input invariance
                    x.detach() for x in output.explainer_inputs.constant_shifts.values()
                )
                if output.explainer_inputs.constant_shifts is not None
                else None
            ),
            input_layer_names=(
                tuple(x for x in output.explainer_inputs.input_layer_names.values())
                if output.explainer_inputs.input_layer_names is not None
                else None
            ),  # these are only for input invariance
            return_intermediate_results=False,
            return_dict=True,
        )
        possible_args = set(inspect.signature(self._metric_func).parameters)
        if "explainer" in possible_args:
            possible_args.update(set(inspect.signature(explainer.explain).parameters))
        kwargs = {
            key: value for key, value in metric_kwargs.items() if key in possible_args
        }
        return kwargs

    @reinit__is_reduced
    def update(self, output: ExplanationModelOutput):
        if self._progress_bar is not None:
            if self._attached_name is not None:
                self._progress_bar.pbar.set_postfix_str(
                    f"computing metric=[{self._attached_name}.{self._metric_name}]"
                )
            else:
                self._progress_bar.pbar.set_postfix_str(
                    f"computing metric=[{self._metric_name}]"
                )

        bsz = output.target.shape[0]
        start_time = time.time()
        metric_func_inputs = self._prepare_metric_kwargs(output)
        metric_output = self._metric_func(**metric_func_inputs)
        metric_output = apply_to_tensor(
            metric_output, lambda tensor: tensor.detach().cpu()
        )
        self._metric_outputs.append(metric_output)
        end_time = time.time()
        self._time_taken += end_time - start_time
        self._num_examples += bsz

        if self._progress_bar is not None:
            self._progress_bar.pbar.set_postfix_str(
                f"computing metric=[{self._metric_name}], metric outputs=[{metric_output.keys()}]"
            )

    @sync_all_reduce("_num_examples", "_time_taken:SUM")
    def compute(self) -> float:
        try:
            if self._num_examples == 0:
                raise NotComputableError(
                    "TorchXAIMetric must have at least one example before it can be computed."
                )

            time_taken_per_sample = [
                self._time_taken / self._num_examples
            ] * self._num_examples
            aggregated_metric = {
                key: torch.cat([d[key] for d in self._metric_outputs], dim=0)
                for key in self._metric_outputs[0]
            }

            ws = idist.get_world_size()
            if ws > 1:
                # All gather across all processes
                for key in aggregated_metric.keys():
                    aggregated_metric[key] = cast(
                        torch.Tensor, idist.all_gather(aggregated_metric[key])
                    )

            return {
                **aggregated_metric,
                f"time_to_compute_{self._metric_name}": time_taken_per_sample,
            }
        except Exception as e:
            logger.exception(
                f"An error occurred while computing the metric {self._metric_name}. Error: {e}"
            )
            exit(1)
