import inspect
import time
from functools import partial
from typing import Callable, Optional, cast

import ignite.distributed as idist
import torch
from atria.core.utilities.logging import get_logger
from ignite.handlers import ProgressBar
from ignite.metrics import Metric
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.utils import apply_to_tensor
from insightx.engines.metrics_cacher import MetricsCacher
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
        cacher: Optional[MetricsCacher] = None,
    ):
        self._attached_name = attached_name
        self._metric_func = metric_func
        self._metric_name = self._metric_func.func.__name__
        self._forward_func = forward_func
        self._explainer = explainer
        self._metric_outputs = None
        self._progress_bar = progress_bar
        self._cacher = cacher
        super().__init__(output_transform=output_transform, device=device)

    @property
    def metric_name(self):
        return self._metric_name

    @reinit__is_reduced
    def reset(self):
        self._metric_outputs = []
        self._num_examples = 0
        self._result: Optional[float] = None
        super().reset()

    def _prepare_metric_kwargs(
        self,
        output: ExplanationModelOutput,
    ):
        explainer = self._explainer(self._forward_func)

        # if target is a list of list we assume it is a multi-target scenario with varying output sizes per sample
        # in which case we need to iterate over the samples in the batch
        is_target_list = isinstance(output.target, list)
        is_target_list_of_lists = isinstance(output.target, list) and isinstance(
            output.target[0], list
        )

        metric_kwargs = dict(
            forward_func=self._forward_func,
            inputs=tuple(x.detach() for x in output.explainer_args.inputs.values()),
            additional_forward_args=tuple(
                x.detach()
                for x in output.explainer_args.additional_forward_kwargs.values()
            ),
            target=(
                output.target.detach()
                if isinstance(output.target, torch.Tensor)
                else output.target
            ),
            attributions=tuple(
                apply_to_tensor(x, torch.detach) for x in output.explanations.values()
            ),
            baselines=tuple(
                x.detach() for x in output.explainer_args.baselines.values()
            ),
            feature_mask=tuple(
                x.detach() for x in output.explainer_args.feature_masks.values()
            ),
            is_multi_target=is_target_list,
            explainer=explainer,
            constant_shifts=(
                tuple(  # these are only for input invariance
                    x.detach() for x in output.explainer_args.constant_shifts.values()
                )
                if output.explainer_args.constant_shifts is not None
                else None
            ),
            input_layer_names=(
                tuple(x for x in output.explainer_args.input_layer_names.values())
                if output.explainer_args.input_layer_names is not None
                else None
            ),  # these are only for input invariance
            frozen_features=(
                output.explainer_args.frozen_features
                if output.explainer_args.frozen_features is not None
                else None
            ),
            return_intermediate_results=True,
            return_dict=True,
            show_progress=True,
        )

        possible_args = set(inspect.signature(self._metric_func).parameters)
        if "explainer" in possible_args:
            possible_args.update(set(inspect.signature(explainer.explain).parameters))
        if is_target_list_of_lists:
            batch_size = output.explainer_args.inputs[
                next(iter(output.explainer_args.inputs))
            ].shape[0]

            metric_kwargs_list = []
            for batch_idx in range(batch_size):
                current_metric_kwargs = {}
                for k, v in metric_kwargs.items():
                    if k in [
                        "forward_func",
                        "is_multi_target",
                        "explainer",
                        "constant_shifts",
                        "input_layer_names",
                        "return_intermediate_results",
                        "return_dict",
                    ]:
                        current_metric_kwargs[k] = v
                        continue

                    if isinstance(v, tuple):
                        current_metric_kwargs[k] = tuple(
                            v_i[batch_idx].unsqueeze(0) for v_i in v
                        )
                    elif isinstance(v, torch.Tensor):
                        current_metric_kwargs[k] = v[batch_idx].unsqueeze(0)
                    else:
                        current_metric_kwargs[k] = v[batch_idx]

                total_targets = len(current_metric_kwargs["target"])
                assert all(
                    tuple(
                        x.shape[1] == len(current_metric_kwargs["target"])
                        for x in current_metric_kwargs["attributions"]
                    )
                ), (
                    "dim=1 of attributions must have the same size as the total number of targets for each input tuple in multi-target scenario"
                    f"dim=1 of attributions: {[x.shape[1] for x in current_metric_kwargs['attributions']]}"
                    f"total number of targets: {total_targets}"
                )

                # convert explanations to list of tensors
                current_metric_kwargs["attributions"] = [
                    tuple(x[:, t] for x in current_metric_kwargs["attributions"])
                    for t in range(total_targets)
                ]

                assert (
                    len(current_metric_kwargs["attributions"]) == total_targets
                ), "The number of targets must be equal to the number of attributions as input to the metric function."
                metric_kwargs_list += [current_metric_kwargs]

            metric_kwargs = [
                {
                    key: value
                    for key, value in metric_kwargs.items()
                    if key in possible_args
                }
                for metric_kwargs in metric_kwargs_list
            ]
            return metric_kwargs
        else:
            metric_kwargs = {
                key: value
                for key, value in metric_kwargs.items()
                if key in possible_args
            }
            return metric_kwargs

    @reinit__is_reduced
    def update(self, output: ExplanationModelOutput):
        if output.explanations is None:
            return

        loaded_metrics = None
        if self._cacher is not None:
            loaded_metrics = self._cacher.load_metrics(
                f"{self._attached_name}.{self._metric_name}",
                output.sample_keys,
            )

        if loaded_metrics is not None:
            logger.debug(f"Loaded metrics from cache: {loaded_metrics}")
            self._num_examples += len(loaded_metrics)
            self._metric_outputs.append(loaded_metrics)
        else:
            if self._progress_bar is not None:
                if self._attached_name is not None:
                    self._progress_bar.pbar.set_postfix_str(
                        f"computing metric=[{self._attached_name}.{self._metric_name}]"
                    )
                else:
                    self._progress_bar.pbar.set_postfix_str(
                        f"computing metric=[{self._metric_name}]"
                    )

            start_time = time.time()
            metric_kwargs = self._prepare_metric_kwargs(output)
            if isinstance(metric_kwargs, list):
                metric_output = []
                for metric_kwargs_per_sample in metric_kwargs:
                    metric_output_per_sample = self._metric_func(
                        **metric_kwargs_per_sample
                    )
                    metric_output_per_sample = apply_to_tensor(
                        metric_output_per_sample, lambda tensor: tensor.detach().cpu()
                    )
                    metric_output.append(
                        {k: torch.cat(v) for k, v in metric_output_per_sample.items()}
                    )
                metric_output = {
                    key: [d[key] for d in metric_output]
                    for key in metric_output[0].keys()
                }
            else:
                metric_output = self._metric_func(**metric_kwargs)
                metric_output = apply_to_tensor(
                    metric_output, lambda tensor: tensor.detach().cpu()
                )
            end_time = time.time()
            time_taken = torch.tensor(end_time - start_time, requires_grad=False)
            metric_output = {
                **metric_output,
                f"time_taken": torch.stack(
                    [
                        time_taken / len(output.sample_keys)
                        for _ in range(len(output.sample_keys))
                    ]
                ),
            }

            if self._cacher is not None:
                self._cacher.save_metrics(
                    {
                        f"{self._attached_name}.{self._metric_name}_{k}": v
                        for k, v in metric_output.items()
                    },
                    output.sample_keys,
                )

            self._num_examples += len(metric_output)
            self._metric_outputs.append(metric_output)

            if self._progress_bar is not None:
                self._progress_bar.pbar.set_postfix_str(
                    f"computing metric=[{self._metric_name}], metric outputs=[{metric_output.keys()}]"
                )

    @sync_all_reduce("_num_examples")
    def compute(self) -> float:
        try:
            if self._num_examples == 0:
                return {}

            if isinstance(next(iter(self._metric_outputs[0].values())), torch.Tensor):
                aggregated_metric = {
                    key: [d[key] for d in self._metric_outputs]
                    for key in self._metric_outputs[0]
                }
            elif isinstance(next(iter(self._metric_outputs[0].values())), list):
                aggregated_metric = {
                    key: [item for d in self._metric_outputs for item in d[key]]
                    for key in self._metric_outputs[0]
                }
            else:
                raise ValueError(
                    "The metric output must be a tensor or a list of tensors as batch."
                )

            ws = idist.get_world_size()
            if ws > 1:
                # All gather across all processes
                for key in aggregated_metric.keys():
                    aggregated_metric[key] = cast(
                        torch.Tensor, idist.all_gather(aggregated_metric[key])
                    )

            return aggregated_metric
        except Exception as e:
            logger.exception(
                f"An error occurred while computing the metric {self._metric_name}. Error: {e}"
            )
            exit(1)
