import inspect
import time
from collections.abc import Callable
from functools import partial

import torch
from atria_core.logger.logger import get_logger
from ignite.handlers import ProgressBar
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from ignite.utils import apply_to_tensor
from torch.cuda.amp import autocast
from torchxai.explainers.explainer import Explainer

from atria_insights.utilities.containers import ExplainerStepOutput

logger = get_logger(__name__)


def default_output_transform(output):
    assert isinstance(output, ExplainerStepOutput), (
        "The output of the model must be an instance of ExplanationModelOutput, "
    )
    return output


def detach(x):
    if isinstance(x, list):
        return [detach(xx) if xx is not None else None for xx in x]
    elif isinstance(x, torch.Tensor):
        return x.detach()
    return x  # For other types, return as is


class TorchXAIMetric(Metric):
    def __init__(
        self,
        metric_func: partial[Callable],
        forward_func: torch.nn.Module,
        explainer: Explainer,
        output_transform=default_output_transform,
        device="cpu",
        progress_bar: ProgressBar | None = None,
        attached_name: str | None = None,
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
        self._num_examples = 0
        self._result: float | None = None
        super().reset()

    def _prepare_metric_kwargs(self, output: ExplainerStepOutput):
        # if target is a list of list we assume it is a multi-target scenario with varying output sizes per sample
        # in which case we need to iterate over the samples in the batch
        is_target_list = isinstance(output.target, list)
        is_target_list_of_lists = isinstance(output.target, list) and isinstance(
            output.target[0], list
        )
        metric_kwargs = dict(
            forward_func=self._forward_func,
            inputs=tuple(
                x.detach()
                for x in output.explainer_step_inputs.model_inputs.explained_inputs.values()
            ),
            additional_forward_args=tuple(
                x.detach()
                for x in output.explainer_step_inputs.model_inputs.additional_forward_kwargs.values()
            )
            if output.explainer_step_inputs.model_inputs.additional_forward_kwargs
            else None,
            target=(
                output.target.detach()
                if isinstance(output.target, torch.Tensor)
                else output.target
            ),
            attributions=tuple(detach(x) for x in output.explanations.values()),
            baselines=(
                tuple(
                    x.detach()
                    for x in output.explainer_step_inputs.metric_baselines.values()
                )
                if all(
                    x is not None
                    for x in output.explainer_step_inputs.metric_baselines.values()
                )
                else None
            ),
            explainer_baselines=(
                tuple(
                    x.detach() for x in output.explainer_step_inputs.baselines.values()
                )
                if all(
                    x is not None
                    for x in output.explainer_step_inputs.baselines.values()
                )
                else None
            ),
            feature_mask=tuple(
                x.detach() for x in output.explainer_step_inputs.feature_masks.values()
            ),
            is_multi_target=is_target_list,
            explainer=self._explainer,
            constant_shifts=(
                tuple(  # these are only for input invariance
                    x.detach()
                    for x in output.explainer_step_inputs.constant_shifts.values()
                )
                if output.explainer_step_inputs.constant_shifts is not None
                else None
            ),
            input_layer_names=(
                tuple(
                    x for x in output.explainer_step_inputs.input_layer_names.values()
                )
                if output.explainer_step_inputs.input_layer_names is not None
                else None
            ),  # these are only for input invariance
            frozen_features=(
                output.explainer_step_inputs.frozen_features
                if output.explainer_step_inputs.frozen_features is not None
                else None
            ),
            train_baselines=tuple(output.explainer_step_inputs.train_baselines.values())
            if output.explainer_step_inputs.train_baselines is not None
            else None,
            return_intermediate_results=True,
            return_dict=True,
            show_progress=True,
        )

        possible_args = set(inspect.signature(self._metric_func).parameters)
        if "explainer" in possible_args:
            explainer_possible_args = set(
                inspect.signature(self._explainer.explain).parameters
            )
            if "baselines" in explainer_possible_args:
                if "baselines" in possible_args:
                    possible_args.remove("baselines")
                possible_args.add("explainer_baselines")
            possible_args.update(
                set(inspect.signature(self._explainer.explain).parameters)
            )

        if is_target_list_of_lists:
            batch_size = output.explainer_step_inputs.model_inputs.explained_inputs[
                next(iter(output.explainer_step_inputs.model_inputs.explained_inputs))
            ].shape[0]

            metric_kwargs_list = []
            for batch_idx in range(batch_size):
                current_metric_kwargs = {}
                for k, v in metric_kwargs.items():
                    try:
                        if v is None:
                            current_metric_kwargs[k] = None
                            continue
                        if k in [
                            "forward_func",
                            "is_multi_target",
                            "explainer",
                            "constant_shifts",
                            "input_layer_names",
                            "return_intermediate_results",
                            "return_dict",
                            "show_progress",
                            "train_baselines",
                        ]:
                            current_metric_kwargs[k] = v
                            continue

                        if isinstance(v, tuple):
                            current_metric_kwargs[k] = tuple(
                                (
                                    v_i[batch_idx].unsqueeze(0)
                                    if v_i[batch_idx] is not None
                                    else v_i[batch_idx]
                                )
                                for v_i in v
                            )
                        elif isinstance(v, torch.Tensor):
                            current_metric_kwargs[k] = v[batch_idx].unsqueeze(0)
                        else:
                            if (
                                k == "frozen_features"
                            ):  # frozen features is a list of tensors
                                current_metric_kwargs[k] = v[batch_idx].unsqueeze(0)
                            else:
                                current_metric_kwargs[k] = v[batch_idx]
                    except Exception as e:
                        logger.exception(
                            f"An error occurred while preparing metric kwargs {k}: {v} for multi-target scenario. Error: {e}"
                        )
                        exit(1)

                total_targets = len(current_metric_kwargs["target"])
                assert all(
                    tuple(
                        x is None or x.shape[1] == len(current_metric_kwargs["target"])
                        for x in current_metric_kwargs["attributions"]
                    )
                ), (
                    "dim=1 of attributions must have the same size as the total number of targets for each input tuple in multi-target scenario"
                    f"dim=1 of attributions: {[x.shape[1] for x in current_metric_kwargs['attributions']]}"
                    f"total number of targets: {total_targets}"
                )

                # convert explanations to list of tensors
                # current_metric_kwargs["attributions"] can be a tuple of None's in case the explanation does not exist
                # or targets are empty something like (None, None, ...)
                if current_metric_kwargs["attributions"][0] is not None:
                    current_metric_kwargs["attributions"] = [
                        tuple(x[:, t] for x in current_metric_kwargs["attributions"])
                        for t in range(total_targets)
                    ]
                else:
                    current_metric_kwargs["attributions"] = []
                assert len(current_metric_kwargs["attributions"]) == total_targets, (
                    "The number of targets must be equal to the number of attributions as input to the metric function."
                )
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
    def update(self, output: ExplainerStepOutput):
        if output.explanations is None:
            return

        # if self._progress_bar is not None:
        if self._attached_name is not None:
            # self._progress_bar.pbar.set_postfix_str(
            print(f"computing metric=[{self._attached_name}.{self._metric_name}]")
            # )
        else:
            # self._progress_bar.pbar.set_postfix_str(
            print(f"computing metric=[{self._metric_name}]")
            # )

        start_time = time.time()
        metric_kwargs = self._prepare_metric_kwargs(output)
        if isinstance(metric_kwargs, list):
            metric_output = []
            for metric_kwargs_per_sample in metric_kwargs:
                if (
                    "target" in metric_kwargs_per_sample
                    and len(metric_kwargs_per_sample["target"]) == 0
                ):
                    metric_output.append({"failure": True})
                    continue
                metric_output_per_sample = self._metric_func(**metric_kwargs_per_sample)
                metric_output_per_sample = apply_to_tensor(
                    metric_output_per_sample, lambda tensor: tensor.detach().cpu()
                )
                metric_output.append(metric_output_per_sample)
            metric_output = {
                key: [d[key] if d is not None else -1000 for d in metric_output]
                for key in metric_output[0].keys()
            }
        else:
            with autocast(enabled=True):
                logger.info(f"Computing metric: {self._metric_func}")
                metric_output = self._metric_func(**metric_kwargs)
            metric_output = apply_to_tensor(
                metric_output, lambda tensor: tensor.detach().cpu()
            )
        end_time = time.time()
        time_taken = torch.tensor(end_time - start_time, requires_grad=False)
        metric_output = {
            **metric_output,
            "time_taken": torch.stack(
                [time_taken / len(output.index) for _ in range(len(output.index))]
            ),
        }

        self._num_examples += len(metric_output)
        self._metric_outputs.append(metric_output)

        # if self._progress_bar is not None:
        #     self._progress_bar.pbar.set_postfix_str(
        #         f"computing metric=[{self._metric_name}], metric outputs=[{metric_output.keys()}]"
        #     )

    @sync_all_reduce("_num_examples")
    def compute(self) -> float:
        try:
            if self._num_examples == 0:
                return {}

            return self._metric_outputs
        except Exception as e:
            logger.exception(
                f"An error occurred while computing the metric {self._metric_name}. Error: {e}"
            )
            exit(1)
