from __future__ import annotations

from typing import Any, Mapping, Union

import torch
from atria._core.constants import DataKeys
from atria._core.utilities.logging import get_logger
from ignite.engine import Engine
from insightx.utilities.common import _flatten_dict
from insightx.utilities.containers import ExplanationModelOutput
from insightx.utilities.h5io import HFIOMultiOutput, HFIOSingleOutput

logger = get_logger(__name__)


class ExplanationResultsSaver:
    def __init__(
        self,
        output_file_path: Union[HFIOSingleOutput, HFIOMultiOutput],
        is_single_output: bool = True,
    ) -> None:
        self._output_file_path = output_file_path
        self._is_single_output = is_single_output

    def _save_metadata(
        self,
        hfio: Union[HFIOSingleOutput, HFIOMultiOutput],
        batch: Mapping[str, torch.Tensor],
        output: ExplanationModelOutput,
    ) -> None:
        # get sample keys
        for batch_idx in range(len(batch["__index__"])):
            # get unique sample key
            sample_key = batch["__key__"][batch_idx]

            # store ground truth labels
            for key in [
                DataKeys.LABEL,
                DataKeys.WORD_LABELS,
                DataKeys.ANSWER_START_INDICES,
                DataKeys.ANSWER_END_INDICES,
            ]:
                if key in batch:
                    hfio.save_attribute(
                        f"ground_truth_{key}",
                        batch[key][batch_idx].detach().cpu().numpy(),
                        sample_key,
                    )

            # store predicted or explanation target labels
            hfio.save_attribute(
                "pred_or_expl_target_labels",
                output.target.detach().cpu().numpy(),
                sample_key,
            )

    def _save_explanations(
        self,
        hfio: Union[HFIOSingleOutput, HFIOMultiOutput],
        explanations: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        for input_key, explanations_per_sample in explanations.items():
            for sample_key, explanation in zip(
                batch["__key__"], explanations_per_sample
            ):
                hfio.save(
                    f"explanation_{input_key}",
                    explanation.detach().cpu().numpy(),
                    sample_key,
                )

    def _save_metrics(
        self,
        hfio: Union[HFIOSingleOutput, HFIOMultiOutput],
        metrics: Mapping[str, Any],
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        for metric_key, metric_per_sample in metrics.items():
            if isinstance(metric_per_sample, torch.Tensor):
                metric_per_sample = metric_per_sample.detach().cpu().numpy()
            for sample_key, metric in zip(batch["__key__"], metric_per_sample):
                logger.debug("Saving metric %s for sample %s", metric_key, sample_key)
                hfio.save(
                    metric_key,
                    metric,
                    sample_key,
                )

    def __call__(self, engine: Engine) -> None:
        assert isinstance(
            engine.state.output, ExplanationModelOutput
        ), f"Output must be of type ExplanationModelOutput for {self.__class__.__name__} handler."

        batch = engine.state.batch
        output: ExplanationModelOutput = engine.state.output
        metrics = {
            k: v
            for k, v in engine.state.metrics.items()
            # only take scalar values as ignite Metric has a bug where it duplicates metric values
            # twice one as dict and one as scalar
            if isinstance(v, dict)
        }
        # flatten the dict of metrics
        metrics = _flatten_dict(metrics)

        with (
            HFIOSingleOutput(self._output_file_path)
            if self._is_single_output
            else HFIOMultiOutput(self._output_file_path)
        ) as hfio:
            self._save_metadata(hfio, batch, output)
            self._save_explanations(hfio, output.reduced_explanations, batch)
            self._save_metrics(hfio, metrics, batch)
