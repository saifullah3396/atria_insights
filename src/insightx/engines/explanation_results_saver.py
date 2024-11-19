from __future__ import annotations

from typing import Any, Mapping

import torch
from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine

from insightx.utilities.common import _flatten_dict
from insightx.utilities.containers import ExplanationModelOutput
from insightx.utilities.h5io import HFSampleSaver

logger = get_logger(__name__)


class ExplanationResultsSaver:
    def __init__(
        self,
        output_file_path: HFSampleSaver,
        cache_full_explanations: bool = False,
    ) -> None:
        self._output_file_path = output_file_path
        self._cache_full_explanations = cache_full_explanations

    def _save_metadata(
        self,
        hfio: HFSampleSaver,
        batch: Mapping[str, torch.Tensor],
        output: ExplanationModelOutput,
    ) -> None:
        # get sample keys
        batch_size = len(batch["__key__"])
        for batch_idx in range(len(batch["__key__"])):
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
            if len(output.target) == batch_size:
                target = output.target[batch_idx]

            hfio.save_attribute(
                "pred_or_expl_target_labels",
                (
                    target.detach().cpu().numpy()
                    if isinstance(target, torch.Tensor)
                    else target
                ),
                sample_key,
            )

    def load_explanations(
        self, batch: Mapping[str, torch.Tensor], explainer_args
    ) -> None:
        if not self._cache_full_explanations:
            return
        with HFSampleSaver(self._output_file_path) as hfio:
            explanations_batch = []
            for sample_key in batch["__key__"]:
                explanations = {}
                for input_key in explainer_args.inputs.keys():
                    explanation = hfio.load(
                        f"explanation_{input_key}",
                        sample_key,
                    )
                    if explanation is None:
                        continue
                    explanations[input_key] = torch.from_numpy(explanation)
                if len(explanations) > 0:
                    explanations_batch.append(explanations)

            return explanations_batch

    def _save_explanations(
        self,
        hfio: HFSampleSaver,
        explanations: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        is_reduced: bool = False,
    ) -> None:

        for input_key, explanations_per_sample in explanations.items():
            for sample_key, explanation in zip(
                batch["__key__"], explanations_per_sample
            ):
                hfio.save(
                    (
                        f"explanation_{input_key}"
                        if not is_reduced
                        else f"reduced_explanation_{input_key}"
                    ),
                    explanation.detach().cpu().numpy(),
                    sample_key,
                )

    def _save_metrics(
        self,
        hfio: HFSampleSaver,
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

    def key_exists(self, key: str, sample_key: str) -> bool:
        with HFSampleSaver(self._output_file_path) as hfio:
            return hfio.key_exists(key, sample_key)

    def sample_exists(self, sample_key: str) -> bool:
        with HFSampleSaver(self._output_file_path) as hfio:
            return hfio.sample_exists(sample_key)

    def __call__(self, engine: Engine) -> None:
        assert isinstance(
            engine.state.output, ExplanationModelOutput
        ), f"Output must be of type ExplanationModelOutput for {self.__class__.__name__} handler."

        batch = engine.state.batch
        output: ExplanationModelOutput = engine.state.output
        if output.explanations is None:
            return

        metrics = {
            k: v
            for k, v in engine.state.metrics.items()
            # only take scalar values as ignite Metric has a bug where it duplicates metric values
            # twice one as dict and one as scalar
            if isinstance(v, dict)
        }
        # flatten the dict of metrics
        metrics = _flatten_dict(metrics)

        with HFSampleSaver(self._output_file_path) as hfio:
            self._save_metadata(hfio, batch, output)
            if not output.explanations_loaded_from_cache:
                if self._cache_full_explanations:
                    self._save_explanations(hfio, output.explanations, batch)
                self._save_explanations(
                    hfio, output.reduced_explanations, batch, is_reduced=True
                )
            self._save_metrics(hfio, metrics, batch)
