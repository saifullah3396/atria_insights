from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import torch
from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from insightx.utilities.containers import ExplanationModelOutput
from insightx.utilities.h5io import HFSampleSaver

logger = get_logger(__name__)


class ExplanationResultsCacher:
    def __init__(
        self,
        output_file_path: Path,
        cache_full_explanations: bool = False,
    ) -> None:
        self._output_file_path = output_file_path
        self._cache_full_explanations = cache_full_explanations

    @property
    def explanation_file_path(self) -> Path:
        return self._output_file_path

    @property
    def reduced_explanation_file_path(self) -> Path:
        return self._output_file_path.with_suffix(".reduced.h5")

    @property
    def metadata_file_path(self) -> Path:
        return self._output_file_path.parent.parent / "metadata.h5"

    def explanation_exists(self, sample_key: str) -> bool:
        if not self.explanation_file_path.exists():
            return False

        with HFSampleSaver(self.explanation_file_path, mode="r") as hfio:
            return hfio.sample_exists(sample_key)

    def batch_explanations_exists(self, batch) -> bool:
        return all([self.explanation_exists(key) for key in batch["__key__"]])

    def metadata_exists(self, sample_key: str) -> bool:
        if not self.metadata_file_path.exists():
            return False

        with HFSampleSaver(self.metadata_file_path, mode="r") as hfio:
            return hfio.sample_exists(sample_key)

    def batch_metadata_exists(self, batch) -> bool:
        return all([self.metadata_exists(key) for key in batch["__key__"]])

    def _save_metadata(
        self,
        batch: Mapping[str, torch.Tensor],
        output: ExplanationModelOutput,
    ) -> None:
        with HFSampleSaver(self.metadata_file_path) as hfio:
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
                        hfio.save(
                            f"ground_truth_{key}",
                            batch[key][batch_idx].detach().cpu().numpy(),
                            sample_key,
                        )

                # store predicted or explanation target labels
                if len(output.target) == batch_size:
                    target = output.target[batch_idx]

                # save feature masks
                for (
                    input_key,
                    feature_masks_per_sample,
                ) in output.explainer_args.feature_masks.items():
                    hfio.save(
                        f"feature_masks_{input_key}",
                        feature_masks_per_sample[batch_idx].detach().cpu().numpy(),
                        sample_key,
                    )

                # save baselines
                for (
                    input_key,
                    baselines_per_sample,
                ) in output.explainer_args.baselines.items():
                    if baselines_per_sample is not None:
                        hfio.save(
                            f"baselines_{input_key}",
                            baselines_per_sample[batch_idx].detach().cpu().numpy(),
                            sample_key,
                        )

                # save frozen features
                hfio.save(
                    "frozen_features",
                    output.explainer_args.frozen_features[batch_idx]
                    .detach()
                    .cpu()
                    .numpy(),
                    sample_key,
                )

                hfio.save(
                    "pred_or_expl_target_labels",
                    (
                        target.detach().cpu().numpy()
                        if isinstance(target, torch.Tensor)
                        else target
                    ),
                    sample_key,
                )

    def _save_explanations(
        self,
        file_path: Path,
        explanations: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        with HFSampleSaver(file_path) as hfio:
            for input_key, explanations_per_sample in explanations.items():
                for sample_key, explanation in zip(
                    batch["__key__"], explanations_per_sample
                ):
                    hfio.save(
                        input_key,
                        explanation.detach().cpu().numpy(),
                        sample_key,
                    )

    def _load_explanations_with_base_key(
        self,
        file_path: Path,
        sample_keys: List[str],
        concatenate_results: bool = True,
    ) -> None:
        if not file_path.exists():
            return None

        with HFSampleSaver(file_path, "r") as hfio:
            explanations_batch = []
            for sample_key in sample_keys:
                explanations_per_sample = {}
                keys = hfio.get_keys(sample_key)
                for key in keys:
                    explanation = hfio.load(
                        key,
                        sample_key,
                    )
                    if explanation is None:
                        continue
                    explanations_per_sample[key] = torch.from_numpy(explanation)
                if len(explanations_per_sample) > 0:
                    explanations_batch.append(explanations_per_sample)

            if len(explanations_batch) != len(sample_keys):
                logger.warning(
                    f"Missing explanations for {len(sample_keys) - len(explanations_batch)} samples"
                )
                return None

            if len(explanations_batch) > 0:
                explanations_batch = {
                    # convert list of dicts to dict of lists
                    key: (
                        torch.cat(
                            [explanation[key] for explanation in explanations_batch]
                        )
                        if concatenate_results
                        else [explanation[key] for explanation in explanations_batch]
                    )
                    for key in explanations_batch[0].keys()
                }
            else:
                explanations_batch = None

            return explanations_batch

    def load_explanations(self, sample_keys: List[str]) -> None:
        explanations = self._load_explanations_with_base_key(
            self.explanation_file_path, sample_keys
        )
        reduced_explanations = self._load_explanations_with_base_key(
            self.reduced_explanation_file_path,
            sample_keys,
            concatenate_results=False,
        )
        return explanations, reduced_explanations

    def save_results(self, batch: BatchDict, output: ExplanationModelOutput) -> None:
        self._save_metadata(batch, output)
        if self._cache_full_explanations:
            if output.explanations is not None:
                self._save_explanations(
                    self.explanation_file_path, output.explanations, batch
                )
        if output.reduced_explanations is not None:
            self._save_explanations(
                self.reduced_explanation_file_path,
                output.reduced_explanations,
                batch,
            )
