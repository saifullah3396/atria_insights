from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

import torch
from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict

from insightx.utilities.containers import ExplanationStepMetadata
from insightx.utilities.h5io import HFSampleSaver

logger = get_logger(__name__)


class ExplanationResultsCacher:
    def __init__(
        self,
        output_file_path: Path,
        cache_full_explanations: bool = False,
        cache_reduced_explanations: bool = False,
        save_fp16: bool = False,
    ) -> None:
        self._output_file_path = output_file_path
        self._cache_full_explanations = cache_full_explanations
        self._cache_reduced_explanations = cache_reduced_explanations
        self._save_fp16 = save_fp16

    @property
    def explanation_file_path(self) -> Path:
        return self._output_file_path

    @property
    def reduced_explanation_file_path(self) -> Path:
        return self._output_file_path.with_suffix(".reduced.h5")

    @property
    def metadata_file_path(self) -> Path:
        return self._output_file_path.parent.parent / "metadata.h5"

    def explanation_exists(
        self, sample_key: str, check_reduced_explanations: bool = False
    ) -> bool:
        file_path = (
            self.reduced_explanation_file_path
            if check_reduced_explanations
            else self.explanation_file_path
        )
        if not file_path.exists():
            return False

        with HFSampleSaver(file_path, mode="r") as hfio:
            return hfio.sample_exists(sample_key)

    def batch_explanations_exists(
        self, batch, check_reduced_explanations: bool = False
    ) -> bool:
        return all(
            [
                self.explanation_exists(
                    key, check_reduced_explanations=check_reduced_explanations
                )
                for key in batch["__key__"]
            ]
        )

    def metadata_exists(self, sample_key: str) -> bool:
        if not self.metadata_file_path.exists():
            return False

        with HFSampleSaver(self.metadata_file_path, mode="r") as hfio:
            return hfio.sample_exists(sample_key)

    def batch_metadata_exists(self, batch) -> bool:
        return all([self.metadata_exists(key) for key in batch["__key__"]])

    def save_metadata(
        self,
        batch: Mapping[str, torch.Tensor],
        explanation_step_metadata: ExplanationStepMetadata,
        file_suffix: str = None,
    ) -> None:
        file_path = (
            self.metadata_file_path.with_suffix(f".{file_suffix}.h5")
            if file_suffix is not None
            else self.metadata_file_path
        )
        with HFSampleSaver(file_path) as hfio:
            # get sample keys
            batch_size = len(batch["__key__"])
            for batch_idx in range(len(batch["__key__"])):
                # get unique sample key
                sample_key = batch["__key__"][batch_idx]

                # store dataset labels
                if explanation_step_metadata.dataset_labels is not None:
                    hfio.save(
                        "dataset_labels",
                        explanation_step_metadata.dataset_labels,
                        sample_key,
                    )

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

                # store sample info
                for key in [
                    DataKeys.IMAGE_FILE_PATH,
                    DataKeys.WORDS,
                    DataKeys.WORD_BBOXES,
                    DataKeys.WORD_IDS,
                ]:
                    if key in batch:
                        hfio.save(
                            key,
                            (
                                batch[key][batch_idx].detach().cpu().numpy()
                                if isinstance(batch[key], torch.Tensor)
                                else batch[key][batch_idx]
                            ),
                            sample_key,
                        )

                # save feature masks
                for (
                    input_key,
                    feature_masks_per_sample,
                ) in explanation_step_metadata.explainer_args.feature_masks.items():
                    hfio.save(
                        f"feature_masks_{input_key}",
                        feature_masks_per_sample[batch_idx].detach().cpu().numpy(),
                        sample_key,
                    )

                # save baselines
                for (
                    input_key,
                    baselines_per_sample,
                ) in explanation_step_metadata.explainer_args.baselines.items():
                    if baselines_per_sample is not None:
                        hfio.save(
                            f"baselines_{input_key}",
                            baselines_per_sample[batch_idx].detach().cpu().numpy(),
                            sample_key,
                        )

                # save total feature groups
                hfio.save(
                    "total_features",
                    explanation_step_metadata.explainer_args.total_features,
                    sample_key,
                )

                # save frozen features
                hfio.save(
                    "frozen_features",
                    explanation_step_metadata.explainer_args.frozen_features[batch_idx]
                    .detach()
                    .cpu()
                    .numpy(),
                    sample_key,
                )
                # store predicted or explanation target labels
                assert len(explanation_step_metadata.target) == batch_size, (
                    f"Expected target to have the same length as the batch size, "
                    f"but got {len(explanation_step_metadata.target)} and {batch_size}"
                )
                assert len(explanation_step_metadata.model_outputs) == batch_size, (
                    f"Expected model outputs to have the same length as the batch size, "
                    f"but got {len(explanation_step_metadata.model_outputs)} and {batch_size}"
                )

                target = explanation_step_metadata.target[batch_idx]
                model_output = explanation_step_metadata.model_outputs[batch_idx]
                target = (
                    target.detach().cpu().numpy()
                    if isinstance(target, torch.Tensor)
                    else target
                )
                model_output = (
                    model_output.detach().cpu().numpy()
                    if isinstance(model_output, torch.Tensor)
                    else model_output
                )

                # save target and target output
                hfio.save(
                    "pred_or_expl_target_labels",
                    target,
                    sample_key,
                )
                hfio.save(
                    "target_word_ids",
                    explanation_step_metadata.target_word_ids[batch_idx],
                    sample_key,
                )
                hfio.save(
                    "model_outputs",
                    model_output,
                    sample_key,
                )

    def _save_explanations(
        self,
        file_path: Path,
        explanations: Dict[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        with HFSampleSaver(file_path) as hfio:
            for input_key, explanations_per_sample in explanations.items():
                for sample_key, explanation in zip(
                    batch["__key__"], explanations_per_sample
                ):
                    if explanation is not None:
                        explanation = (
                            explanation.detach().cpu().half().numpy()
                            if self._save_fp16
                            else explanation.detach().cpu().numpy()
                        )
                    else:
                        # this is just a place holder if there is no value for explanation
                        explanation = -1000
                    hfio.save(
                        input_key,
                        explanation,
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
                logger.debug(
                    f"No cached explanations found for {len(sample_keys) - len(explanations_batch)} samples in batch."
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

    def save_explanations(
        self,
        batch: BatchDict,
        explanations: Dict[str, torch.Tensor],
        reduced_explanations: Dict[str, torch.Tensor],
    ) -> None:
        if self._cache_full_explanations:
            if explanations is not None:
                self._save_explanations(
                    self.explanation_file_path,
                    explanations,
                    batch,
                )

        if self._cache_reduced_explanations:
            if reduced_explanations is not None:
                self._save_explanations(
                    self.reduced_explanation_file_path,
                    reduced_explanations,
                    batch,
                )
