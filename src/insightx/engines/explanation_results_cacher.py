from __future__ import annotations

from pathlib import Path
from typing import List, Mapping

import torch
from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger

from insightx.utilities.containers import ExplanationModelOutput
from insightx.utilities.h5io import HFSampleSaver
from atria.core.utilities.typing import BatchDict

logger = get_logger(__name__)


class ExplanationResultsCacher:
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

    def key_exists(self, key: str, sample_key: str) -> bool:
        with HFSampleSaver(self._output_file_path) as hfio:
            return hfio.key_exists(key, sample_key)

    def sample_exists(self, sample_key: str) -> bool:
        with HFSampleSaver(self._output_file_path) as hfio:
            return hfio.sample_exists(sample_key)

    def _load_explanations_with_base_key(
        self,
        file_path: Path,
        sample_keys: List[str],
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

            if len(explanations_batch) > 0:
                explanations_batch = {
                    # convert list of dicts to dict of lists
                    key: torch.cat(
                        [explanation[key] for explanation in explanations_batch]
                    )
                    for key in explanations_batch[0].keys()
                }
            else:
                explanations_batch = None
            return explanations_batch

    def load_explanations(self, sample_keys: List[str]) -> None:
        explanations = self._load_explanations_with_base_key(
            self._output_file_path, sample_keys
        )
        reduced_explanations = self._load_explanations_with_base_key(
            self._output_file_path.with_suffix(".reduced.h5"),
            sample_keys,
        )
        return explanations, reduced_explanations

    def save_results(self, batch: BatchDict, output: ExplanationModelOutput) -> None:
        with HFSampleSaver(self._output_file_path) as hfio:
            self._save_metadata(hfio, batch, output)

        if self._cache_full_explanations:
            self._save_explanations(self._output_file_path, output.explanations, batch)
        self._save_explanations(
            self._output_file_path.with_suffix(".reduced.h5"),
            output.reduced_explanations,
            batch,
        )
