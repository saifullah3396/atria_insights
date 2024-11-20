from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping

import torch
from atria.core.utilities.logging import get_logger

from insightx.utilities.common import _flatten_dict
from insightx.utilities.h5io import HFSampleSaver

logger = get_logger(__name__)


class MetricsCacher:
    def __init__(
        self,
        output_file_path: HFSampleSaver,
    ) -> None:
        self._output_file_path = output_file_path

    def key_exists(self, key: str, sample_key: str) -> bool:
        with HFSampleSaver(self._output_file_path) as hfio:
            return hfio.key_exists(key, sample_key)

    def sample_exists(self, sample_key: str) -> bool:
        with HFSampleSaver(self._output_file_path) as hfio:
            return hfio.sample_exists(sample_key)

    def save_metrics(
        self,
        metrics: Mapping[str, Any],
        sample_keys: List[str],
    ) -> None:
        # flatten the dict of metrics
        metrics = _flatten_dict(metrics)

        with HFSampleSaver(self._output_file_path) as hfio:
            for metric_key, metrics_batch in metrics.items():
                if isinstance(metrics_batch, torch.Tensor):
                    metrics_batch = metrics_batch.detach().cpu().numpy()
                for sample_key, metric in zip(sample_keys, metrics_batch):
                    logger.debug(
                        "Saving metric %s for sample %s", metric_key, sample_key
                    )
                    hfio.save(
                        metric_key,
                        metric,
                        sample_key,
                    )

    def load_metrics(
        self,
        metric_key: str,
        sample_keys: List[str],
    ) -> None:
        if not Path(self._output_file_path).exists():
            return None
        with HFSampleSaver(self._output_file_path) as hfio:
            loaded_metrics_batch = []
            for sample_key in sample_keys:
                logger.debug("Loading metric %s for sample %s", metric_key, sample_key)
                keys = hfio.get_keys(
                    sample_key,
                )
                loaded_metrics = {}
                for key in keys:
                    if f"{metric_key}" in key:
                        loaded_metrics[key] = torch.from_numpy(
                            hfio.load(
                                key,
                                sample_key,
                            )
                        )
                if len(loaded_metrics) > 0:
                    loaded_metrics_batch.append(loaded_metrics)

            if len(loaded_metrics_batch) > 0 and len(loaded_metrics_batch) == len(
                sample_keys
            ):
                test = {
                    # convert list of dicts to dict of lists
                    key[len(metric_key) + 1 :]: [
                        metric[key] for metric in loaded_metrics_batch
                    ]
                    for key in loaded_metrics_batch[0].keys()
                }
                loaded_metrics_batch = {
                    # convert list of dicts to dict of lists
                    key[len(metric_key) + 1 :]: torch.stack(
                        [metric[key] for metric in loaded_metrics_batch]
                    )
                    for key in loaded_metrics_batch[0].keys()
                }
            else:
                loaded_metrics_batch = None

            return loaded_metrics_batch
