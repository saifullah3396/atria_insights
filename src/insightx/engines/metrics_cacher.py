from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from atria.core.utilities.logging import get_logger
from insightx.utilities.common import _flatten_dict
from insightx.utilities.h5io import HFSampleSaver

logger = get_logger(__name__)


class MetricsCacher:
    def __init__(
        self,
        output_file_path: Path,
    ) -> None:
        self._output_file_path = output_file_path
        self._output_file_path.parent.mkdir(exist_ok=True, parents=True)

    def _get_metric_path(self, metric_key: str) -> Path:
        return self._output_file_path.with_suffix(f".{metric_key}.h5")

    def _key_exists(self, sample_key: str, metric_key: str) -> bool:
        metric_file_path = self._get_metric_path(metric_key)
        if not Path(metric_file_path).exists():
            return False

        with HFSampleSaver(metric_file_path, mode="r") as hfio:
            return hfio.sample_exists(sample_key)
        return False

    def metrics_exist(self, batch: Dict[str, Any], metrics: Dict[str, str]) -> bool:
        return all(
            [
                self._key_exists(sample_key, metric_key)
                for metric_key in metrics.keys()
                for sample_key in batch["__key__"]
            ]
            if metrics is not None
            else []
        )

    def save_metrics(
        self,
        metrics: Mapping[str, Any],
        sample_keys: List[str],
        metric_key: str,
    ) -> None:
        # flatten the dict of metrics
        metrics = _flatten_dict(metrics)

        with HFSampleSaver(
            self._get_metric_path(metric_key),
        ) as hfio:
            for metric_param_name, metrics_batch in metrics.items():
                if isinstance(metrics_batch, torch.Tensor):
                    metrics_batch = metrics_batch.detach().cpu().numpy()
                for sample_key, metric_param_value in zip(sample_keys, metrics_batch):
                    logger.debug(
                        "Saving metric %s for sample %s", metric_param_name, sample_key
                    )
                    hfio.save(
                        metric_param_name,
                        metric_param_value,
                        sample_key,
                    )

    def load_metrics(
        self,
        metric_key: str,
        sample_keys: List[str],
    ) -> None:
        metric_file_path = self._get_metric_path(metric_key)
        if not metric_file_path.exists():
            return None
        with HFSampleSaver(metric_file_path, mode="r") as hfio:
            loaded_metrics_batch = []
            for sample_key in sample_keys:
                logger.debug("Loading metric %s for sample %s", metric_key, sample_key)
                keys = hfio.get_keys(
                    sample_key,
                )
                loaded_metrics = {}
                for key in keys:
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
                loaded_metrics_batch = {
                    # convert list of dicts to dict of lists
                    key[len(metric_key) + 1 :]: [
                        metric[key] for metric in loaded_metrics_batch
                    ]
                    for key in loaded_metrics_batch[0].keys()
                }
            else:
                loaded_metrics_batch = None

            return loaded_metrics_batch
