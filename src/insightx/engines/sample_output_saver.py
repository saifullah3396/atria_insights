from __future__ import annotations

from atria._core.utilities.logging import get_logger
from ignite.engine import Engine

from insightx.utilities.h5io import HFSampleSaver

logger = get_logger(__name__)


class ModelOutputSaver:
    def __init__(
        self,
        output_file_path: HFSampleSaver,
    ) -> None:
        self._output_file_path = output_file_path

    def __call__(self, engine: Engine) -> None:
        if engine.state.output is None:
            return
        with HFSampleSaver(self._output_file_path) as hfio:
            for batch_idx in range(len(engine.state.batch["__key__"])):
                # get unique sample key
                sample_key = engine.state.batch["__key__"][batch_idx]
                for key, data in engine.state.output.items():
                    hfio.save(key, data, sample_key)
