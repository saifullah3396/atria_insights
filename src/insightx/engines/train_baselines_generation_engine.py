from functools import partial
from pathlib import Path
from typing import Optional, Union

import ignite.distributed as idist
import torch
import webdataset as wds
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.engines.atria_engine import AtriaEngine
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from insightx.engines.train_baselines_generation_step import (
    TrainBaselinesGenerationStep,
)
from insightx.task_modules.explanation_task_module import ExplanationTaskModule
from insightx.utilities.h5io import HFDataset

logger = get_logger(__name__)


class TrainBaselinesGenerationEngine(AtriaEngine):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        task_module: ExplanationTaskModule,
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        max_train_baselines: int = 100,
        test_run: bool = False,
    ):
        self._max_baseline_samples = max_train_baselines
        self._train_baselines = []
        super().__init__(
            output_dir=output_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=partial(TrainBaselinesGenerationStep),
            device=device,
            tb_logger=None,
            max_epochs=1,
            epoch_length=round(max_train_baselines / dataloader.batch_size),
            outputs_to_running_avg=[],
            logging=LoggingConfig(logging_steps=1, refresh_rate=1),
            metrics=None,
            metric_logging_prefix=None,
            test_run=test_run,
        )

    def _configure_engine(self, engine: Engine):
        from ignite.engine import Events

        super()._configure_engine(engine=engine)

        logger.info(
            f"Attaching train baselines saver to engine [{self.__class__.__name__}]"
        )

        def save_train_baselines(engine: Engine):
            self._train_baselines.append(engine.state.output)

        def finalize_train_baselines(engine: Engine):
            # convert list of dict to dict of list
            self._train_baselines = {
                k: torch.cat([d[k] for d in self._train_baselines], dim=0)
                for k in self._train_baselines[0]
            }
            train_baselines_file_path = Path(self._output_dir) / f"train_baselines.h5"
            logger.info(f"Saving train baselines at [{train_baselines_file_path}]")
            if not train_baselines_file_path.parent.exists():
                train_baselines_file_path.parent.mkdir(parents=True, exist_ok=True)
            with HFDataset(train_baselines_file_path, "w") as hf_dataset:
                for key, data in self._train_baselines.items():
                    hf_dataset.save(key, data)

        if idist.get_rank() == 0:
            engine.add_event_handler(Events.ITERATION_COMPLETED, save_train_baselines)
            engine.add_event_handler(Events.EPOCH_COMPLETED, finalize_train_baselines)

    def run(self):
        train_baselines_file_path = Path(self._output_dir) / f"train_baselines.h5"
        if train_baselines_file_path.exists():
            logger.info(
                f"Train baselines already generated and saved at [{train_baselines_file_path}]. Loading the saved baselines."
            )
            with HFDataset(train_baselines_file_path, "r") as hfio:
                self._train_baselines = hfio.load()
        else:
            super().run()
        return convert_tensor(self._train_baselines, device=self._device)
