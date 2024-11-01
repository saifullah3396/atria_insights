from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import ignite.distributed as idist
import torch
import webdataset as wds
from atria._core.training.configs.logging_config import LoggingConfig
from atria._core.training.engines.atria_engine import AtriaEngine
from atria._core.utilities.common import _validate_partial_class
from atria._core.utilities.logging import get_logger
from ignite.engine import Engine
from ignite.handlers import TensorboardLogger
from ignite.metrics import Metric
from insightx.engines.explanation_results_saver import ExplanationResultsSaver
from insightx.engines.explanation_step import ExplanationStep
from insightx.task_modules.explanation_task_module import ExplanationTaskModule
from torch.utils.data import DataLoader
from torchxai.explainers.explainer import Explainer

logger = get_logger(__name__)


class ExplanationEngine(AtriaEngine):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        task_module: ExplanationTaskModule,
        explainer: partial[Explainer],
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        engine_step: partial[ExplanationStep],
        tb_logger: Optional[TensorboardLogger] = None,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(logging_steps=1, refresh_rate=1),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        test_run: bool = False,
        is_single_output: bool = True,
    ):
        _validate_partial_class(engine_step, ExplanationStep, "engine_step")
        self._explainer = explainer
        self._is_single_output = is_single_output
        self._progress_bar = None
        self._explanation_results_saver = None
        super().__init__(
            output_dir=output_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
        )

    def _initialize_components(self):
        from atria._core.training.utilities.progress_bar import TqdmToLogger
        from ignite.handlers import ProgressBar

        # initialize the engine step
        self._engine_step = self._engine_step(
            task_module=self._task_module,
            explainer=self._explainer,
            device=self._device,
        )

        # initialize the output saver
        self._explanation_results_saver = ExplanationResultsSaver(
            output_file_path=Path(self._output_dir)
            / f"{self._explainer.func.__name__}.h5",
            is_single_output=self._is_single_output,
        )

        # create progress bar for this engine
        self._progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.stage}]",
            persist=True,
            file=TqdmToLogger(get_logger()),  # main logger causes problems here
        )

        # attach the progress bar to task module
        self._task_module.attach_progress_bar(self._progress_bar)

        # initialize the metrics to the required device
        if self._metrics is not None:
            self._metrics = {
                key: metric(
                    attached_name=key,
                    forward_func=self._task_module.torch_model,
                    explainer=self._explainer,
                    progress_bar=self._progress_bar,
                )
                for key, metric in self._metrics.items()
            }

    def _configure_metrics(self, engine: Engine) -> None:
        from ignite.metrics.metric import RunningBatchWise

        if self._metrics is not None:
            for metric_name, metric in self._metrics.items():
                logger.info(
                    f"Attaching metrics {metric_name}={metric.metric_name} to engine [{self.__class__.__name__}]"
                )
                metric.attach(
                    engine,
                    metric_name,
                    usage=RunningBatchWise(),
                )

    def _configure_progress_bar(self, engine: Engine) -> None:
        if idist.get_rank() == 0:
            from ignite.engine import Events

            self._progress_bar.attach(
                engine,
                event_name=Events.ITERATION_STARTED(every=self._logging.refresh_rate),
                metric_names=None,
            )

    def _configure_tb_logger(self, engine: Engine):
        pass  # no need to configure tb logger for explanation engine as we save outputs using ExplanationResultsSaver

    def _configure_explanation_output_saver(
        self, engine: Engine, output_dir: str
    ) -> None:
        from ignite.engine import Events

        logger.info(
            f"Attaching explanation output saver to engine [{self.__class__.__name__}]"
        )

        if idist.get_rank() == 0:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, self._explanation_results_saver
            )

    def _configure_engine(
        self, engine: Engine, output_dir: Optional[Union[str, Path]] = None
    ):
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
        self._configure_explanation_output_saver(engine=engine, output_dir=output_dir)
