from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import ignite.distributed as idist
import torch
import webdataset as wds
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.engines.atria_engine import AtriaEngine
from atria.core.utilities.common import _validate_partial_class
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine
from ignite.handlers import TensorboardLogger
from ignite.metrics import Metric
from torch.utils.data import DataLoader
from torchxai.explainers.explainer import Explainer

from insightx.engines.explanation_results_cacher import ExplanationResultsCacher
from insightx.engines.explanation_step import ExplanationStep
from insightx.engines.metrics_cacher import MetricsCacher
from insightx.task_modules.explanation_task_module import ExplanationTaskModule

logger = get_logger(__name__)


class ExplanationEngine(AtriaEngine):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        task_module: ExplanationTaskModule,
        explainer: partial[Explainer],
        dataloader: Union[DataLoader, wds.WebLoader],
        train_baselines: Dict[str, torch.Tensor],
        device: Union[str, torch.device],
        engine_step: partial[ExplanationStep],
        tb_logger: Optional[TensorboardLogger] = None,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(logging_steps=1, refresh_rate=1),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        test_run: bool = False,
        force_recompute: bool = False,
        cache_full_explanations: bool = False,
        cache_reduced_explanations: bool = False,
        save_metadata_only: bool = False,
        iterative_computation: bool = False,
    ):
        _validate_partial_class(engine_step, ExplanationStep, "engine_step")
        self._explainer = explainer
        self._train_baselines = train_baselines
        self._explanation_results_cacher = None
        self._metrics_cacher = None
        self._force_recompute = force_recompute
        self._cache_full_explanations = cache_full_explanations
        self._cache_reduced_explanations = cache_reduced_explanations
        self._save_metadata_only = save_metadata_only
        self._iterative_computation = iterative_computation
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
        from atria.core.training.utilities.progress_bar import (
            AtriaProgressBar,
            TqdmToLogger,
        )

        # initialize the output saver
        self._explanation_results_cacher = ExplanationResultsCacher(
            output_file_path=Path(self._output_dir)
            / self._explainer.func.__name__
            / f"{self._explainer.func.__name__}.h5",
            cache_full_explanations=self._cache_full_explanations,
            cache_reduced_explanations=self._cache_reduced_explanations,
        )

        # initialize metrics cacher
        self._metrics_cacher = MetricsCacher(
            output_file_path=Path(self._output_dir)
            / self._explainer.func.__name__
            / f"metrics.h5",
        )

        # create progress bar for this engine
        self._progress_bar = AtriaProgressBar(
            persist=True,
            file=TqdmToLogger(get_logger()),  # main logger causes problems here
        )

        # initialize the metrics to the required device
        if self._metrics is not None:
            self._metrics = {
                key: metric(
                    attached_name=key,
                    forward_func=self._task_module.torch_model,
                    explainer=self._explainer,
                    progress_bar=self._progress_bar,
                    cacher=self._metrics_cacher,
                )
                for key, metric in self._metrics.items()
            }

        # initialize the engine step
        self._engine_step = self._engine_step(
            task_module=self._task_module,
            explainer=self._explainer,
            device=self._device,
            train_baselines=self._train_baselines,
            explanation_results_cacher=self._explanation_results_cacher,
            progress_bar=self._progress_bar,
            save_metadata_only=self._save_metadata_only,
            iterative_computation=self._iterative_computation,
        )

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

    def _configure_batch_done(self, engine: Engine):
        from ignite.engine import Events

        def skip_if_batch_done(
            engine,
        ):
            if self._force_recompute:
                engine.state.skip_batch = False
                return

            metadata_exists = self._explanation_results_cacher.batch_metadata_exists(
                engine.state.batch
            )
            explanations_exists = (
                self._explanation_results_cacher.batch_explanations_exists(
                    engine.state.batch,
                    check_reduced_explanations=not self._cache_full_explanations,
                )
            )
            metrics_exist = self._metrics_cacher.metrics_exist(
                engine.state.batch, self._metrics
            )
            if self._cache_full_explanations or self._cache_reduced_explanations:
                batch_done = metadata_exists and explanations_exists and metrics_exist
            else:
                # we do not check explanations in case we don't want to save it. This means it doesn't need to
                # be computed again if metrics are done
                batch_done = metadata_exists and metrics_exist
            if batch_done:
                engine.state.skip_batch = True
            else:
                engine.state.skip_batch = False

        engine.add_event_handler(Events.ITERATION_STARTED, skip_if_batch_done)

    def _configure_engine(self, engine: Engine):
        self._configure_batch_done(engine=engine)
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
