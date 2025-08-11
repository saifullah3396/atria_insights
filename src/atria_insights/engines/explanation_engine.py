from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atria_core.logger.logger import get_logger
from atria_ml import ENGINE
from atria_ml.training.engines.atria_engine import AtriaEngine

from atria_insights.engines.explanation_step import ExplanationStep
from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
)

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine
    from ignite.handlers import TensorboardLogger
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


@ENGINE.register("default_explanation_engine")
class ExplanationEngine(AtriaEngine):
    def __init__(
        self,
        test_run: bool = False,
        with_amp: bool = False,
    ):
        super().__init__(
            max_epochs=1,
            test_run=test_run,
        )
        self._with_amp = with_amp
        self._explainer_pipeline: AtriaExplainerPipeline | None = None

    def build(
        self,
        output_dir: str | Path,
        explainer_pipeline: AtriaExplainerPipeline,
        dataloader: DataLoader,
        device: str | torch.device | None = "cpu",
        train_baselines: dict[str, torch.Tensor] | torch.Tensor | None = None,
        tb_logger: TensorboardLogger | None = None,
    ) -> AtriaEngine:
        """
        Build the engine with the specified configurations.

        Args:
            output_dir (Optional[Union[str, Path]]): Directory for output files.
            model_pipeline (AtriaModelPipeline): Model pipeline to use.
            dataloader (Union[DataLoader, WebLoader]): Data loader for input data.
            device (Union[str, torch.device]): Device to run the engine on.
            tb_logger (Optional[TensorboardLogger]): Tensorboard logger for logging.

        Returns:
            AtriaEngine: The configured engine instance.
        """
        import torch
        from ignite.handlers import ProgressBar

        self._output_dir = output_dir
        self._model_pipeline = explainer_pipeline.model_pipeline
        self._explainer_pipeline = explainer_pipeline
        self._dataloader = dataloader
        self._train_baselines = train_baselines
        self._device = torch.device(device)
        self._tb_logger = tb_logger

        # move task module models to device
        self._explainer_pipeline.model_pipeline.to_device(
            self._device, sync_bn=self._sync_batchnorm
        )

        # initialize the engine step
        self._engine_step = self._setup_engine_step()

        # initialize the progress bar
        self._progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.stage}]", persist=True
        )

        # attach the progress bar to task module
        self._explainer_pipeline.attach_progress_bar(self._progress_bar)

        # initialize the Ignite engine
        self._engine = self._initialize_ignite_engine()

        return self

    def _configure_metrics(self, engine: Engine) -> None:
        from ignite.metrics.metric import RunningBatchWise

        if self._explainer_pipeline.metrics is not None:
            for (
                metric_name,
                metric,
            ) in self._explainer_pipeline.metrics.items():
                logger.info(
                    f"Attaching metrics {metric_name}={metric.metric_name} to engine [{self.__class__.__name__}]"
                )
                metric.attach(
                    engine,
                    metric_name,
                    usage=RunningBatchWise(),
                )

    def _setup_engine_step(self):
        return ExplanationStep(
            explainer_pipeline=self._explainer_pipeline,
            device=self._device,
            train_baselines=self._train_baselines,
            with_amp=self._with_amp,
        )

    def _configure_progress_bar(self, engine: Engine) -> None:
        import ignite.distributed as idist

        if idist.get_rank() == 0:
            from ignite.engine import Events

            self._progress_bar.attach(
                engine,
                event_name=Events.ITERATION_STARTED(every=self._logging.refresh_rate),
                metric_names=None,
            )

    def _configure_engine(self, engine: Engine):
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
        return engine
