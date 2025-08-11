from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple, Union

import torch
from atria_core.logger.logger import get_logger
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.engine_steps.base import BaseEngineStep
from ignite.engine import Engine

from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
)

if TYPE_CHECKING:
    import torch
    from ignite.engine import Engine
    from ignite.handlers import TensorboardLogger
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


class ExplanationStep(BaseEngineStep):
    def __init__(
        self,
        explainer_pipeline: AtriaExplainerPipeline,
        device: Union[str, torch.device],
        train_baselines: Dict[str, torch.Tensor],
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
    ):
        self._explainer_pipeline = explainer_pipeline
        self._device = torch.device(device)
        self._train_baselines = train_baselines
        self._non_blocking_tensor_conv = non_blocking_tensor_conv
        self._with_amp = with_amp

    @property
    def stage(self) -> str:
        return "Explain"

    def __call__(
        self, engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        import torch
        from torch.cuda.amp import autocast

        # ready model for evaluation
        self._explainer_pipeline.model_pipeline.eval()
        if self._with_amp:
            self._explainer_pipeline.model_pipeline.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                if hasattr(batch, "to_device"):
                    batch = batch.to_device(self._device)
                return self._explainer_pipeline.explanation_step(
                    batch=batch,
                    train_baselines=self._train_baselines,
                    explanation_engine=engine,
                )


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
        device: str | torch.device,
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
        self._explainer_pipeline = explainer_pipeline
        self._dataloader = dataloader
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

    def _setup_engine_step(self):
        return ExplanationStep(
            explainer_pipeline=self._explainer_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )
