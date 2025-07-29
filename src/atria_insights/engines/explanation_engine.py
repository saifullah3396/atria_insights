from functools import partial
from typing import Any, Dict, Sequence, Tuple, Union

import torch
from atria_core.logger.logger import get_logger
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.engine_steps.base import BaseEngineStep
from ignite.engine import Engine
from torchxai.explainers.explainer import Explainer

from atria_insights.explanation_pipelines.atria_explanation_pipeline import (
    AtriaExplanationPipeline,
)

logger = get_logger(__name__)


class ExplanationStep(BaseEngineStep):
    def __init__(
        self,
        expl_pipeline: AtriaExplanationPipeline,
        explainer: partial[Explainer],
        device: Union[str, torch.device],
        train_baselines: Dict[str, torch.Tensor],
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
    ):
        self._expl_pipeline = expl_pipeline
        self._explainer = explainer
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
        self._expl_pipeline.model_pipeline.eval()
        if self._with_amp:
            self._expl_pipeline.model_pipeline.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                if hasattr(batch, "to_device"):
                    batch = batch.to_device(self._device)
                return self._expl_pipeline.explanation_step(
                    batch=batch,
                    explainer=self._explainer,
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

    def _setup_engine_step(self):
        self._expl_pipeline = AtriaExplanationPipeline.from_model_pipeline(
            model_pipeline=self._model_pipeline,
        )
        return ExplanationStep(
            expl_pipeline=self._expl_pipeline,
            device=self._device,
            with_amp=self._with_amp,
            test_run=self._test_run,
        )
