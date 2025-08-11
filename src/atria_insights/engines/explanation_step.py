from typing import Any, Dict, Sequence, Tuple, Union

import torch
from atria_core.logger.logger import get_logger
from atria_ml.training.engines.engine_steps.base import BaseEngineStep
from ignite.engine import Engine

from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
)

logger = get_logger(__name__)


class ExplanationStep(BaseEngineStep):
    def __init__(
        self,
        explainer_pipeline: AtriaExplainerPipeline,
        device: Union[str, torch.device],
        train_baselines: Dict[str, torch.Tensor],
        with_amp: bool = False,
    ):
        self._explainer_pipeline = explainer_pipeline
        self._device = torch.device(device)
        self._train_baselines = train_baselines
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
