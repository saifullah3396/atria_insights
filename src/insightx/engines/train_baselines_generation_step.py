from typing import Sequence, Tuple, Union

import torch
from atria.core.training.engines.engine_steps.evaluation import EvaluationStep
from atria.core.utilities.logging import get_logger
from dacite import Any
from ignite.engine import Engine

logger = get_logger(__name__)


class TrainBaselinesGenerationStep(EvaluationStep):
    @property
    def stage(self) -> str:
        return "TrainBaselinesGeneration"

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        # forward pass
        return self._task_module.train_baselines_generation_step(
            engine=engine,
            batch=batch,
            stage=self.stage,
        )
