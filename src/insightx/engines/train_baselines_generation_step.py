from typing import Sequence, Tuple, Union

import torch
from atria.core.models.utilities.common import _validate_keys_in_batch
from atria.core.training.engines.engine_steps.evaluation import EvaluationStep
from atria.core.utilities.logging import get_logger
from dacite import Any
from ignite.engine import Engine
from insightx.task_modules.explanation_task_module import ExplanationTaskModule

logger = get_logger(__name__)


class TrainBaselinesGenerationStep(EvaluationStep):
    @property
    def stage(self) -> str:
        return "TrainBaselinesGeneration"

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        self._task_module: ExplanationTaskModule

        # validate model is built
        self._task_module.validate_model_built()

        # validate batch keys
        _validate_keys_in_batch(
            keys=self._task_module.required_keys_in_batch(stage=self.stage), batch=batch
        )

        return self._task_module.prepare_train_baselines(
            engine=engine,
            batch=batch,
            stage=self.stage,
        )
