from functools import partial
from typing import Any, Dict, Sequence, Tuple, Union

import torch
from atria.core.training.engines.engine_steps.base import BaseEngineStep
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine
from torchxai.explainers.explainer import Explainer

from insightx.task_modules.explanation_task_module import ExplanationTaskModule

logger = get_logger(__name__)


class ExplanationStep(BaseEngineStep):
    def __init__(
        self,
        task_module: ExplanationTaskModule,
        explainer: partial[Explainer],
        device: Union[str, torch.device],
        train_baselines: Dict[str, torch.Tensor],
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
    ):
        self._task_module = task_module
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
        from ignite.utils import convert_tensor
        from torch.cuda.amp import autocast

        # ready model for evaluation
        self._task_module.torch_model.eval()
        if self._with_amp:
            self._task_module.torch_model.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                try:
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = convert_tensor(
                                value,
                                device=self._device,
                                non_blocking=self._non_blocking_tensor_conv,
                            )

                except Exception as e:
                    logger.exception(
                        f"Unable to convert batch to device. "
                        f"Did you forget to setup a runtime_data_transforms or collate_fn?\nError: {e}"
                    )
                    exit(1)
                return self._task_module.explanation_step(
                    batch=batch,
                    explainer=self._explainer,
                    train_baselines=self._train_baselines,
                    explanation_engine=engine,
                )
