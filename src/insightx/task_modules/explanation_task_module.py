from abc import ABCMeta
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.utilities.common import _get_possible_args, _get_required_args
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.utilities import _get_model_forward_fn
from insightx.utilities.containers import ExplainerArguments

logger = get_logger(__name__)


class ExplanationTaskModule(AtriaTaskModule, metaclass=ABCMeta):
    def __init__(
        self,
        model_explainability_wrapper: partial[ModelExplainabilityWrapper],
        is_multi_target: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_explainability_wrapper = model_explainability_wrapper
        self._is_multi_target = is_multi_target
        self._torch_model: ModelExplainabilityWrapper

    def _build_model(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ):
        model = super()._build_model(checkpoint=checkpoint)
        model = self._model_explainability_wrapper(model)
        return model

    def _filter_batch_keys_for_model_forward(self, batch):
        valid_params = set(
            _get_possible_args(_get_model_forward_fn(self._torch_model)).keys()
        )
        filtered_batch = {k: v for k, v in batch.items() if k in valid_params}
        return filtered_batch

    def _validate_batch_keys_for_model_forward(self, batch: BatchDict):
        if self._batch_validated:
            return
        possible_model_args = _get_possible_args(
            _get_model_forward_fn(self._torch_model)
        )
        required_model_args = _get_required_args(
            _get_model_forward_fn(self._torch_model)
        )
        for required_arg in required_model_args:
            assert (
                required_arg not in batch
            ), f"Required argument '{required_arg}' is missing in the batch, batch keys: {list(batch.keys())}"
        for key in batch.keys():
            assert key in possible_model_args, (
                f"Key '{key}' is not a valid argument for the model = {self._torch_model.__class__.__name__}, "
                f"possible arguments are: {list(possible_model_args.keys())}"
            )
        self._batch_validated = True

    def toggle_explainability(
        self, convert_model_to_explainable: bool, convert_model_to_original: bool
    ):
        self._torch_model.toggle_explainability(
            convert_model_to_explainable=convert_model_to_explainable,
            convert_model_to_original=convert_model_to_original,
        )

    def prepare_explainer_arguments(self, batch: BatchDict) -> ExplainerArguments:
        raise NotImplementedError(
            "prepare_explainer_arguments method must be implemented"
        )

    def prepare_train_baselines(self, batch: BatchDict) -> torch.Tensor:
        raise NotImplementedError("prepare_train_baselines method must be implemented")

    def prepare_target(
        self,
        batch: BatchDict,
        explainer_args: ExplainerArguments,
        model_outputs: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError("prepare_target method must be implemented")

    def reduce_explanations(
        self,
        batch: BatchDict,
        explainer_args: ExplainerArguments,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return explanations
