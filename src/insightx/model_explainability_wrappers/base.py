from functools import wraps
from typing import Dict

import torch
from dacite import Any
from insightx.utilities.containers import ExplainerArguments
from pyparsing import abstractmethod


def forward_wrapper(forward_func):
    @wraps(forward_func)
    def inner(*args, **kwargs):
        return forward_func(*args, **kwargs)

    return inner


class ModelExplainabilityWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self._model = model
        self._is_explainable = False

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def prepare_explainer_args(self, *args, **kwargs) -> ExplainerArguments:
        with torch.no_grad():
            inputs = self._prepare_explainable_inputs(*args, **kwargs)
            baselines = self._prepare_baselines_from_inputs(*args, **kwargs)
            feature_masks, frozen_features = self._prepare_feature_masks_from_inputs(
                *args, **kwargs
            )
            feature_masks = self._expand_feature_masks_to_explainable_inputs(
                inputs, feature_masks
            )
            additional_forward_kwargs = self._prepare_additional_forward_kwargs(
                *args, **kwargs
            )
            constant_shifts = self._prepare_constant_shifts(*args, **kwargs)
            input_layer_names = self._prepare_input_layer_names()

            return ExplainerArguments(
                inputs=inputs,
                baselines=baselines,
                feature_masks=feature_masks,
                additional_forward_kwargs=(
                    {}
                    if additional_forward_kwargs is None
                    else additional_forward_kwargs
                ),
                constant_shifts=constant_shifts,
                input_layer_names=input_layer_names,
                frozen_features=frozen_features,
            )

    @abstractmethod
    def _prepare_explainable_inputs(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _prepare_baselines_from_inputs(
        self, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _prepare_feature_masks_from_inputs(
        self, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _prepare_additional_forward_kwargs(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _expand_feature_masks_to_explainable_inputs(
        self, *args, **kwargs
    ) -> ExplainerArguments:
        pass

    def _prepare_constant_shifts(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def _prepare_input_layer_names(self) -> Dict[str, str]:
        pass

    def toggle_explainability(self, convert_to_explainable: bool = True) -> Any:
        if convert_to_explainable:
            if not self._is_explainable:
                self.patch_forward_for_explainability()
                self._is_explainable = True
        else:
            if self._is_explainable:
                self.restore_forward()
                self._is_explainable = False

    def patch_forward_for_explainability(self, **kwargs) -> Any:
        pass

    def restore_forward(self, **kwargs) -> Any:
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
