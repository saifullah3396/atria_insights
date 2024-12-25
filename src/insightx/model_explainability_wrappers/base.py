from typing import Dict

import torch
from dacite import Any
from insightx.utilities.containers import ExplainerArguments
from pyparsing import abstractmethod
from torch import nn


class ModelExplainabilityWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self._model = model
        self._is_explainable = False
        self._is_output_explainable = False
        self.softmax = nn.Softmax(dim=-1)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def prepare_explainer_args(self, *args, **kwargs) -> ExplainerArguments:
        with torch.no_grad():
            inputs = self.prepare_explainable_inputs(*args, **kwargs)
            baselines = self.prepare_baselines_from_inputs(*args, **kwargs)
            metric_baselines = self.prepare_metric_baselines_from_inputs(
                *args, **kwargs
            )
            feature_masks, total_features, frozen_features = (
                self.prepare_feature_masks_from_inputs(*args, **kwargs)
            )
            feature_masks = self.expand_feature_masks_to_explainable_inputs(
                inputs, feature_masks
            )
            additional_forward_kwargs = self.prepare_additional_forward_kwargs(
                *args, **kwargs
            )
            constant_shifts = self.prepare_constant_shifts(*args, **kwargs)
            input_layer_names = self.prepare_input_layer_names()

            return ExplainerArguments(
                inputs=inputs,
                baselines=baselines,
                metric_baselines=metric_baselines,
                feature_masks=feature_masks,
                total_features=total_features,
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
    def prepare_explainable_inputs(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def prepare_baselines_from_inputs(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def prepare_metric_baselines_from_inputs(
        self, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.prepare_baselines_from_inputs(*args, **kwargs)

    @abstractmethod
    def prepare_feature_masks_from_inputs(
        self, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def prepare_additional_forward_kwargs(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def expand_feature_masks_to_explainable_inputs(
        self, *args, **kwargs
    ) -> ExplainerArguments:
        pass

    def prepare_constant_shifts(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def prepare_input_layer_names(self) -> Dict[str, str]:
        pass

    def toggle_explainability(
        self,
        convert_model_to_explainable: bool = True,
        convert_output_to_explainable: bool = False,
    ) -> Any:
        if convert_model_to_explainable:
            if not self._is_explainable:
                self.patch_forward_for_explainability()
                self._is_explainable = True
        else:
            if self._is_explainable:
                self.restore_forward()
                self._is_explainable = False

        self._is_output_explainable = convert_output_to_explainable

    def patch_forward_for_explainability(self, **kwargs) -> Any:
        pass

    def restore_forward(self, **kwargs) -> Any:
        pass

    def forward(self, *args, **kwargs):
        model_outputs = self._model(*args, **kwargs)
        logits = (
            model_outputs.logits if hasattr(model_outputs, "logits") else model_outputs
        )
        if self._is_output_explainable:
            return self.softmax(logits)
        else:
            return logits
