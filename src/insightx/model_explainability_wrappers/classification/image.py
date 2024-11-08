from typing import Callable, Dict

import torch
from dacite import Optional
from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.utilities import _get_first_layer
from insightx.utilities.containers import ExplainerArguments


class ImageClassificationExplainabilityWrapper(ModelExplainabilityWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        segmentation_fn: Optional[Callable] = None,
    ):
        super().__init__(model=model)
        self._segmentation_fn = segmentation_fn

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def _prepare_explainable_inputs(
        self, image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"image": image}

    def _prepare_baselines_from_inputs(
        self, image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {"image": torch.zeros_like(image)}

    def _prepare_feature_masks_from_inputs(
        self, image: torch.Tensor
    ) -> ExplainerArguments:
        if self._segmentation_fn is not None:
            return {"image": self._segmentation_fn(image)}

    def _expand_feature_masks_to_explainable_inputs(
        self, inputs: Dict[str, torch.Tensor], feature_masks: ExplainerArguments
    ) -> ExplainerArguments:
        return {"image": feature_masks["image"].expand_as(inputs["image"])}

    def _prepare_constant_shifts(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # constant shifts are used to shift the input image for input invarince score computation
        return {"image": torch.ones_like(image[0], device=image.device).unsqueeze(0)}

    def _prepare_input_layer_names(self) -> Dict[str, str]:
        first_layer = _get_first_layer(self._model)
        return {"image": "_model" + "." + first_layer[0]}

    def forward(self, *args, **kwargs):
        from torch.nn.functional import softmax

        return softmax(self.model(*args, **kwargs), dim=-1)
