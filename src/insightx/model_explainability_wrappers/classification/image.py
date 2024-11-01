from typing import Callable

import torch
from dacite import Optional

from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.utilities import _get_first_layer
from insightx.utilities.common import filter_kwargs
from insightx.utilities.containers import ExplainerInputs


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

    @filter_kwargs
    def prepare_explainable_inputs_from_inputs(
        self, image: torch.Tensor
    ) -> ExplainerInputs:
        inputs = {"image": image}
        baselines = {"image": torch.zeros_like(image)}
        if self._segmentation_fn(image) is not None:
            feature_masks = {"image": self._segmentation_fn(image)}
        else:
            feature_masks = None

        # expand feature mask to the same shape as the input, this means channel dimensions are considered
        # as a single feature group
        for key in inputs.keys():
            feature_masks[key] = feature_masks[key].expand_as(inputs[key])

        first_layer = _get_first_layer(self._model)

        # constant shifts are used to shift the input image for input invarince score computation
        constant_shifts = {
            key: torch.ones_like(image[0], device=image.device).unsqueeze(0)
            for key in inputs.keys()
        }
        input_layer_names = {"image": "_model" + "." + first_layer[0]}

        return ExplainerInputs(
            inputs=inputs,
            baselines=baselines,
            feature_masks=feature_masks,
            additional_forward_kwargs={},
            constant_shifts=constant_shifts,
            input_layer_names=input_layer_names,
        )
