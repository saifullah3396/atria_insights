from typing import Any, Dict

import torch
from atria_core.logger.logger import get_logger
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.data_instance.document_instance import DocumentInstance
from atria_core.types.data_instance.image_instance import ImageInstance

from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
    AtriaExplainerPipelineConfig,
)
from atria_insights.explainer_pipelines.defaults import _METRICS_DEFAULTS
from atria_insights.explainer_pipelines.utilities import _get_first_layer
from atria_insights.registry import EXPLAINER_PIPELINE
from atria_insights.utilities.containers import ExplainerInputs, ModelInputs
from atria_insights.utilities.image import _create_segmentation_fn

logger = get_logger(__name__)


class ImageClassificationExplainerPipelineConfig(AtriaExplainerPipelineConfig):
    segmentation_fn: str = "grid"
    segmentation_fn_kwargs: Dict[str, Any] = {}  # noqa: F821


@EXPLAINER_PIPELINE.register(
    "image_classification",
    defaults=[
        "_self_",
        {"/model_pipeline@model_pipeline": "image_classification"},
        {"/explainer@explainer": "grad/saliency"},
    ]
    + _METRICS_DEFAULTS,
)
class ImageClassificationExplainerPipeline(AtriaExplainerPipeline):
    __config_cls__ = ImageClassificationExplainerPipelineConfig

    def __init__(
        self,
        *args,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self._segmentation_fn = _create_segmentation_fn(
            segmentation_type=self.config.segmentation_fn,
            **self.config.segmentation_fn_kwargs,
        )

    def _prepare_explanation_step_inputs(
        self, batch: ImageInstance | DocumentInstance
    ) -> ExplainerInputs:
        return ExplainerInputs(
            model_inputs=ModelInputs(
                explained_inputs={"image": batch.image},
            ),
            baselines={"image": torch.zeros_like(batch.image)},
            metric_baselines={"image": torch.zeros_like(batch.image)},
            feature_masks={"image": self._segmentation_fn(batch.image)},
            total_features=batch.image.numel(),
            constant_shifts={
                "image": torch.ones_like(
                    batch.image[0], device=batch.image.device
                ).unsqueeze(0)
            },
            input_layer_names=_get_first_layer(self._model_pipeline.model)[0],
        )

    def _prepare_train_baselines(
        self, batch: ImageInstance | DocumentInstance
    ) -> torch.Tensor:
        return {"image": batch.image}

    def _prepare_target(
        self,
        batch: ImageInstance | DocumentInstance,
        explainer_args: ExplainerInputs,
        model_outputs: torch.Tensor,
    ):
        return model_outputs.argmax(dim=-1)

    def reduce_explanations(
        self,
        batch: BaseDataInstance,
        explainer_args: ExplainerInputs,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {k: explanation.sum(dim=1) for k, explanation in explanations.items()}
