from typing import Any

import torch
from atria_core.logger.logger import get_logger
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.data_instance.document_instance import DocumentInstance
from atria_core.types.data_instance.image_instance import ImageInstance
from atria_registry.module_builder import ModuleBuilder

from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
    AtriaExplainerPipelineConfig,
)
from atria_insights.explainer_pipelines.defaults import _METRICS_DEFAULTS
from atria_insights.explainer_pipelines.utilities import _get_first_layer
from atria_insights.registry import EXPLAINER_PIPELINE
from atria_insights.utilities.containers import ExplainerStepInputs, ModelInputs

logger = get_logger(__name__)


class ImageClassificationExplainerPipelineConfig(AtriaExplainerPipelineConfig):
    image_segmentor: ModuleBuilder | None = None


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

    def __init__(self, *args, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._built_image_segmentor = self.config.image_segmentor()

    def _prepare_explainer_step_inputs(
        self, batch: ImageInstance | DocumentInstance
    ) -> ExplainerStepInputs:
        return ExplainerStepInputs(
            model_inputs=ModelInputs(explained_inputs={"image": batch.image.content}),
            baselines={"image": torch.zeros_like(batch.image.content)},
            metric_baselines={"image": torch.zeros_like(batch.image.content)},
            feature_masks={
                "image": self._built_image_segmentor(batch.image.content).expand_as(
                    batch.image.content
                )
            },
            constant_shifts={
                "image": torch.ones_like(
                    batch.image.content[0], device=batch.image.content.device
                ).unsqueeze(0)
            },
            input_layer_names={"image": _get_first_layer(self.model_pipeline.model)[0]},
        )

    def _prepare_train_baselines(
        self, batch: ImageInstance | DocumentInstance
    ) -> torch.Tensor:
        return {"image": batch.image}

    def _prepare_target(
        self,
        batch: ImageInstance | DocumentInstance,
        explainer_step_inputs: ExplainerStepInputs,
        model_outputs: torch.Tensor,
    ):
        return model_outputs.argmax(dim=-1)

    def reduce_explanations(
        self,
        batch: BaseDataInstance,
        explainer_step_inputs: ExplainerStepInputs,
        explanations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {k: explanation.sum(dim=1) for k, explanation in explanations.items()}
