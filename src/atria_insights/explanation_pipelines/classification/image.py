from typing import Dict

import torch
from atria_core.logger.logger import get_logger
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.types.data_instance.document_instance import DocumentInstance
from atria_core.types.data_instance.image_instance import ImageInstance
from atria_models.pipelines.classification.image import ImageClassificationPipeline

from atria_insights.explanation_pipelines.atria_explanation_pipeline import (
    AtriaExplanationPipeline,
)
from atria_insights.explanation_pipelines.utilities import _get_first_layer
from atria_insights.utilities.containers import ExplanationStepInputs, ModelInputs
from atria_insights_old.model_explainability_wrappers.utils import (
    _create_segmentation_fn,
)

logger = get_logger(__name__)


class ImageClassificationExplanationPipeline(AtriaExplanationPipeline):
    def __init__(
        self,
        model_pipeline: ImageClassificationPipeline,
        segmentation_fn: str = "grid",
        is_multi_target: bool = False,
    ):
        assert isinstance(model_pipeline, ImageClassificationPipeline), (
            "Model pipeline must be an instance of ImageClassificationPipeline"
        )
        super().__init__(model_pipeline=model_pipeline, is_multi_target=is_multi_target)
        self._segmentation_fn = _create_segmentation_fn(
            segmentation_type=segmentation_fn
        )

    def _prepare_explanation_step_inputs(
        self, batch: ImageInstance | DocumentInstance
    ) -> ExplanationStepInputs:
        return ExplanationStepInputs(
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
        explainer_args: ExplanationStepInputs,
        model_outputs: torch.Tensor,
    ):
        return model_outputs.argmax(dim=-1)

    def reduce_explanations(
        self,
        batch: BaseDataInstance,
        explainer_args: ExplanationStepInputs,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {k: explanation.sum(dim=1) for k, explanation in explanations.items()}
