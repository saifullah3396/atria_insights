from abc import abstractmethod
from typing import Dict, Optional

import torch
from atria_core.logger.logger import get_logger
from atria_models.pipelines.classification.sequence import (
    SequenceClassificationPipeline,
)
from atria_transforms.data_types import TokenizedDocumentInstance
from torchxai.metrics._utils.common import _reduce_tensor_with_indices_non_deterministic

from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
)
from atria_insights.utilities.containers import ExplainerStepInputs

logger = get_logger(__name__)


class SequenceClassificationExplanationPipeline(AtriaExplainerPipeline):
    def __init__(
        self,
        model_pipeline: SequenceClassificationPipeline,
        is_multi_target: bool = False,
        group_tokens_to_words: bool = True,
        baselines_config: Optional[Dict[str, str]] = None,
        metric_baselines_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model_pipeline=model_pipeline, is_multi_target=is_multi_target)
        self._group_tokens_to_words = group_tokens_to_words
        self._baselines_config = baselines_config
        self._metric_baselines_config = metric_baselines_config

    @abstractmethod
    def _prepare_explainer_step_inputs(
        self, batch: TokenizedDocumentInstance
    ) -> ExplainerStepInputs:
        pass

    @abstractmethod
    def _prepare_train_baselines(
        self, batch: TokenizedDocumentInstance
    ) -> torch.Tensor:
        pass

    def _prepare_target(
        self,
        batch: TokenizedDocumentInstance,
        explanation_step_inputs: ExplainerStepInputs,
        model_outputs: torch.Tensor,
    ):
        return model_outputs.argmax(dim=-1)

    def _extract_feature_group_explanations(
        self, explanation: torch.Tensor, feature_mask: torch.Tensor
    ):
        assert explanation.shape == feature_mask.shape and explanation.shape[0] == 1
        reduced_explanation, _ = _reduce_tensor_with_indices_non_deterministic(
            explanation.flatten(), (feature_mask - feature_mask.min()).flatten()
        )
        return reduced_explanation

    def _reduce_explanations(
        self,
        _: TokenizedDocumentInstance,
        explainer_inputs: ExplainerStepInputs,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        reduced_explanations = {}
        for key in explanations.keys():
            if key in [
                "input_embeddings",
                "spatial_position_embeddings",
                "position_embeddings",
                "patch_embeddings",
            ]:
                reduced_explanations[key] = [
                    self._extract_feature_group_explanations(
                        explanations[key][batch_idx].unsqueeze(0),
                        explainer_inputs.feature_masks[key][batch_idx].unsqueeze(0),
                    )
                    for batch_idx in range(explanations[key].shape[0])
                ]
            elif key in [
                "pixel_values",
            ]:
                reduced_explanations[key] = explanations[key].sum(
                    dim=1, keepdim=True
                )  # sum across channel dim
        return reduced_explanations
