from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, OrderedDict, Union

from atria_core.logger.logger import get_logger

from atria_insights.explainer_pipelines.utilities import _explainer_forward

if TYPE_CHECKING:
    from functools import partial

    import torch
    from atria_core.types.data_instance.base import BaseDataInstance
    from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
    from ignite.handlers import ProgressBar
    from torchxai.explainers import Explainer

    from atria_insights.utilities.containers import (
        ExplainerInputs,
        ExplainerStepOutput,
    )

logger = get_logger(__name__)


class AtriaExplainerPipeline:
    def __init__(
        self,
        model_pipeline: AtriaModelPipeline,
        is_multi_target: bool = False,
    ):
        self._model_pipeline = model_pipeline
        self._is_multi_target = is_multi_target
        self._progress_bar = None

    def attach_progress_bar(self, progress_bar: ProgressBar | None) -> None:
        self._progress_bar = progress_bar

    @property
    def model_pipeline(self) -> AtriaModelPipeline:
        return self._model_pipeline

    @staticmethod
    def from_model_pipeline(
        model_pipeline: AtriaModelPipeline,
    ) -> AtriaExplainerPipeline:
        """
        Factory method to create an instance of AtriaExplanationPipeline from a model pipeline.
        """

        from atria_models.pipelines.classification.image import (
            ImageClassificationPipeline,
        )

        from atria_insights.explainer_pipelines.image_classification import (
            ImageClassificationExplainerPipeline,
        )

        if isinstance(model_pipeline, ImageClassificationPipeline):
            return ImageClassificationExplainerPipeline(model_pipeline)
        else:
            raise NotImplementedError(
                f"Explanation pipeline for {type(model_pipeline)} is not implemented."
            )

    def explanation_step(
        self,
        batch: BaseDataInstance,
        explainer: partial[Explainer],
        train_baselines: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> ExplainerStepOutput:
        # prepare inputs for explanation
        with torch.no_grad():
            explainer_step_inputs = self._prepare_explainer_step_inputs(batch=batch)

            # convert dict to ordered dict
            if train_baselines is not None:
                train_baselines = OrderedDict(
                    {
                        key: train_baselines[key]
                        for key in explainer_step_inputs.inputs.keys()
                    }
                )

            # model forward
            model_outputs = self.model_pipeline.model(
                *tuple(explainer_step_inputs.inputs.values()),
                *tuple(explainer_step_inputs.additional_forward_kwargs.values()),
            )

            # prepare target
            target = self._prepare_target(
                batch=batch,
                explainer_step_inputs=explainer_step_inputs,
                model_outputs=model_outputs,
            )

        explanations = _explainer_forward(
            explainer=explainer,
            explainer_step_inputs=explainer_step_inputs,
            target=target,
            train_baselines=train_baselines,
            model_outputs=model_outputs,
        )

        # reduce explanations
        reduced_explanations = self._reduce_explanations(
            batch=batch,
            explainer_step_inputs=explainer_step_inputs,
            explanations=explanations,
        )

        # perform explainer forward
        explanations, reduced_explanations = self._explainer_forward(
            batch=batch,
            explainer=explainer,
            explainer_args=explainer_step_inputs,
            target=target,
            train_baselines=train_baselines,
            model_outputs=model_outputs,
        )

        # prepare step outputs
        return self._prepare_step_outputs(
            batch=batch,
            explainer_step_inputs=explainer_step_inputs,
            target=target,
            model_outputs=model_outputs,
            explanations=explanations,
            reduced_explanations=reduced_explanations,
        )

    def train_baselines_generation_step(
        self, batch: BaseDataInstance
    ) -> Dict[str, torch.Tensor]:
        return self._prepare_train_baselines(batch=batch)

    def _prepare_step_outputs(
        self,
        batch: BaseDataInstance,  # noqa: F821
        explainer_step_inputs: ExplainerInputs,
        target: Union[torch.Tensor, List[torch.Tensor]],
        model_outputs: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        reduced_explanations: Dict[str, torch.Tensor],
    ) -> ExplainerStepOutput:
        return ExplainerStepOutput(
            index=batch.index,
            sample_id=batch.sample_id,
            # sample explanation step data
            feature_masks=explainer_step_inputs.feature_masks,
            baselines=explainer_step_inputs.baselines,
            target=target,
            model_outputs=model_outputs,
            total_features=explainer_step_inputs.total_features,
            frozen_features=explainer_step_inputs.frozen_features,
            # explanations
            explanations=explanations,
            reduced_explanations=reduced_explanations,
        )

    @abstractmethod
    def _prepare_explainer_step_inputs(
        self, batch: BaseDataInstance
    ) -> ExplainerInputs:
        pass

    @abstractmethod
    def _prepare_train_baselines(self, batch: BaseDataInstance) -> torch.Tensor:
        pass

    @abstractmethod
    def _prepare_target(
        self,
        batch: BaseDataInstance,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    def _reduce_explanations(
        self,
        batch: BaseDataInstance,
        explainer_args: ExplainerInputs,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return explanations
