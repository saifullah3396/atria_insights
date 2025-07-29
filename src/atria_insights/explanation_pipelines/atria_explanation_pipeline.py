from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Dict, List, OrderedDict, Union

from atria_core.logger.logger import get_logger
from atria_core.types.data_instance.base import BaseDataInstance
from atria_core.utilities.common import _get_possible_args, _get_required_args
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from torchxai.explainers.base import Explainer

from atria_insights.engines.utilities import _explainer_forward, _get_model_forward_fn
from atria_insights.utilities.containers import (
    ExplanationStepInputs,
    ExplanationStepOutput,
)

if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


class AtriaExplanationPipeline:
    def __init__(
        self,
        model_pipeline: AtriaModelPipeline,
        is_multi_target: bool = False,
    ):
        self._model_pipeline = model_pipeline
        self._is_multi_target = is_multi_target
        self._progress_bar = None

    def attach_progress_bar(self, progress_bar):
        self._progress_bar = progress_bar

    @property
    def model_pipeline(self) -> AtriaModelPipeline:
        return self._model_pipeline

    @staticmethod
    def from_model_pipeline(
        model_pipeline: AtriaModelPipeline,
    ) -> AtriaExplanationPipeline:
        """
        Factory method to create an instance of AtriaExplanationPipeline from a model pipeline.
        """
        from atria_models.pipelines.classification.image import (
            ImageClassificationPipeline,
        )

        if isinstance(model_pipeline, ImageClassificationPipeline):
            return ImageClassificationPipeline(model_pipeline)
        else:
            raise NotImplementedError(
                f"Explanation pipeline for {type(model_pipeline)} is not implemented."
            )

    def explanation_step(
        self,
        batch: BaseDataInstance,
        explainer: partial[Explainer],
        train_baselines: Dict[str, torch.Tensor],
        **kwargs,
    ) -> ExplanationStepOutput:
        # prepare inputs for explanation
        with torch.no_grad():
            explanation_step_inputs = self._prepare_explanation_step_inputs(batch=batch)

            # convert dict to ordered dict
            train_baselines = OrderedDict(
                {
                    key: train_baselines[key]
                    for key in explanation_step_inputs.inputs.keys()
                }
            )

            # model forward
            model_outputs = self.model_pipeline.model(
                *tuple(explanation_step_inputs.inputs.values()),
                *tuple(explanation_step_inputs.additional_forward_kwargs.values()),
            )

            # prepare target
            target = self._prepare_target(
                batch=batch,
                explanation_step_inputs=explanation_step_inputs,
                model_outputs=model_outputs,
            )

        # perform explainer forward
        explanations, reduced_explanations = self._explainer_forward(
            batch=batch,
            explainer=explainer,
            explainer_args=explanation_step_inputs,
            target=target,
            model_outputs=model_outputs,
        )

        # prepare step outputs
        return self._prepare_step_outputs(
            batch=batch,
            explanation_step_inputs=explanation_step_inputs,
            target=target,
            model_outputs=model_outputs,
            explanations=explanations,
            reduced_explanations=reduced_explanations,
        )

    def train_baselines_generation_step(
        self, batch: BaseDataInstance
    ) -> Dict[str, torch.Tensor]:
        return self._prepare_train_baselines(batch=batch)

    def _prepare_explainer_input_kwargs(
        self,
        explainer: partial[Explainer],
        explainer_args: ExplanationStepInputs,
        target: Union[torch.Tensor, List[torch.Tensor]],
    ):
        possible_args = _get_possible_args(explainer.explain)
        explainer_input_kwargs = dict(
            inputs=tuple(explainer_args.inputs.values()),
            additional_forward_args=tuple(
                explainer_args.additional_forward_kwargs.values()
            ),
            target=target,
        )
        if "baselines" in possible_args:
            assert explainer_args.baselines.keys() == explainer_args.inputs.keys(), (
                f"Baselines must have the same keys as inputs. Got {explainer_args.baselines.keys()} "
            )
            explainer_input_kwargs["baselines"] = tuple(
                explainer_args.baselines.values()
            )
        if "feature_mask" in possible_args:
            assert (
                explainer_args.feature_masks.keys() == explainer_args.inputs.keys()
            ), (
                f"Feature masks must have the same keys as inputs. Got {explainer_args.feature_masks.keys()} "
            )
            explainer_input_kwargs["feature_mask"] = tuple(
                explainer_args.feature_masks.values()
            )
        if "frozen_features" in possible_args:
            assert (
                len(explainer_args.frozen_features)
                == list(explainer_args.inputs.values())[0].shape[0]
            ), (
                f"Length of frozen features must be equal to the batch size. "
                f"Got {len(explainer_args.frozen_features)} and {list(explainer_args.inputs.values())[0].shape[0]}"
            )
            explainer_input_kwargs["frozen_features"] = explainer_args.frozen_features
        if "train_baselines" in possible_args:
            assert (
                explainer_args.train_baselines.keys() == explainer_args.inputs.keys()
            ), (
                f"Train baselines must have the same keys as inputs. Got {explainer_args.train_baselines.keys()} "
            )

            for train_baselines, inputs in zip(
                explainer_args.train_baselines.values(), explainer_args.inputs.values()
            ):
                assert train_baselines.shape[1:] == inputs.shape[1:], (
                    f"Train baselines must have the same shape as inputs. Got {train_baselines.shape} and {inputs.shape}"
                )
            explainer_input_kwargs["train_baselines"] = tuple(
                explainer_args.train_baselines.values()
            )
        return explainer_input_kwargs

    def _prepare_explainer_instance(self, explainer: partial[Explainer]) -> Explainer:
        return explainer(
            self.model_pipeline.model, is_multi_target=self._is_multi_target
        )

    def _explainer_forward(
        self,
        batch: BaseDataInstance,
        explainer: partial[Explainer],
        explanation_step_inputs: ExplanationStepInputs,
        target: Union[torch.Tensor, List[torch.Tensor]],
        model_outputs: torch.Tensor,
    ):
        explainer = self._prepare_explainer_instance(explainer)

        if self._progress_bar is not None and self._progress_bar.pbar is not None:
            self._progress_bar.pbar.set_postfix_str(
                f"generating explanations using explainer=[{explainer.__class__.__name__}]"
            )

        explanations = _explainer_forward(
            explainer=explainer,
            explanation_step_inputs=explanation_step_inputs,
            target=target,
            model_outputs=model_outputs,
        )

        # reduce explanations
        reduced_explanations = self._reduce_explanations(
            batch=batch,
            explanation_step_inputs=explanation_step_inputs,
            explanations=explanations,
        )

        return explanations, reduced_explanations

    def _prepare_step_outputs(
        self,
        batch: BaseDataInstance,  # noqa: F821
        explanation_step_inputs: ExplanationStepInputs,
        target: Union[torch.Tensor, List[torch.Tensor]],
        model_outputs: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        reduced_explanations: Dict[str, torch.Tensor],
    ) -> ExplanationStepOutput:
        return ExplanationStepOutput(
            index=batch.index,
            sample_id=batch.sample_id,
            # sample explanation step data
            feature_masks=explanation_step_inputs.feature_masks,
            baselines=explanation_step_inputs.baselines,
            target=target,
            model_outputs=model_outputs,
            total_features=explanation_step_inputs.total_features,
            frozen_features=explanation_step_inputs.frozen_features,
            # explanations
            explanations=explanations,
            reduced_explanations=reduced_explanations,
        )

    def _explainable_model_forward(self, inputs, additional_forward_kwargs):
        required_args = tuple(
            _get_required_args(_get_model_forward_fn(self.model_pipeline.model))
        )
        assert (
            tuple(inputs.keys()) + tuple(additional_forward_kwargs.keys())
            == required_args
        ), (
            f"Explainable model forward requires {required_args}. Got {tuple(inputs.keys()) + tuple(additional_forward_kwargs.keys())}"
        )
        model_outputs = self.model_pipeline.model(
            *tuple(inputs.values()), *tuple(additional_forward_kwargs.values())
        )
        return model_outputs

    @abstractmethod
    def _prepare_explanation_step_inputs(
        self, batch: BaseDataInstance
    ) -> ExplanationStepInputs:
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
        explainer_args: ExplanationStepInputs,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return explanations
