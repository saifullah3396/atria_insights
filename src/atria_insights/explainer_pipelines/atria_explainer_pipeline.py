from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, OrderedDict, Union

from atria_core.logger import get_logger
from atria_core.utilities.repr import RepresentationMixin
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from pydantic import BaseModel, ConfigDict

from atria_insights.explainer_pipelines.utilities import _explainer_forward
from atria_insights.registry.registry_groups import (
    ExplainerBuilder,
    ExplainerMetricBuilder,
)

if TYPE_CHECKING:
    import torch
    from atria_core.types import (
        BaseDataInstance,
        DatasetMetadata,
    )
    from ignite.contrib.handlers import TensorboardLogger
    from ignite.handlers import ProgressBar

    from atria_insights.utilities.containers import (
        ExplainerInputs,
        ExplainerStepOutput,
    )


logger = get_logger(__name__)


class AtriaExplainerPipelineConfig(BaseModel):
    """
    Configuration model for AtriaExplainerPipeline.

    This model is used to define the configuration parameters for the AtriaExplainerPipeline.
    It includes fields for model, checkpoint configurations, metric configurations, and runtime transforms.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    pipeline_name: str | None = None
    config_name: str = "default"
    model_pipeline: AtriaModelPipeline | None = None
    explainer: ExplainerBuilder | None = None
    explainer_metrics: dict[str, ExplainerMetricBuilder] | None = None
    is_multi_target: bool = False


class ExplainerPipelineConfigMixin:
    __config_cls__: type[AtriaExplainerPipelineConfig]

    def __init__(self, **kwargs):
        config_cls = getattr(self.__class__, "__config_cls__", None)
        assert issubclass(config_cls, AtriaExplainerPipelineConfig), (
            f"{self.__class__.__name__} must define a __config_cls__ attribute "
            "that is a subclass of ModelPipelineConfig."
        )
        self._config = config_cls(**kwargs)
        if self._config.pipeline_name is None:
            self._config.pipeline_name = self.__class__.__name__.lower()
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate presence of Config at class definition time
        if not hasattr(cls, "__config_cls__"):
            raise TypeError(
                f"{cls.__name__} must define a nested `__config_cls__` class."
            )

        if not issubclass(cls.__config_cls__, AtriaExplainerPipelineConfig):
            raise TypeError(
                f"{cls.__name__}.Config must subclass pydantic.ModelPipelineConfig. Got {cls.__config_cls__} instead."
            )

    @cached_property
    def config(self) -> AtriaExplainerPipelineConfig:
        return self._config


class AtriaExplainerPipeline(ABC, ExplainerPipelineConfigMixin, RepresentationMixin):
    __config_cls__ = AtriaExplainerPipelineConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._progress_bar: ProgressBar | None = None

    @property
    def model_pipeline(self) -> AtriaModelPipeline:
        return self._config.model_pipeline

    @model_pipeline.setter
    def model_pipeline(self, value: AtriaModelPipeline) -> None:
        self._config.model_pipeline = value

    @property
    def explainer(self) -> ExplainerBuilder | None:
        return self._config.explainer

    def attach_progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Attach a progress bar to the explainer pipeline.

        Args:
            progress_bar (ProgressBar): The progress bar to attach.
        """
        self._progress_bar = progress_bar

    def build(
        self,
        dataset_metadata: DatasetMetadata,
        tb_logger: TensorboardLogger | None = None,
    ) -> None:
        """
        Build the explainer pipeline with the provided configuration.
        """
        self.model_pipeline = self.model_pipeline.build(
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        return self

    # @staticmethod
    # def from_model_pipeline(
    #     model_pipeline: AtriaModelPipeline,
    #     explainer: ExplainerBuilder | None = None,
    #     explainer_metrics: dict[str, ExplainerMetricBuilder] | None = None,
    #     **kwargs: Any,
    # ) -> AtriaExplainerPipeline:
    #     """
    #     Factory method to create an instance of AtriaExplanationPipeline from a model pipeline.
    #     """

    #     from atria_models.pipelines.classification.image import (
    #         ImageClassificationPipeline,
    #     )

    #     from atria_insights.explainer_pipelines.classification.image import (
    #         ImageClassificationExplainerPipeline,
    #     )

    #     if isinstance(model_pipeline, ImageClassificationPipeline):
    #         return ImageClassificationExplainerPipeline(
    #             model_pipeline=model_pipeline,
    #             explainer=explainer,
    #             explainer_metric=explainer_metrics,
    #             **kwargs,
    #         )
    #     else:
    #         raise NotImplementedError(
    #             f"Explanation pipeline for {type(model_pipeline)} is not implemented."
    #         )

    def explanation_step(
        self,
        batch: BaseDataInstance,
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
            explainer=self.explainer,
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
