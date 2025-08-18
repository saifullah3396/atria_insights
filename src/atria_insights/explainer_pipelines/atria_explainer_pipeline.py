from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import cached_property
from typing import TYPE_CHECKING, Any

from atria_core.logger import get_logger
from atria_models.pipelines.atria_model_pipeline import AtriaModelPipeline
from pydantic import BaseModel, ConfigDict

from atria_insights.explainer_pipelines.utilities import _explainer_forward
from atria_insights.registry.registry_groups import (
    ExplainerBuilder,
    ExplainerMetricBuilder,
)

if TYPE_CHECKING:
    import torch
    from atria_core.types import BaseDataInstance
    from ignite.handlers import ProgressBar
    from ignite.metrics import Metric
    from torchxai.explainers import Explainer

    from atria_insights.utilities.containers import (
        ExplainerStepInputs,
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
    explainer: ExplainerBuilder | None = None
    explainer_metrics: dict[str, ExplainerMetricBuilder] | None = None
    is_multi_target: bool = False


class ExplainerPipelineConfigMixin:
    __config_cls__: type[AtriaExplainerPipelineConfig]

    def __init__(self, **kwargs):
        from atria_core.utilities.strings import _convert_to_snake_case

        config_cls = getattr(self.__class__, "__config_cls__", None)
        assert issubclass(config_cls, AtriaExplainerPipelineConfig), (
            f"{self.__class__.__name__} must define a __config_cls__ attribute "
            "that is a subclass of ModelPipelineConfig."
        )
        self._config = config_cls(**kwargs)
        if self._config.pipeline_name is None:
            self._config.pipeline_name = _convert_to_snake_case(self.__class__.__name__)
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


class AtriaExplainerPipeline(ABC, ExplainerPipelineConfigMixin):
    __config_cls__ = AtriaExplainerPipelineConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._progress_bar: ProgressBar | None = None
        self._model_pipeline: AtriaModelPipeline | None = None
        self._built_explainer: Explainer | None = None
        self._built_explainer_metrics: dict[str, Metric] | None = None

    @property
    def model_pipeline(self) -> AtriaModelPipeline:
        return self._model_pipeline

    @model_pipeline.setter
    def model_pipeline(self, value: AtriaModelPipeline) -> None:
        self._model_pipeline = value

    @property
    def metrics(self) -> dict[str, Metric] | None:
        """
        Returns the metrics defined in the explainer pipeline configuration.
        """
        return self._built_explainer_metrics

    @property
    def explainer(self) -> Explainer | None:
        """
        Returns the explainer instance built from the configuration.
        """
        return self._built_explainer

    def attach_progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Attach a progress bar to the explainer pipeline.

        Args:
            progress_bar (ProgressBar): The progress bar to attach.
        """
        self._progress_bar = progress_bar
        for key, value in self._built_explainer_metrics.items():
            value._progress_bar = progress_bar

    def build(
        self,
        model_pipeline: AtriaModelPipeline,
        device: str | torch.device | None = "cpu",
    ) -> None:
        """
        Build the explainer pipeline with the provided configuration.
        """
        # self._model_pipeline = self._model_pipeline.build(
        #     dataset_metadata=dataset_metadata, tb_logger=tb_logger
        # )
        from atria_insights.metrics.torchxai_metric import TorchXAIMetric

        self._model_pipeline = model_pipeline
        self._built_explainer = self.config.explainer(model=self._model_pipeline.model)
        self._built_explainer_metrics = {
            metric_name: TorchXAIMetric(
                explainer=self._built_explainer,
                metric_func=metric_builder(),
                forward_func=self._model_pipeline.model,
                device=device,
                progress_bar=self._progress_bar,
            )
            for metric_name, metric_builder in self.config.explainer_metrics.items()
        }
        logger.info(self._built_explainer_metrics)

        return self

    def explanation_step(
        self,
        batch: BaseDataInstance,
        train_baselines: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> ExplainerStepOutput:
        import torch

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
            model_outputs = self._model_pipeline.model(
                *tuple(explainer_step_inputs.model_inputs.explained_inputs.values()),
                *tuple(
                    explainer_step_inputs.model_inputs.additional_forward_kwargs.values()
                    if explainer_step_inputs.model_inputs.additional_forward_kwargs
                    else []
                ),
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
    ) -> dict[str, torch.Tensor]:
        return self._prepare_train_baselines(batch=batch)

    def _prepare_step_outputs(
        self,
        batch: BaseDataInstance,  # noqa: F821
        explainer_step_inputs: ExplainerStepInputs,
        target: torch.Tensor | list[torch.Tensor],
        model_outputs: torch.Tensor,
        explanations: dict[str, torch.Tensor],
        reduced_explanations: dict[str, torch.Tensor],
    ) -> ExplainerStepOutput:
        from atria_insights.utilities.containers import ExplainerStepOutput

        return ExplainerStepOutput(
            index=batch.index,
            sample_id=batch.sample_id,
            # sample explanation step data
            explainer_step_inputs=explainer_step_inputs,
            target=target,
            model_outputs=model_outputs,
            # explanations
            explanations=explanations,
            reduced_explanations=reduced_explanations,
        )

    @abstractmethod
    def _prepare_explainer_step_inputs(
        self, batch: BaseDataInstance
    ) -> ExplainerStepInputs:
        pass

    @abstractmethod
    def _prepare_train_baselines(self, batch: BaseDataInstance) -> torch.Tensor:
        pass

    @abstractmethod
    def _prepare_target(
        self,
        batch: BaseDataInstance,
        explainer_step_inputs: ExplainerStepInputs,
        model_outputs: torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        pass

    def _reduce_explanations(
        self,
        batch: BaseDataInstance,
        explainer_step_inputs: ExplainerStepInputs,
        explanations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return explanations

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}(
            model_pipeline={self._model_pipeline},
            explainer={self.explainer},
            explainer_metrics={self.metrics},
            is_multi_target={self.config.is_multi_target}
        )"""
