from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from atria_core.logger import get_logger
from atria_core.types import TaskType
from atria_datasets.pipelines.atria_data_pipeline import AtriaDataPipeline
from atria_ml.registry import TASK_PIPELINE
from atria_ml.training.engines.atria_engine import AtriaEngine
from atria_ml.training.engines.evaluation import TestEngine
from atria_ml.training.engines.utilities import RunConfig
from atria_registry.registry_config import RegistryConfig

from atria_insights.engines.explanation_engine import ExplanationEngine
from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
    AtriaExplainerPipeline,
)
from atria_insights.registry.registry_groups import (
    ExplainerBuilder,
    ExplainerMetricBuilder,
)

if TYPE_CHECKING:
    from ignite.engine import State

logger = get_logger(__name__)

DEFAULTS_SELF = ["_self_"]
DATA_PIPELINE_DEFAULT_LIST = [{"/data_pipeline@data_pipeline": "default"}]
MODEL_DEFAULT_LIST = [{"/model_pipeline@explainer_pipeline.model_pipeline": "???"}]
ENGINE_DEFAULT_LIST = [
    {
        "/engine@test_engine": "default_test_engine",
    },
    {
        "/engine@explanation_engine": "default_explanation_engine",
    },
]


@TASK_PIPELINE.register(
    "explainer",
    configs=[
        RegistryConfig(
            name=TaskType.image_classification.value,
            defaults=DEFAULTS_SELF
            + ENGINE_DEFAULT_LIST
            + DATA_PIPELINE_DEFAULT_LIST
            + [{"/explainer_pipeline@explainer_pipeline": "image_classification"}],
            output_dir="outputs/explainer/image_classification/${resolve_experiment_name:${experiment_name}}/",
        ),
    ],
    zen_meta={
        "pipeline_name": "atria_explainer",
        "n_devices": 1,
        "backend": "nccl",
        "experiment_name": "_to_be_resolved_",
    },
    zen_exclude=["hydra", "package", "version"],
    is_global_package=True,
)
class Explainer:
    def __init__(
        self,
        test_checkpoint: str,
        data_pipeline: AtriaDataPipeline,
        explainer_pipeline: AtriaExplainerPipeline,
        test_engine: TestEngine,
        explanation_engine: ExplanationEngine,
        explainer: ExplainerBuilder,
        explainer_metrics: dict[str, ExplainerMetricBuilder] | None = None,
        output_dir: str | None = None,
        do_eval: bool = True,
    ):
        self._output_dir = output_dir
        self._data_pipeline = data_pipeline
        self._explainer_pipeline = explainer_pipeline
        self._test_engine = test_engine
        self._explanation_engine = explanation_engine
        self._explainer = explainer
        self._explainer_metrics = explainer_metrics
        self._test_checkpoint = test_checkpoint
        self._do_eval = do_eval

    @property
    def explainer_pipeline(self):
        return self._explainer_pipeline

    @property
    def test_engine(self):
        return self._test_engine

    @property
    def device(self):
        return self._device

    @property
    def logger(self):
        return logger

    def _initialize_runtime(self, local_rank: int) -> None:
        from atria_ml.training.utilities.torch_utils import _initialize_torch

        # initialize training
        logger.info("Initializing torch runtime environment...")
        self._seed = _initialize_torch(
            seed=self._seed, deterministic=self._deterministic
        )

        # initialize torch device (cpu or gpu)
        self._device = local_rank

        logger.info(f"Seed set to {self._seed} on device: {self._device}")

    def _build_data_pipeline(self):
        logger.info("Setting up data pipeline")
        self._data_pipeline.build(
            runtime_transforms=self._model_pipeline.config.runtime_transforms
        )

    def _build_explainer_pipeline(self) -> None:
        # initialize the task module from partial
        logger.info("Setting up task module")
        self._explainer_pipeline = self._explainer_pipeline.build(
            dataset_metadata=self._data_pipeline.dataset_metadata,
        )

    def _build_test_engine(self) -> None:
        if self._test_engine is not None:
            logger.info("Setting up test engine")
            test_dataloader = self._data_pipeline.test_dataloader()
            self._test_engine: AtriaEngine = self._test_engine.build(
                output_dir=self._output_dir,
                model_pipeline=self._explainer_pipeline.model_pipeline,
                dataloader=test_dataloader,
                device=self._device,
            )

    def _build_explanation_engine(self) -> None:
        self._explanation_engine = self._explanation_engine.build(
            output_dir=self._output_dir,
            explainer_pipeline=self._explainer_pipeline,
            dataloader=self._data_pipeline.test_dataloader(),
            device=self._device,
        )

    def build(
        self, local_rank: int, experiment_name: str, run_config: RunConfig
    ) -> None:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(f"Log file path: {handler.baseFilename}")

        self._experiment_name = experiment_name
        self._run_config = run_config
        self._initialize_runtime(local_rank=local_rank)
        self._build_data_pipeline()
        self._build_explainer_pipeline()

    def test(self) -> dict[str, State]:
        self._build_test_engine()
        return self._test_engine.run(test_checkpoint=self._test_checkpoint)

    def explain(self):
        from atria_ml.training.utilities.torch_utils import _reset_random_seeds

        _reset_random_seeds(self._seed)
        self._build_explanation_engine()
        return self._explanation_engine.run()

    def run(self) -> dict[str, State]:
        assert self._prepare_atria_ckpt or self._do_eval, (
            "Either `prepare_atria_ckpt` or `do_eval` must be True. "
        )
        if self.do_eval:
            self.test()
        return self.explain()
