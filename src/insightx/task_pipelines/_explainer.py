"""
This module defines the `ModelInferencer` class, which is responsible for evaluating a machine learning model
using a specified data pipeline, task module, and test engine. The class is designed to be registered as a
task runner in the `atria` framework and provides functionality for initializing and running the evaluation process.

Classes:
    - ModelInferencer: A task runner for evaluating models.

Dependencies:
    - ignite.distributed: For distributed training and device management.
    - atria.data.data_pipelines.default_pipeline.DefaultDataPipeline: For handling data pipelines.
    - atria.logger.logger.get_logger: For logging.
    - atria.models.model_pipelines.atria_model_pipeline.AtriaTaskModule: For task-specific model handling.
    - atria.registry.TASK_RUNNER: For registering the task runner.
    - atria.training.engines.evaluation.InferenceEngine: For running the evaluation process.
    - atria.training.utilities.constants.TrainingStage: For defining training stages.
    - atria.training.utilities.initialization: For initializing PyTorch and TensorBoard.
    - atria.utilities.common._msg_with_separator: For formatting log messages.

Usage:
    The `ModelInferencer` class is registered as a task runner and can be instantiated with the required
    components (data pipeline, task module, and test engine). It provides a `run` method to execute the
    evaluation process and an `_initialize` method for setting up the necessary components.
"""

import logging
from functools import partial
from typing import Any, Callable, ClassVar, Dict, Iterator, Optional, Type

from atria.data.pipelines.utilities import (
    auto_dataloader,
    default_collate,
    mmdet_pseudo_collate,
)
from atria.models.pipelines.atria_model_pipeline import AtriaModelPipeline
from atria.registry import TASK_PIPELINE
from atria.training.engines.evaluation import InferenceEngine
from atria_core.logger.logger import get_logger
from insightx.engines.explanation_engine import ExplanationEngine

logger = get_logger(__name__)


@TASK_PIPELINE.register(
    "explainer",
    zen_meta=dict(
        hydra={
            "run": {
                "dir": "outputs/explainer/${dataset.name}/${model_pipeline.model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
            },
            "output_subdir": "hydra",
            "job": {"chdir": False},
            "searchpath": [
                "pkg://atria/conf",
                "pkg://atria_examples/conf",
            ],
        },
        n_devices=1,
        backend="nccl",
    ),
    is_global_package=True,
)
class Explainer:
    _REGISTRY_CONFIGS: ClassVar[Type[dict]] = dict(
        image_classification=dict(
            hydra_defaults=[
                {"/model_pipeline@model_pipeline": "image_classification"},
                {"/metric_factory@metric_factory.accuracy": "accuracy"},
                {"/metric_factory@metric_factory.precision": "precision"},
                {"/metric_factory@metric_factory.recall": "recall"},
                {"/metric_factory@metric_factory.f1_score": "f1_score"},
                {"/engine@explanation_engine": "default_explanation_engine"},
                {"/data_transform@runtime_transforms": "image_transform/default"},
                "_self_",
            ],
            allowed_keys=["id", "index", "doc_id", "image", "ground_truth"],
        ),
        sequence_classification=dict(
            hydra_defaults=[
                {"/model_pipeline@model_pipeline": "sequence_classification"},
                {"/metric_factory@metric_factory.accuracy": "accuracy"},
                {"/metric_factory@metric_factory.precision": "precision"},
                {"/metric_factory@metric_factory.recall": "recall"},
                {"/metric_factory@metric_factory.f1_score": "f1_score"},
                {"/engine@explanation_engine": "default_explanation_engine"},
                {
                    "/data_transform@runtime_transforms": "document_instance_tokenizer/sequence_classification"
                },
                "_self_",
            ],
            allowed_keys=["id", "index", "doc_id", "image", "ground_truth"],
        ),
        semantic_entity_recognition=dict(
            hydra_defaults=[
                {"/model_pipeline@model_pipeline": "token_classification"},
                {"/metric_factory@metric_factory.accuracy": "seqeval_accuracy_score"},
                {"/metric_factory@metric_factory.precision": "seqeval_precision_score"},
                {"/metric_factory@metric_factory.recall": "seqeval_recall_score"},
                {"/metric_factory@metric_factory.f1_score": "seqeval_f1_score"},
                {
                    "/metric_factory@metric_factory.classification_report": "seqeval_classification_report"
                },
                {"/engine@explanation_engine": "default_explanation_engine"},
                {
                    "/data_transform@runtime_transforms": "document_instance_tokenizer/semantic_entity_recognition"
                },
                "_self_",
            ],
            allowed_keys=["id", "index", "doc_id", "image", "ground_truth"],
        ),
        layout_entity_recognition=dict(
            hydra_defaults=[
                {"/model_pipeline@model_pipeline": "layout_token_classification"},
                {"/metric_factory@metric_factory.precision": "layout_precision"},
                {"/metric_factory@metric_factory.recall": "layout_recall"},
                {"/metric_factory@metric_factory.f1_score": "layout_f1"},
                {"/engine@explanation_engine": "default_explanation_engine"},
                {
                    "/data_transform@runtime_transforms": "document_instance_tokenizer/semantic_entity_recognition"
                },
                "_self_",
            ],
            allowed_keys=["id", "index", "doc_id", "image", "ground_truth"],
        ),
        visual_question_answering=dict(
            hydra_defaults=[
                {"/model_pipeline@model_pipeline": "question_answering"},
                {"/metric_factory@metric_factory.sequence_anls": "sequence_anls"},
                {"/engine@explanation_engine": "default_explanation_engine"},
                {
                    "/data_transform@runtime_transforms": "document_instance_tokenizer/visual_question_answering"
                },
                "_self_",
            ],
            allowed_keys=["id", "index", "doc_id", "image", "ground_truth"],
        ),
        layout_analysis=dict(
            hydra_defaults=[
                {"/model_pipeline@model_pipeline": "object_detection"},
                {"/metric_factory@metric_factory.cocoeval": "cocoeval"},
                {"/engine@explanation_engine": "default_explanation_engine"},
                {
                    "/data_transform@runtime_transforms": "document_instance_mmdet_transform/evaluation"
                },
                "_self_",
            ],
            collate_fn="mmdet_pseudo_collate",
            allowed_keys=["id", "index", "doc_id", "image", "ground_truth"],
        ),
    )
    """
    A task runner for evaluating machine learning models.

    Args:
        model_pipeline (partial[AtriaTaskModule]): A partially initialized task module for handling the model.
        explanation_engine (partial[InferenceEngine]): A partially initialized test engine for running the evaluation.
        runtime_transforms (DataTransformsDict): A dictionary of data transforms for runtime evaluation.
        seed (int): The random seed for reproducibility.
        deterministic (bool): Whether to use deterministic algorithms.
        allowed_keys (set): A set of allowed keys for the data pipeline.
        batch_size (int): The batch size for data loading.
        num_workers (int): The number of workers for data loading.
        collate_fn (str): The collate function to use for data loading.


    Attributes:
        _model_pipeline (partial[AtriaTaskModule]): A partially initialized task module for handling the model.
        _explanation_engine (partial[InferenceEngine]): A partially initialized test engine for running the evaluation.
        _runtime_transforms (DataTransformsDict): A dictionary of data transforms for runtime evaluation.
        _seed (int): The random seed for reproducibility.
        _deterministic (bool): Whether to use deterministic algorithms.
        _allowed_keys (set): A set of allowed keys for the data pipeline.
        _batch_size (int): The batch size for data loading.
        _num_workers (int): The number of workers for data loading.
        _collate_fn (str): The collate function to use for data loading.
    """

    def __init__(
        self,
        model_pipeline: AtriaModelPipeline,
        explanation_engine: ExplanationEngine,
        metric_factory: Optional[Dict[str, partial[Callable]]] = None,
        runtime_transforms: Callable = None,
        seed: int = 42,
        deterministic: bool = False,
        allowed_keys: Optional[set] = None,
        batch_size: int = 8,
        num_workers: int = 8,
        collate_fn: str = "default_collate",
    ):
        self._runtime_transforms = runtime_transforms
        self._model_pipeline = model_pipeline
        self._explanation_engine = explanation_engine
        self._metric_factory = metric_factory
        self._seed = seed
        self._deterministic = deterministic
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._allowed_keys = allowed_keys

        assert collate_fn in ["default_collate", "mmdet_pseudo_collate"], (
            f"collate_fn must be one of ['collate_fn', 'mmdet_pseudo_collate'], "
            f"but got {collate_fn}"
        )
        if collate_fn == "default_collate":
            self._collate_fn = default_collate
        elif collate_fn == "mmdet_pseudo_collate":
            self._collate_fn = mmdet_pseudo_collate

    # ---------------------------
    # Public Methods
    # ---------------------------

    @property
    def model_pipeline(self) -> AtriaModelPipeline:
        """
        Returns the model pipeline.
        """
        return self._model_pipeline

    def build(
        self,
        model_checkpoint: Optional[Dict[str, Any]] = None,
        compute_metrics: bool = True,
    ) -> None:
        """
        Initializes the components required for evaluation, including logging, data pipeline, task module,
        and test engine.
        """
        self._initialize()
        self._build_model_pipeline(
            model_checkpoint=model_checkpoint,
        )
        self._build_explanation_engine(compute_metrics=compute_metrics)
        return self

    def run(self, dataset: Iterator) -> None:
        """
        Executes the evaluation process by running the test engine.
        """

        data_loader = auto_dataloader(
            dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
        return self._explanation_engine.run(data_loader)

    # ---------------------------
    # Private Methods
    # ---------------------------

    def _initialize(self):
        import ignite.distributed as idist

        from atria.training.utilities.torch_utils import _initialize_torch

        # Print log configuration
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(
                    f"Verbose logs can be found at file: {handler.baseFilename}"
                )

        # Initialize PyTorch
        _initialize_torch(
            seed=self._seed,
            deterministic=self._deterministic,
        )

        # Initialize the device (CPU or GPU)
        self._device = idist.device()

    def _build_model_pipeline(
        self,
        model_checkpoint: Optional[Dict[str, Any]] = None,
    ) -> AtriaModelPipeline:
        """
        Builds the model pipeline for the evaluation process.
        """

        # Set up the model pipeline
        if model_checkpoint is not None:
            from ignite.handlers import Checkpoint

            self._model_pipeline: AtriaModelPipeline = self._model_pipeline.build(
                dataset_metadata=model_checkpoint.get("dataset_metadata")
            )
            try:
                Checkpoint.load_objects(
                    to_load={
                        "model_pipeline": self._model_pipeline,
                    },
                    checkpoint=model_checkpoint,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model checkpoint: {e}")
        else:
            self._model_pipeline = self._model_pipeline.build()

    def _build_explanation_engine(
        self, compute_metrics: bool = True
    ) -> InferenceEngine:
        """
        Builds the test engine for the evaluation process.
        """

        # Initialize the test engine from the partial
        logger.info("Setting up inference engine")
        self._explanation_engine = self._explanation_engine.build(
            model_pipeline=self._model_pipeline,
            device=self._device,
            metric_factory=self._metric_factory if compute_metrics else None,
        )
