from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from atria.core.models.utilities.common import _validate_keys_in_batch
from atria.core.training.engines.engine_steps.base import BaseEngineStep
from atria.core.training.utilities.progress_bar import AtriaProgressBar
from atria.core.utilities.common import _get_required_args
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine
from insightx.engines.explanation_results_cacher import ExplanationResultsCacher
from insightx.engines.utilities import _explainer_forward, _map_inputs_to_ordered_dict
from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.explanation_task_module import ExplanationTaskModule
from insightx.task_modules.utilities import _get_model_forward_fn
from insightx.utilities.containers import ExplanationStepMetadata, ExplanationStepOutput
from torchxai.explainers.explainer import Explainer

logger = get_logger(__name__)


class ExplanationStepUpdate:
    def __init__(
        self,
        engine: Engine,
        batch: Sequence[torch.Tensor],
        task_module: ExplanationTaskModule,
        explainer: partial[Explainer],
        train_baselines: Dict[str, torch.Tensor],
        explanation_results_cacher: ExplanationResultsCacher = None,
        progress_bar: Optional[AtriaProgressBar] = None,
        save_metadata_only: bool = False,
        step_update_validated: bool = True,
        device: torch.device = torch.device("cpu"),
        stage: str = "Explain",
    ):
        self._engine = engine
        self._task_module = task_module
        self._explainer = explainer
        self._batch = batch
        self._train_baselines = train_baselines
        self._explanation_results_cacher = explanation_results_cacher
        self._progress_bar = progress_bar
        self._save_metadata_only = save_metadata_only
        self._step_update_validated = step_update_validated
        self._device = device
        self._stage = stage

        # computed outputs
        self._step_metadata = None
        self._explanations = None
        self._reduced_explanations = None

    def _validate_inputs(self):
        # validate model is built
        self._task_module.validate_model_built()

        # validate batch keys
        _validate_keys_in_batch(
            keys=self._task_module.required_keys_in_batch(stage=self._stage),
            batch=self._batch,
        )

    def _prepare_explainer_arguments(self):
        # prepare input arguments for explainer
        self._explainer_args = self._task_module.prepare_explainer_arguments(
            batch=self._batch
        )

        # attach train baselines
        self._explainer_args.train_baselines = _map_inputs_to_ordered_dict(
            self._train_baselines, self._explainer_args.inputs.keys()
        )

        return self._explainer_args

    def _explainable_model_forward(self, inputs, additional_forward_kwargs):
        required_args = tuple(
            _get_required_args(_get_model_forward_fn(self._task_module._torch_model))
        )
        assert (
            tuple(inputs.keys()) + tuple(additional_forward_kwargs.keys())
            == required_args
        ), f"Explainable model forward requires {required_args}. Got {tuple(inputs.keys()) + tuple(additional_forward_kwargs.keys())}"
        model_outputs = self._task_module.torch_model(
            *tuple(inputs.values()), *tuple(additional_forward_kwargs.values())
        )
        return model_outputs

    def _prepare_explainable_model_outputs(self) -> torch.Tensor:
        assert isinstance(self._task_module.torch_model, ModelExplainabilityWrapper), (
            f"Model must be a ModelExplainabilityWrapper to perform model explainability. "
            f"Please check the model definition."
        )

        # here we just perform a forward pass to get the model outputs so we set the convert_output_to_explainable to False
        self._task_module.torch_model.toggle_explainability(
            convert_model_to_explainable=True, convert_output_to_explainable=False
        )
        self._model_outputs = self._explainable_model_forward(
            inputs=self._explainer_args.inputs,
            additional_forward_kwargs=self._explainer_args.additional_forward_kwargs,
        )
        if not self._step_update_validated:
            self._task_module.torch_model.toggle_explainability(
                convert_model_to_explainable=False, convert_output_to_explainable=False
            )
            standard_model_outputs = self._task_module._model_forward(self._batch)
            self._task_module.torch_model.toggle_explainability(
                convert_model_to_explainable=True, convert_output_to_explainable=False
            )
            assert torch.allclose(
                standard_model_outputs,
                self._model_outputs,
            ), (
                f"Model prediction changed after switching to explainable mode. "
                "Please fix the explainable model definition."
            )

    def _prepare_target(self):
        self._target = self._task_module.prepare_target(
            batch=self._batch,
            explainer_args=self._explainer_args,
            model_outputs=self._model_outputs,
        )

    def _prepare_explanation_step_metadata(self):
        # prepare inputs for explanation
        self._prepare_explainer_arguments()

        # prepare model_outputs
        self._prepare_explainable_model_outputs()

        # prepare target
        self._prepare_target()

        # prepare metadata
        self._step_metadata = ExplanationStepMetadata(
            sample_keys=self._batch["__key__"],
            explainer_args=self._explainer_args,
            target=self._target,
            model_outputs=self._model_outputs,
            dataset_labels=self._task_module._dataset_metadata.labels,
        )

    def _save_step_metadata(self):
        assert (
            self._step_metadata is not None
        ), "Step metadata must be prepared before saving."
        explainer_output = self._prepare_explanation_step_output()
        self._explanation_results_cacher.save_metadata(
            batch=self._batch,
            explanation_step_metadata=self._step_metadata,
        )
        return explainer_output

    def _save_step_outputs(self):
        assert (
            self._step_metadata is not None
            and self._explanations is not None
            and self._reduced_explanations is not None
        ), "Step metadata and outputs must be prepared before saving."
        self._explanation_results_cacher.save_results(
            batch=self._batch,
            explanation_step_output=self._prepare_explanation_step_output(),
        )

    def _prepare_explanation_step_output(self):
        return ExplanationStepOutput(
            explanations=self._explanations,
            reduced_explanations=self._reduced_explanations,
            metadata=ExplanationStepMetadata(
                sample_keys=self._batch["__key__"],
                explainer_args=self._explainer_args,
                target=self._target,
                model_outputs=self._model_outputs,
            ),
        )

    def _reorder_explanations(self, explanations: Dict[str, torch.Tensor]):
        if explanations is not None:
            return _map_inputs_to_ordered_dict(
                (explanations, self._explainer_args.inputs.keys())
            )

    def _prepare_reduced_explanations(self, explanations: Dict[str, torch.Tensor]):
        if explanations is not None:
            reduced_explanations = (
                self._task_module.reduce_explanations(
                    self._batch, self._explainer_args, explanations
                ),
            )
            return reduced_explanations

    def _prepare_explanations_from_cache(self) -> bool:
        # load from cache if available
        self._explanations, self._reduced_explanations = (
            self._explanation_results_cacher.load_explanations(self._batch["__key__"])
        )

        # reorder explanations if exists
        self._explanations = self._reorder_explanations(self._explanations)
        self._reduced_explanations = self._reorder_explanations(
            self._reduced_explanations
        )

        # prepare reduced explanations from explanations if not available
        if self._reduced_explanations is None:
            self._reduced_explanations = self._prepare_reduced_explanations(
                self._explanations
            )

        if self._explanations is not None and self._reduced_explanations is not None:
            return True
        return False

    def _initialize_explainer_instance(self):
        is_multi_target = isinstance(
            self._target, list
        )  # by default we assume multi target if target is a list
        return self._explainer(
            self._task_module.torch_model, is_multi_target=is_multi_target
        )

    def _prepare_explanations(self):
        self._task_module.torch_model.toggle_explainability(
            convert_model_to_explainable=True, convert_output_to_explainable=True
        )

        # initialize explainer
        explainer = self._initialize_explainer_instance()

        # if the progress bar is available, update the description
        if self._progress_bar is not None and self._progress_bar.pbar is not None:
            self._progress_bar.pbar.set_postfix_str(
                f"generating explanations using explainer=[{explainer.__class__.__name__}]"
            )

        # generate explanations
        self._explanations = _explainer_forward(
            explainer=explainer,
            explainer_args=self._explainer_args,
            target=self._target,
        )

        # generate reduced explanations
        self._reduced_explanations = self._task_module.reduce_explanations(
            self._batch, self._explainer_args, self._explanations
        )

    def __call__(self):
        # if the engine is skipping the batch, return the empty output
        if self._engine.state.skip_batch:
            return ExplanationStepOutput()

        # perform validation of model step and batch arguments
        if not self._step_update_validated:
            self._validate_inputs()

        # prepare metadata
        self._prepare_explanation_step_metadata()

        # save metadata only
        if self._save_metadata_only and self._explanation_results_cacher is not None:
            self._save_step_metadata()
            return self._prepare_explanation_step_output()

        # load explanations from cache if available
        if self._explanation_results_cacher is not None:
            cached_explanations_exist = self._prepare_explanations_from_cache()

            if cached_explanations_exist:
                return self._prepare_explanation_step_output()

        # perform explanations from explainer
        self._prepare_explanations()

        # save explanations to cache
        if self._explanation_results_cacher is not None:
            self._save_step_outputs()

        # set validation flag
        self._step_update_validated = True

        # return explanation step output
        return self._prepare_explanation_step_output()


class ExplanationStep(BaseEngineStep):
    def __init__(
        self,
        task_module: ExplanationTaskModule,
        explainer: partial[Explainer],
        device: Union[str, torch.device],
        train_baselines: Dict[str, torch.Tensor],
        explanation_results_cacher: ExplanationResultsCacher,
        save_metadata_only: bool,
        progress_bar: Optional[AtriaProgressBar] = None,
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
    ):
        self._task_module = task_module
        self._explainer = explainer
        self._device = torch.device(device)
        self._train_baselines = train_baselines
        self._explanation_results_cacher = explanation_results_cacher
        self._progress_bar = progress_bar
        self._save_metadata_only = save_metadata_only
        self._non_blocking_tensor_conv = non_blocking_tensor_conv
        self._with_amp = with_amp
        self._step_update_validated = False

    @property
    def stage(self) -> str:
        return "Explain"

    def __call__(
        self, engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:

        import torch
        from ignite.utils import convert_tensor
        from torch.cuda.amp import autocast

        # ready model for evaluation
        self._task_module.torch_model.eval()
        if self._with_amp:
            self._task_module.torch_model.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                try:
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            batch[key] = convert_tensor(
                                value,
                                device=self._device,
                                non_blocking=self._non_blocking_tensor_conv,
                            )

                except Exception as e:
                    logger.exception(
                        f"Unable to convert batch to device. "
                        f"Did you forget to setup a runtime_data_transforms or collate_fn?\nError: {e}"
                    )
                    exit(1)

                if (
                    self._progress_bar is not None
                    and self._progress_bar.pbar is not None
                ):
                    self._progress_bar.pbar.set_description_str(
                        f"Stage [{self.stage}]",
                    )

                return ExplanationStepUpdate(
                    engine=engine,
                    batch=batch,
                    task_module=self._task_module,
                    explainer=self._explainer,
                    train_baselines=self._train_baselines,
                    explanation_results_cacher=self._explanation_results_cacher,
                    progress_bar=self._progress_bar,
                    save_metadata_only=self._save_metadata_only,
                    step_update_validated=self._step_update_validated,
                    device=self._device,
                    stage=self.stage,
                )()
