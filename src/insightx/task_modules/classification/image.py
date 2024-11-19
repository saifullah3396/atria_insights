from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.classification.image import ImageClassificationModule
from ignite.contrib.handlers import TensorboardLogger

from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.explanation_task_module import ExplanationTaskModule
from insightx.utilities.containers import ExplainerArguments

logger = get_logger(__name__)


class ImageClassificationExplanationModule(
    ExplanationTaskModule, ImageClassificationModule
):
    def __init__(
        self,
        model_explainability_wrapper: partial[ModelExplainabilityWrapper],
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint: Optional[str] = None,
        load_checkpoint_strict: bool = False,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        mixup_builder: Optional[partial] = None,
    ):
        super().__init__(
            model_explainability_wrapper=model_explainability_wrapper,
            torch_model_builder=torch_model_builder,
            checkpoint=checkpoint,
            load_checkpoint_strict=load_checkpoint_strict,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
            mixup_builder=mixup_builder,
        )
        self._explainable_model_output_validated = False

    def _build_model(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ):
        model = super(ImageClassificationModule, self)._build_model(
            checkpoint=checkpoint
        )
        model = self._model_explainability_wrapper(model)
        return model

    def _required_keys_for_explainability(self) -> List[str]:
        required_keys = super()._required_keys_for_explainability()
        required_keys += [DataKeys.IMAGE]
        return required_keys

    def _prepare_explainer_arguments(
        self, batch: BatchDict, **kwargs
    ) -> ExplainerArguments:
        # prepare inputs for explainable model
        return self.torch_model.prepare_explainer_args(image=batch[DataKeys.IMAGE])

    def _prepare_train_baselines(self, batch: BatchDict, **kwargs) -> torch.Tensor:
        # prepare inputs for explainable model
        return self.torch_model._prepare_explainable_inputs(image=batch[DataKeys.IMAGE])

    def _prepare_target(self, batch: BatchDict, explainer_args: ExplainerArguments):
        with torch.no_grad():
            if not self._explainable_model_output_validated:
                self._torch_model.toggle_explainability(False)
                standard_model_outputs = self._model_forward(batch)
                self._torch_model.toggle_explainability(True)
                explainable_model_outputs = self._explainable_model_forward(
                    inputs=explainer_args.inputs,
                    additional_forward_kwargs=explainer_args.additional_forward_kwargs,
                )
                assert torch.allclose(
                    standard_model_outputs,
                    explainable_model_outputs,
                ), (
                    f"Model prediction changed after switching to explainable mode. "
                    "Please fix the explainable model definition."
                )
                self._explainable_model_output_validated = True
                return explainable_model_outputs.argmax(dim=-1)
            else:
                explainable_model_outputs = self._explainable_model_forward(
                    inputs=explainer_args.inputs,
                    additional_forward_kwargs=explainer_args.additional_forward_kwargs,
                )
                return explainable_model_outputs.argmax(dim=-1)

    def _reduce_explanations(
        self,
        batch: BatchDict,
        explainer_args: ExplainerArguments,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {k: explanation.sum(dim=1) for k, explanation in explanations.items()}
