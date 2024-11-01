from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from atria._core.models.task_modules.atria_task_module import AtriaTaskModule
from atria._core.models.utilities.common import _validate_keys_in_batch
from atria._core.utilities.common import _get_possible_args, _get_required_args
from atria._core.utilities.logging import get_logger
from atria._core.utilities.typing import BatchDict
from ignite.utils import apply_to_tensor
from torchxai.explainers.explainer import Explainer

from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.model_output_wrappers import SoftmaxWrapper
from insightx.task_modules.utilities import _get_model_forward_fn
from insightx.utilities.containers import ExplainerInputs, ExplanationModelOutput

logger = get_logger(__name__)


class ExplanationTaskModule(AtriaTaskModule, metaclass=ABCMeta):
    def __init__(
        self,
        model_explainability_wrapper: partial[ModelExplainabilityWrapper],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_explainability_wrapper = model_explainability_wrapper

    def _build_model(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ):
        model = super()._build_model(checkpoint=checkpoint)
        model = self._model_explainability_wrapper(SoftmaxWrapper(model))
        model.toggle_explainability(True)
        return model

    def _required_keys_for_explainability(self) -> List[str]:
        return []

    def _filter_batch_keys_for_model_forward(self, batch):
        valid_params = set(
            _get_possible_args(_get_model_forward_fn(self._torch_model)).keys()
        )
        filtered_batch = {k: v for k, v in batch.items() if k in valid_params}
        return filtered_batch

    def _validate_batch_keys_for_model_forward(self, batch: BatchDict):
        if self._batch_validated:
            return
        possible_model_args = _get_possible_args(
            _get_model_forward_fn(self._torch_model)
        )
        required_model_args = _get_required_args(
            _get_model_forward_fn(self._torch_model)
        )
        for required_arg in required_model_args:
            assert (
                required_arg not in batch
            ), f"Required argument '{required_arg}' is missing in the batch, batch keys: {list(batch.keys())}"
        for key in batch.keys():
            assert key in possible_model_args, (
                f"Key '{key}' is not a valid argument for the model = {self._torch_model.__class__.__name__}, "
                f"possible arguments are: {list(possible_model_args.keys())}"
            )
        self._batch_validated = True

    def _explainable_model_forward(self, inputs, additional_forward_kwargs):
        _get_model_forward_fn(self._torch_model)
        required_args = tuple(
            _get_required_args(_get_model_forward_fn(self._torch_model))
        )
        assert (
            tuple(inputs.keys()) + tuple(additional_forward_kwargs.keys())
            == required_args
        ), f"Explainable model forward requires {required_args}. Got {tuple(inputs.keys()) + tuple(additional_forward_kwargs.keys())}"
        model_outputs = self.torch_model(
            *tuple(inputs.values()), *tuple(additional_forward_kwargs.values())
        )
        return model_outputs

    def _explainer_forward(
        self,
        explainer: partial[Explainer],
        explainer_inputs: ExplainerInputs,
        target: Union[torch.Tensor, List[torch.Tensor]],
    ):
        # initialize explainer
        explainer = explainer(self.torch_model)

        possible_args = _get_possible_args(explainer.explain)
        input_keys = explainer_inputs.inputs.keys()
        explainer_kwargs = {
            "inputs": tuple(explainer_inputs.inputs.values()),
            "additional_forward_args": tuple(
                explainer_inputs.additional_forward_kwargs.values()
            ),
            "target": target,
        }
        if "baselines" in possible_args:
            assert (
                explainer_inputs.baselines.keys() == explainer_inputs.inputs.keys()
            ), f"Baselines must have the same keys as inputs. Got {explainer_inputs.baselines.keys()} "
            explainer_kwargs["baselines"] = tuple(explainer_inputs.baselines.values())
        if "feature_masks" in possible_args:
            assert (
                explainer_inputs.feature_masks.keys() == explainer_inputs.inputs.keys()
            ), f"Feature masks must have the same keys as inputs. Got {explainer_inputs.feature_masks.keys()} "
            explainer_kwargs["feature_masks"] = tuple(
                explainer_inputs.feature_masks.values()
            )
        explainer_kwargs["inputs"] = tuple(
            x.requires_grad_() for x in explainer_kwargs["inputs"]
        )

        if self._progress_bar is not None:
            self._progress_bar.pbar.set_postfix_str(
                f"generating explanations using explainer=[{explainer.__class__.__name__}]"
            )

        explanations = {
            input_key: explanation
            for input_key, explanation in zip(
                input_keys, explainer.explain(**explainer_kwargs)
            )
        }

        # detach tensors
        apply_to_tensor(explainer_inputs.inputs, torch.detach)
        apply_to_tensor(explainer_inputs.baselines, torch.detach)
        apply_to_tensor(explainer_inputs.additional_forward_kwargs, torch.detach)
        apply_to_tensor(explainer_inputs.feature_masks, torch.detach)
        apply_to_tensor(explanations, torch.detach)
        apply_to_tensor(target, torch.detach)

        return ExplanationModelOutput(
            explanations=explanations,
            reduced_explanations=self._reduce_explanations(explanations),
            explainer_inputs=explainer_inputs,
            target=target,
        )

    def explanation_step(
        self,
        batch: BatchDict,
        explainer: partial[Explainer],
        **kwargs,
    ) -> ExplanationModelOutput:
        # validate model is built
        self._validate_model_built()

        # validate batch keys
        _validate_keys_in_batch(
            keys=self._required_keys_for_explainability(), batch=batch
        )

        # prepare inputs for explanation
        explainer_inputs = self._prepare_explainer_inputs(batch=batch)

        # prepare target
        target = self._prepare_target(batch=batch, explainer_inputs=explainer_inputs)

        # perform explainer forward
        explainer_output = self._explainer_forward(
            explainer=explainer,
            explainer_inputs=explainer_inputs,
            target=target,
        )

        assert isinstance(
            explainer_output, ExplanationModelOutput
        ), f"Model output must be of type ModelOutput. Got {type(explainer_output)}"
        return explainer_output

    @abstractmethod
    def _prepare_explainer_inputs(self, batch: BatchDict, **kwargs) -> ExplainerInputs:
        pass

    @abstractmethod
    def _prepare_target(
        self, batch: BatchDict, **kwargs
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    def _reduce_explanations(self, explanations):
        return explanations
