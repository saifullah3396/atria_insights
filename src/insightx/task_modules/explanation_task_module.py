from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.models.utilities.common import _validate_keys_in_batch
from atria.core.utilities.common import _get_possible_args, _get_required_args
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from ignite.engine import Engine
from ignite.utils import apply_to_tensor, convert_tensor
from insightx.engines.explanation_results_cacher import ExplanationResultsCacher
from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.model_output_wrappers import SoftmaxWrapper
from insightx.task_modules.utilities import _get_model_forward_fn
from insightx.utilities.containers import ExplainerArguments, ExplanationModelOutput
from torchxai.explainers.explainer import Explainer

logger = get_logger(__name__)


class ExplanationTaskModule(AtriaTaskModule, metaclass=ABCMeta):
    def __init__(
        self,
        model_explainability_wrapper: partial[ModelExplainabilityWrapper],
        is_multi_target: bool = False,
        save_metadata_only: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._model_explainability_wrapper = model_explainability_wrapper
        self._is_multi_target = is_multi_target
        self._explanation_results_cacher: ExplanationResultsCacher = None
        self._save_metadata_only = save_metadata_only

    def toggle_explainability(self, state: bool):
        self.torch_model.toggle_explainability(state)

    def attach_explanation_results_cacher(self, explanation_results_cacher):
        self._explanation_results_cacher = explanation_results_cacher

    def _build_model(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ):
        model = super()._build_model(checkpoint=checkpoint)
        model = self._model_explainability_wrapper(SoftmaxWrapper(model))
        return model

    def _required_keys_for_explainability(self) -> List[str]:
        return ["__key__"]

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

    def _prepare_explainer_input_kwargs(
        self,
        explainer: partial[Explainer],
        explainer_args: ExplainerArguments,
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
            assert (
                explainer_args.baselines.keys() == explainer_args.inputs.keys()
            ), f"Baselines must have the same keys as inputs. Got {explainer_args.baselines.keys()} "
            explainer_input_kwargs["baselines"] = tuple(
                explainer_args.baselines.values()
            )
        if "feature_mask" in possible_args:
            assert (
                explainer_args.feature_masks.keys() == explainer_args.inputs.keys()
            ), f"Feature masks must have the same keys as inputs. Got {explainer_args.feature_masks.keys()} "
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
            ), f"Train baselines must have the same keys as inputs. Got {explainer_args.train_baselines.keys()} "

            for train_baselines, inputs in zip(
                explainer_args.train_baselines.values(), explainer_args.inputs.values()
            ):
                assert (
                    train_baselines.shape[1:] == inputs.shape[1:]
                ), f"Train baselines must have the same shape as inputs. Got {train_baselines.shape} and {inputs.shape}"
            explainer_input_kwargs["train_baselines"] = tuple(
                explainer_args.train_baselines.values()
            )
        return explainer_input_kwargs

    def _explainer_forward(
        self,
        batch: BatchDict,
        explainer: partial[Explainer],
        explainer_args: ExplainerArguments,
        target: Union[torch.Tensor, List[torch.Tensor]],
    ):
        # initialize explainer
        explainer = explainer(self.torch_model, is_multi_target=self._is_multi_target)
        input_keys = explainer_args.inputs.keys()

        if self._progress_bar is not None and self._progress_bar.pbar is not None:
            self._progress_bar.pbar.set_postfix_str(
                f"generating explanations using explainer=[{explainer.__class__.__name__}]"
            )

        explanations = {
            input_key: explanation
            for input_key, explanation in zip(
                input_keys,
                explainer.explain(
                    **self._prepare_explainer_input_kwargs(
                        explainer, explainer_args, target
                    )
                ),
            )
        }

        logger.debug(f"Explanations generated with the following information:")
        logger.debug(f"Explainer: {explainer.__class__.__name__}")
        logger.debug(f"Explaination keys: {explanations.keys()}")
        logger.debug(f"Explaination shapes: {[v.shape for v in explanations.values()]}")
        logger.debug(f"Explaination types: {[v.dtype for v in explanations.values()]}")

        # detach tensors
        apply_to_tensor(explainer_args.inputs, torch.detach)
        if all(x is not None for x in explainer_args.baselines.values()):
            apply_to_tensor(explainer_args.baselines, torch.detach)
        apply_to_tensor(explainer_args.additional_forward_kwargs, torch.detach)
        apply_to_tensor(explainer_args.feature_masks, torch.detach)
        apply_to_tensor(explanations, torch.detach)
        apply_to_tensor(target, torch.detach)

        return ExplanationModelOutput(
            explanations=explanations,
            reduced_explanations=self._reduce_explanations(
                batch, explainer_args, explanations
            ),
            explainer_args=explainer_args,
            target=target,
            sample_keys=batch["__key__"],
        )

    def train_baselines_generation_step(
        self,
        batch: BatchDict,
        **kwargs,
    ) -> torch.Tensor:
        # validate model is built
        self._validate_model_built()

        # validate batch keys
        _validate_keys_in_batch(
            keys=self._required_keys_for_explainability(), batch=batch
        )

        return self._prepare_train_baselines(batch=batch)

    def explanation_step(
        self,
        batch: BatchDict,
        explainer: partial[Explainer],
        train_baselines: Dict[str, torch.Tensor],
        explanation_engine: Engine,
        **kwargs,
    ) -> ExplanationModelOutput:
        if explanation_engine is not None and explanation_engine.state.skip_batch:
            return ExplanationModelOutput()

        # validate model is built
        self._validate_model_built()

        # validate batch keys
        _validate_keys_in_batch(
            keys=self._required_keys_for_explainability(), batch=batch
        )

        # prepare inputs for explanation
        explainer_args = self._prepare_explainer_arguments(batch=batch)

        # attach train baselines
        explainer_args.train_baselines = train_baselines

        # convert dict to ordered dict
        explainer_args.train_baselines = OrderedDict(
            {
                key: explainer_args.train_baselines[key]
                for key in explainer_args.inputs.keys()
            }
        )

        # prepare target
        target = self._prepare_target(batch=batch, explainer_args=explainer_args)

        # save metadata only
        if self._save_metadata_only and self._explanation_results_cacher is not None:
            explainer_output = ExplanationModelOutput(
                explanations=None,
                reduced_explanations=None,
                explainer_args=explainer_args,
                target=target,
                sample_keys=batch["__key__"],
            )
            self._explanation_results_cacher.save_results(batch, explainer_output)
            return explainer_output

        # load explanations from cache if available
        if self._explanation_results_cacher is not None:
            explanations, reduced_explanations = (
                self._explanation_results_cacher.load_explanations(batch["__key__"])
            )

            if explanations is not None:
                # convert dict to ordered dict
                explanations = OrderedDict(
                    {key: explanations[key] for key in explainer_args.inputs.keys()}
                )
                if reduced_explanations is not None:
                    # convert dict to ordered dict
                    reduced_explanations = OrderedDict(
                        {
                            key: reduced_explanations[key]
                            for key in explainer_args.inputs.keys()
                        }
                    )
                else:
                    reduced_explanations = (
                        self._reduce_explanations(batch, explainer_args, explanations),
                    )

            if explanations is not None:
                return ExplanationModelOutput(
                    explanations=convert_tensor(explanations, device=target.device),
                    reduced_explanations=reduced_explanations,
                    explainer_args=explainer_args,
                    target=target,
                    sample_keys=batch["__key__"],
                )

        # perform explainer forward
        explainer_output = self._explainer_forward(
            batch=batch,
            explainer=explainer,
            explainer_args=explainer_args,
            target=target,
        )
        assert isinstance(
            explainer_output, ExplanationModelOutput
        ), f"Model output must be of type ModelOutput. Got {type(explainer_output)}"

        # save explanations to cache
        if self._explanation_results_cacher is not None:
            self._explanation_results_cacher.save_results(batch, explainer_output)

        return explainer_output

    @abstractmethod
    def _prepare_explainer_arguments(self, batch: BatchDict) -> ExplainerArguments:
        pass

    @abstractmethod
    def _prepare_train_baselines(self, batch: BatchDict) -> torch.Tensor:
        pass

    @abstractmethod
    def _prepare_target(
        self,
        batch: BatchDict,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        pass

    def _reduce_explanations(
        self,
        batch: BatchDict,
        explainer_args: ExplainerArguments,
        explanations: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return explanations
