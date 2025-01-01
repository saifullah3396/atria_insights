from collections import OrderedDict
from typing import Dict, List, Union

import torch
import tqdm
from atria.core.utilities.common import _get_possible_args
from atria.core.utilities.logging import get_logger
from ignite.utils import apply_to_tensor
from torchxai.explainers.explainer import Explainer

from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.utilities.containers import ExplainerArguments

logger = get_logger(__name__)


def _map_inputs_to_ordered_dict(inputs: Dict[str, torch.Tensor], keys):
    return OrderedDict({key: inputs[key] for key in keys})


def _unwrap_model(model):
    if isinstance(model, ModelExplainabilityWrapper):
        return _unwrap_model(model.model)
    return model


def _get_model_forward_fn(model):
    return _unwrap_model(model).forward


def _prepare_explainer_input_kwargs(
    explainer: Explainer,
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
        explainer_input_kwargs["baselines"] = tuple(explainer_args.baselines.values())
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
    explainer: Explainer,
    explainer_args: ExplainerArguments,
    target: torch.Tensor,
    compute_multi_target_iteratively: bool = False,
):
    # prepare explainer input kwargs
    explainer_input_kwargs = _prepare_explainer_input_kwargs(
        explainer=explainer, explainer_args=explainer_args, target=target
    )

    if explainer._is_multi_target:
        # iterate over each sample in the batch separately
        explanations = []
        for batch_idx in range(explainer_args.inputs["input_embeddings"].shape[0]):
            current_explainer_kwargs = {}
            for k, v in explainer_input_kwargs.items():
                if k in ["train_baselines"]:
                    current_explainer_kwargs[k] = explainer_input_kwargs[k]
                    continue

                if isinstance(v, tuple):
                    current_explainer_kwargs[k] = tuple(
                        v_i[batch_idx].unsqueeze(0) for v_i in v
                    )
                elif isinstance(v, torch.Tensor):
                    current_explainer_kwargs[k] = v[batch_idx].unsqueeze(0)
                else:
                    current_explainer_kwargs[k] = v[batch_idx]

            # check if the batch size is 1
            for key, value in current_explainer_kwargs.items():
                if key in ["train_baselines", "frozen_features"]:
                    continue
                if isinstance(value, tuple):
                    for v in value:
                        assert v.shape[0] == 1
                    logger.debug(f"{key}: {[v.shape for v in value]}")
                elif isinstance(value, torch.Tensor):
                    assert (
                        value.shape[0] == 1
                    ), f"{key} found with batch size > 1: {value.shape}"
                    logger.debug(f"{key}: {value.shape}")

            # this returns a list of explanations for each target
            # example target 0 -> (explanation_embeddings, explanation_position_embeddings, ...)
            # example target 1 -> (explanation_embeddings, explanation_position_embeddings, ...)
            if len(current_explainer_kwargs["target"]) == 0:
                explanations.append(
                    {input_key: None for input_key in explainer_args.inputs.keys()}
                )
                continue

            if compute_multi_target_iteratively:
                logger.info("Computing explanations per target iteratively...")
                explainer._is_multi_target = False
                input_explanations_per_target = []
                targets_list = current_explainer_kwargs.pop("target")
                for target in tqdm.tqdm(targets_list):
                    curr_explanation = explainer.explain(
                        **current_explainer_kwargs, target=target
                    )
                    input_explanations_per_target.append(curr_explanation[0])
                explainer._is_multi_target = True
                current_explainer_kwargs["target"] = targets_list
            else:
                logger.info("Computing multi-target explanations in batch...")
                input_explanations_per_target = explainer.explain(
                    **current_explainer_kwargs
                )

            # convert the mapping to -> tuples -> list of targets
            target_explanations_per_input = tuple(
                map(list, zip(*input_explanations_per_target))
            )
            target_explanations_per_input = {
                input_key: torch.cat(explanations, dim=0)
                for input_key, explanations in zip(
                    explainer_args.inputs.keys(), target_explanations_per_input
                )
            }
            explanations.append(target_explanations_per_input)

        # convert list of dict explanations to dict of list explanations
        explanations = {
            k: [explanation[k] for explanation in explanations]
            for k in explanations[0].keys()
        }
    else:
        explanations = {
            input_key: explanation
            for input_key, explanation in zip(
                explainer_args.inputs.keys(),
                explainer.explain(**explainer_input_kwargs),
            )
        }

    # log information
    logger.debug(f"Explanations generated with the following information:")
    logger.debug(f"Explainer: {explainer.__class__.__name__}")
    logger.debug(f"Explaination keys: {explanations.keys()}")
    for input_key, explanation_per_input in explanations.items():
        if isinstance(explanation_per_input, list):
            logger.debug(
                f"Explaination shapes [{input_key}] {[v.shape if v is not None else v for v in explanation_per_input]}"
            )
            logger.debug(
                f"Explaination types [{input_key}] {[v.dtype if v is not None else v  for v in explanation_per_input]}"
            )
        else:
            logger.debug(
                f"Explaination shape [{input_key}] {explanation_per_input.shape}"
            )
            logger.debug(
                f"Explaination type [{input_key}] {explanation_per_input.dtype}"
            )

    # detach tensors
    apply_to_tensor(explainer_args.inputs, torch.detach)
    if all(x is not None for x in explainer_args.baselines.values()):
        apply_to_tensor(explainer_args.baselines, torch.detach)
    apply_to_tensor(explainer_args.additional_forward_kwargs, torch.detach)
    apply_to_tensor(explainer_args.feature_masks, torch.detach)
    for input_key, explanation_per_input in explanations.items():
        if isinstance(explanation_per_input, list):
            for explanation in explanation_per_input:
                if explanation is not None:
                    apply_to_tensor(explanation, torch.detach)
        else:
            apply_to_tensor(explanation_per_input, torch.detach)
    if isinstance(target, torch.Tensor):
        apply_to_tensor(target, torch.detach)
    return explanations
