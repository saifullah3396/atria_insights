from collections import OrderedDict

import torch
import tqdm
from atria_core.logger.logger import get_logger
from ignite.utils import apply_to_tensor
from torchxai.explainers.explainer import Explainer
from torchxai.metrics._utils.common import _reduce_tensor_with_indices_non_deterministic

from atria_insights.registry.registry_groups import ExplainerBuilder
from atria_insights.utilities.containers import ExplainerStepInputs

logger = get_logger(__name__)


def _get_first_layer(module, name=None):
    children = list(module.named_children())
    if len(children) > 0:
        return _get_first_layer(
            children[0][1],
            name=children[0][0] if name is None else name + "." + children[0][0],
        )
    return name, module


def _extract_feature_group_explanations(
    explanation: torch.Tensor, feature_mask: torch.Tensor
):
    assert explanation.shape == feature_mask.shape and explanation.shape[0] == 1
    reduced_explanation, _ = _reduce_tensor_with_indices_non_deterministic(
        explanation.flatten(), (feature_mask - feature_mask.min()).flatten()
    )
    return reduced_explanation


def _extract_multi_taret_feature_group_explanations(
    explanation: torch.Tensor, feature_mask: torch.Tensor
):
    if explanation is None:
        return

    reduced_explanations = []
    for single_target_explanation in explanation:
        assert single_target_explanation.shape == feature_mask.shape, (
            "Explanation and feature mask must have the same shape. "
            f"Got {single_target_explanation.shape} and {feature_mask.shape}"
        )
        reduced_explanation, _ = _reduce_tensor_with_indices_non_deterministic(
            single_target_explanation.flatten(),
            (feature_mask - feature_mask.min()).flatten(),
        )
        reduced_explanations.append(reduced_explanation)
    return torch.stack(reduced_explanations, dim=0)


def _generate_word_level_targets(
    token_labels_per_sample,
    predicted_token_labels_per_sample,
    word_ids_per_sample,
    percent_other_labels_kept: float = 0.0,
    max_targets: int | None = None,
    dataset_labels: list[str] = 0,
    seed: int = 0,
):
    # find the tokens that are either predicted as Other or have ground-truth of other besides the padding labels
    other_label_idx = dataset_labels.index("O")
    other_labels_mask = token_labels_per_sample == other_label_idx
    other_labels_mask |= predicted_token_labels_per_sample == other_label_idx
    other_labels_mask &= token_labels_per_sample != -100

    # now extract indices of these other label tokens
    other_labels_indices = other_labels_mask.nonzero().flatten()

    # we randomly shuffle the indices of other labels
    rand_other_labels_indices = other_labels_indices[
        torch.randperm(
            other_labels_indices.shape[0],
            device=other_labels_indices.device,
            # this seeding is necessary so that always the same targets for each sample are generated across methods/runs
            generator=torch.Generator(other_labels_indices.device).manual_seed(seed),
        )
    ]

    # extract (1-percent_other_labels_kept) of Other (O) labels to ignore
    other_labels_ignored_indices = rand_other_labels_indices[
        int(rand_other_labels_indices.shape[0] * percent_other_labels_kept) :
    ]

    targets = []
    target_word_ids = []
    for token_id in range(word_ids_per_sample.shape[0]):
        if (
            word_ids_per_sample[token_id] != -100
            and word_ids_per_sample[token_id] != last_word_id
            and token_id not in other_labels_ignored_indices
        ):
            targets.append(
                (token_id, predicted_token_labels_per_sample[token_id].item())
            )
            target_word_ids.append(word_ids_per_sample[token_id].item())
        last_word_id = word_ids_per_sample[token_id]

    # get N% of the total targets for final evaluation
    random_indices = torch.randperm(
        len(targets),
        # this seeding is necessary so that always the same targets for each sample are generated across methods/runs
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    # filter tokens
    def filter(arr):
        # rearrange indices randomly
        arr = [arr[idx] for idx in random_indices]

        if max_targets is not None:
            arr = arr[:max_targets]

        # take first max_target targets
        return arr

    targets = filter(targets)
    target_word_ids = filter(target_word_ids)

    # sanity check
    assert len(targets) == len(target_word_ids)
    return targets, target_word_ids


def _map_inputs_to_ordered_dict(inputs: dict[str, torch.Tensor], keys):
    return OrderedDict({key: inputs[key] for key in keys})


def _prepare_explainer_input_kwargs(
    explainer: ExplainerBuilder,
    explainer_step_inputs: ExplainerStepInputs,
    target: torch.Tensor | list[torch.Tensor],
    train_baselines: dict[str, torch.Tensor] | None = None,
):
    from atria_core.utilities.common import _get_possible_args

    possible_args = _get_possible_args(explainer.explain)
    explainer_input_kwargs = dict(
        inputs=tuple(explainer_step_inputs.model_inputs.explained_inputs.values()),
        additional_forward_args=tuple(
            explainer_step_inputs.model_inputs.additional_forward_kwargs.values()
            if explainer_step_inputs.model_inputs.additional_forward_kwargs
            else []
        ),
        target=target,
    )
    if "baselines" in possible_args:
        assert (
            explainer_step_inputs.baselines.keys()
            == explainer_step_inputs.model_inputs.explained_inputs.keys()
        ), (
            f"Baselines must have the same keys as inputs. Got {explainer_step_inputs.baselines.keys()} "
        )
        explainer_input_kwargs["baselines"] = tuple(
            explainer_step_inputs.baselines.values()
        )
    if "feature_mask" in possible_args:
        assert (
            explainer_step_inputs.feature_masks.keys()
            == explainer_step_inputs.model_inputs.explained_inputs.keys()
        ), (
            f"Feature masks must have the same keys as inputs. Got {explainer_step_inputs.feature_masks.keys()} "
        )
        explainer_input_kwargs["feature_mask"] = tuple(
            explainer_step_inputs.feature_masks.values()
        )
    if (
        "frozen_features" in possible_args
        and explainer_step_inputs.frozen_features is not None
    ):
        assert (
            len(explainer_step_inputs.frozen_features)
            == list(explainer_step_inputs.model_inputs.explained_inputs.values())[
                0
            ].shape[0]
        ), (
            f"Length of frozen features must be equal to the batch size. "
            f"Got {len(explainer_step_inputs.frozen_features)} and {list(explainer_step_inputs.model_inputs.explained_inputs.values())[0].shape[0]}"
        )
        explainer_input_kwargs["frozen_features"] = (
            explainer_step_inputs.frozen_features
        )
    if "train_baselines" in possible_args:
        explainer_input_kwargs["train_baselines"] = train_baselines
    return explainer_input_kwargs


def _explainer_forward(
    explainer: Explainer,
    explainer_step_inputs: ExplainerStepInputs,
    target: torch.Tensor,
    iterative_computation: bool = False,
    train_baselines: dict[str, torch.Tensor] | None = None,
):
    # prepare explainer input kwargs
    explainer_input_kwargs = _prepare_explainer_input_kwargs(
        explainer=explainer,
        explainer_step_inputs=explainer_step_inputs,
        target=target,
        train_baselines=train_baselines,
    )

    if explainer._is_multi_target:
        # iterate over each sample in the batch separately
        explanations = []
        for batch_idx in range(
            explainer_step_inputs.inputs["input_embeddings"].shape[0]
        ):
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
                    logger.info(f"{key}: {[v.shape for v in value]}")
                elif isinstance(value, torch.Tensor):
                    assert value.shape[0] == 1, (
                        f"{key} found with batch size > 1: {value.shape}"
                    )
                    logger.info(f"{key}: {value.shape}")

            # this returns a list of explanations for each target
            # example target 0 -> (explanation_embeddings, explanation_position_embeddings, ...)
            # example target 1 -> (explanation_embeddings, explanation_position_embeddings, ...)
            if len(current_explainer_kwargs["target"]) == 0:
                explanations.append(dict.fromkeys(explainer_step_inputs.inputs.keys()))
                continue

            if iterative_computation:
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
                    explainer_step_inputs.inputs.keys(), target_explanations_per_input
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
                explainer_step_inputs.model_inputs.explained_inputs.keys(),
                explainer.explain(**explainer_input_kwargs),
            )
        }

    # log information
    logger.info("Explanations generated with the following information:")
    logger.info(f"Explainer: {explainer.__class__.__name__}")
    logger.info(f"Explaination keys: {explanations.keys()}")
    for input_key, explanation_per_input in explanations.items():
        if isinstance(explanation_per_input, list):
            logger.info(
                f"Explaination shapes [{input_key}] {[v.shape if v is not None else v for v in explanation_per_input]}"
            )
            logger.info(
                f"Explaination types [{input_key}] {[v.dtype if v is not None else v for v in explanation_per_input]}"
            )
        else:
            logger.info(
                f"Explaination shape [{input_key}] {explanation_per_input.shape}"
            )
            logger.info(
                f"Explaination type [{input_key}] {explanation_per_input.dtype}"
            )

    # detach tensors
    apply_to_tensor(explainer_step_inputs.model_inputs.explained_inputs, torch.detach)
    if all(x is not None for x in explainer_step_inputs.baselines.values()):
        apply_to_tensor(explainer_step_inputs.baselines, torch.detach)
    if explainer_step_inputs.model_inputs.additional_forward_kwargs:
        apply_to_tensor(
            explainer_step_inputs.model_inputs.additional_forward_kwargs, torch.detach
        )
    apply_to_tensor(explainer_step_inputs.feature_masks, torch.detach)
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
