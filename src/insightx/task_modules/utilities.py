from typing import List, Optional

import torch
from torchxai.metrics._utils.common import _reduce_tensor_with_indices_non_deterministic

from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper


def _unwrap_model(model):
    if isinstance(model, ModelExplainabilityWrapper):
        return _unwrap_model(model.model)
    return model


def _get_first_layer(module, name=None):
    children = list(module.named_children())
    if len(children) > 0:
        return _get_first_layer(
            children[0][1],
            name=children[0][0] if name is None else name + "." + children[0][0],
        )
    return name, module


def _get_model_forward_fn(model):
    return _unwrap_model(model).forward


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
    max_targets: Optional[int] = None,
    dataset_labels: List[str] = 0,
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
