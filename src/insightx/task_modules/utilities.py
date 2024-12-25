import torch
from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from torchxai.metrics._utils.common import _reduce_tensor_with_indices_non_deterministic


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
