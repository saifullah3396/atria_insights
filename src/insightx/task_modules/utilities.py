from insightx.model_explainability_wrappers.base import ModelExplainabilityWrapper
from insightx.task_modules.model_output_wrappers import SoftmaxWrapper


def _unwrap_model(model):
    if isinstance(model, SoftmaxWrapper):
        return _unwrap_model(model.model)
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
