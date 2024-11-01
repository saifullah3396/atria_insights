from insightx.registry.module_registry import ModuleRegistry
from insightx.utilities.common import _get_parent_module

# register task modules type=[atria]
ModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".classification.image",
    registered_class_or_func="ImageClassificationExplanationModule",
    hydra_defaults=[
        "_self_",
        {
            "/model_explainability_wrapper@model_explainability_wrapper": "image_classification_explainability_wrapper"
        },
        {"/torch_model_builder@torch_model_builder": "timm"},
    ],
)
