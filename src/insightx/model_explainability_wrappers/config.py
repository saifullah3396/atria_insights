from hydra_zen import builds

from insightx.model_explainability_wrappers.utils import create_segmentation_fn
from insightx.registry.module_registry import ModuleRegistry
from insightx.utilities.common import _get_parent_module

# register task modules type=[atria]
ModuleRegistry.register_model_explainability_wrapper(
    module=_get_parent_module(__name__) + ".classification.image",
    registered_class_or_func="ImageClassificationExplainabilityWrapper",
    hydra_defaults=[
        "_self_",
    ],
    segmentation_fn=builds(
        create_segmentation_fn,
        populate_full_signature=True,
        zen_partial=False,
        segmentation_type="grid",
        cell_size=16,
    ),
)
