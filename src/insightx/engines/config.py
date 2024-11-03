from insightx.registry.module_registry import ModuleRegistry
from insightx.utilities.common import _get_parent_module

ModuleRegistry.register_engine_step(
    module=".".join((_get_parent_module(__name__), f"explanation_step")),
    registered_class_or_func=[
        "ExplanationStep",
    ],
    name=[
        "default_explanation_step",
    ],
)

# register explanation engine
ModuleRegistry.register_engine(
    module=".".join((_get_parent_module(__name__), f"explanation_engine")),
    registered_class_or_func=f"ExplanationEngine",
    name=f"default_explanation_engine",
    hydra_defaults=[
        "_self_",
        {"/engine_step@engine_step": f"default_explanation_step"},
        {"/explainer@explainer": "grad/saliency"},
        # {"/metric@metrics.completeness": "axiomatic/completeness"},
    ],
)
