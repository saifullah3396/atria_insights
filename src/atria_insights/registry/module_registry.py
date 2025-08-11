from atria_insights.registry.registry_groups import (
    ExplainerMetricRegistryGroup,
    ExplainerPipelineRegistryGroup,
    ExplainerRegistryGroup,
)

_initialized = False


def init_registry():
    from atria_registry.module_registry import ModuleRegistry

    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="EXPLAINER_METRIC",
        registry_group=ExplainerMetricRegistryGroup(
            name="explainer_metric", default_provider="atria_insights"
        ),
    )
    ModuleRegistry().add_registry_group(
        name="EXPLAINER",
        registry_group=ExplainerRegistryGroup(
            name="explainer", default_provider="atria_insights"
        ),
    )
    ModuleRegistry().add_registry_group(
        name="EXPLAINER_PIPELINE",
        registry_group=ExplainerPipelineRegistryGroup(
            name="explainer_pipeline", default_provider="atria_insights"
        ),
    )
