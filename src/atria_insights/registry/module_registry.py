from atria_insights.registry.registry_groups import (
    ExplainerRegistryGroup,
    ExplMetricsRegistryGroup,
)

_initialized = False


def init_registry():
    from atria_registry.module_registry import ModuleRegistry

    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="EXPL_METRIC",
        registry_group=ExplMetricsRegistryGroup(
            name="expl_metric", default_provider="atria_insights"
        ),
    )
    ModuleRegistry().add_registry_group(
        name="EXPLAINER",
        registry_group=ExplainerRegistryGroup(
            name="explainer", default_provider="atria_insights"
        ),
    )
