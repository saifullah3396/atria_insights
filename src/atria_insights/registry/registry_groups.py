from atria_registry import RegistryGroup
from atria_registry.module_builder import ModuleBuilder


class ExplMetricBuilder(ModuleBuilder):
    pass


class ExplainerBuilder(ModuleBuilder):
    pass


class ExplMetricsRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = ExplMetricBuilder
    __exclude_from_builder__ = [
        "device",
        "explainer",
        "forward_func",
    ]  # these are passed at runtime from trainer


class ExplainerRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = ExplainerBuilder
    __exclude_from_builder__ = []  # these are passed at runtime from trainer
