from __future__ import annotations

from typing import TYPE_CHECKING

from atria_registry import RegistryGroup
from atria_registry.module_builder import ModuleBuilder

if TYPE_CHECKING:
    from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
        AtriaExplainerPipelineConfig,
    )


class ExplainerMetricBuilder(ModuleBuilder):
    pass


class ExplainerBuilder(ModuleBuilder):
    pass


class ExplainerMetricRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = ExplainerMetricBuilder
    __exclude_from_builder__ = [
        "device",
        "explainer",
        "forward_func",
    ]  # these are passed at runtime from trainer


class ExplainerRegistryGroup(RegistryGroup):
    __registers_as_module_builder__ = True
    __module_builder_class__ = ExplainerBuilder
    __exclude_from_builder__ = ["model"]  # these are passed at runtime from trainer


class ExplainerPipelineRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing models.

    This class provides additional methods for registering and managing models
    within the registry system.
    """

    def register(
        self,
        name: str,
        configs: list[AtriaExplainerPipelineConfig] | None = None,
        builds_to_file_store: bool = True,
        **kwargs,
    ):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """
        from atria_insights.explainer_pipelines.atria_explainer_pipeline import (
            AtriaExplainerPipelineConfig,
        )

        if builds_to_file_store and not self._file_store_build_enabled:

            def noop_(module):
                return module

            return noop_

        # get spec params
        provider = kwargs.pop("provider", None)
        is_global_package = kwargs.pop("is_global_package", False)
        registers_target = kwargs.pop("registers_target", True)
        defaults = kwargs.pop("defaults", None)

        def decorator(module):
            from atria_registry.module_spec import ModuleSpec

            # build the module spec
            module_spec = ModuleSpec(
                module=module,
                name=name,
                group=self.name,
                provider=provider or self._default_provider,
                is_global_package=is_global_package,
                registers_target=registers_target,
                defaults=defaults,
            )

            if configs is not None:
                import copy

                assert isinstance(configs, list) and all(
                    isinstance(config, AtriaExplainerPipelineConfig)
                    for config in configs
                ), (
                    f"Expected configs to be a list of AtriaExplainerPipelineConfig, got {type(configs)} instead."
                )
                for config in configs:
                    config.pipeline_name = name
                    config_module_spec = copy.deepcopy(module_spec)
                    config_defaults = config.model_extra.pop("defaults", None)
                    if config_defaults is not None:
                        config_module_spec.defaults = config_defaults
                    config_module_spec.name = (
                        config.pipeline_name + "/" + config.config_name
                    )
                    config_module_spec.model_extra.update(
                        {**config.model_extra, **kwargs}
                    )
                    self.register_module(config_module_spec)
                return module
            else:
                from hydra_zen import MISSING

                config = module.__config_cls__.model_construct(
                    pipeline_name=name,
                    model_pipeline=MISSING,
                    explainer=MISSING,
                    explainer_metrics=MISSING,
                )
                module_spec.model_extra.update(
                    {k: getattr(config, k) for k in config.__class__.model_fields}
                )
                module_spec.model_extra.update({**kwargs})
                self.register_module(module_spec)
                return module

        return decorator
