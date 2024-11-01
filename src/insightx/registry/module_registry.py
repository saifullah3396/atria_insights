from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from hydra.core.config_store import ConfigStore

from insightx.utilities.common import _convert_to_snake_case, _resolve_module_from_path

PROVIDER_NAME = "insightx"


class RegistryGroups:
    METRIC = "metric"
    EXPLAINER = "explainer"
    ENGINE = "engine"
    ENGINE_STEP = "engine_step"
    TASK_MODULE = "task_module"
    MODEL_EXPLAINABILITY_WRAPPER = "model_explainability_wrapper"


@dataclass
class ModuleConfig:
    group: str
    name: str
    module: str
    package: str = None
    build_kwargs: dict = None

    def __attrs_post_init__(self):
        if self.build_kwargs is None:
            self.build_kwargs = {}


class ModuleRegistry:
    REGISTERED_MODULE_CONFIG: List[ModuleConfig] = []

    @staticmethod
    def build_module_configurations() -> ConfigStore:
        from hydra_zen import builds

        for module_config in ModuleRegistry.REGISTERED_MODULE_CONFIG:
            cs = ConfigStore.instance()

            try:
                cs.store(
                    group=module_config.group,
                    name=module_config.name,
                    node=builds(
                        _resolve_module_from_path(module_config.module),
                        **module_config.build_kwargs,
                    ),
                    provider=PROVIDER_NAME,
                    package=module_config.package,
                )
            except Exception as e:
                raise Exception(
                    f"Failed to register {module_config.group}/{module_config.name} due to error: {e}"
                )

        return cs

    @staticmethod
    def register_module_configuration(
        group: str,
        name: str,
        module: str,
        build_kwargs: dict = None,
        lazy_build: bool = True,
        is_global_package: bool = False,
    ):
        if build_kwargs is None:
            build_kwargs = {}

        if lazy_build:
            ModuleRegistry.REGISTERED_MODULE_CONFIG.append(
                ModuleConfig(
                    group=group,
                    name=name,
                    module=module,
                    build_kwargs=build_kwargs,
                    package="__global__" if is_global_package else None,
                )
            )
        else:
            from hydra.core.config_store import ConfigStore
            from hydra_zen import builds

            cs = ConfigStore.instance()
            cs.store(
                group=PROVIDER_NAME + "/" + group,
                name=name,
                node=builds(
                    _resolve_module_from_path(module),
                    **build_kwargs,
                ),
                provider=PROVIDER_NAME,
                package="__global__" if is_global_package else None,
            )

    @staticmethod
    def register_class_in_group(
        group: str,
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        if not isinstance(registered_class_or_func, (tuple, list)):
            registered_class_or_func = [registered_class_or_func]
        if name is None:
            name = [None] * len(registered_class_or_func)
        if not isinstance(name, (tuple, list)):
            name = [name]
        assert len(registered_class_or_func) == len(name), (
            f"Length of registered_class_or_func ({len(registered_class_or_func)}) "
            f"and name ({len(name)}) must be the same."
        )
        is_global_package = kwargs.pop("is_global_package", False)
        for single_name, single_class_or_func in zip(name, registered_class_or_func):
            # register torch model torch_model_builders as child node of task_module
            ModuleRegistry.register_module_configuration(
                group=group,
                name=(
                    single_name
                    if single_name is not None
                    else _convert_to_snake_case(single_class_or_func)
                ),
                module=module + "." + single_class_or_func,
                build_kwargs=dict(populate_full_signature=True, **kwargs),
                is_global_package=is_global_package,
            )

    @staticmethod
    def register_metric(
        sub_group: str,
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        ModuleRegistry.register_class_in_group(
            group=f"{RegistryGroups.METRIC}/{sub_group}",
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_explainer(
        sub_group: str,
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        ModuleRegistry.register_class_in_group(
            group=f"{RegistryGroups.EXPLAINER}/{sub_group}",
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_engine(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        ModuleRegistry.register_class_in_group(
            group=RegistryGroups.ENGINE,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_engine_step(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        ModuleRegistry.register_class_in_group(
            group=RegistryGroups.ENGINE_STEP,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_task_module(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        ModuleRegistry.register_class_in_group(
            group=RegistryGroups.TASK_MODULE,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_model_explainability_wrapper(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        ModuleRegistry.register_class_in_group(
            group=RegistryGroups.MODEL_EXPLAINABILITY_WRAPPER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_task_runner(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        ModuleRegistry.register_class_in_group(
            group="",  # since task runner is a top level module, it must be placed in the root of the registry, therefore it has no group
            module=module,
            registered_class_or_func=registered_class_or_func,
            is_global_package=True,
            **kwargs,
        )
