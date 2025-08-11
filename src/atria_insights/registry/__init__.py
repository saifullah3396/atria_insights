"""
Registry Initialization Module

This module initializes the registry system for the Atria models package. It imports
and initializes the model registry from the `ModuleRegistry` class, making it
accessible as a module-level constant.

The registry system provides a centralized way to register and retrieve model
components throughout the application.

Constants:
    MODEL: Registry group for model components

Example:
    >>> from atria_models.registry import MODEL
    >>> # Register a new model
    >>> @MODEL.register()
    >>> class MyModel:
    ...     pass
    >>> # Get a registered model
    >>> model_cls = MODEL.get("my_model")

Dependencies:
    atria_registry.ModuleRegistry: Provides the main registry class
    atria_models.registry.module_registry: Provides registry initialization
    atria_models.registry.registry_groups: Provides ModelRegistryGroup class

Author: Atria Development Team
Date: 2025-07-10
Version: 1.2.0
License: MIT
"""

from atria_registry.module_registry import ModuleRegistry

from atria_insights.registry.module_registry import init_registry
from atria_insights.registry.registry_groups import (
    ExplainerMetricRegistryGroup,
    ExplainerPipelineRegistryGroup,
    ExplainerRegistryGroup,
)

init_registry()

EXPLAINER_METRIC: ExplainerMetricRegistryGroup = ModuleRegistry().EXPLAINER_METRIC
EXPLAINER: ExplainerRegistryGroup = ModuleRegistry().EXPLAINER
EXPLAINER_PIPELINE: ExplainerPipelineRegistryGroup = ModuleRegistry().EXPLAINER_PIPELINE

__all__ = ["EXPLAINER_METRIC", "EXPLAINER", "EXPLAINER_PIPELINE"]
