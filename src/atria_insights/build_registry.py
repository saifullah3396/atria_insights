"""
Build Registry Script

This script is responsible for building the registry of modules used in the Atria project. It imports various components from the Atria framework and writes the registry configuration to YAML files.

Usage:
    Run this script to generate the registry configuration files in the `conf` directory.

Modules Imported:
    - Core utilities for registry management.
    - Data batch samplers, pipelines, and storage managers.
    - Model definitions and pipelines.
    - Task pipelines for training and evaluation.
    - Training optimizers and schedulers.

Author: Saifullah
Date: April 14, 2025
"""

from pathlib import Path

from atria_registry.utilities import write_registry_to_yaml

from atria_insights.explainer_pipelines.classification.image import *  # noqa
from atria_insights.explainers.registry import *  # noqa
from atria_insights.metrics.registry import *  # noqa
from atria_insights.task_pipelines._explainer import *  # noqa

if __name__ == "__main__":
    write_registry_to_yaml(
        str(Path(__file__).parent / "conf"),
        types=["explainer_metric", "explainer", "explainer_pipeline", "task_pipeline"],
        delete_existing=True,
    )
