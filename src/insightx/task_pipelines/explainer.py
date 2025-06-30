"""
This module defines the entry point for the Explainer task pipeline.

The Explainer is responsible for running evaluation tasks using configurations
provided via Hydra. It sets up logging, instantiates the Explainer, and executes
the evaluation process.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from atria_core.logger.logger import LoggerBase, get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="task_pipeline/explainer",
)
def app(config: DictConfig) -> None:
    """
    Main entry point for the Explainer application.

    Args:
        config (DictConfig): The configuration object provided by Hydra.

    This function sets up logging, instantiates the Explainer, and runs the
    evaluation process. Any exceptions during instantiation or execution are logged.
    """
    from hydra.core.hydra_config import HydraConfig  # noqa
    from hydra_zen import instantiate  # noqa

    from insightx.task_pipelines._explainer import Explainer  # noqa
    from atria_core.utilities.pydantic import pydantic_parser  # noqa

    hydra_config = HydraConfig.get()
    LoggerBase().log_file_path = (
        Path(hydra_config.runtime.output_dir) / f"{hydra_config.job.name}.log"
    )
    config.output_dir = hydra_config.runtime.output_dir
    try:
        explainer: Explainer = instantiate(
            config, _convert_="object", _target_wrapper_=pydantic_parser
        )
        explainer.build()
        explainer.run()
    except Exception as e:
        logger.exception(f"Failed to instantiate ModelExplainer: {e}")


if __name__ == "__main__":
    app()
