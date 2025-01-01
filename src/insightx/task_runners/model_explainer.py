from functools import partial
from typing import Optional

import hydra
import ignite.distributed as idist
from atria.core.data.data_modules.atria_data_module import AtriaDataModule
from atria.core.training.engines.evaluation import TestEngine
from atria.core.training.utilities.initialization import (
    _initialize_torch,
    reset_random_seeds,
)
from atria.core.utilities.logging import get_logger
from atria.core.utilities.pydantic_parser import atria_pydantic_parser
from hydra.core.hydra_config import HydraConfig
from insightx.engines.explanation_engine import ExplanationEngine
from insightx.engines.train_baselines_generation_engine import (
    TrainBaselinesGenerationEngine,
)
from insightx.task_modules.explanation_task_module import ExplanationTaskModule
from omegaconf import DictConfig


class ModelExplainer:
    def __init__(
        self,
        data_module: AtriaDataModule,
        task_module: partial[ExplanationTaskModule],
        test_engine: partial[TestEngine],
        explanation_engine: partial[ExplanationEngine],
        output_dir: str,
        seed: int = 42,
        deterministic: bool = False,
        backend: Optional[str] = "nccl",
        n_devices: int = 1,
        test_model: bool = False,
        max_train_baselines: int = 100,
    ):
        self._output_dir = output_dir
        self._seed = seed
        self._deterministic = deterministic
        self._backend = backend
        self._n_devices = n_devices
        self._data_module = data_module
        self._task_module = task_module
        self._test_engine = test_engine
        self._explanation_engine = explanation_engine
        self._test_model = test_model
        self._max_train_baselines = max_train_baselines

    def run(self, hydra_config: HydraConfig, runtime_cfg: DictConfig) -> None:
        logger = get_logger(hydra_config=hydra_config)

        # initialize training
        _initialize_torch(
            seed=self._seed,
            deterministic=self._deterministic,
        )

        # initialize torch device (cpu or gpu)
        device = idist.device()

        # initialize logging directory and tensorboard logger
        output_dir = hydra_config.runtime.output_dir

        # build data module
        self._data_module.setup(stage=None)

        # initialize the task module from partial
        task_module = self._task_module(
            dataset_metadata=self._data_module.dataset_metadata,
        )
        task_module.build_model()

        if self._test_model:
            # initilize the test engine from partial
            self._test_engine = self._test_engine(
                output_dir=output_dir,
                task_module=task_module,
                dataloader=self._data_module.test_dataloader(),
                device=device,
            )

            # run the test engine
            self._test_engine.run()

        # generate training baselines
        train_baselines_generation_engine = TrainBaselinesGenerationEngine(
            output_dir=output_dir,
            task_module=task_module,
            dataloader=self._data_module.train_dataloader(),
            device=device,
            max_train_baselines=self._max_train_baselines,
        )
        train_baselines = train_baselines_generation_engine.run()

        # Reset the seed again before running explainer
        reset_random_seeds(self._seed)

        # initilize the test engine from partial
        self._explanation_engine = self._explanation_engine(
            output_dir=output_dir,
            task_module=task_module,
            dataloader=self._data_module.test_dataloader(),
            device=device,
            train_baselines=train_baselines,
        )

        # run the test engine
        self._explanation_engine.run()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="model_explainer",
)
def app(cfg: ModelExplainer) -> None:
    from hydra_zen import instantiate

    model_explainer: ModelExplainer = instantiate(
        cfg, _convert_="object", _target_wrapper_=atria_pydantic_parser
    )
    hydra_config = HydraConfig.get()
    logger = get_logger(hydra_config=hydra_config)
    try:
        return model_explainer.run(hydra_config=hydra_config, runtime_cfg=cfg)
    except Exception as e:
        logger.exception(e)
    finally:
        return None


if __name__ == "__main__":
    app()
