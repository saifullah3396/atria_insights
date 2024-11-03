from insightx.registry.module_registry import ModuleRegistry
from insightx.utilities.common import _get_parent_module

TRAINER_DIR = "${output_dir}/model_explainer/"
TRAINER_DIR += "${resolve_dir_name:${data_module.dataset_name}}/"
TRAINER_DIR += "${resolve_dir_name:${task_module.torch_model_builder.model_name}}/"

ModuleRegistry.register_task_runner(
    module=".".join((_get_parent_module(__name__), f"model_explainer")),
    registered_class_or_func="ModelExplainer",
    hydra_defaults=[
        "_self_",
        {"/data_module@data_module": "huggingface"},
        {"/task_module@task_module": "image_classification_explanation_module"},
        {"/engine@test_engine": "default_test_engine"},
        {"/engine@explanation_engine": "default_explanation_engine"},
    ],
    zen_meta=dict(
        hydra={
            "run": {"dir": TRAINER_DIR},
            "output_subdir": "hydra",
            "job": {"chdir": False},
            "searchpath": [
                "pkg://atria/conf",
            ],
        },
    ),
)
