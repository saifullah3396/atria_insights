from hydra_zen import builds
from insightx.registry.module_registry import ModuleRegistry
from insightx.utilities.common import _get_parent_module
from torchxai.metrics import (
    aopc,
    completeness,
    complexity_entropy,
    complexity_sundararajan,
    effective_complexity,
    faithfulness_corr,
    faithfulness_estimate,
    infidelity,
    input_invariance,
    monotonicity,
    monotonicity_corr_and_non_sens,
    sensitivity_max_and_avg,
    sensitivity_n,
    sparseness,
)
from torchxai.metrics._utils.perturbation import (
    default_infidelity_perturb_fn,
    default_random_perturb_func,
)

# axiomatic
ModuleRegistry.register_metric(
    sub_group="axiomatic",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=completeness.__name__,
    zen_partial=True,
    metric_func=builds(
        completeness,
        populate_full_signature=True,
        zen_partial=True,
    ),
)
ModuleRegistry.register_metric(
    sub_group="axiomatic",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=input_invariance.__name__,
    zen_partial=True,
    metric_func=builds(
        input_invariance,
        populate_full_signature=True,
        zen_partial=True,
    ),
)
ModuleRegistry.register_metric(
    sub_group="axiomatic",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=monotonicity_corr_and_non_sens.__name__,
    zen_partial=True,
    metric_func=builds(
        monotonicity_corr_and_non_sens,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_random_perturb_func,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)

# complexity
ModuleRegistry.register_metric(
    sub_group="complexity",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=complexity_entropy.__name__,
    zen_partial=True,
    metric_func=builds(
        complexity_entropy,
        populate_full_signature=True,
        zen_partial=True,
    ),
)
ModuleRegistry.register_metric(
    sub_group="complexity",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=complexity_sundararajan.__name__,
    zen_partial=True,
    metric_func=builds(
        complexity_sundararajan,
        populate_full_signature=True,
        zen_partial=True,
    ),
)
ModuleRegistry.register_metric(
    sub_group="complexity",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=effective_complexity.__name__,
    zen_partial=True,
    metric_func=builds(
        effective_complexity,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_random_perturb_func,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)
ModuleRegistry.register_metric(
    sub_group="complexity",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=sparseness.__name__,
    zen_partial=True,
    metric_func=builds(
        sparseness,
        populate_full_signature=True,
        zen_partial=True,
    ),
)

# faithfulness
ModuleRegistry.register_metric(
    sub_group="faithfulness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=aopc.__name__,
    zen_partial=True,
    metric_func=builds(
        aopc,
        populate_full_signature=True,
        zen_partial=True,
    ),
)

ModuleRegistry.register_metric(
    sub_group="faithfulness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=faithfulness_corr.__name__,
    zen_partial=True,
    metric_func=builds(
        faithfulness_corr,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_random_perturb_func,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)

ModuleRegistry.register_metric(
    sub_group="faithfulness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=faithfulness_estimate.__name__,
    zen_partial=True,
    metric_func=builds(
        faithfulness_estimate,
        populate_full_signature=True,
        zen_partial=True,
    ),
)

ModuleRegistry.register_metric(
    sub_group="faithfulness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=infidelity.__name__,
    zen_partial=True,
    metric_func=builds(
        infidelity,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_infidelity_perturb_fn,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)

ModuleRegistry.register_metric(
    sub_group="faithfulness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=monotonicity.__name__,
    zen_partial=True,
    metric_func=builds(
        monotonicity,
        populate_full_signature=True,
        zen_partial=True,
    ),
)

ModuleRegistry.register_metric(
    sub_group="faithfulness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=sensitivity_n.__name__,
    zen_partial=True,
    metric_func=builds(
        sensitivity_n,
        populate_full_signature=True,
        zen_partial=True,
    ),
)

# robustness
ModuleRegistry.register_metric(
    sub_group="robustness",
    module=_get_parent_module(__name__) + ".torchxai_metric",
    registered_class_or_func="TorchXAIMetric",
    name=sensitivity_max_and_avg.__name__,
    zen_partial=True,
    metric_func=builds(
        sensitivity_max_and_avg,
        populate_full_signature=True,
        zen_partial=True,
    ),
)
