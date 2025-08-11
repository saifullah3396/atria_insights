from hydra_zen import builds
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
    default_fixed_baseline_perturb_func,
    default_infidelity_perturb_fn,
)
from torchxai.metrics.complexity.complexity_entropy import (
    complexity_entropy_feature_grouped,
)
from torchxai.metrics.complexity.complexity_sundararajan import (
    complexity_sundararajan_feature_grouped,
)
from torchxai.metrics.complexity.sparseness import sparseness_feature_grouped

from atria_insights.metrics.torchxai_metric import TorchXAIMetric
from atria_insights.registry import EXPLAINER_METRIC

# axiomatic
EXPLAINER_METRIC.register(
    name="axiomatic/" + completeness.__name__,
    metric_func=builds(
        completeness,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="axiomatic/" + input_invariance.__name__,
    metric_func=builds(
        input_invariance,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="axiomatic/" + monotonicity_corr_and_non_sens.__name__,
    metric_func=builds(
        monotonicity_corr_and_non_sens,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_fixed_baseline_perturb_func,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)(TorchXAIMetric)

# complexity
EXPLAINER_METRIC.register(
    name="complexity/ " + complexity_entropy.__name__,
    metric_func=builds(
        complexity_entropy,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="complexity/ " + complexity_entropy_feature_grouped.__name__,
    metric_func=builds(
        complexity_entropy_feature_grouped,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="complexity/ " + complexity_sundararajan.__name__,
    metric_func=builds(
        complexity_sundararajan,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="complexity/ " + complexity_sundararajan_feature_grouped.__name__,
    metric_func=builds(
        complexity_sundararajan_feature_grouped,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="complexity/ " + effective_complexity.__name__,
    metric_func=builds(
        effective_complexity,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_fixed_baseline_perturb_func,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="complexity/ " + sparseness.__name__,
    metric_func=builds(
        sparseness,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
EXPLAINER_METRIC.register(
    name="complexity/ " + sparseness_feature_grouped.__name__,
    metric_func=builds(
        sparseness_feature_grouped,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)

# faithfulness
EXPLAINER_METRIC.register(
    name="faithfulness/" + aopc.__name__,
    metric_func=builds(
        aopc,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)

EXPLAINER_METRIC.register(
    name="faithfulness/" + faithfulness_corr.__name__,
    metric_func=builds(
        faithfulness_corr,
        populate_full_signature=True,
        zen_partial=True,
        perturb_func=builds(
            default_fixed_baseline_perturb_func,
            populate_full_signature=True,
            zen_partial=False,
        ),
    ),
)(TorchXAIMetric)

EXPLAINER_METRIC.register(
    name="faithfulness/" + faithfulness_estimate.__name__,
    metric_func=builds(
        faithfulness_estimate,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)

EXPLAINER_METRIC.register(
    name="faithfulness/" + infidelity.__name__,
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
)(TorchXAIMetric)

EXPLAINER_METRIC.register(
    name="faithfulness/" + monotonicity.__name__,
    metric_func=builds(
        monotonicity,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)

EXPLAINER_METRIC.register(
    name="faithfulness/" + sensitivity_n.__name__,
    metric_func=builds(
        sensitivity_n,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)

# robustness
EXPLAINER_METRIC.register(
    name="robustness/" + sensitivity_max_and_avg.__name__,
    metric_func=builds(
        sensitivity_max_and_avg,
        populate_full_signature=True,
        zen_partial=True,
    ),
)(TorchXAIMetric)
