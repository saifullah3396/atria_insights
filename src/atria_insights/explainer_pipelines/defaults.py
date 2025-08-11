# fmt: off
_METRICS_DEFAULTS = [
    # Axiomatic metrics
    {"/explainer_metric@explainer_metrics.completeness": "axiomatic/completeness"},
    {"/explainer_metric@explainer_metrics.input_invariance": "axiomatic/input_invariance"},
    {"/explainer_metric@explainer_metrics.monotonicity_corr_and_non_sens": "axiomatic/monotonicity_corr_and_non_sens"},

    # Complexity metrics
    {"/explainer_metric@explainer_metrics.complexity_entropy_feature_grouped": "complexity/complexity_entropy_feature_grouped"},
    {"/explainer_metric@explainer_metrics.complexity_sundararajan_feature_grouped": "complexity/complexity_sundararajan_feature_grouped"},
    {"/explainer_metric@explainer_metrics.effective_complexity": "complexity/effective_complexity"},
    {"/explainer_metric@explainer_metrics.sparseness_feature_grouped": "complexity/sparseness_feature_grouped"},

    # Faithfulness metrics
    {"/explainer_metric@explainer_metrics.aopc": "faithfulness/aopc"},
    {"/explainer_metric@explainer_metrics.faithfulness_corr": "faithfulness/faithfulness_corr"},
    {"/explainer_metric@explainer_metrics.faithfulness_estimate": "faithfulness/faithfulness_estimate"},
    {"/explainer_metric@explainer_metrics.infidelity": "faithfulness/infidelity"},
    {"/explainer_metric@explainer_metrics.monotonicity": "faithfulness/monotonicity"},
    {"/explainer_metric@explainer_metrics.sensitivity_n_2_normalized": "faithfulness/sensitivity_n_2_normalized"},# @package __global__
    {"/explainer_metric@explainer_metrics.sensitivity_n_4_normalized": "faithfulness/sensitivity_n_4_normalized"},
    {"/explainer_metric@explainer_metrics.sensitivity_n_6_normalized": "faithfulness/sensitivity_n_6_normalized"},

    # Robustness metrics
    {"/explainer_metric@explainer_metrics.sensitivity_max_and_avg": "robustness/sensitivity_max_and_avg"},
]
# fmt: on
