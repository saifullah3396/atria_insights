# _grad
from torchxai.explainers._grad.deeplift import DeepLiftExplainer  # noqa
from torchxai.explainers._grad.deeplift_shap import DeepLiftShapExplainer  # noqa
from torchxai.explainers._grad.gradient_shap import GradientShapExplainer  # noqa
from torchxai.explainers._grad.guided_backprop import GuidedBackpropExplainer  # noqa
from torchxai.explainers._grad.input_x_gradient import InputXGradientExplainer  # noqa
from torchxai.explainers._grad.integrated_gradients import (
    IntegratedGradientsExplainer,
)  # noqa
from torchxai.explainers._grad.saliency import SaliencyExplainer  # noqa

# _perturbation
from torchxai.explainers._perturbation.feature_ablation import (
    FeatureAblationExplainer,
)  # noqa
from torchxai.explainers._perturbation.kernel_shap import KernelShapExplainer  # noqa
from torchxai.explainers._perturbation.lime import LimeExplainer  # noqa
from torchxai.explainers._perturbation.occlusion import OcclusionExplainer  # noqa
from torchxai.explainers.explainer import Explainer  # noqa

# _random
from torchxai.explainers.random import RandomExplainer  # noqa

from insightx.registry.module_registry import ModuleRegistry

# grad
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.saliency",
    registered_class_or_func="SaliencyExplainer",
    name="saliency",
)
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.deeplift",
    registered_class_or_func="DeepLiftExplainer",
    name="deeplift",
)
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.deeplift_shap",
    registered_class_or_func="DeepLiftShapExplainer",
    name="deeplift_shap",
)
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.gradient_shap",
    registered_class_or_func="GradientShapExplainer",
    name="gradient_shap",
)
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.guided_backprop",
    registered_class_or_func="GuidedBackpropExplainer",
    name="guided_backprop",
)
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.input_x_gradient",
    registered_class_or_func="InputXGradientExplainer",
    name="input_x_gradient",
)
ModuleRegistry.register_explainer(
    sub_group="grad",
    module="torchxai.explainers._grad.integrated_gradients",
    registered_class_or_func="IntegratedGradientsExplainer",
    name="integrated_gradients",
)

# perturbation
ModuleRegistry.register_explainer(
    sub_group="perturbation",
    module="torchxai.explainers._perturbation.feature_ablation",
    registered_class_or_func="FeatureAblationExplainer",
    name="feature_ablation",
)
ModuleRegistry.register_explainer(
    sub_group="perturbation",
    module="torchxai.explainers._perturbation.kernel_shap",
    registered_class_or_func="KernelShapExplainer",
    name="kernel_shap",
)
ModuleRegistry.register_explainer(
    sub_group="perturbation",
    module="torchxai.explainers._perturbation.lime",
    registered_class_or_func="LimeExplainer",
    name="lime",
)
ModuleRegistry.register_explainer(
    sub_group="perturbation",
    module="torchxai.explainers._perturbation.occlusion",
    registered_class_or_func="OcclusionExplainer",
    name="occlusion",
)

# random
ModuleRegistry.register_explainer(
    sub_group="",
    module="torchxai.explainers.random",
    registered_class_or_func="RandomExplainer",
    name="random",
)
