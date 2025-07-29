# _grad
from torchxai.explainers._grad.deeplift import DeepLiftExplainer  # noqa
from torchxai.explainers._grad.deeplift_shap import DeepLiftShapExplainer  # noqa
from torchxai.explainers._grad.gradient_shap import GradientShapExplainer  # noqa
from torchxai.explainers._grad.guided_backprop import GuidedBackpropExplainer  # noqa
from torchxai.explainers._grad.input_x_gradient import InputXGradientExplainer  # noqa
from torchxai.explainers._grad.saliency import SaliencyExplainer  # noqa

# _perturbation
from torchxai.explainers._perturbation.kernel_shap import KernelShapExplainer  # noqa
from torchxai.explainers._perturbation.lime import LimeExplainer  # noqa
from torchxai.explainers._perturbation.feature_ablation import FeatureAblationExplainer  # noqa
from torchxai.explainers._perturbation.occlusion import OcclusionExplainer  # noqa
from torchxai.explainers.explainer import Explainer  # noqa

# _random
from torchxai.explainers.random import RandomExplainer  # noqa

from atria_insights.registry import EXPLAINER

# grad
EXPLAINER.register(name="grad/saliency")(SaliencyExplainer)
EXPLAINER.register(name="grad/deeplift")(DeepLiftExplainer)
EXPLAINER.register(name="grad/deeplift_shap")(DeepLiftShapExplainer)
EXPLAINER.register(name="grad/gradient_shap")(GradientShapExplainer)
EXPLAINER.register(name="grad/guided_backprop")(GuidedBackpropExplainer)
EXPLAINER.register(name="grad/input_x_gradient")(InputXGradientExplainer)
EXPLAINER.register(name="grad/integrated_gradients")

# perturbation
EXPLAINER.register(name="perturbation/feature_ablation")(FeatureAblationExplainer)
EXPLAINER.register(name="perturbation/kernel_shap")(KernelShapExplainer)
EXPLAINER.register(name="perturbation/lime")(LimeExplainer)
EXPLAINER.register(name="perturbation/occlusion")(OcclusionExplainer)

# random
EXPLAINER.register(name="random")(RandomExplainer)
