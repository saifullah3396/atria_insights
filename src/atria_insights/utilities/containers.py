import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Union

import torch


@dataclasses.dataclass
class ModelInputs:
    explained_inputs: Mapping[str, torch.Tensor] = None
    additional_forward_kwargs: Dict[str, Any] = None


@dataclasses.dataclass(frozen=True)
class ExplainerStepInputs:
    model_inputs: ModelInputs = None
    baselines: Mapping[str, torch.Tensor] = None
    metric_baselines: Mapping[str, torch.Tensor] = None
    feature_masks: Mapping[str, torch.Tensor] = None
    total_features: Optional[int] = None
    constant_shifts: Mapping[str, torch.Tensor] = None
    input_layer_names: Mapping[str, str] = None
    train_baselines: Mapping[str, torch.Tensor] = None
    frozen_features: torch.Tensor = None


@dataclasses.dataclass(frozen=True)
class ExplainerStepOutput:
    # sample metadata
    index: int = None
    sample_id: list[str] = None

    # sample inputs
    explainer_step_inputs: ExplainerStepInputs = None
    target: Union[torch.Tensor, List[torch.Tensor]] = None
    model_outputs: torch.Tensor = None

    # explanations
    explanations: Union[torch.Tensor, List[torch.Tensor]] = None
    reduced_explanations: Union[torch.Tensor, List[torch.Tensor]] = None
