import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Union

import torch


@dataclasses.dataclass
class ExplainerArguments:
    inputs: Mapping[str, torch.Tensor] = None
    baselines: Mapping[str, torch.Tensor] = None
    metric_baselines: Mapping[str, torch.Tensor] = None
    feature_masks: Mapping[str, torch.Tensor] = None
    total_features: Optional[int] = None
    additional_forward_kwargs: Dict[str, Any] = None
    constant_shifts: Mapping[str, torch.Tensor] = None
    input_layer_names: Mapping[str, str] = None
    train_baselines: Mapping[str, torch.Tensor] = None
    frozen_features: torch.Tensor = None


@dataclasses.dataclass(frozen=True)
class ExplanationStepMetadata:
    sample_keys: List[str] = None
    explainer_args: ExplainerArguments = None
    target: Union[torch.Tensor, List[torch.Tensor]] = None
    target_word_ids: List[List[int]] = None
    model_outputs: Union[torch.Tensor, List[torch.Tensor]] = None
    dataset_labels: Any = None


@dataclasses.dataclass(frozen=True)
class ExplanationStepOutput:
    metadata: ExplanationStepMetadata = None
    explanations: Union[torch.Tensor, List[torch.Tensor]] = None
    reduced_explanations: Union[torch.Tensor, List[torch.Tensor]] = None
