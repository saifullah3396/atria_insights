import dataclasses
from typing import Any, Dict, List, Mapping, Union

import torch


@dataclasses.dataclass
class ExplainerInputs:
    inputs: Mapping[str, torch.Tensor] = None
    baselines: Mapping[str, torch.Tensor] = None
    feature_masks: Mapping[str, torch.Tensor] = None
    additional_forward_kwargs: Dict[str, Any] = None
    constant_shifts: Mapping[str, torch.Tensor] = None
    input_layer_names: Mapping[str, str] = None


@dataclasses.dataclass
class ExplanationModelOutput:
    explanations: Union[torch.Tensor, List[torch.Tensor]] = None
    reduced_explanations: Union[torch.Tensor, List[torch.Tensor]] = None
    explainer_inputs: ExplainerInputs = None
    target: Union[torch.Tensor, List[torch.Tensor]] = None
