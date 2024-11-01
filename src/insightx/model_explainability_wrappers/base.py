from functools import wraps

import torch
from dacite import Any
from pyparsing import abstractmethod

from insightx.utilities.common import filter_kwargs


def forward_wrapper(forward_func):
    @wraps(forward_func)
    def inner(*args, **kwargs):
        return forward_func(*args, **kwargs)

    return inner


class ModelExplainabilityWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self._model = model

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @filter_kwargs
    @abstractmethod
    def prepare_explainable_inputs_from_inputs(self, **kwargs) -> Any:
        pass

    def toggle_explainability(self, convert_to_explainable: bool = True) -> Any:
        if convert_to_explainable:
            self.patch_forward_for_explainability()
        else:
            self.restore_forward()

    def patch_forward_for_explainability(self, **kwargs) -> Any:
        pass

    def restore_forward(self, **kwargs) -> Any:
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
