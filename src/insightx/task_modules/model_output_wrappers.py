import torch
from torch import nn


class SoftmaxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if hasattr(self.model, "logits"):
            return self.softmax(self.model.logits)
        else:
            return self.softmax(self.model(*args, **kwargs))
