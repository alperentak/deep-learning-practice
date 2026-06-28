import torch
from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, in_shape: int, hidden_units: int, out_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=out_shape),
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
