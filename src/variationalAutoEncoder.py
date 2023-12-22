import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):

    def __init__(self, in_channels: int, hid_dim: int):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass
