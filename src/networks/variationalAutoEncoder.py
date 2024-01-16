import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import pytorch_lightning as pl


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, channels_order: list, latent_dim: int):
        super().__init__()

    def encode(self):
        ...

    def decode(self):
        ...

    def reparametrize(self):
        ...

    def kl_divergence_loss(self):
        ...

    def forward(self, x: torch.Tensor):
        ...
