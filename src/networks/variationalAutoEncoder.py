import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import pytorch_lightning as pl


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, channels_order: list, latent_dim: int):
        super().__init__()

        encode_layers = []
        for i in range(len(channels_order) - 1):
            encode_layers.extend(
                [
                    nn.Conv2d(
                        channels_order[i], channels_order[i + 1], kernel_size=3, stride=2),
                    nn.BatchNorm2d(channels_order[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ]
            )
        self.encoder = nn.Sequential(*encode_layers)

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
