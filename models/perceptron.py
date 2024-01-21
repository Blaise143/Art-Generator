import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int,
                 out_features: int,
                 dropout: float = None) -> None:
        super().__init__()
        self.fc = nn.ModuleList()
        self.fc.extend(
            [
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
            ]
        )
        if dropout:
            self.fc.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        for module in self.fc:
            x = module(x)
        return x


if __name__ == "__main__":
    a = range(20, 0, -1)
    print(list(a))
