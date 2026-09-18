"""A deliberately simple baseline: no convolutions.

Global 8×8 average pooling reduces each image to a 192-d colour/layout vector,
then a two-layer MLP predicts the three damage classes. This is not expected to
be a strong model. It exists to put a real, trained number on the board before
any CNN is introduced.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearBaseline(nn.Module):
    def __init__(self, num_classes: int = 3, pool_size: int = 8, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * pool_size * pool_size, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(x))
