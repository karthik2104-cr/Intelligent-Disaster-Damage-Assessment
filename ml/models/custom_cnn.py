"""Custom CNN trained from scratch (no pretrained weights).

Four conv blocks extract spatial features at increasing channel depth; each
block is conv-BN-ReLU twice, then 2×2 max-pool. BatchNorm stabilises training
on a moderately imbalanced dataset. Max-pool provides translation-tolerant
downsampling. Global average pooling replaces a large flatten+FC layer so the
parameter count stays modest on a 4 GB GPU. Dropout on the head reduces
memorisation of social-media artefacts. The final linear layer emits 3 logits.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32),    # 224 -> 112
            _block(32, 64),   # 112 -> 56
            _block(64, 128),  # 56 -> 28
            _block(128, 256), # 28 -> 14
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
