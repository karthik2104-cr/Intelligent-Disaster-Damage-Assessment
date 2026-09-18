"""EfficientNet-B0 transfer learning with a 3-class head.

torchvision EfficientNet-B0 expects the same 224×224 ImageNet-normalised RGB
input that Part 1 already produces. The pipeline is therefore unchanged; only
the classifier (`Linear(1280, 3)`) is new. Backbone freeze / last-block
fine-tune mirrors the ResNet-50 protocol.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Classifier(nn.Module):
    architecture = "efficientnet_b0"
    pretrained_weights = "IMAGENET1K_V1"

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("classifier.")

    def unfreeze_last_block(self) -> None:
        """Train the last features block (features.7) plus the classifier."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = (
                name.startswith("features.7.") or name.startswith("classifier.")
            )

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]
