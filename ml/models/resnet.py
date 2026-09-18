"""ResNet-50 transfer learning with a 3-class head.

Uses torchvision's ImageNet-pretrained ResNet-50. The original 1000-way `fc`
layer is replaced by `Linear(2048, 3)`. By default the backbone is frozen and
only the new head is trained; `unfreeze_last_block()` then trains `layer4` +
the head at a lower learning rate.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Classifier(nn.Module):
    architecture = "resnet50"
    pretrained_weights = "IMAGENET1K_V2"

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    def unfreeze_last_block(self) -> None:
        """Train layer4 (highest-level residual block) plus the classifier."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]
