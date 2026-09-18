"""Model factory used by training scripts and tests."""

from __future__ import annotations

from torch import nn

from ml.models.baseline import LinearBaseline
from ml.models.custom_cnn import CustomCNN
from ml.models.efficientnet import EfficientNetB0Classifier
from ml.models.resnet import ResNet50Classifier

MODEL_NAMES = ("baseline", "custom_cnn", "resnet50", "efficientnet_b0")


def build_model(name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "baseline":
        return LinearBaseline(num_classes=num_classes)
    if key == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if key == "resnet50":
        return ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    if key in {"efficientnet_b0", "efficientnetb0"}:
        return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
