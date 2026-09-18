"""Grad-CAM: class-activation heatmap for convolutional models.

This is a visualization of spatial activations weighted by the gradient of the
chosen class score. It does not prove that the model used those pixels as a
causal reason for the label.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn


def last_conv2d(model: nn.Module) -> nn.Conv2d:
    conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv = module
    if conv is None:
        raise ValueError("model has no Conv2d layer (Grad-CAM is not defined)")
    return conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target = target_layer or last_conv2d(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.target.register_forward_hook(self._on_forward),
            self.target.register_full_backward_hook(self._on_backward),
        ]

    def _on_forward(self, _module, _inputs, output) -> None:
        self._activations = output

    def _on_backward(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, images: torch.Tensor, class_index: int) -> np.ndarray:
        if images.ndim != 4 or images.shape[0] != 1:
            raise ValueError("Grad-CAM expects a single image batch of shape (1, C, H, W)")
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        score = logits[0, class_index]
        score.backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("hooks did not capture activations/gradients")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1)
        cam = torch.relu(cam)[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        if was_training:
            self.model.train()
        return cam.detach().cpu().numpy()


def tensor_to_image(tensor: torch.Tensor, denormalise: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    return denormalise(array)
