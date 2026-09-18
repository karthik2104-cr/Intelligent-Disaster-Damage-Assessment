"""PyTorch Dataset wrapper around the Part 1 ``ImageClassificationDataset``.

Part 1 returns HWC float32 arrays (Keras-style). PyTorch conv layers need CHW.
The permutation happens here, at the framework boundary, so Part 1 preprocessing
does not change.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.data.dataset import ImageClassificationDataset


class DamageSeverityTorchDataset(Dataset):
    """``(C, H, W) float32 tensor, int64 label`` view of a Part 1 dataset."""

    def __init__(self, source: ImageClassificationDataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    @property
    def num_classes(self) -> int:
        return self.source.num_classes

    @property
    def label_mapping(self):
        return self.source.label_mapping

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        array, label = self.source[index]
        if array.ndim != 3:
            raise ValueError(f"expected HWC image, got shape {array.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()
        return tensor, torch.tensor(label, dtype=torch.int64)
