"""Load the Part 1 class-weight vector in label-index order."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ml.config import DataConfig
from ml.data.labels import LabelMapping


def class_weight_tensor(
    data_config: DataConfig,
    mapping: LabelMapping,
    device: torch.device | None = None,
) -> torch.Tensor:
    path = data_config.paths.metadata / "class_weights.json"
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_classes = payload.get("classes")
    if stored_classes != list(mapping.classes):
        raise ValueError(
            "class_weights.json class order does not match label_mapping.json: "
            f"{stored_classes} vs {list(mapping.classes)}"
        )
    weights = torch.tensor(payload["weights"], dtype=torch.float32)
    if device is not None:
        weights = weights.to(device)
    return weights
