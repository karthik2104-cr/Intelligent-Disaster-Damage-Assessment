"""Checkpoint save/load. Stores enough to recreate inference later."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int,
    metrics: Mapping[str, Any],
    class_to_index: Mapping[str, int],
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "class_to_index": dict(class_to_index),
        "config": dict(config),
        "extra": dict(extra or {}),
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
