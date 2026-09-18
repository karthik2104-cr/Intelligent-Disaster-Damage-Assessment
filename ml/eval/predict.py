"""Load best.pt checkpoints and collect validation predictions. No training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.engine.checkpoints import load_checkpoint
from ml.models.factory import build_model

CLASS_NAMES: tuple[str, str, str] = ("little_or_none", "mild", "severe")


@dataclass
class PredictionBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    indices: np.ndarray
    class_names: tuple[str, ...] = CLASS_NAMES

    @property
    def confidence(self) -> np.ndarray:
        return self.probs.max(axis=1)

    def __len__(self) -> int:
        return int(self.y_true.shape[0])


def load_eval_model(
    name: str,
    checkpoint: str | Path,
    device: torch.device,
    *,
    num_classes: int = 3,
) -> torch.nn.Module:
    """Build architecture without downloading weights, then load ``best.pt``."""
    model = build_model(name, num_classes=num_classes, pretrained=False)
    payload = load_checkpoint(checkpoint, model=model, map_location=device)
    mapping = payload.get("class_to_index")
    if mapping:
        expected = {name_: idx for idx, name_ in enumerate(CLASS_NAMES)}
        if dict(mapping) != expected:
            raise ValueError(f"checkpoint class map {mapping} != {expected}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool = False,
) -> PredictionBundle:
    """Forward-only pass. Sampler must be sequential so ``indices`` match the dataset."""
    y_true: list[int] = []
    y_pred: list[int] = []
    probs: list[np.ndarray] = []
    offset = 0
    amp = use_amp and device.type == "cuda"
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            batch_probs = torch.softmax(logits.float(), dim=1)
        pred = batch_probs.argmax(dim=1)
        n = int(labels.shape[0])
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        probs.append(batch_probs.cpu().numpy())
        offset += n
    stacked = np.concatenate(probs, axis=0)
    true = np.asarray(y_true, dtype=np.int64)
    return PredictionBundle(
        y_true=true,
        y_pred=np.asarray(y_pred, dtype=np.int64),
        probs=stacked.astype(np.float64),
        indices=np.arange(len(true), dtype=np.int64),
    )


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def compare_to_reference(
    summary: dict,
    reference: dict,
    *,
    keys: Sequence[str] = ("accuracy", "macro_precision", "macro_recall", "macro_f1"),
    atol: float = 2e-3,
) -> dict[str, dict[str, float]]:
    """Compare reproduced metrics to Part 2 JSON. Returns per-key deltas."""
    report: dict[str, dict[str, float]] = {}
    for key in keys:
        got = float(summary[key])
        expected = float(reference[key])
        delta = abs(got - expected)
        report[key] = {"reproduced": got, "part2": expected, "abs_delta": delta, "ok": delta <= atol}
    mild_got = float(summary["per_class"]["mild"]["f1"])
    mild_exp = float(reference["per_class"]["mild"]["f1"])
    delta = abs(mild_got - mild_exp)
    report["mild_f1"] = {
        "reproduced": mild_got,
        "part2": mild_exp,
        "abs_delta": delta,
        "ok": delta <= atol,
    }
    return report
