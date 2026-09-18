"""Classification metrics. Validation during training uses loss + accuracy;
comparison uses precision / recall / F1 as well."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMeter:
    """Streaming loss / accuracy tracker for one split of one epoch."""

    loss_sum: float = 0.0
    correct: int = 0
    count: int = 0
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)

    def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        batch = int(labels.shape[0])
        self.loss_sum += float(loss.detach()) * batch
        preds = logits.detach().argmax(dim=1)
        self.correct += int((preds == labels).sum().item())
        self.count += batch
        self.y_true.extend(labels.detach().cpu().tolist())
        self.y_pred.extend(preds.cpu().tolist())

    @property
    def loss(self) -> float:
        return self.loss_sum / max(self.count, 1)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.count, 1)

    def as_dict(self) -> dict[str, float]:
        return {"loss": self.loss, "accuracy": self.accuracy, "n": float(self.count)}


def summarise_predictions(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    *,
    class_names: list[str],
) -> dict[str, object]:
    """Full comparison metrics. All numbers come from the given predictions."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    per_class_precision = precision_score(true, pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(true, pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(true, pred, labels=labels, average=None, zero_division=0)
    report = classification_report(
        true,
        pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "n": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "per_class": {
            class_names[i]: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
                "support": int((true == i).sum()),
            }
            for i in labels
        },
        "sklearn_report": report,
    }
