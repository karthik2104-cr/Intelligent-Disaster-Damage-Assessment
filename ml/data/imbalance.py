"""Class imbalance handling.

The pipeline only computes the quantities a training loop needs; it does not
resample the data on disk. That is a deliberate choice, argued in
docs/data_pipeline.md: class weights leave the dataset and its measured
distribution intact, whereas oversampling duplicates images (which augmentation
already covers more cheaply) and undersampling throws away real data.

All weights are computed from the **train** split only. Deriving them from the
full dataset would let test-set composition influence training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ml.config import DataConfig
from ml.data.labels import LabelMapping


@dataclass
class ImbalanceReport:
    task: str
    strategy: str
    counts: dict[str, int] = field(default_factory=dict)
    proportions: dict[str, float] = field(default_factory=dict)
    class_weights: dict[str, float] = field(default_factory=dict)
    class_weights_by_index: dict[int, float] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    minority_share: float = 1.0
    severity: str = "none"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "Class Imbalance",
            "---------------",
            f"Task:            {self.task}",
            f"Imbalance ratio: {self.imbalance_ratio:.2f} (majority / minority)",
            f"Minority share:  {100 * self.minority_share:.2f}%",
            f"Severity:        {self.severity}",
            f"Strategy:        {self.strategy}",
            "",
            "Weights (train split, 'balanced')",
        ]
        width = max((len(name) for name in self.counts), default=0)
        for name, count in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            weight = self.class_weights.get(name, float("nan"))
            lines.append(f"  {name:<{width}} n={count:>6}  weight={weight:.4f}")
        return "\n".join(lines)


def classify_severity(imbalance_ratio: float) -> str:
    """Bucket the imbalance ratio so the chosen strategy can be justified.

    Thresholds are conventional rules of thumb, not results from this dataset:
    below 1.5 the classes are near-balanced; up to 3 the imbalance is mild and
    weighting is usually enough; up to 10 it is moderate; beyond that it is severe
    and needs sampling or loss changes as well.
    """
    if imbalance_ratio < 1.5:
        return "none"
    if imbalance_ratio < 3.0:
        return "mild"
    if imbalance_ratio < 10.0:
        return "moderate"
    return "severe"


def compute_class_weights(
    counts: Mapping[str, int], *, mode: str = "balanced"
) -> dict[str, float]:
    """Class weights from label counts.

    ``balanced`` matches scikit-learn's ``class_weight="balanced"``:
    ``n_samples / (n_classes * count)``, so weights average to 1.0 and can be
    dropped straight into a weighted cross-entropy loss.
    ``inverse`` uses ``1 / count`` normalised to a mean of 1.0.
    """
    positive = {name: int(count) for name, count in counts.items() if int(count) > 0}
    if not positive:
        raise ValueError("cannot compute class weights from an empty distribution")

    total = sum(positive.values())
    num_classes = len(positive)
    if mode == "balanced":
        weights = {name: total / (num_classes * count) for name, count in positive.items()}
    elif mode == "inverse":
        raw = {name: 1.0 / count for name, count in positive.items()}
        mean = sum(raw.values()) / len(raw)
        weights = {name: value / mean for name, value in raw.items()}
    else:
        raise ValueError(f"unsupported class_weight_mode {mode!r}")
    return weights


def analyse_imbalance(
    manifest: pd.DataFrame,
    config: DataConfig,
    *,
    label_mapping: LabelMapping,
    task: str | None = None,
    split: str = "train",
) -> ImbalanceReport:
    """Measure imbalance on one split and compute the weights for it."""
    target = task or config.task.primary
    frame = manifest.loc[manifest["split"] == split] if "split" in manifest.columns else manifest
    if frame.empty:
        raise ValueError(f"no rows for split {split!r}")

    counts = {str(name): int(value) for name, value in frame[target].value_counts().items()}
    # Include classes that exist in the mapping but not in this split, so a
    # missing class shows up as 0 instead of vanishing from the report.
    for name in label_mapping.classes:
        counts.setdefault(name, 0)

    total = sum(counts.values()) or 1
    present = [count for count in counts.values() if count > 0]
    ratio = max(present) / min(present) if present else 1.0

    weights = compute_class_weights(counts, mode=config.imbalance.class_weight_mode)
    return ImbalanceReport(
        task=target,
        strategy=config.imbalance.strategy,
        counts=counts,
        proportions={name: count / total for name, count in counts.items()},
        class_weights=weights,
        class_weights_by_index={
            label_mapping.encode(name): weight
            for name, weight in weights.items()
            if name in label_mapping.class_to_index
        },
        imbalance_ratio=ratio,
        minority_share=(min(present) / total if present else 1.0),
        severity=classify_severity(ratio),
    )


def class_weight_vector(report: ImbalanceReport, label_mapping: LabelMapping) -> np.ndarray:
    """Weights ordered by class index, ready for a weighted loss function."""
    return np.asarray(
        [report.class_weights.get(name, 1.0) for name in label_mapping.classes],
        dtype=np.float32,
    )


def sample_weights(
    labels: np.ndarray, report: ImbalanceReport, label_mapping: LabelMapping
) -> np.ndarray:
    """Per-sample weights, for a balanced sampler if Part 2 wants one.

    Provided so the alternative strategy is available without re-deriving it, but
    ``imbalance.strategy`` in the config selects class weighting by default.
    """
    weights = class_weight_vector(report, label_mapping)
    return weights[np.asarray(labels, dtype=int)]
