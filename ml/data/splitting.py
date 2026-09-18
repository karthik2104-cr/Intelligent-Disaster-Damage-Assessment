"""Train / validation / test splitting.

MEDIC publishes official train/dev/test TSVs, so the default strategy is to
preserve them: results stay comparable with the published baselines, and the
official test set stays a fixed, untouched evaluation set. ``regenerate`` exists
as an alternative for ablations and is grouped and stratified rather than a plain
random shuffle.

Whichever strategy is used, :func:`check_leakage` runs afterwards and reports any
sample identity or image content that appears in more than one split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ml.config import DataConfig

SPLITS: tuple[str, ...] = ("train", "val", "test")

# MEDIC calls its validation split "dev"; the pipeline standardises on "val".
OFFICIAL_SPLIT_ALIASES: Mapping[str, str] = {
    "train": "train",
    "dev": "val",
    "val": "val",
    "validation": "val",
    "test": "test",
}


class SplitError(ValueError):
    """Raised when a split cannot be produced or fails its integrity checks."""


@dataclass
class LeakageReport:
    """Overlap between splits, by sample id and by image content hash."""

    id_column: str
    duplicate_ids_within_splits: dict[str, int] = field(default_factory=dict)
    shared_ids: dict[str, int] = field(default_factory=dict)
    shared_content_hashes: dict[str, int] = field(default_factory=dict)
    shared_groups: dict[str, int] = field(default_factory=dict)

    @property
    def has_leakage(self) -> bool:
        return bool(self.shared_ids or self.shared_content_hashes)

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "has_leakage": self.has_leakage}

    def render(self) -> str:
        lines = ["Leakage check", "-------------"]
        lines.append(f"Duplicate ids within a split: {self.duplicate_ids_within_splits or 'none'}")
        lines.append(f"Ids shared between splits:    {self.shared_ids or 'none'}")
        lines.append(f"Content shared between splits: {self.shared_content_hashes or 'none'}")
        lines.append(f"Groups shared between splits:  {self.shared_groups or 'none'}")
        lines.append(f"Result: {'LEAKAGE DETECTED' if self.has_leakage else 'clean'}")
        return "\n".join(lines)


@dataclass
class SplitReport:
    strategy: str
    seed: int
    counts: dict[str, int] = field(default_factory=dict)
    proportions: dict[str, float] = field(default_factory=dict)
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    class_proportions: dict[str, dict[str, float]] = field(default_factory=dict)
    leakage: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    def render(self) -> str:
        lines = [
            "Split Summary",
            "-------------",
            f"Strategy: {self.strategy}",
            f"Seed:     {self.seed}",
            "",
        ]
        for split in SPLITS:
            if split in self.counts:
                lines.append(
                    f"{split:<5}: {self.counts[split]:>6} "
                    f"({100 * self.proportions.get(split, 0.0):5.2f}%)"
                )
        for split, distribution in self.class_distribution.items():
            lines += ["", f"{split} class distribution"]
            total = sum(distribution.values()) or 1
            width = max((len(name) for name in distribution), default=0)
            for name, count in sorted(distribution.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:<{width}} : {count:>6}  ({100 * count / total:5.2f}%)")
        return "\n".join(lines)


def normalise_split_names(manifest: pd.DataFrame, column: str = "official_split") -> pd.Series:
    """Map the dataset's own split names onto ``train``/``val``/``test``."""
    if column not in manifest.columns:
        raise SplitError(f"manifest has no {column!r} column")
    raw = manifest[column].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw) - set(OFFICIAL_SPLIT_ALIASES))
    if unknown:
        raise SplitError(f"unrecognised split names in {column!r}: {unknown}")
    return raw.map(OFFICIAL_SPLIT_ALIASES)


def apply_official_split(manifest: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Attach a ``split`` column taken from the dataset's published assignment."""
    result = manifest.copy()
    result["split"] = normalise_split_names(result)
    empty = [split for split in SPLITS if not (result["split"] == split).any()]
    if empty:
        raise SplitError(f"official split assignment produced empty splits: {empty}")
    return result


def regenerate_split(manifest: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Build a fresh grouped, stratified split with a fixed seed.

    Groups (``split.group_column``, e.g. the disaster event) are kept whole so
    that images from one event cannot appear in two splits - near-duplicate
    photographs of the same scene are common in social media data and would
    otherwise leak. Groups are assigned greedily to whichever split is furthest
    below its target size, processing the largest groups first, which keeps the
    label distribution close to the overall one without breaking any group.
    """
    group_column = config.split.group_column
    stratify_column = config.split.stratify_column
    if group_column not in manifest.columns:
        raise SplitError(f"manifest has no group column {group_column!r}")
    if stratify_column not in manifest.columns:
        raise SplitError(f"manifest has no stratify column {stratify_column!r}")

    ratios = {split: float(config.split.ratios[split]) for split in SPLITS}
    total = len(manifest)
    targets = {split: ratios[split] * total for split in SPLITS}

    classes = sorted(manifest[stratify_column].dropna().unique())
    overall = manifest[stratify_column].value_counts(normalize=True)

    # Per-group class histograms, largest group first.
    histograms = (
        manifest.groupby(group_column)[stratify_column]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=classes, fill_value=0)
    )
    order = histograms.sum(axis=1).sort_values(ascending=False, kind="mergesort").index

    rng = np.random.default_rng(config.seed)
    assigned: dict[str, list] = {split: [] for split in SPLITS}
    sizes = {split: 0.0 for split in SPLITS}
    class_totals = {split: np.zeros(len(classes)) for split in SPLITS}

    for group in order:
        counts = histograms.loc[group].to_numpy(dtype=float)
        size = counts.sum()
        best, best_cost = None, None
        for split in SPLITS:
            if targets[split] <= 0:
                continue
            # Prefer the split that is proportionally emptiest, then the one whose
            # class mix would drift least from the overall distribution.
            fill = (sizes[split] + size) / targets[split]
            projected = class_totals[split] + counts
            projected_total = projected.sum()
            drift = (
                float(
                    np.abs(
                        projected / projected_total
                        - overall.reindex(classes, fill_value=0.0).to_numpy()
                    ).sum()
                )
                if projected_total
                else 0.0
            )
            cost = (fill, drift, rng.random())
            if best_cost is None or cost < best_cost:
                best, best_cost = split, cost
        if best is None:
            raise SplitError("no split has a positive target size")
        assigned[best].append(group)
        sizes[best] += size
        class_totals[best] += counts

    _fill_empty_splits(assigned, histograms, sizes, class_totals, targets)

    lookup = {group: split for split, groups in assigned.items() for group in groups}
    result = manifest.copy()
    result["split"] = result[group_column].map(lookup)
    if result["split"].isna().any():
        raise SplitError("some samples were not assigned to a split")
    empty = [split for split in SPLITS if not (result["split"] == split).any()]
    if empty:
        raise SplitError(
            f"regenerated split produced empty splits {empty}; there are probably too few "
            f"distinct {group_column!r} groups for these ratios"
        )
    return result


def _fill_empty_splits(
    assigned: dict[str, list],
    histograms: pd.DataFrame,
    sizes: dict[str, float],
    class_totals: dict[str, np.ndarray],
    targets: dict[str, float],
) -> None:
    """Move the smallest surplus group into any split that received nothing.

    The greedy pass prefers the emptiest fractional fill, which with a handful
    of large groups can leave val/test empty. If there are at least as many
    groups as non-zero-target splits, every such split should get one group.
    """
    nonempty_targets = [split for split, target in targets.items() if target > 0]

    def group_size(group) -> float:
        return float(histograms.loc[group].sum())

    for _ in range(len(nonempty_targets)):
        empty = [split for split in nonempty_targets if not assigned[split]]
        if not empty:
            return
        donors = [split for split in nonempty_targets if len(assigned[split]) > 1]
        if not donors:
            return
        donor = max(donors, key=lambda split: (len(assigned[split]), sizes[split]))
        group = min(assigned[donor], key=group_size)
        dest = empty[0]
        assigned[donor].remove(group)
        assigned[dest].append(group)
        counts = histograms.loc[group].to_numpy(dtype=float)
        size = float(counts.sum())
        sizes[donor] -= size
        sizes[dest] += size
        class_totals[donor] -= counts
        class_totals[dest] += counts


def make_splits(manifest: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    if config.split.strategy == "official":
        return apply_official_split(manifest, config)
    return regenerate_split(manifest, config)


def check_leakage(manifest: pd.DataFrame, config: DataConfig) -> LeakageReport:
    """Look for the same sample, or the same image bytes, in two splits."""
    id_column = config.split.id_column
    if "split" not in manifest.columns:
        raise SplitError("manifest has no 'split' column; run make_splits() first")
    if id_column not in manifest.columns:
        raise SplitError(f"manifest has no id column {id_column!r}")

    report = LeakageReport(id_column=id_column)

    for split, frame in manifest.groupby("split"):
        repeats = int(frame[id_column].duplicated().sum())
        if repeats:
            report.duplicate_ids_within_splits[str(split)] = repeats

    def overlaps(column: str) -> dict[str, int]:
        found: dict[str, int] = {}
        by_split = {
            str(split): set(frame[column].dropna())
            for split, frame in manifest.groupby("split")
        }
        names = sorted(by_split)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = by_split[left] & by_split[right]
                if shared:
                    found[f"{left}|{right}"] = len(shared)
        return found

    report.shared_ids = overlaps(id_column)
    if "content_hash" in manifest.columns:
        report.shared_content_hashes = overlaps("content_hash")
    group_column = config.split.group_column
    if group_column in manifest.columns:
        report.shared_groups = overlaps(group_column)
    return report


def build_split_report(manifest: pd.DataFrame, config: DataConfig) -> SplitReport:
    report = SplitReport(strategy=config.split.strategy, seed=config.seed)
    total = len(manifest) or 1
    for split in SPLITS:
        frame = manifest.loc[manifest["split"] == split]
        report.counts[split] = int(len(frame))
        report.proportions[split] = len(frame) / total
        target = config.task.primary
        if target in frame.columns and not frame.empty:
            counts = {str(k): int(v) for k, v in frame[target].value_counts().items()}
            report.class_distribution[split] = counts
            split_total = sum(counts.values()) or 1
            report.class_proportions[split] = {
                name: count / split_total for name, count in counts.items()
            }
    report.leakage = check_leakage(manifest, config).as_dict()
    return report
