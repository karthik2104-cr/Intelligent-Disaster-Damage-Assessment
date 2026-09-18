"""Dataset validation and cleaning.

Two responsibilities, deliberately separated:

* :func:`scan_manifest` measures the dataset - it reads every file, records
  dimensions, formats, channels and content hashes, and flags problems. It never
  removes anything.
* :func:`clean_manifest` decides what to exclude, and records the reason for
  every exclusion in an ``exclusion_reason`` column plus a
  :class:`CleaningReport`.

Nothing here writes to ``data/raw``. Cleaning produces a new manifest; the raw
dataset stays exactly as downloaded.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from ml.config import DataConfig
from ml.data.io import probe_image
from ml.data.labels import EXPECTED_CLASSES

# Columns added to a manifest by scan_manifest().
PROBE_COLUMNS = (
    "exists",
    "readable",
    "error",
    "format",
    "mode",
    "width",
    "height",
    "channels",
    "file_bytes",
    "is_animated",
    "content_hash",
)

# Issue flags. Each is a boolean column; a sample can carry several at once.
ISSUE_MISSING_FILE = "issue_missing_file"
ISSUE_UNREADABLE = "issue_unreadable"
ISSUE_BAD_EXTENSION = "issue_bad_extension"
ISSUE_TOO_SMALL = "issue_too_small"
ISSUE_MISSING_LABEL = "issue_missing_label"
ISSUE_INVALID_LABEL = "issue_invalid_label"
ISSUE_DUPLICATE = "issue_duplicate_content"
ISSUE_CONFLICTING_DUPLICATE = "issue_conflicting_duplicate"

ISSUE_COLUMNS = (
    ISSUE_MISSING_FILE,
    ISSUE_UNREADABLE,
    ISSUE_BAD_EXTENSION,
    ISSUE_TOO_SMALL,
    ISSUE_MISSING_LABEL,
    ISSUE_INVALID_LABEL,
    ISSUE_DUPLICATE,
    ISSUE_CONFLICTING_DUPLICATE,
)

# Which issues disqualify a sample from training. Exact content duplicates are
# excluded because they inflate metrics and leak across splits; a duplicate whose
# copies disagree on the label is excluded outright, including the first copy,
# since we cannot tell which label is correct.
FATAL_ISSUES = (
    ISSUE_MISSING_FILE,
    ISSUE_UNREADABLE,
    ISSUE_BAD_EXTENSION,
    ISSUE_TOO_SMALL,
    ISSUE_MISSING_LABEL,
    ISSUE_INVALID_LABEL,
    ISSUE_DUPLICATE,
    ISSUE_CONFLICTING_DUPLICATE,
)


@dataclass
class CleaningReport:
    """Auditable record of what validation found and what cleaning removed."""

    total_samples: int = 0
    valid_samples: int = 0
    excluded_samples: int = 0
    issue_counts: dict[str, int] = field(default_factory=dict)
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    duplicate_groups: int = 0
    duplicate_extra_copies: int = 0
    conflicting_duplicate_groups: int = 0
    formats: dict[str, int] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)
    channels: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    dimension_summary: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    def render(self) -> str:
        """The human-readable dataset summary printed by verify_dataset.py."""
        lines = [
            "Dataset Summary",
            "---------------",
            f"Total images:     {self.total_samples}",
            f"Valid images:     {self.valid_samples}",
            f"Excluded images:  {self.excluded_samples}",
            f"Corrupted images: {self.issue_counts.get(ISSUE_UNREADABLE, 0)}",
            f"Missing files:    {self.issue_counts.get(ISSUE_MISSING_FILE, 0)}",
            f"Missing labels:   {self.issue_counts.get(ISSUE_MISSING_LABEL, 0)}",
            f"Invalid labels:   {self.issue_counts.get(ISSUE_INVALID_LABEL, 0)}",
        ]
        for task, distribution in self.class_distribution.items():
            lines += ["", f"Class Distribution ({task})", "-" * (len(task) + 22)]
            total = sum(distribution.values()) or 1
            width = max((len(name) for name in distribution), default=0)
            for name, count in sorted(distribution.items(), key=lambda kv: -kv[1]):
                lines.append(f"{name:<{width}} : {count:>6}  ({100 * count / total:5.2f}%)")
        if self.formats:
            lines += ["", "Formats", "-------"]
            lines += [f"{name}: {count}" for name, count in sorted(self.formats.items())]
        if self.modes:
            lines += ["", "Colour modes", "------------"]
            lines += [f"{name}: {count}" for name, count in sorted(self.modes.items())]
        if self.channels:
            lines += ["", "Channels", "--------"]
            lines += [f"{name}: {count}" for name, count in sorted(self.channels.items())]
        if self.dimension_summary:
            lines += ["", "Image dimensions (pixels)", "-------------------------"]
            lines += [f"{key}: {value:g}" for key, value in self.dimension_summary.items()]
        lines += [
            "",
            "Duplicates",
            "----------",
            f"Duplicate content groups: {self.duplicate_groups}",
            f"Redundant copies:         {self.duplicate_extra_copies}",
            f"Label-conflicting groups: {self.conflicting_duplicate_groups}",
        ]
        if self.exclusion_reasons:
            lines += ["", "Exclusions by reason", "--------------------"]
            lines += [
                f"{reason}: {count}"
                for reason, count in sorted(self.exclusion_reasons.items(), key=lambda kv: -kv[1])
            ]
        return "\n".join(lines)


def scan_manifest(
    manifest: pd.DataFrame,
    config: DataConfig,
    *,
    path_column: str = "abs_path",
    workers: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Probe every file referenced by ``manifest`` and append the probe columns.

    Decoding is I/O and C-bound, so a thread pool gives a real speed-up despite
    the GIL. Returns a copy; the input is not modified.
    """
    if path_column not in manifest.columns:
        raise KeyError(f"manifest has no {path_column!r} column")

    paths = manifest[path_column].tolist()
    total = len(paths)
    algorithm = config.validation.hash_algorithm if config.validation.duplicate_scan else None
    results: list[dict[str, object]] = [None] * total  # type: ignore[list-item]

    def probe(index_and_path: tuple[int, str]) -> tuple[int, dict[str, object]]:
        index, path = index_and_path
        info = probe_image(path, hash_algorithm=algorithm)
        return index, {
            "exists": info.exists,
            "readable": info.ok,
            "error": info.error,
            "format": info.format,
            "mode": info.mode,
            "width": info.width,
            "height": info.height,
            "channels": info.channels,
            "file_bytes": info.file_bytes,
            "is_animated": info.is_animated,
            "content_hash": info.content_hash,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, (index, record) in enumerate(
            pool.map(probe, enumerate(paths)), start=1
        ):
            results[index] = record
            if progress is not None and (completed % 2000 == 0 or completed == total):
                progress(completed, total)

    probed = pd.DataFrame(results, index=manifest.index)
    return pd.concat([manifest, probed], axis=1)


def _label_columns(config: DataConfig, manifest: pd.DataFrame) -> list[str]:
    candidates = [config.task.primary, *config.task.auxiliary]
    return [column for column in candidates if column in manifest.columns]


def flag_issues(manifest: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Add one boolean column per issue type. Requires scan_manifest() output."""
    missing = [column for column in PROBE_COLUMNS if column not in manifest.columns]
    if missing:
        raise KeyError(f"run scan_manifest() first; manifest is missing {missing}")

    flagged = manifest.copy()
    flagged[ISSUE_MISSING_FILE] = ~flagged["exists"].fillna(False).astype(bool)
    flagged[ISSUE_UNREADABLE] = (
        ~flagged["readable"].fillna(False).astype(bool) & ~flagged[ISSUE_MISSING_FILE]
    )

    extensions = flagged["image_path"].astype(str).str.lower().map(lambda p: Path(p).suffix)
    flagged[ISSUE_BAD_EXTENSION] = ~extensions.isin(config.validation.allowed_extensions)

    min_side = config.validation.min_side
    smaller_side = flagged[["width", "height"]].min(axis=1)
    flagged[ISSUE_TOO_SMALL] = flagged["readable"].fillna(False).astype(bool) & (
        smaller_side < min_side
    )

    label_columns = _label_columns(config, flagged)
    primary = config.task.primary
    if primary in flagged.columns:
        values = flagged[primary]
        blank = values.isna() | (values.astype(str).str.strip() == "")
        flagged[ISSUE_MISSING_LABEL] = blank
        known = EXPECTED_CLASSES.get(primary)
        flagged[ISSUE_INVALID_LABEL] = (
            (~values.isin(known)) & ~blank if known else pd.Series(False, index=flagged.index)
        )
    else:
        flagged[ISSUE_MISSING_LABEL] = True
        flagged[ISSUE_INVALID_LABEL] = False

    flagged = _flag_duplicates(flagged, config, label_columns=label_columns)
    return flagged


def _flag_duplicates(
    manifest: pd.DataFrame, config: DataConfig, *, label_columns: Sequence[str]
) -> pd.DataFrame:
    """Flag exact byte-identical duplicates by content hash.

    The first occurrence in a duplicate group is kept and the rest are flagged.
    If the copies in a group disagree on the primary label, every member is
    flagged as conflicting instead - keeping an arbitrary one would bake a
    coin-flip into the ground truth.
    """
    flagged = manifest.copy()
    flagged[ISSUE_DUPLICATE] = False
    flagged[ISSUE_CONFLICTING_DUPLICATE] = False

    if not config.validation.duplicate_scan or "content_hash" not in flagged.columns:
        return flagged

    hashes = flagged["content_hash"]
    hashable = hashes.notna() & flagged["readable"].fillna(False).astype(bool)
    if not hashable.any():
        return flagged

    subset = flagged.loc[hashable]
    grouped = subset.groupby("content_hash", sort=False)
    duplicated_mask = grouped["content_hash"].transform("size") > 1
    duplicate_rows = subset.loc[duplicated_mask]

    if duplicate_rows.empty:
        return flagged

    primary = config.task.primary
    conflicting_hashes: set[str] = set()
    if primary in duplicate_rows.columns:
        distinct = duplicate_rows.groupby("content_hash")[primary].nunique(dropna=False)
        conflicting_hashes = set(distinct[distinct > 1].index)

    conflicting_index = duplicate_rows.index[
        duplicate_rows["content_hash"].isin(conflicting_hashes)
    ]
    flagged.loc[conflicting_index, ISSUE_CONFLICTING_DUPLICATE] = True

    consistent = duplicate_rows.loc[~duplicate_rows["content_hash"].isin(conflicting_hashes)]
    if not consistent.empty:
        redundant = consistent.index[consistent.duplicated(subset="content_hash", keep="first")]
        flagged.loc[redundant, ISSUE_DUPLICATE] = True
    return flagged


def clean_manifest(
    manifest: pd.DataFrame,
    config: DataConfig,
    *,
    fatal_issues: Iterable[str] = FATAL_ISSUES,
) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Split a flagged manifest into kept and excluded rows and build the report.

    Returns:
        ``(clean, excluded, report)``. ``clean`` contains only samples with no
        fatal issue. Excluded rows keep their ``exclusion_reason`` so they can
        be written out rather than silently dropped.
    """
    flagged = manifest if ISSUE_MISSING_FILE in manifest.columns else flag_issues(manifest, config)
    fatal = [issue for issue in fatal_issues if issue in flagged.columns]

    # First matching issue wins, so FATAL_ISSUES order determines which reason is
    # reported when a sample trips several checks.
    reasons = pd.Series("", index=flagged.index, dtype="object")
    for issue in fatal:
        unassigned = flagged[issue].fillna(False).astype(bool) & (reasons == "")
        reasons.loc[unassigned] = issue
    flagged = flagged.assign(exclusion_reason=reasons)

    excluded_mask = flagged["exclusion_reason"] != ""
    clean = flagged.loc[~excluded_mask].reset_index(drop=True)
    excluded = flagged.loc[excluded_mask].reset_index(drop=True)

    report = _build_report(flagged, clean, excluded, config)
    return clean, excluded, report


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _build_report(
    flagged: pd.DataFrame,
    clean: pd.DataFrame,
    excluded: pd.DataFrame,
    config: DataConfig,
) -> CleaningReport:
    report = CleaningReport(
        total_samples=int(len(flagged)),
        valid_samples=int(len(clean)),
        excluded_samples=int(len(excluded)),
    )
    report.issue_counts = {
        issue: int(flagged[issue].fillna(False).astype(bool).sum())
        for issue in ISSUE_COLUMNS
        if issue in flagged.columns
    }
    if not excluded.empty:
        report.exclusion_reasons = _value_counts(excluded["exclusion_reason"])

    if ISSUE_DUPLICATE in flagged.columns and "content_hash" in flagged.columns:
        readable = flagged.loc[flagged["readable"].fillna(False).astype(bool)]
        sizes = readable["content_hash"].value_counts()
        groups = sizes[sizes > 1]
        report.duplicate_groups = int(len(groups))
        report.duplicate_extra_copies = int((groups - 1).sum())
        report.conflicting_duplicate_groups = int(
            flagged.loc[
                flagged[ISSUE_CONFLICTING_DUPLICATE].fillna(False).astype(bool), "content_hash"
            ].nunique()
        )

    if not clean.empty:
        report.formats = _value_counts(clean["format"])
        report.modes = _value_counts(clean["mode"])
        report.channels = _value_counts(clean["channels"])
        for column in _label_columns(config, clean):
            report.class_distribution[column] = _value_counts(clean[column])
        widths, heights = clean["width"].dropna(), clean["height"].dropna()
        if not widths.empty:
            report.dimension_summary = {
                "width_min": float(widths.min()),
                "width_median": float(widths.median()),
                "width_max": float(widths.max()),
                "height_min": float(heights.min()),
                "height_median": float(heights.median()),
                "height_max": float(heights.max()),
                "megapixels_median": float((widths * heights).median() / 1e6),
            }
    return report


def imbalance_summary(distribution: Mapping[str, int]) -> dict[str, float]:
    """Quantify class imbalance so the strategy can be argued from numbers.

    ``imbalance_ratio`` is majority/minority. ``normalised_entropy`` is 1.0 for a
    perfectly uniform distribution and approaches 0 as mass concentrates in one
    class.
    """
    import math

    counts = [int(value) for value in distribution.values() if int(value) > 0]
    if not counts:
        return {}
    total = sum(counts)
    probabilities = [count / total for count in counts]
    entropy = -sum(p * math.log(p) for p in probabilities)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return {
        "num_classes": float(len(counts)),
        "total": float(total),
        "majority_count": float(max(counts)),
        "minority_count": float(min(counts)),
        "imbalance_ratio": max(counts) / min(counts),
        "minority_share": min(counts) / total,
        "normalised_entropy": entropy / max_entropy if max_entropy else 1.0,
    }
