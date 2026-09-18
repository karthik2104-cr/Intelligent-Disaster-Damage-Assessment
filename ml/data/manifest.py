"""Building the raw manifest from the dataset's own label files.

This is the one module that knows MEDIC's on-disk layout. Everything downstream
works from the resulting DataFrame, so pointing the project at a differently
shaped dataset means replacing this file and nothing else.

MEDIC ships three tab-separated label files at the dataset root::

    MEDIC_train.tsv   MEDIC_dev.tsv   MEDIC_test.tsv

with columns ``image_id``, ``event_name``, ``image_path`` and one column per
classification task. ``image_path`` is relative to the dataset root and is the
only column that uniquely identifies a sample - ``image_id`` is a *source*
identifier and repeats across many images.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import DataConfig
from ml.data.labels import TASKS

LABEL_FILES = {"train": "MEDIC_train.tsv", "dev": "MEDIC_dev.tsv", "test": "MEDIC_test.tsv"}

# Columns every label file must provide.
REQUIRED_COLUMNS = ("image_id", "event_name", "image_path")


class ManifestBuildError(RuntimeError):
    """Raised when the raw dataset does not look like the expected layout."""


def find_label_files(dataset_root: Path) -> dict[str, Path]:
    """Locate the split label files, searching one level down if necessary."""
    found: dict[str, Path] = {}
    for split, filename in LABEL_FILES.items():
        candidate = dataset_root / filename
        if candidate.is_file():
            found[split] = candidate
            continue
        matches = sorted(dataset_root.glob(f"*/{filename}"))
        if matches:
            found[split] = matches[0]
    missing = sorted(set(LABEL_FILES) - set(found))
    if missing:
        raise ManifestBuildError(
            f"label files for splits {missing} not found under {dataset_root}. "
            "Run scripts/download_dataset.py first."
        )
    return found


def read_label_file(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ManifestBuildError(f"{path.name} is missing required columns {missing}")
    frame["official_split"] = split
    return frame


def derive_source_and_event(image_path: str) -> tuple[str, str]:
    """Split ``data/<source>/<event>/<file>`` into its source and event parts.

    MEDIC aggregates five upstream collections, so the first path component after
    ``data/`` identifies the source dataset and the second (when present) the
    disaster event. Both are used for grouped splitting and for per-event EDA.
    """
    parts = Path(str(image_path).replace("\\", "/")).parts
    parts = parts[1:] if parts and parts[0] == "data" else parts
    source = parts[0] if parts else "unknown"
    event = parts[1] if len(parts) > 2 else source
    return source, event


def build_raw_manifest(config: DataConfig) -> pd.DataFrame:
    """Read every label file and return one row per labelled sample.

    The returned frame carries the labels and derived grouping columns but no
    image statistics; those are added by
    :func:`ml.data.validation.scan_manifest`.
    """
    dataset_root = config.dataset_root
    if not dataset_root.is_dir():
        raise ManifestBuildError(
            f"dataset root not found: {dataset_root}. Run scripts/download_dataset.py first."
        )

    frames = [read_label_file(path, split) for split, path in find_label_files(dataset_root).items()]
    manifest = pd.concat(frames, ignore_index=True)

    manifest["image_path"] = manifest["image_path"].astype(str).str.replace("\\", "/", regex=False)
    derived = manifest["image_path"].map(derive_source_and_event)
    manifest["source_dataset"] = [item[0] for item in derived]
    manifest["event"] = [item[1] for item in derived]
    manifest["abs_path"] = manifest["image_path"].map(lambda p: str(dataset_root / p))

    present_tasks = [task for task in TASKS if task in manifest.columns]
    if config.task.primary not in present_tasks:
        raise ManifestBuildError(
            f"primary task column {config.task.primary!r} is not present in the label files; "
            f"found {present_tasks}"
        )

    ordered = [
        "image_path",
        "abs_path",
        "image_id",
        "event_name",
        "source_dataset",
        "event",
        "official_split",
        *present_tasks,
    ]
    return manifest[ordered + [c for c in manifest.columns if c not in ordered]]
