"""Clean, split and materialise the MEDIC dataset for training.

Never writes into ``data/raw``. Excluded samples are recorded with a reason,
not silently dropped.

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --no-cache
    python scripts/prepare_dataset.py --rescan

Pipeline:

    scanned manifest (from verify_dataset.py, or a fresh scan)
            ↓
    flag issues + exclude with recorded reasons
            ↓
    attach train/val/test (official MEDIC splits by default)
            ↓
    leakage check
            ↓
    label mapping + class weights from the train split
            ↓
    optional resized JPEG cache under data/interim/cache
            ↓
    data/processed/{train,val,test}.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import load_config  # noqa: E402
from ml.data.cache import cache_relative_path, write_cached_image  # noqa: E402
from ml.data.imbalance import analyse_imbalance, class_weight_vector  # noqa: E402
from ml.data.labels import TASKS, build_label_mapping, save_label_mappings  # noqa: E402
from ml.data.manifest import build_raw_manifest  # noqa: E402
from ml.data.splitting import (  # noqa: E402
    SPLITS,
    build_split_report,
    check_leakage,
    make_splits,
)
from ml.data.validation import (  # noqa: E402
    clean_manifest,
    flag_issues,
    scan_manifest,
)


SCANNED_NAME = "scanned_manifest.csv"


def _progress(done: int, total: int) -> None:
    print(f"  {done}/{total}", flush=True)


def load_or_scan(config, *, rescan: bool, workers: int) -> pd.DataFrame:
    scanned_path = config.paths.interim / SCANNED_NAME
    if scanned_path.is_file() and not rescan:
        print(f"Reusing {scanned_path}")
        return pd.read_csv(scanned_path, low_memory=False)
    print("Scanning images ...")
    manifest = build_raw_manifest(config)
    scanned = scan_manifest(manifest, config, workers=workers, progress=_progress)
    config.paths.interim.mkdir(parents=True, exist_ok=True)
    flagged = flag_issues(scanned, config)
    flagged.to_csv(scanned_path, index=False)
    return flagged


def cache_split(frame: pd.DataFrame, config, *, workers: int) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(cached_path=pd.Series(dtype="object"))
    cache_root = config.cache_dir
    relatives = frame["image_path"].map(cache_relative_path)
    destinations = [str(cache_root / rel) for rel in relatives]

    def one(item: tuple[str, str]) -> str | None:
        source, dest = item
        dest_path = Path(dest)
        if dest_path.is_file():
            return dest
        try:
            write_cached_image(source, dest_path, config)
            return dest
        except Exception as exc:  # noqa: BLE001 - record the failure, keep going
            print(f"  cache failed for {source}: {exc}", file=sys.stderr)
            return None

    print(f"Caching {len(frame)} images to {cache_root} ...")
    results: list[str | None] = [None] * len(frame)
    pairs = list(zip(frame["abs_path"].tolist(), destinations))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, value in enumerate(pool.map(one, pairs)):
            results[index] = value
            if (index + 1) % 2000 == 0 or index + 1 == len(pairs):
                print(f"  {index + 1}/{len(pairs)}", flush=True)
    out = frame.copy()
    out["cached_path"] = results
    missing = out["cached_path"].isna().sum()
    if missing:
        print(f"  warning: {int(missing)} images could not be cached")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rescan", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not config.dataset_root.is_dir():
        print("error: dataset root not found. Run download_dataset.py first.", file=sys.stderr)
        return 1

    flagged = load_or_scan(config, rescan=args.rescan, workers=args.workers)
    if "issue_missing_file" not in flagged.columns:
        flagged = flag_issues(flagged, config)

    clean, excluded, cleaning_report = clean_manifest(flagged, config)

    print(cleaning_report.render())
    print()

    split_frame = make_splits(clean, config)
    leakage = check_leakage(split_frame, config)
    print(leakage.render())
    if leakage.has_leakage:
        print(
            "warning: identity or byte-identical content appears in more than one split. "
            "Inspect reports/split_summary.json before training.",
            file=sys.stderr,
        )
    split_report = build_split_report(split_frame, config)
    print()
    print(split_report.render())

    mappings = {}
    for task in TASKS:
        if task not in split_frame.columns:
            continue
        mappings[task] = build_label_mapping(task, split_frame[task].tolist(), strict=True)
    mapping_path = save_label_mappings(mappings, config.paths.metadata / "label_mapping.json")
    print(f"\nLabel mapping -> {mapping_path}")

    primary = mappings[config.task.primary]
    imbalance = analyse_imbalance(split_frame, config, label_mapping=primary)
    print()
    print(imbalance.render())

    if config.cache.enabled and not args.no_cache:
        split_frame = cache_split(split_frame, config, workers=args.workers)

    config.paths.processed.mkdir(parents=True, exist_ok=True)
    config.paths.interim.mkdir(parents=True, exist_ok=True)
    config.paths.metadata.mkdir(parents=True, exist_ok=True)
    config.paths.reports.mkdir(parents=True, exist_ok=True)

    split_frame.to_csv(config.clean_samples_path, index=False)
    flagged.to_csv(config.all_samples_path, index=False)
    if not excluded.empty:
        excluded_path = config.paths.interim / "excluded_samples.csv"
        excluded.to_csv(excluded_path, index=False)
        print(f"Excluded samples -> {excluded_path}")

    for split in SPLITS:
        path = config.manifest_path(split)
        split_frame.loc[split_frame["split"] == split].to_csv(path, index=False)
        print(f"  {split} -> {path}")

    cleaning_report.save(config.paths.reports / "cleaning_report.json")
    split_report.save(config.paths.reports / "split_summary.json")
    (config.paths.reports / "imbalance.json").write_text(
        json.dumps(imbalance.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (config.paths.reports / "leakage.json").write_text(
        json.dumps(leakage.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    weights = class_weight_vector(imbalance, primary)
    (config.paths.metadata / "class_weights.json").write_text(
        json.dumps(
            {
                "task": config.task.primary,
                "strategy": config.imbalance.strategy,
                "mode": config.imbalance.class_weight_mode,
                "split": "train",
                "classes": list(primary.classes),
                "weights": [float(w) for w in weights],
                "weights_by_class": imbalance.class_weights,
                "imbalance_ratio": imbalance.imbalance_ratio,
                "severity": imbalance.severity,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    task_doc = {
        "input": (
            "A single RGB photograph (social-media / ground-level disaster image), "
            f"resized to {list(config.preprocessing.image_size)} and ImageNet-normalised."
        ),
        "target": config.task.primary,
        "task": "multi-class image classification",
        "classes": list(primary.classes),
        "label_representation": "class name in the TSV / CSV; integer index via label_mapping.json",
        "auxiliary_tasks": list(config.task.auxiliary),
        "localization_or_segmentation": False,
        "notes": (
            "MEDIC provides image-level labels only. There are no bounding boxes or "
            "segmentation masks in this dataset. xBD is documented separately for "
            "future localisation work; it is not required for Part 2 classification."
        ),
    }
    (config.paths.metadata / "task.json").write_text(
        json.dumps(task_doc, indent=2) + "\n", encoding="utf-8"
    )

    print("\nDone. Next: open notebooks/01_dataset_exploration.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
