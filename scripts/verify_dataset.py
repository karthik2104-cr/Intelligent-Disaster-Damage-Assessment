"""Verify the downloaded MEDIC dataset against its own label files.

This script only *measures*. It never writes into ``data/raw`` and never
deletes images. Every number it prints is counted from files on disk.

    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --workers 8

Outputs (all derived from the scan):

* ``reports/dataset_summary.txt``  - human-readable summary
* ``reports/dataset_summary.json`` - the same numbers as JSON
* ``data/interim/scanned_manifest.csv`` - one row per labelled sample, with
  probe columns, so ``prepare_dataset.py`` does not have to decode every
  image a second time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import load_config  # noqa: E402
from ml.data.labels import TASKS  # noqa: E402
from ml.data.manifest import build_raw_manifest  # noqa: E402
from ml.data.validation import (  # noqa: E402
    ISSUE_MISSING_FILE,
    ISSUE_MISSING_LABEL,
    ISSUE_UNREADABLE,
    clean_manifest,
    flag_issues,
    scan_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip content hashing (faster, but duplicate detection is disabled)",
    )
    args = parser.parse_args(argv)

    overrides = {"validation": {"duplicate_scan": False}} if args.skip_hash else None
    config = load_config(args.config, overrides=overrides)

    print(f"Dataset : {config.dataset.name} {config.dataset.version}")
    print(f"Root    : {config.dataset_root}")
    if not config.dataset_root.is_dir():
        print(
            "error: dataset root not found. Run python scripts/download_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    print("Building manifest from official TSV files ...")
    manifest = build_raw_manifest(config)
    print(f"  labelled rows: {len(manifest)}")
    for split, count in manifest["official_split"].value_counts().items():
        print(f"    {split}: {count}")

    print(f"Scanning {len(manifest)} images (workers={args.workers}) ...")

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total}", flush=True)

    scanned = scan_manifest(manifest, config, workers=args.workers, progress=progress)
    flagged = flag_issues(scanned, config)
    _, _, report = clean_manifest(flagged, config)

    config.paths.interim.mkdir(parents=True, exist_ok=True)
    config.paths.reports.mkdir(parents=True, exist_ok=True)
    scanned_path = config.paths.interim / "scanned_manifest.csv"
    flagged.to_csv(scanned_path, index=False)
    print(f"  scanned manifest -> {scanned_path}")

    summary_txt = report.render()
    extra = [
        "",
        "Official split sizes (from TSV files, before cleaning)",
        "-----------------------------------------------------",
    ]
    for split, count in manifest["official_split"].value_counts().sort_index().items():
        extra.append(f"{split}: {int(count)}")
    extra += ["", "Label columns present"]
    extra.append(", ".join(c for c in TASKS if c in manifest.columns) or "none")
    text = summary_txt + "\n" + "\n".join(extra) + "\n"

    (config.paths.reports / "dataset_summary.txt").write_text(text, encoding="utf-8")
    payload = report.as_dict()
    payload["official_split_counts"] = {
        str(k): int(v) for k, v in manifest["official_split"].value_counts().items()
    }
    payload["labelled_rows"] = int(len(manifest))
    payload["unreadable_errors"] = {
        str(k): int(v)
        for k, v in flagged.loc[
            flagged[ISSUE_UNREADABLE].fillna(False).astype(bool), "error"
        ]
        .value_counts()
        .items()
    }
    (config.paths.reports / "dataset_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(text)
    print(f"Corrupted/unreadable: {report.issue_counts.get(ISSUE_UNREADABLE, 0)}")
    print(f"Missing files:        {report.issue_counts.get(ISSUE_MISSING_FILE, 0)}")
    print(f"Missing labels:       {report.issue_counts.get(ISSUE_MISSING_LABEL, 0)}")
    print("Wrote reports/dataset_summary.txt and reports/dataset_summary.json")
    print("Next: python scripts/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
