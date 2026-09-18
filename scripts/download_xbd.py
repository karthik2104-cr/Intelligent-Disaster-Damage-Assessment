"""Documented acquisition helper for the xBD / xView2 dataset.

xBD is NOT the primary dataset for this project. MEDIC is. xBD is recorded here
because it is the natural next dataset if Part 2/4 later adds building
localisation or damage segmentation.

Official access requires creating an account on the xView2 website and accepting
the dataset terms. This script does not bypass that.

    python scripts/download_xbd.py
    python scripts/download_xbd.py --from-local path/to/extracted_xbd

If you have already downloaded the official archives after registering, pass
``--from-local`` to copy/link them into ``data/raw/xBD``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import load_config  # noqa: E402

OFFICIAL_PAGE = "https://xview2.org/dataset"
SIGNUP_PAGE = "https://xview2.org/signup"
DOWNLOAD_PAGE = "https://xview2.org/download-links"
LICENSE = "CC BY-NC-SA 4.0"
PAPER = "https://arxiv.org/abs/1911.09296"


INSTRUCTIONS = f"""
xBD / xView2 — official acquisition
===================================

xBD is the annotated satellite dataset created for the xView2 challenge
(Gupta et al., "xBD: A Dataset for Assessing Building Damage from Satellite
Imagery", 2019). It provides pre- and post-disaster RGB chips with building
polygons and Joint Damage Scale labels:

    0 no-damage | 1 minor-damage | 2 major-damage | 3 destroyed

Published scale (full release paper): 22,068 images, 850,736 building polygons,
19 disaster events. Imagery is 1024×1024 PNG, 3-band RGB, ~0.3 m/pixel from the
Maxar/DigitalGlobe Open Data Program. Approximate size quoted by the official
baseline repository: ~10 GB compressed / ~11 GB uncompressed for the challenge
release (tier-3 extras are larger).

License: {LICENSE} (non-commercial, share-alike).

This project does NOT download xBD automatically. The official host requires
registration:

  1. Open {SIGNUP_PAGE} and create an account.
  2. Accept the xView2 / xBD terms.
  3. Download the train / test / holdout archives from {DOWNLOAD_PAGE}.
  4. Place the extracted tree somewhere local, then run:

       python scripts/download_xbd.py --from-local PATH_TO_EXTRACTED_XBD

Do not use unofficial mirrors to skip registration. The license and challenge
terms must be accepted on the official site.

MEDIC remains the primary dataset for Part 2 (image-level damage-severity
classification). xBD is optional infrastructure for a later localisation or
segmentation experiment.
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--from-local",
        default=None,
        help="path to an already-downloaded, extracted xBD directory",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    print(INSTRUCTIONS)
    print()

    record = {
        "dataset_name": "xBD",
        "official_page": OFFICIAL_PAGE,
        "signup_page": SIGNUP_PAGE,
        "download_page": DOWNLOAD_PAGE,
        "paper": PAPER,
        "license": LICENSE,
        "authentication_required": True,
        "role_in_this_project": "optional future localisation/segmentation dataset",
        "primary_dataset": "MEDIC",
        "downloaded": False,
    }

    if not args.from_local:
        config.paths.metadata.mkdir(parents=True, exist_ok=True)
        path = config.paths.metadata / "xbd_access.json"
        record["recorded_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote access notes to {path}")
        print("No files were downloaded.")
        return 0

    source = Path(args.from_local).expanduser().resolve()
    if not source.is_dir():
        print(f"error: --from-local is not a directory: {source}", file=sys.stderr)
        return 1

    destination = config.paths.raw / "xBD"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"error: {destination} already exists; remove it first", file=sys.stderr)
        return 1
    print(f"Copying {source} -> {destination}")
    shutil.copytree(source, destination)
    record["downloaded"] = True
    record["local_source"] = str(source)
    record["destination"] = str(destination)
    record["copied_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config.paths.metadata.mkdir(parents=True, exist_ok=True)
    (config.paths.metadata / "xbd_access.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
