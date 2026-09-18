"""Shared fixtures: a tiny MEDIC-shaped dataset that never needs a download."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from ml.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "data_config.yaml"

TSV_COLUMNS = [
    "image_id",
    "event_name",
    "image_path",
    "damage_severity",
    "informative",
    "humanitarian",
    "disaster_types",
]


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def _row(image_id, event, rel, damage, informative="informative", humanitarian="infrastructure_and_utility_damage", disaster="earthquake"):
    return {
        "image_id": image_id,
        "event_name": event,
        "image_path": rel,
        "damage_severity": damage,
        "informative": informative,
        "humanitarian": humanitarian,
        "disaster_types": disaster,
    }


@pytest.fixture
def mini_dataset(tmp_path: Path):
    """Build a miniature MEDIC layout under tmp_path and return a DataConfig."""
    root = tmp_path / "data" / "raw" / "MEDIC"
    img_dir = root / "data" / "source_a"

    # Distinct images for the three damage classes, two events, three splits.
    files = {
        "data/source_a/eq/train_none.jpg": ((10, 180, 20), "little_or_none", "train", "eq"),
        "data/source_a/eq/train_mild.jpg": ((200, 180, 20), "mild", "train", "eq"),
        "data/source_a/eq/train_severe.jpg": ((200, 20, 20), "severe", "train", "eq"),
        "data/source_a/flood/train_none.jpg": ((20, 20, 180), "little_or_none", "train", "flood"),
        "data/source_a/flood/train_mild.jpg": ((20, 180, 180), "mild", "train", "flood"),
        "data/source_a/eq/dev_severe.jpg": ((180, 40, 40), "severe", "dev", "eq"),
        "data/source_a/flood/dev_none.jpg": ((30, 30, 160), "little_or_none", "dev", "flood"),
        "data/source_a/eq/test_mild.jpg": ((190, 190, 40), "mild", "test", "eq"),
        "data/source_a/flood/test_severe.jpg": ((160, 30, 30), "severe", "test", "flood"),
    }

    rows = {"train": [], "dev": [], "test": []}
    for rel, (color, damage, split, event) in files.items():
        _write_image(root / rel, color)
        rows[split].append(
            _row("src", event, rel, damage, disaster=event if event != "eq" else "earthquake")
        )

    # Byte-identical duplicate of the first training image (should be excluded).
    dup_rel = "data/source_a/eq/train_none_dup.jpg"
    (root / dup_rel).write_bytes((root / "data/source_a/eq/train_none.jpg").read_bytes())
    rows["train"].append(_row("src", "eq", dup_rel, "little_or_none"))

    # Unreadable file with a JPEG extension.
    bad_rel = "data/source_a/eq/corrupt.jpg"
    (root / bad_rel).write_bytes(b"this is not an image")
    rows["train"].append(_row("src", "eq", bad_rel, "mild"))

    # Label pointing at a missing file.
    rows["train"].append(_row("src", "eq", "data/source_a/eq/missing.jpg", "severe"))

    # PNG so format handling is exercised.
    png_rel = "data/source_a/eq/train_png.png"
    png_path = root / png_rel
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), (90, 90, 90)).save(png_path, format="PNG")
    rows["train"].append(_row("src", "eq", png_rel, "mild", disaster="earthquake"))

    # Cover every published class once so strict label-mapping checks pass.
    extra_classes = [
        ("data/source_a/extra/not_informative.jpg", "little_or_none", "not_informative", "not_humanitarian", "not_disaster"),
        ("data/source_a/extra/people.jpg", "severe", "informative", "affected_injured_or_dead_people", "fire"),
        ("data/source_a/extra/rescue.jpg", "mild", "informative", "rescue_volunteering_or_donation_effort", "hurricane"),
        ("data/source_a/extra/landslide.jpg", "mild", "informative", "infrastructure_and_utility_damage", "landslide"),
        ("data/source_a/extra/other.jpg", "little_or_none", "not_informative", "not_humanitarian", "other_disaster"),
    ]
    for rel, damage, informative, humanitarian, disaster in extra_classes:
        _write_image(root / rel, (120, 80, 40), size=(40, 40))
        rows["train"].append(_row("src", "extra", rel, damage, informative, humanitarian, disaster))

    for split, filename in (("train", "MEDIC_train.tsv"), ("dev", "MEDIC_dev.tsv"), ("test", "MEDIC_test.tsv")):
        pd.DataFrame(rows[split], columns=TSV_COLUMNS).to_csv(
            root / filename, sep="\t", index=False
        )

    config = load_config(
        CONFIG_PATH,
        root=tmp_path,
        overrides={
            "cache": {"enabled": True, "size": [32, 32]},
            "preprocessing": {"image_size": [32, 32]},
            "validation": {"min_side": 16, "duplicate_scan": True},
            "split": {"strategy": "official"},
        },
    )
    return config
