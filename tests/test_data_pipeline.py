"""Unit tests for the Part 1 data pipeline.

Uses the miniature MEDIC-shaped fixture in conftest.py. These tests do not
download or read the full dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.config import load_config
from ml.data.augmentation import AugmentationConfig, apply_augmentations, build_eval_pipeline, build_train_pipeline
from ml.data.cache import write_cached_image
from ml.data.dataset import ImageClassificationDataset, build_dataset
from ml.data.imbalance import analyse_imbalance, compute_class_weights
from ml.data.io import ImageReadError, load_image, probe_image
from ml.data.labels import LabelError, build_label_mapping
from ml.data.manifest import build_raw_manifest
from ml.data.preprocessing import Preprocessor, resize, to_array
from ml.data.splitting import check_leakage, make_splits, regenerate_split
from ml.data.validation import (
    ISSUE_DUPLICATE,
    ISSUE_MISSING_FILE,
    ISSUE_UNREADABLE,
    clean_manifest,
    flag_issues,
    scan_manifest,
)


def test_load_valid_jpeg(mini_dataset):
    path = mini_dataset.dataset_root / "data/source_a/eq/train_none.jpg"
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size[0] > 0 and image.size[1] > 0
    info = probe_image(path)
    assert info.ok
    assert info.format == "JPEG"
    assert info.channels == 3


def test_invalid_image_is_reported_not_raised(mini_dataset):
    path = mini_dataset.dataset_root / "data/source_a/eq/corrupt.jpg"
    info = probe_image(path)
    assert info.exists
    assert not info.ok
    assert info.error
    with pytest.raises(ImageReadError):
        load_image(path)


def test_missing_image(mini_dataset):
    path = mini_dataset.dataset_root / "data/source_a/eq/missing.jpg"
    info = probe_image(path)
    assert not info.exists
    assert info.error == "file_not_found"


def test_label_loading_from_official_tsvs(mini_dataset):
    manifest = build_raw_manifest(mini_dataset)
    assert set(manifest["official_split"]) <= {"train", "dev", "test"}
    assert "damage_severity" in manifest.columns
    assert manifest["damage_severity"].notna().all()
    assert set(manifest["damage_severity"]) <= {"little_or_none", "mild", "severe"}


def test_label_mapping_ordinal_and_unknown():
    mapping = build_label_mapping(
        "damage_severity",
        ["severe", "little_or_none", "mild", "mild"],
        strict=True,
    )
    assert mapping.classes == ("little_or_none", "mild", "severe")
    assert mapping.encode("little_or_none") == 0
    assert mapping.encode("severe") == 2
    assert mapping.decode(1) == "mild"
    with pytest.raises(LabelError):
        mapping.encode("destroyed")


def test_preprocessing_output_shape_and_dtype(mini_dataset):
    path = mini_dataset.dataset_root / "data/source_a/eq/train_severe.jpg"
    preprocessor = Preprocessor.from_config(mini_dataset)
    array = preprocessor(path)
    assert array.dtype == np.float32
    assert array.shape == (32, 32, 3)
    # ImageNet normalisation of a valid image should not be all zeros.
    assert float(np.abs(array).mean()) > 0


def test_resize_stretch_and_center_crop():
    image = Image.new("RGB", (100, 50), (10, 20, 30))
    stretched = resize(image, (32, 32), mode="stretch")
    assert stretched.size == (32, 32)
    cropped = resize(image, (32, 32), mode="resize_shorter_then_center_crop")
    assert cropped.size == (32, 32)


def test_to_array_channel_order():
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    hwc = to_array(image, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), channel_order="hwc")
    chw = to_array(image, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), channel_order="chw")
    assert hwc.shape == (8, 8, 3)
    assert chw.shape == (3, 8, 8)
    np.testing.assert_allclose(hwc[:, :, 0], chw[0])


def test_official_split_and_no_id_leakage(mini_dataset):
    manifest = build_raw_manifest(mini_dataset)
    scanned = scan_manifest(manifest, mini_dataset, workers=2)
    flagged = flag_issues(scanned, mini_dataset)
    clean, excluded, report = clean_manifest(flagged, mini_dataset)
    assert report.excluded_samples == len(excluded)
    assert excluded[ISSUE_UNREADABLE].any() or (excluded["exclusion_reason"] == ISSUE_UNREADABLE).any()
    assert excluded[ISSUE_MISSING_FILE].any() or (excluded["exclusion_reason"] == ISSUE_MISSING_FILE).any()
    assert excluded[ISSUE_DUPLICATE].any() or (excluded["exclusion_reason"] == ISSUE_DUPLICATE).any()

    split = make_splits(clean, mini_dataset)
    assert set(split["split"]) == {"train", "val", "test"}
    leakage = check_leakage(split, mini_dataset)
    assert not leakage.has_leakage
    assert not leakage.shared_ids


def test_regenerate_split_is_deterministic(mini_dataset):
    rows = []
    for event in range(6):
        for i, damage in enumerate(("little_or_none", "mild", "severe")):
            rows.append(
                {
                    "image_path": f"data/s/e{event}/img{i}.jpg",
                    "event": f"e{event}",
                    "damage_severity": damage,
                }
            )
    frame = pd.DataFrame(rows)
    a = regenerate_split(frame, mini_dataset)
    b = regenerate_split(frame, mini_dataset)
    pd.testing.assert_series_equal(a["split"], b["split"])
    assert set(a["split"]) == {"train", "val", "test"}
    # Groups stay intact.
    for event, part in a.groupby("event"):
        assert part["split"].nunique() == 1


def test_augmentation_is_train_only_and_reproducible(mini_dataset):
    path = mini_dataset.dataset_root / "data/source_a/eq/train_severe.jpg"
    train = build_train_pipeline(mini_dataset)
    eval_pipeline = build_eval_pipeline(mini_dataset)
    assert train.augments
    assert not eval_pipeline.augments

    rng_a = np.random.default_rng((0, 0, 0))
    rng_b = np.random.default_rng((0, 0, 0))
    rng_c = np.random.default_rng((0, 1, 0))
    first = train(path, rng_a)
    second = train(path, rng_b)
    third = train(path, rng_c)
    eval_array = eval_pipeline(path)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (32, 32, 3)
    assert eval_array.shape == (32, 32, 3)
    # A different epoch seed should be allowed to differ (and usually will).
    assert third.shape == first.shape


def test_augmentation_disabled_transforms_do_not_flip_vertically():
    image = Image.new("RGB", (20, 30), (255, 0, 0))
    image.putpixel((0, 0), (0, 255, 0))
    cfg = AugmentationConfig(enabled=True, vertical_flip=0.0, horizontal_flip=0.0)
    rng = np.random.default_rng(0)
    out = apply_augmentations(image, cfg, (20, 30), rng)
    assert out.getpixel((0, 0)) == (0, 255, 0)


def test_dataset_getitem_types(mini_dataset):
    manifest = build_raw_manifest(mini_dataset)
    scanned = scan_manifest(manifest, mini_dataset, workers=2)
    clean, _, _ = clean_manifest(flag_issues(scanned, mini_dataset), mini_dataset)
    split = make_splits(clean, mini_dataset)
    train = split.loc[split["split"] == "train"].reset_index(drop=True)
    mapping = build_label_mapping("damage_severity", split["damage_severity"].tolist())
    dataset = ImageClassificationDataset(
        manifest=train,
        pipeline=build_eval_pipeline(mini_dataset),
        label_mapping=mapping,
        target_column="damage_severity",
        seed=mini_dataset.seed,
    )
    array, label = dataset[0]
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert array.shape == (32, 32, 3)
    assert isinstance(label, int)
    assert 0 <= label < dataset.num_classes
    batch_x, batch_y = dataset.batch([0, 1])
    assert batch_x.shape[0] == 2
    assert batch_y.dtype == np.int64


def test_class_weights_balanced():
    weights = compute_class_weights({"a": 90, "b": 10}, mode="balanced")
    assert weights["b"] > weights["a"]
    # sklearn convention: n_samples / (n_classes * count)
    assert weights["a"] == pytest.approx(100 / (2 * 90))
    assert weights["b"] == pytest.approx(100 / (2 * 10))


def test_imbalance_from_train_split_only(mini_dataset):
    manifest = build_raw_manifest(mini_dataset)
    scanned = scan_manifest(manifest, mini_dataset, workers=2)
    clean, _, _ = clean_manifest(flag_issues(scanned, mini_dataset), mini_dataset)
    split = make_splits(clean, mini_dataset)
    mapping = build_label_mapping("damage_severity", split["damage_severity"].tolist())
    report = analyse_imbalance(split, mini_dataset, label_mapping=mapping, split="train")
    train_counts = split.loc[split["split"] == "train", "damage_severity"].value_counts().to_dict()
    for name, count in train_counts.items():
        assert report.counts[name] == count


def test_cache_write_and_build_dataset(mini_dataset):
    src = mini_dataset.dataset_root / "data/source_a/eq/train_mild.jpg"
    dest = mini_dataset.cache_dir / "eq/train_mild.jpg"
    write_cached_image(src, dest, mini_dataset)
    assert dest.is_file()
    cached = Image.open(dest)
    assert cached.size == (32, 32)


def test_config_image_size_comes_from_yaml():
    config = load_config()
    assert config.preprocessing.image_size == (224, 224)
    assert config.task.primary == "damage_severity"
    assert config.split.strategy == "official"
    assert config.seed == 42
