"""Configuration loading for the data pipeline.

The YAML file in ``configs/`` is the single source of truth for paths, image
size, seeds, split ratios and augmentation parameters. Modules take a
:class:`DataConfig` instance rather than reading globals, which is what lets the
tests build a pipeline over fixture data without touching the real dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "data_config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing a required value."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    version: str
    official_url: str
    download_url: str
    archive_name: str
    expected_archive_bytes: int | None
    root_dirname: str
    license: str
    source: str
    hf_repo: str
    hf_revision: str


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    metadata: Path
    reports: Path


@dataclass(frozen=True)
class TaskConfig:
    primary: str
    auxiliary: tuple[str, ...]


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    ratios: Mapping[str, float]
    group_column: str
    stratify_column: str
    id_column: str

    def __post_init__(self) -> None:
        if self.strategy not in {"official", "regenerate"}:
            raise ConfigError(
                f"split.strategy must be 'official' or 'regenerate', got {self.strategy!r}"
            )
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"split.ratios must sum to 1.0, got {total}")


@dataclass(frozen=True)
class NormalizeConfig:
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: tuple[int, int]
    color_mode: str
    interpolation: str
    resize_mode: str
    scale: float
    normalize: NormalizeConfig
    output_dtype: str
    channel_order: str

    def __post_init__(self) -> None:
        if self.channel_order not in {"hwc", "chw"}:
            raise ConfigError(
                f"preprocessing.channel_order must be 'hwc' or 'chw', got {self.channel_order!r}"
            )
        if self.resize_mode not in {"resize_shorter_then_center_crop", "stretch"}:
            raise ConfigError(f"unsupported preprocessing.resize_mode {self.resize_mode!r}")
        if len(self.image_size) != 2:
            raise ConfigError("preprocessing.image_size must be [height, width]")
        if len(self.normalize.mean) != 3 or len(self.normalize.std) != 3:
            raise ConfigError("preprocessing.normalize.mean/std must have 3 values")
        if any(s == 0 for s in self.normalize.std):
            raise ConfigError("preprocessing.normalize.std must not contain zeros")


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool
    format: str
    quality: int
    size: tuple[int, int]


@dataclass(frozen=True)
class ValidationConfig:
    allowed_extensions: frozenset[str]
    min_side: int
    hash_algorithm: str
    duplicate_scan: bool


@dataclass(frozen=True)
class ImbalanceConfig:
    strategy: str
    class_weight_mode: str


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    paths: PathsConfig
    task: TaskConfig
    seed: int
    split: SplitConfig
    preprocessing: PreprocessingConfig
    cache: CacheConfig
    augmentation: Mapping[str, Any]
    validation: ValidationConfig
    imbalance: ImbalanceConfig
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    # --- Derived locations -------------------------------------------------
    @property
    def archive_path(self) -> Path:
        return self.paths.raw / self.dataset.archive_name

    @property
    def dataset_root(self) -> Path:
        """Extracted raw dataset directory (never written to by the pipeline)."""
        return self.paths.raw / self.dataset.root_dirname

    @property
    def cache_dir(self) -> Path:
        return self.paths.interim / "cache"

    def manifest_path(self, split: str) -> Path:
        return self.paths.processed / f"{split}.csv"

    @property
    def all_samples_path(self) -> Path:
        return self.paths.interim / "all_samples.csv"

    @property
    def clean_samples_path(self) -> Path:
        return self.paths.interim / "clean_samples.csv"


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required config key {context}.{key}")
    return mapping[key]


def _as_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _as_int_pair(value: Sequence[int], context: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ConfigError(f"{context} must have exactly 2 values, got {value!r}")
    return int(value[0]), int(value[1])


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> DataConfig:
    """Load and validate the data pipeline configuration.

    Args:
        path: YAML config file. Defaults to ``configs/data_config.yaml``.
        root: Base directory for relative paths. Defaults to the repo root.
            Tests point this at a temporary directory.
        overrides: Shallow-merged per top-level section, applied after the file
            is read. Used by tests and by CLI flags.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    if overrides:
        for section, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(data.get(section), Mapping):
                data[section] = {**data[section], **value}
            else:
                data[section] = value

    base_root = Path(root).resolve() if root is not None else REPO_ROOT

    ds = _require(data, "dataset", "root")
    dataset = DatasetConfig(
        name=_require(ds, "name", "dataset"),
        version=str(_require(ds, "version", "dataset")),
        official_url=_require(ds, "official_url", "dataset"),
        download_url=_require(ds, "download_url", "dataset"),
        archive_name=_require(ds, "archive_name", "dataset"),
        expected_archive_bytes=ds.get("expected_archive_bytes"),
        root_dirname=_require(ds, "root_dirname", "dataset"),
        license=_require(ds, "license", "dataset"),
        source=ds.get("source", "hf"),
        hf_repo=ds.get("hf_repo", ""),
        hf_revision=ds.get("hf_revision", "main"),
    )

    p = _require(data, "paths", "root")
    paths = PathsConfig(
        root=base_root,
        raw=_as_path(_require(p, "raw", "paths"), base_root),
        interim=_as_path(_require(p, "interim", "paths"), base_root),
        processed=_as_path(_require(p, "processed", "paths"), base_root),
        metadata=_as_path(_require(p, "metadata", "paths"), base_root),
        reports=_as_path(p.get("reports", "reports"), base_root),
    )

    t = _require(data, "task", "root")
    task = TaskConfig(
        primary=_require(t, "primary", "task"),
        auxiliary=tuple(t.get("auxiliary", ())),
    )

    s = _require(data, "split", "root")
    split = SplitConfig(
        strategy=_require(s, "strategy", "split"),
        ratios={k: float(v) for k, v in _require(s, "ratios", "split").items()},
        group_column=_require(s, "group_column", "split"),
        stratify_column=_require(s, "stratify_column", "split"),
        id_column=s.get("id_column", "image_path"),
    )

    pre = _require(data, "preprocessing", "root")
    norm = _require(pre, "normalize", "preprocessing")
    preprocessing = PreprocessingConfig(
        image_size=_as_int_pair(_require(pre, "image_size", "preprocessing"), "preprocessing.image_size"),
        color_mode=pre.get("color_mode", "rgb"),
        interpolation=pre.get("interpolation", "bilinear"),
        resize_mode=pre.get("resize_mode", "resize_shorter_then_center_crop"),
        scale=float(pre.get("scale", 1.0 / 255.0)),
        normalize=NormalizeConfig(
            mean=tuple(float(v) for v in _require(norm, "mean", "preprocessing.normalize")),
            std=tuple(float(v) for v in _require(norm, "std", "preprocessing.normalize")),
        ),
        output_dtype=pre.get("output_dtype", "float32"),
        channel_order=pre.get("channel_order", "hwc"),
    )

    c = data.get("cache", {})
    cache = CacheConfig(
        enabled=bool(c.get("enabled", False)),
        format=c.get("format", "jpeg"),
        quality=int(c.get("quality", 95)),
        size=_as_int_pair(c.get("size", preprocessing.image_size), "cache.size"),
    )

    v = data.get("validation", {})
    validation = ValidationConfig(
        allowed_extensions=frozenset(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in v.get("allowed_extensions", [".jpg", ".jpeg", ".png"])
        ),
        min_side=int(v.get("min_side", 1)),
        hash_algorithm=v.get("hash_algorithm", "sha256"),
        duplicate_scan=bool(v.get("duplicate_scan", True)),
    )

    i = data.get("imbalance", {})
    imbalance = ImbalanceConfig(
        strategy=i.get("strategy", "class_weights"),
        class_weight_mode=i.get("class_weight_mode", "balanced"),
    )

    return DataConfig(
        dataset=dataset,
        paths=paths,
        task=task,
        seed=int(data.get("seed", 42)),
        split=split,
        preprocessing=preprocessing,
        cache=cache,
        augmentation=data.get("augmentation", {}),
        validation=validation,
        imbalance=imbalance,
        raw=data,
    )
