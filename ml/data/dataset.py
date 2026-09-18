"""Manifest-backed dataset objects.

:class:`ImageClassificationDataset` is plain Python: index it and you get
``(array, label)``. PyTorch wrapping lives in ``ml.engine.dataset`` so this
module stays importable without Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from ml.config import DataConfig
from ml.data.augmentation import ImagePipeline, build_eval_pipeline, build_train_pipeline
from ml.data.labels import LabelMapping, build_label_mapping, load_label_mappings

MANIFEST_REQUIRED_COLUMNS = ("image_path", "abs_path", "split")


class ManifestError(ValueError):
    """Raised when a manifest is missing columns or points at missing files."""


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in MANIFEST_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ManifestError(f"manifest {path} is missing columns {missing}")
    return frame


def resolve_paths(manifest: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Rebuild absolute paths from ``image_path`` relative to the raw dataset root.

    Manifests store both a relative and an absolute path. The relative one is
    authoritative, so a manifest produced on another machine still works here.
    """
    result = manifest.copy()
    root = config.dataset_root
    result["abs_path"] = result["image_path"].map(lambda p: str(root / str(p)))
    return result


@dataclass
class ImageClassificationDataset:
    """A split of the dataset, as ``(image_array, label_index)`` pairs.

    Args:
        manifest: Rows for one split.
        pipeline: Preprocessing (+ augmentation for train) to apply.
        label_mapping: Class-name to index mapping for ``target_column``.
        target_column: Label column to return.
        seed: Base seed for augmentation randomness.
        image_column: Column holding the path to read. ``prepare_dataset.py``
            writes a ``cached_path`` column; using it avoids re-decoding
            multi-megapixel originals on every epoch.
    """

    manifest: pd.DataFrame
    pipeline: ImagePipeline
    label_mapping: LabelMapping
    target_column: str
    seed: int = 0
    image_column: str = "abs_path"
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.image_column not in self.manifest.columns:
            raise ManifestError(f"manifest has no image column {self.image_column!r}")
        if self.target_column not in self.manifest.columns:
            raise ManifestError(f"manifest has no target column {self.target_column!r}")
        self.manifest = self.manifest.reset_index(drop=True)
        self._labels = np.asarray(
            self.label_mapping.encode_all(self.manifest[self.target_column].tolist()),
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def num_classes(self) -> int:
        return self.label_mapping.num_classes

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.pipeline.preprocessor.output_shape

    def set_epoch(self, epoch: int) -> None:
        """Change the augmentation stream while keeping runs reproducible.

        Each sample's randomness is derived from ``(seed, epoch, index)``, so the
        same epoch always produces the same augmented batch, and different epochs
        produce different ones.
        """
        self.epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng((int(self.seed), int(self.epoch), int(index)))

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        path = self.manifest.at[index, self.image_column]
        array = self.pipeline(path, self._rng(index) if self.pipeline.augments else None)
        return array, int(self._labels[index])

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        for index in range(len(self)):
            yield self[index]

    def batch(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Stack several samples into a batch. Mostly useful for smoke tests."""
        arrays, labels = zip(*(self[int(index)] for index in indices))
        return np.stack(arrays), np.asarray(labels, dtype=np.int64)


def build_dataset(
    config: DataConfig,
    split: str,
    *,
    manifest: pd.DataFrame | None = None,
    target_column: str | None = None,
    label_mapping: LabelMapping | None = None,
    use_cache: bool | None = None,
) -> ImageClassificationDataset:
    """Build the dataset for one split, reading the manifest written by prepare_dataset.py.

    ``split == "train"`` is the only split that gets an augmenting pipeline.
    """
    target = target_column or config.task.primary

    if manifest is None:
        path = config.manifest_path(split)
        if not path.is_file():
            raise ManifestError(
                f"manifest not found: {path}. Run scripts/prepare_dataset.py first."
            )
        manifest = read_manifest(path)

    if label_mapping is None:
        mapping_path = config.paths.metadata / "label_mapping.json"
        if mapping_path.is_file():
            label_mapping = load_label_mappings(mapping_path)[target]
        else:
            # Falling back to the split's own labels risks a train/test mismatch,
            # so only do it when the persisted mapping is genuinely absent.
            label_mapping = build_label_mapping(target, manifest[target].tolist())

    pipeline = build_train_pipeline(config) if split == "train" else build_eval_pipeline(config)

    cached = config.cache.enabled if use_cache is None else use_cache
    image_column = "cached_path" if cached and "cached_path" in manifest.columns else "abs_path"

    return ImageClassificationDataset(
        manifest=manifest,
        pipeline=pipeline,
        label_mapping=label_mapping,
        target_column=target,
        seed=config.seed,
        image_column=image_column,
    )


def build_all_datasets(
    config: DataConfig, **kwargs
) -> dict[str, ImageClassificationDataset]:
    from ml.data.splitting import SPLITS

    return {split: build_dataset(config, split, **kwargs) for split in SPLITS}
