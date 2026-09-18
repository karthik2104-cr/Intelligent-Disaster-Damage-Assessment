"""Data pipeline for the disaster damage assessment project.

Stage order, matching docs/data_pipeline.md::

    raw dataset -> manifest -> validation -> cleaning -> split
                -> preprocessing -> (train only) augmentation -> model-ready arrays

Each stage is a module here and each takes a :class:`ml.config.DataConfig`, so
the whole pipeline is driven by ``configs/data_config.yaml``.
"""

from ml.data.augmentation import (
    AugmentationConfig,
    ImagePipeline,
    build_eval_pipeline,
    build_train_pipeline,
)
from ml.data.cache import cache_relative_path, write_cached_image
from ml.data.dataset import (
    ImageClassificationDataset,
    build_all_datasets,
    build_dataset,
    read_manifest,
    resolve_paths,
)
from ml.data.imbalance import ImbalanceReport, analyse_imbalance, compute_class_weights
from ml.data.io import ImageInfo, ImageReadError, load_image, probe_image
from ml.data.labels import (
    TASKS,
    LabelMapping,
    build_label_mapping,
    load_label_mappings,
    save_label_mappings,
)
from ml.data.manifest import build_raw_manifest
from ml.data.preprocessing import Preprocessor, resize, to_array
from ml.data.splitting import (
    SPLITS,
    LeakageReport,
    SplitReport,
    build_split_report,
    check_leakage,
    make_splits,
)
from ml.data.validation import CleaningReport, clean_manifest, flag_issues, scan_manifest

__all__ = [
    "SPLITS",
    "TASKS",
    "AugmentationConfig",
    "CleaningReport",
    "ImageClassificationDataset",
    "ImageInfo",
    "ImagePipeline",
    "ImageReadError",
    "ImbalanceReport",
    "LabelMapping",
    "LeakageReport",
    "Preprocessor",
    "SplitReport",
    "analyse_imbalance",
    "build_all_datasets",
    "build_dataset",
    "build_eval_pipeline",
    "build_label_mapping",
    "build_raw_manifest",
    "build_split_report",
    "build_train_pipeline",
    "cache_relative_path",
    "check_leakage",
    "clean_manifest",
    "compute_class_weights",
    "flag_issues",
    "load_image",
    "load_label_mappings",
    "make_splits",
    "probe_image",
    "read_manifest",
    "resize",
    "resolve_paths",
    "save_label_mappings",
    "scan_manifest",
    "to_array",
    "write_cached_image",
]
