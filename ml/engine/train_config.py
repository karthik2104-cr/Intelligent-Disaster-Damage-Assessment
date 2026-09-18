"""Training configuration, kept separate from the Part 1 data config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ml.config import REPO_ROOT

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train_config.yaml"


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    optimizer: str
    lr: float
    weight_decay: float
    scheduler: str
    scheduler_factor: float
    scheduler_patience: int
    min_lr: float
    epochs: int
    early_stopping_patience: int
    monitor: str
    monitor_mode: str
    amp: bool
    grad_clip_norm: float
    log_every: int
    use_class_weights: bool
    models_dir: Path
    experiments_dir: Path
    model_overrides: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any]


def load_train_config(path: str | Path | None = None) -> TrainConfig:
    config_path = Path(path) if path is not None else DEFAULT_TRAIN_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    loader = data.get("dataloader", {})
    optim = data.get("optim", {})
    sched = data.get("scheduler", {})
    train = data.get("train", {})
    loss = data.get("loss", {})
    paths = data.get("paths", {})

    return TrainConfig(
        seed=int(data.get("seed", 42)),
        batch_size=int(loader.get("batch_size", 32)),
        num_workers=int(loader.get("num_workers", 0)),
        pin_memory=bool(loader.get("pin_memory", True)),
        persistent_workers=bool(loader.get("persistent_workers", False)),
        optimizer=str(optim.get("name", "adamw")),
        lr=float(optim.get("lr", 1e-3)),
        weight_decay=float(optim.get("weight_decay", 1e-4)),
        scheduler=str(sched.get("name", "reduce_on_plateau")),
        scheduler_factor=float(sched.get("factor", 0.5)),
        scheduler_patience=int(sched.get("patience", 2)),
        min_lr=float(sched.get("min_lr", 1e-6)),
        epochs=int(train.get("epochs", 8)),
        early_stopping_patience=int(train.get("early_stopping_patience", 3)),
        monitor=str(train.get("monitor", "val_macro_f1")),
        monitor_mode=str(train.get("monitor_mode", "max")),
        amp=bool(train.get("amp", True)),
        grad_clip_norm=float(train.get("grad_clip_norm", 1.0)),
        log_every=int(train.get("log_every", 50)),
        use_class_weights=bool(loss.get("use_class_weights", True)),
        models_dir=REPO_ROOT / paths.get("models", "models"),
        experiments_dir=REPO_ROOT / paths.get("experiments", "reports/experiments"),
        model_overrides=data.get("models", {}),
        raw=data,
    )


def resolve_model_settings(config: TrainConfig, model_name: str) -> dict[str, Any]:
    """Merge global train settings with per-model overrides."""
    settings = {
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "epochs_head": config.epochs,
        "epochs_finetune": 0,
        "finetune_lr": config.lr / 10.0,
    }
    settings.update(config.model_overrides.get(model_name, {}))
    return settings
