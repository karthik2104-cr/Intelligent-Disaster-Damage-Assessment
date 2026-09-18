"""DataLoaders built on the Part 1 pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ml.config import DataConfig
from ml.data.dataset import build_dataset
from ml.data.splitting import SPLITS
from ml.engine.dataset import DamageSeverityTorchDataset
from ml.engine.train_config import TrainConfig


def _worker_init(worker_id: int) -> None:
    import numpy as np

    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def build_dataloader(
    data_config: DataConfig,
    train_config: TrainConfig,
    split: str,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> DataLoader:
    source = build_dataset(data_config, split)
    dataset = DamageSeverityTorchDataset(source)
    train = split == "train"
    pin = bool(train_config.pin_memory and device is not None and device.type == "cuda")
    workers = int(train_config.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(batch_size or train_config.batch_size),
        shuffle=train,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=bool(train_config.persistent_workers and workers > 0),
        drop_last=False,
        worker_init_fn=_worker_init if workers else None,
    )


def build_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
    *,
    batch_size: int | None = None,
    device: torch.device | None = None,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, DataLoader]:
    return {
        split: build_dataloader(
            data_config, train_config, split, batch_size=batch_size, device=device
        )
        for split in splits
    }
