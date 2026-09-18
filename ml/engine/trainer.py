"""Explicit training loop: forward, loss, backward, optimizer step, validate."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.engine.checkpoints import save_checkpoint
from ml.engine.metrics import ClassificationMeter, summarise_predictions


def _move(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    images, labels = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float = 0.0,
    epoch: int = 0,
    log_every: int = 50,
) -> dict[str, float]:
    """One training epoch. Gradients are computed and parameters are updated."""
    model.train()
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    meter = ClassificationMeter()
    use_amp = scaler is not None and device.type == "cuda"

    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(progress, start=1):
        images, labels = _move(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        meter.update(loss, logits, labels)
        if step % log_every == 0:
            progress.set_postfix(loss=f"{meter.loss:.4f}", acc=f"{meter.accuracy:.3f}")

    return meter.as_dict() | {"macro_f1": float("nan")}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    class_names: list[str],
    use_amp: bool = False,
) -> dict[str, Any]:
    """Forward-only evaluation. No backward, no parameter updates."""
    model.eval()
    meter = ClassificationMeter()
    amp = use_amp and device.type == "cuda"
    for batch in tqdm(loader, desc="val", leave=False):
        images, labels = _move(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        meter.update(loss, logits, labels)
    summary = summarise_predictions(meter.y_true, meter.y_pred, class_names=class_names)
    return {
        "loss": meter.loss,
        "accuracy": meter.accuracy,
        "n": meter.count,
        "macro_f1": summary["macro_f1"],
        "macro_recall": summary["macro_recall"],
        "per_class": summary["per_class"],
        "metrics": summary,
    }


class Trainer:
    """Reusable fit() around train_one_epoch / validate, with checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        class_to_index: dict[str, int],
        checkpoint_dir: Path,
        monitor: str = "val_macro_f1",
        monitor_mode: str = "max",
        early_stopping_patience: int = 3,
        amp: bool = True,
        grad_clip_norm: float = 1.0,
        scheduler: Any | None = None,
        run_config: dict[str, Any] | None = None,
        log_every: int = 50,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.class_to_index = class_to_index
        self.class_names = [name for name, _ in sorted(class_to_index.items(), key=lambda kv: kv[1])]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.early_stopping_patience = early_stopping_patience
        self.amp = amp and device.type == "cuda"
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = scheduler
        self.run_config = run_config or {}
        self.log_every = log_every
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp) if self.amp else None
        self.history: list[dict[str, Any]] = []
        self.best_score: float | None = None
        self.best_path = self.checkpoint_dir / "best.pt"
        self.last_path = self.checkpoint_dir / "last.pt"

    def _is_improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.monitor_mode == "max":
            return score > self.best_score
        return score < self.best_score

    def _monitor_value(self, val_stats: dict[str, Any]) -> float:
        key = self.monitor[4:] if self.monitor.startswith("val_") else self.monitor
        return float(val_stats[key])

    def fit(self, epochs: int, *, start_epoch: int = 1, phase: str = "train") -> dict[str, Any]:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        stale = 0
        best_epoch = start_epoch - 1

        for epoch in range(start_epoch, start_epoch + epochs):
            train_stats = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                scaler=self.scaler,
                grad_clip_norm=self.grad_clip_norm,
                epoch=epoch,
                log_every=self.log_every,
            )
            val_stats = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                class_names=self.class_names,
                use_amp=self.amp,
            )
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()

            score = self._monitor_value(val_stats)
            row = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "val_macro_f1": val_stats["macro_f1"],
                "val_macro_recall": val_stats["macro_recall"],
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "val_per_class": val_stats["per_class"],
            }
            self.history.append(row)
            print(
                f"[{phase} {epoch:03d}] "
                f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
                f"val loss={row['val_loss']:.4f} acc={row['val_accuracy']:.4f} "
                f"macroF1={row['val_macro_f1']:.4f}"
            )

            payload_metrics = {k: row[k] for k in row if k != "val_per_class"}
            save_checkpoint(
                self.last_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=payload_metrics,
                class_to_index=self.class_to_index,
                config=self.run_config,
                extra={"phase": phase, "history": self.history},
            )
            if self._is_improved(score):
                self.best_score = score
                best_epoch = epoch
                stale = 0
                save_checkpoint(
                    self.best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=payload_metrics,
                    class_to_index=self.class_to_index,
                    config=self.run_config,
                    extra={"phase": phase, "history": self.history},
                )
                print(f"  saved best checkpoint ({self.monitor}={score:.4f}) -> {self.best_path}")
            else:
                stale += 1
                if stale >= self.early_stopping_patience:
                    print(f"  early stopping at epoch {epoch} (patience {self.early_stopping_patience})")
                    break

        elapsed = time.perf_counter() - started
        return {
            "best_score": self.best_score,
            "best_epoch": best_epoch,
            "best_checkpoint": str(self.best_path) if self.best_path.is_file() else None,
            "history": self.history,
            "seconds": elapsed,
        }
