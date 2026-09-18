"""Train a damage-severity classifier on the Part 1 MEDIC pipeline.

    python scripts/train.py --model baseline
    python scripts/train.py --model custom_cnn
    python scripts/train.py --model resnet50
    python scripts/train.py --model efficientnet_b0
    python scripts/train.py --model custom_cnn --max-train-batches 20  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import load_config  # noqa: E402
from ml.data.labels import load_label_mappings  # noqa: E402
from ml.engine.checkpoints import load_checkpoint  # noqa: E402
from ml.engine.dataloaders import build_dataloader  # noqa: E402
from ml.engine.experiment_log import log_experiment  # noqa: E402
from ml.engine.metrics import summarise_predictions  # noqa: E402
from ml.engine.plots import plot_history  # noqa: E402
from ml.engine.reproducibility import seed_everything  # noqa: E402
from ml.engine.train_config import load_train_config, resolve_model_settings  # noqa: E402
from ml.engine.trainer import Trainer, validate  # noqa: E402
from ml.engine.weights import class_weight_tensor  # noqa: E402
from ml.models.factory import build_model  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    key = name.lower()
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer {name!r}")


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _limit_loader(loader: DataLoader, max_batches: int | None) -> DataLoader:
    if not max_batches:
        return loader

    class _Limited(DataLoader):
        pass

    # Keep the real loader; wrap iteration in trainer via a thin subset sampler instead.
    from torch.utils.data import Subset

    n = min(len(loader.dataset), max_batches * loader.batch_size)
    subset = Subset(loader.dataset, list(range(n)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )


def evaluate_split(model, loader, criterion, device, class_names, use_amp: bool) -> dict:
    stats = validate(
        model, loader, criterion, device, class_names=class_names, use_amp=use_amp
    )
    return stats["metrics"] | {
        "loss": stats["loss"],
        "accuracy": stats["accuracy"],
        "macro_f1": stats["macro_f1"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=["baseline", "custom_cnn", "resnet50", "efficientnet_b0"])
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None, help="sanity/smoke: cap train+val batches")
    parser.add_argument("--skip-test", action="store_true", help="do not touch the isolated test set")
    parser.add_argument("--experiment-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_config = load_config(args.data_config)
    train_config = load_train_config(args.train_config)
    settings = resolve_model_settings(train_config, args.model)
    if args.epochs is not None:
        settings["epochs"] = args.epochs
        settings["epochs_head"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.lr is not None:
        settings["lr"] = args.lr
    if args.num_workers is not None:
        object.__setattr__(train_config, "num_workers", args.num_workers)

    seed_everything(train_config.seed)
    device = _device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}")

    mapping = load_label_mappings(data_config.paths.metadata / "label_mapping.json")[
        data_config.task.primary
    ]
    class_to_index = mapping.class_to_index
    print("classes:", class_to_index)

    train_loader = build_dataloader(
        data_config, train_config, "train", batch_size=settings["batch_size"], device=device
    )
    val_loader = build_dataloader(
        data_config, train_config, "val", batch_size=settings["batch_size"], device=device
    )
    test_loader = None
    if not args.skip_test:
        test_loader = build_dataloader(
            data_config, train_config, "test", batch_size=settings["batch_size"], device=device
        )
    train_loader = _limit_loader(train_loader, args.max_train_batches)
    val_loader = _limit_loader(val_loader, args.max_train_batches)

    pretrained = not args.no_pretrained
    model = build_model(args.model, num_classes=mapping.num_classes, pretrained=pretrained)
    model.to(device)

    transfer = args.model in {"resnet50", "efficientnet_b0"}
    if transfer and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        print("frozen backbone; trainable:", len(model.trainable_parameter_names()))

    weights = class_weight_tensor(data_config, mapping, device=device) if train_config.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("loss: CrossEntropyLoss", "weighted" if weights is not None else "unweighted")
    if weights is not None:
        print("  weights:", {name: float(weights[i]) for name, i in class_to_index.items()})

    amp = train_config.amp and not args.no_amp
    experiment_id = args.experiment_id or (
        f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    ckpt_dir = train_config.models_dir / args.model
    run_config = {
        "experiment_id": experiment_id,
        "model": args.model,
        "dataset": f"{data_config.dataset.name} {data_config.dataset.version}",
        "image_size": list(data_config.preprocessing.image_size),
        "channel_order_model": "chw",
        "augmentation_train": True,
        "augmentation_val": False,
        "class_weights": train_config.use_class_weights,
        "optimizer": train_config.optimizer,
        "batch_size": settings["batch_size"],
        "seed": train_config.seed,
        "pretrained": pretrained,
        "device": str(device),
        "amp": amp,
        "settings": settings,
        "class_to_index": class_to_index,
    }

    def make_trainer(lr: float, epochs_note: str) -> Trainer:
        opt = _optimizer(train_config.optimizer, _trainable_params(model), lr, settings["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=train_config.scheduler_factor,
            patience=train_config.scheduler_patience,
            min_lr=train_config.min_lr,
        )
        return Trainer(
            model,
            train_loader,
            val_loader,
            criterion,
            opt,
            device,
            class_to_index=class_to_index,
            checkpoint_dir=ckpt_dir,
            monitor=train_config.monitor,
            monitor_mode=train_config.monitor_mode,
            early_stopping_patience=int(settings["early_stopping_patience"]),
            amp=amp,
            grad_clip_norm=train_config.grad_clip_norm,
            scheduler=sched,
            run_config=run_config | {"phase_note": epochs_note, "lr": lr},
            log_every=train_config.log_every,
        )

    t0 = time.perf_counter()
    if transfer:
        head_epochs = int(settings.get("epochs_head", settings["epochs"]))
        trainer = make_trainer(float(settings["lr"]), "head")
        head_result = trainer.fit(head_epochs, start_epoch=1, phase="head")
        finetune_result = None
        do_ft = (not args.skip_finetune) and int(settings.get("epochs_finetune", 0)) > 0
        if do_ft:
            print("fine-tune last block at lower LR")
            if head_result["best_checkpoint"]:
                load_checkpoint(head_result["best_checkpoint"], model=model, map_location=device)
            model.unfreeze_last_block()
            print("trainable after unfreeze:", len(model.trainable_parameter_names()))
            trainer = make_trainer(float(settings["finetune_lr"]), "finetune")
            trainer.history = head_result["history"]
            trainer.best_score = head_result["best_score"]
            finetune_result = trainer.fit(
                int(settings["epochs_finetune"]),
                start_epoch=head_result["best_epoch"] + 1,
                phase="finetune",
            )
            combined_history = finetune_result["history"]
            best_checkpoint = finetune_result["best_checkpoint"]
            best_score = finetune_result["best_score"]
        else:
            combined_history = head_result["history"]
            best_checkpoint = head_result["best_checkpoint"]
            best_score = head_result["best_score"]
            finetune_result = None
    else:
        trainer = make_trainer(float(settings["lr"]), "scratch")
        result = trainer.fit(int(settings["epochs"]), start_epoch=1, phase="scratch")
        combined_history = result["history"]
        best_checkpoint = result["best_checkpoint"]
        best_score = result["best_score"]
        head_result, finetune_result = result, None

    elapsed = time.perf_counter() - t0
    if best_checkpoint:
        load_checkpoint(best_checkpoint, model=model, map_location=device)

    val_metrics = evaluate_split(
        model, val_loader, criterion, device, mapping.classes, amp
    )
    test_metrics = None
    if test_loader is not None and args.max_train_batches is None:
        print("Evaluating isolated test set AFTER model selection on validation.")
        test_metrics = evaluate_split(
            model, test_loader, criterion, device, mapping.classes, amp
        )

    curves = plot_history(
        combined_history, train_config.experiments_dir / f"{experiment_id}_curves.png"
    )
    record = {
        "experiment_id": experiment_id,
        "model": args.model,
        "dataset_version": data_config.dataset.version,
        "preprocessing": {
            "image_size": list(data_config.preprocessing.image_size),
            "normalize": {
                "mean": list(data_config.preprocessing.normalize.mean),
                "std": list(data_config.preprocessing.normalize.std),
            },
            "resize_mode": data_config.preprocessing.resize_mode,
        },
        "augmentation": "train only (Part 1 pipeline)",
        "class_weight_usage": train_config.use_class_weights,
        "optimizer": train_config.optimizer,
        "learning_rate": settings["lr"],
        "weight_decay": settings["weight_decay"],
        "batch_size": settings["batch_size"],
        "epochs_configured": settings,
        "best_validation_macro_f1": best_score,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_seconds": elapsed,
        "device": str(device),
        "amp": amp,
        "best_checkpoint": best_checkpoint,
        "history": combined_history,
        "curves": str(curves),
        "notes": (
            "Validation used for early stopping and checkpoint selection. "
            "Test metrics, if present, were computed once after selection."
        ),
        "smoke": args.max_train_batches is not None,
        "head_result_keys": list(head_result) if transfer else None,
        "finetuned": finetune_result is not None,
    }
    log_experiment(train_config.experiments_dir, record)
    print(json.dumps({k: record[k] for k in ("experiment_id", "best_validation_macro_f1", "val_metrics", "test_metrics", "training_seconds", "best_checkpoint") if k in record}, indent=2))
    print(f"done in {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    raise SystemExit(main())
