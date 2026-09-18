"""Sanity-check the PyTorch stack on a few real batches before a long run.

    python scripts/sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import load_config  # noqa: E402
from ml.data.labels import load_label_mappings  # noqa: E402
from ml.engine.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from ml.engine.dataloaders import build_dataloader  # noqa: E402
from ml.engine.reproducibility import seed_everything  # noqa: E402
from ml.engine.train_config import load_train_config  # noqa: E402
from ml.engine.trainer import train_one_epoch, validate  # noqa: E402
from ml.engine.weights import class_weight_tensor  # noqa: E402
from ml.models.factory import MODEL_NAMES, build_model  # noqa: E402


def main() -> int:
    data_config = load_config()
    train_config = load_train_config()
    seed_everything(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**20, "MiB")

    mapping = load_label_mappings(data_config.paths.metadata / "label_mapping.json")[
        data_config.task.primary
    ]
    print("classes:", mapping.class_to_index)

    # Force a small batch and zero workers so this is a diagnostic, not a job.
    object.__setattr__(train_config, "num_workers", 0)
    train_loader = build_dataloader(data_config, train_config, "train", batch_size=8, device=device)
    val_loader = build_dataloader(data_config, train_config, "val", batch_size=8, device=device)

    images, labels = next(iter(train_loader))
    print("train batch images", tuple(images.shape), images.dtype, "labels", tuple(labels.shape), labels.dtype)
    assert images.shape[1:] == (3, 224, 224), images.shape
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert labels.ndim == 1
    assert int(labels.min()) >= 0 and int(labels.max()) < mapping.num_classes

    val_images, val_labels = next(iter(val_loader))
    print("val batch images", tuple(val_images.shape), "aug-off check: finite", torch.isfinite(val_images).all().item())

    weights = class_weight_tensor(data_config, mapping, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    tmp = Path("models") / "_sanity"
    tmp.mkdir(parents=True, exist_ok=True)

    for name in MODEL_NAMES:
        print(f"\n=== {name} ===")
        model = build_model(name, num_classes=mapping.num_classes, pretrained=False)
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print("parameters:", n_params)
        batch = images.to(device)
        y = labels.to(device)
        logits = model(batch)
        print("logits", tuple(logits.shape))
        assert logits.shape == (batch.shape[0], 3)
        loss = criterion(logits, y)
        print("loss", float(loss.detach()))
        assert torch.isfinite(loss)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert grad_ok, "no finite gradients"
        optimizer.step()
        changed = any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.named_parameters() if k in before)
        assert changed, "optimizer did not update parameters"
        print("backward + optimizer step: ok")

        ckpt = tmp / f"{name}.pt"
        save_checkpoint(
            ckpt,
            model=model,
            optimizer=optimizer,
            epoch=0,
            metrics={"loss": float(loss.detach())},
            class_to_index=mapping.class_to_index,
            config={"model": name, "sanity": True},
        )
        fresh = build_model(name, num_classes=mapping.num_classes, pretrained=False)
        load_checkpoint(ckpt, model=fresh, map_location="cpu")
        print("checkpoint save/load: ok")

    # One tiny real train/val step on the custom CNN.
    print("\n=== one-epoch smoke on custom_cnn (2 train batches via subset) ===")
    from torch.utils.data import DataLoader, Subset

    model = build_model("custom_cnn", num_classes=3, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_small = DataLoader(Subset(train_loader.dataset, list(range(16))), batch_size=8, shuffle=True)
    val_small = DataLoader(Subset(val_loader.dataset, list(range(16))), batch_size=8)
    train_stats = train_one_epoch(model, train_small, criterion, opt, device, epoch=1, log_every=1)
    val_stats = validate(model, val_small, criterion, device, class_names=list(mapping.classes))
    print("train", train_stats)
    print("val loss/acc/f1", val_stats["loss"], val_stats["accuracy"], val_stats["macro_f1"])
    print("\nSANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
