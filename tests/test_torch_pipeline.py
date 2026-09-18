"""Unit tests for Part 2 PyTorch components. No multi-hour training."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.data.dataset import ImageClassificationDataset
from ml.data.manifest import build_raw_manifest
from ml.data.splitting import make_splits
from ml.data.validation import clean_manifest, flag_issues, scan_manifest
from ml.engine.checkpoints import load_checkpoint, save_checkpoint
from ml.engine.dataset import DamageSeverityTorchDataset
from ml.engine.metrics import summarise_predictions
from ml.engine.reproducibility import seed_everything
from ml.engine.trainer import train_one_epoch, validate
from ml.models.factory import build_model
from ml.data.augmentation import build_eval_pipeline
from ml.data.labels import build_label_mapping


def _tiny_torch_dataset(mini_dataset):
    manifest = build_raw_manifest(mini_dataset)
    scanned = scan_manifest(manifest, mini_dataset, workers=2)
    clean, _, _ = clean_manifest(flag_issues(scanned, mini_dataset), mini_dataset)
    split = make_splits(clean, mini_dataset)
    train = split.loc[split["split"] == "train"].reset_index(drop=True)
    mapping = build_label_mapping("damage_severity", split["damage_severity"].tolist())
    source = ImageClassificationDataset(
        manifest=train,
        pipeline=build_eval_pipeline(mini_dataset),
        label_mapping=mapping,
        target_column="damage_severity",
        seed=mini_dataset.seed,
    )
    return DamageSeverityTorchDataset(source), mapping


def test_torch_dataset_chw_and_label_dtype(mini_dataset):
    dataset, mapping = _tiny_torch_dataset(mini_dataset)
    image, label = dataset[0]
    assert image.shape[0] == 3
    assert image.ndim == 3
    assert image.dtype == torch.float32
    assert label.dtype == torch.int64
    assert 0 <= int(label) < mapping.num_classes == 3


def test_dataloader_batch_shapes(mini_dataset):
    dataset, _ = _tiny_torch_dataset(mini_dataset)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    assert images.ndim == 4
    assert images.shape[1] == 3
    assert labels.dtype == torch.int64
    assert labels.shape[0] == images.shape[0]


def test_models_output_three_logits():
    seed_everything(0)
    x = torch.randn(2, 3, 64, 64)
    for name in ("baseline", "custom_cnn", "resnet50", "efficientnet_b0"):
        model = build_model(name, num_classes=3, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3), (name, logits.shape)


def test_custom_cnn_parameter_count_is_modest():
    model = build_model("custom_cnn", num_classes=3)
    n = sum(p.numel() for p in model.parameters())
    assert n < 5_000_000


def test_weighted_loss_and_backward():
    model = build_model("baseline", num_classes=3)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 1])
    weights = torch.tensor([0.5, 4.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("baseline", num_classes=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=opt,
        epoch=3,
        metrics={"val_macro_f1": 0.4},
        class_to_index={"little_or_none": 0, "mild": 1, "severe": 2},
        config={"image_size": [224, 224]},
    )
    fresh = build_model("baseline", num_classes=3)
    payload = load_checkpoint(path, model=fresh, map_location="cpu")
    assert payload["epoch"] == 3
    assert payload["class_to_index"]["mild"] == 1
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a, b)


def test_inference_argmax():
    model = build_model("custom_cnn", num_classes=3, pretrained=False)
    model.eval()
    x = torch.randn(5, 3, 64, 64)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    assert pred.shape == (5,)
    assert pred.dtype == torch.int64
    assert int(pred.min()) >= 0 and int(pred.max()) <= 2


def test_summarise_predictions_class_order():
    names = ["little_or_none", "mild", "severe"]
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 1, 0, 2]
    summary = summarise_predictions(y_true, y_pred, class_names=names)
    assert set(summary["per_class"]) == set(names)
    assert 0.0 <= summary["macro_f1"] <= 1.0
    assert summary["n"] == 5


def test_one_train_and_val_step(mini_dataset):
    dataset, mapping = _tiny_torch_dataset(mini_dataset)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    model = build_model("custom_cnn", num_classes=3, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    train_stats = train_one_epoch(model, loader, criterion, opt, device, epoch=1, log_every=10)
    val_stats = validate(model, loader, criterion, device, class_names=list(mapping.classes))
    assert train_stats["loss"] > 0
    assert 0 <= val_stats["accuracy"] <= 1
