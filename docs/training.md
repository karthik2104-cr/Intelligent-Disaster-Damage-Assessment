# Part 2 models and training

PyTorch was chosen because the rest of this project is already NumPy/PIL-based,
the existing dataset object has a lazy Torch adapter, and transfer-learning
backbones (ResNet-50, EfficientNet-B0) are first-class in `torchvision`. Keras
would have required a second preprocessing path; we convert HWC → CHW only at
the DataLoader boundary and leave Part 1 untouched.

Environment used for the runs below: Python 3.11, **PyTorch 2.6.0+cu124**,
**torchvision 0.21.0+cu124**, CUDA available, NVIDIA GPU (AMP enabled).

## Task

MEDIC v1.0, official split, 3-class `damage_severity`:
`little_or_none=0`, `mild=1`, `severe=2` (from `data/metadata/label_mapping.json`).

Counts after Part 1 cleaning: train 46,487 / val 6,128 / test 15,630.

## DataLoader boundary

`ml.data.dataset.build_dataset` still returns HWC float32 arrays (Part 1).
`ml.engine.dataset.DamageSeverityTorchDataset` wraps that object and permutes
to CHW tensors. Training loaders shuffle; val/test do not. Augmentation is
still the Part 1 train pipeline only.

## Loss

`torch.nn.CrossEntropyLoss` with the balanced class-weight vector from
`data/metadata/class_weights.json`. Cross-entropy is the standard loss for
single-label multi-class classification. The weights (train-split only) raise
the cost of mistakes on `mild`, the minority class. No on-disk oversampling.

## Optimizer and controls

- Optimizer: AdamW (`lr=1e-3`, `weight_decay=1e-4` unless a model override)
- Scheduler: ReduceLROnPlateau on validation loss (factor 0.5, patience 2)
- Early stopping and best checkpoint: **validation macro-F1**
- Test set: never used for stopping, LR schedule, or checkpoint selection
- AMP on CUDA; gradient clip norm 1.0
- Seed 42 (Python, NumPy, Torch)

Transfer models train a new 3-class head with the backbone frozen, then unfreeze
the last block (`layer4` for ResNet-50, `features.7` for EfficientNet-B0) at
`lr=1e-4` if the head run is stable. Fine-tuning is kept short because the
4 GB GPU and the mild-class gap make full-network tuning easy to overfit.

## Models

| Name | Role | Params (approx.) |
|---|---|---|
| `LinearBaseline` | 8×8 GAP + MLP. No convolutions. Floor. | tiny |
| `CustomCNN` | 4 conv-BN-ReLU blocks (32→64→128→256), GAP, dropout 0.4, 3-logit head. From scratch. | ~1.17M |
| `ResNet50Classifier` | ImageNet `IMAGENET1K_V2`, new 3-class `fc`. Head then `layer4`. | ImageNet backbone |
| `EfficientNetB0Classifier` | ImageNet `IMAGENET1K_V1`, new 3-class classifier. Head then `features.7`. | ImageNet backbone |

EfficientNet-B0 uses the same 224×224 ImageNet normalisation as Part 1. The
dataset pipeline is not changed for it.

## How to train

```powershell
python scripts/sanity_check.py
python scripts/train.py --model baseline
python scripts/train.py --model custom_cnn
python scripts/train.py --model resnet50
python scripts/train.py --model efficientnet_b0
```

Config: `configs/train_config.yaml`. Checkpoints: `models/<name>/best.pt` and
`last.pt` (gitignored). Experiment JSON: `reports/experiments/`.

## Actual results

Full tables: [`reports/experiments/comparison.md`](../reports/experiments/comparison.md).

Validation (n=6128), selection metric = macro-F1:

| Model | Trained? | Evaluated? | Val acc | Val macro-F1 | Val mild F1 | Wall time |
|---|---|---|---:|---:|---:|---|
| LinearBaseline | yes | val + test | 0.4953 | 0.4374 | 0.2111 | 22.6 min |
| CustomCNN | yes | val + test | 0.5852 | 0.5097 | 0.2361 | 501.4 min |
| ResNet-50 | yes (head + last-block FT) | val + test | 0.7805 | 0.6463 | 0.3172 | 20.1 min |
| EfficientNet-B0 | yes (head + last-block FT) | val + test | 0.7660 | **0.6536** | **0.3541** | 17.3 min |

Isolated test (n=15630), after selection, not used to pick the model:

| Model | Test acc | Test macro-F1 |
|---|---:|---:|
| LinearBaseline | 0.5358 | 0.4391 |
| CustomCNN | 0.6294 | 0.5185 |
| ResNet-50 | 0.8070 | 0.6447 |
| EfficientNet-B0 | 0.7828 | 0.6418 |

Fine-tuning improved transfer models on validation macro-F1 (ResNet head peak
~0.617 → best 0.646 after `layer4`; EfficientNet head peak ~0.589 → best 0.654
after `features.7`). ResNet train accuracy climbed to 0.96 while val loss rose
in later FT epochs — the saved checkpoint is the best **val macro-F1**, not the
last epoch.

Custom CNN wall time is long because it trains from scratch at batch 32 with
on-the-fly decode; transfer models reuse ImageNet features and finished in
about 20 minutes each.

## Model selected for Part 3

**EfficientNet-B0**, checkpoint `models/efficientnet_b0/best.pt`.

Highest validation macro-F1 and best mild-class F1/recall among the four.
ResNet-50 is the accuracy runner-up (val acc 0.7805, test acc 0.8070) if a
later ablation wants that trade-off.

## Limitations (Part 2)

- `mild` F1 stays below 0.36 even for the best model.
- Transfer fine-tune used only the last block, not the full backbone.
- Custom CNN was not given a multi-day hyperparameter search.
- Test metrics are a one-shot report; they must not be used to retune Part 2.

Part 3 (not started) will cover confusion matrices, error analysis, confidence,
and explainability on the selected checkpoint.
