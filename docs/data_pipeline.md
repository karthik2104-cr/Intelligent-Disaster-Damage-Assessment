# Data pipeline

```
Raw Dataset (data/raw/MEDIC, never modified)
     ↓
Validation          scripts/verify_dataset.py
     ↓
Cleaning            recorded exclusions, not silent deletes
     ↓
Train/Val/Test      official MEDIC splits by default
     ↓
Preprocessing       load → RGB → resize → scale → ImageNet normalise
     ↓
Training Augmentation   train split only
     ↓
Ready for Part 2    data/processed/{train,val,test}.csv
                    data/metadata/label_mapping.json
                    data/metadata/class_weights.json
```

Configuration lives in `configs/data_config.yaml`. Image size, seed, split
strategy, normalisation and augmentation probabilities are not hard-coded in
the Python modules.

## Cleaning

`ml/data/validation.py` probes every labelled path:

- missing file
- unreadable / truncated image
- disallowed extension
- too-small image (min side from config, default 32 px)
- missing or unpublished label
- exact byte-identical duplicates (SHA-256)
- duplicate groups whose copies **disagree** on the primary label (every copy
  excluded; we do not pick a winner)

Raw files stay in place. Excluded rows go to
`data/interim/excluded_samples.csv` with `exclusion_reason`.

## Split strategy

**Default: `official`.** MEDIC publishes `MEDIC_train.tsv`, `MEDIC_dev.tsv`,
`MEDIC_test.tsv`. Dev is renamed `val`. The official test set is not reshuffled.

This is the right default because:

1. The test set stays isolated and comparable with published MEDIC numbers.
2. MEDIC's own authors already made the train/dev/test cut.

**Alternative: `regenerate`.** Groups by `event` (directory under each source
dataset) so photographs of one incident cannot appear in two splits, then
fills splits to the configured ratios. Use this only for leakage-sensitive
ablations, not as the headline split.

After either strategy, `check_leakage` reports overlapping `image_path` values
and overlapping content hashes. Shared *events* under the official split are
expected (the official cut is by image, not by event) and are not treated as
leakage.

Seed: `42` (config).

## Preprocessing

Implemented in `ml/data/preprocessing.py`, driven by config:

| Step | Default |
|---|---|
| Colour | convert to RGB (palette / RGBA composited on white) |
| Resize | short-side scale + centre crop to `image_size` (224, 224) |
| Interpolation | bilinear |
| Scale | 1/255 |
| Normalise | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Dtype / layout | float32, HWC (set `channel_order: chw` for PyTorch-native layout) |

Aspect-preserving crop is used instead of a stretch because these are
photographs: distorting building proportions would change the damage cue.

`prepare_dataset.py` can write a JPEG cache at 256×256 under
`data/interim/cache` so Part 2 does not re-decode original social-media files
every epoch. Augmentation still runs at load time on the cached (or original)
image; the final 224×224 resize always happens in the preprocessor.

## Augmentation (train only)

`ml/data/augmentation.py`. Validation and test use `build_eval_pipeline()`,
which ignores the augmentation block.

| Transform | Why |
|---|---|
| Horizontal flip p=0.5 | Damage has no left/right handedness |
| Rotation ±15° p=0.5 | Hand-held camera tilt |
| Random resized crop p=0.8, scale 0.7–1.0 | Photographer standing nearer / framing wider |
| Brightness ±0.2 p=0.5 | Time of day, smoke, exposure |
| Contrast ±0.2 p=0.5 | Same |
| Saturation ±0.2 p=0.3 | Device white-balance |

Deliberately **off**:

- Vertical flip — gravity (collapse, waterline) is a real cue; upside-down
  photos do not occur at inference.
- Grayscale — fire, smoke and floodwater are colour signals.
- Cutout / random erasing — damage evidence is often a small region; masking
  it would keep the label while removing the evidence.

Randomness is `numpy.random.Generator` seeded by `(seed, epoch, index)`, so an
epoch is reproducible.

## Class imbalance

Measured on the **train split only** after cleaning. Strategy in config:
`class_weights` with sklearn's `balanced` formula

```
n_samples / (n_classes * n_class)
```

Weights are written to `data/metadata/class_weights.json` for the Part 2 loss.
We do **not** oversample on disk: that would duplicate social-media photos
that augmentation already varies, and would change the dataset the
verification report describes.

If verification finds the imbalance ratio is below ~1.5, the weights are
close to 1 and can be ignored in Part 2. If it is severe (>10), Part 2 may
also enable a weighted sampler using the same JSON; that decision is left
open on purpose.

## Reproducibility

- Dataset source and version: `configs/data_config.yaml` and
  `data/metadata/acquisition.json`
- HF class-label order is checked against the live schema before decoding
  integer labels
- Fixed seed 42
- Official splits used by default
- Dependencies pinned in `requirements.txt`
- Unit tests use a synthetic MEDIC-shaped fixture, not the 11 GB download
