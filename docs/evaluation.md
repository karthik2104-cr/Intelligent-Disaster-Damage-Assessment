# Part 3 — validation evaluation, errors, explainability

Part 3 does **not** train. It loads `models/<name>/best.pt` and scores the
**validation** split only. The test set is not used for analysis or selection.

Reproduce:

```powershell
python scripts/evaluate.py
python -m pytest tests -q
```

Logic lives in `ml/eval/`. Figures: `reports/figures/`. Machine-readable
results: `reports/evaluation/part3_validation.json`.

## Checkpoints

| Model | File used |
|---|---|
| LinearBaseline | `models/baseline/best.pt` |
| CustomCNN | `models/custom_cnn/best.pt` |
| ResNet-50 | `models/resnet50/best.pt` |
| EfficientNet-B0 | `models/efficientnet_b0/best.pt` |

`last.pt` was not used.

## Reproduced validation metrics (n = 6,128)

Compared to Part 2 experiment JSON. All headline metrics matched within
absolute tolerance **0.002**. Baseline matched to 0.0. Transfer models differ
by ~1 val example / ~0.001 F1 (fp32 eval vs Part 2 AMP).

| Model | Acc | Macro P | Macro R | Macro F1 | Mild F1 | Params | ms/img (CUDA, bs=16) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LinearBaseline | 0.495268 | 0.464482 | 0.463381 | 0.437427 | 0.211080 | 12,547 | — |
| CustomCNN | 0.585346 | 0.535573 | 0.530203 | 0.509860 | 0.236162 | 1,173,987 | — |
| ResNet-50 | 0.780352 | 0.651051 | 0.641573 | 0.645843 | 0.315949 | 23,514,179 | 3.32 |
| EfficientNet-B0 | 0.765829 | 0.649652 | 0.659884 | 0.653169 | 0.352790 | 4,011,391 | 1.53 |

Support: `little_or_none` 3605, `mild` 693, `severe` 1830.

Factual observations (not a universal ranking):

- EfficientNet-B0 has the highest **validation macro-F1** among the four.
- ResNet-50 has the highest **validation accuracy**.
- EfficientNet-B0 has the highest **mild F1**.
- EfficientNet-B0 has fewer parameters than ResNet-50 and higher images/s in a
  320-image CUDA batch-16 probe (652 vs 301 img/s).

Application development may continue with EfficientNet-B0 because it has the
highest **validation** macro-F1. That is a validation-based decision. The test
set remains unused for selection.

## Per-class F1 (reproduced, validation)

| Model | little_or_none | mild | severe |
|---|---:|---:|---:|
| LinearBaseline | 0.625 | 0.211 | 0.476 |
| CustomCNN | 0.731 | 0.236 | 0.562 |
| ResNet-50 | 0.885 | 0.316 | 0.736 |
| EfficientNet-B0 | 0.874 | 0.353 | 0.733 |

## Mild-class error counts (validation)

### ResNet-50 (mild support 693)

| Pattern | Count | % of mild | % of val |
|---|---:|---:|---:|
| Mild → little_or_none (FN) | 223 | 32.2 | 3.64 |
| Mild → severe (FN) | 261 | 37.7 | 4.26 |
| little_or_none → mild (FP) | 178 | — | 2.90 |
| severe → mild (FP) | 243 | — | 3.97 |

TP 209, FP 421, FN 484.

Most frequent off-diagonal: **severe → little_or_none** (269), then
**mild → severe** (261). ResNet-50 more often sends true mild to **severe**
than to little_or_none. It also confuses severe with little_or_none more than
EfficientNet-B0 does.

### EfficientNet-B0 (mild support 693)

| Pattern | Count | % of mild | % of val |
|---|---:|---:|---:|
| Mild → little_or_none (FN) | 175 | 25.3 | 2.86 |
| Mild → severe (FN) | 243 | 35.1 | 3.97 |
| little_or_none → mild (FP) | 293 | — | 4.78 |
| severe → mild (FP) | 298 | — | 4.86 |

TP 275, FP 591, FN 418.

Most frequent off-diagonal: **severe → mild** (298) and
**little_or_none → mild** (293). EfficientNet-B0 **over-predicts mild**
(higher recall, more FP) relative to ResNet-50.

Dominant mild-class pattern on both transfer models: true mild is more often
called **severe** than little_or_none. EfficientNet-B0’s extra mild F1 comes
with more mild false positives from the other two classes.

## Confidence (softmax max class; high ≥ 0.80, low < 0.50)

| | ResNet-50 | EfficientNet-B0 |
|---|---:|---:|
| Correct / incorrect | 4782 / 1346 | 4693 / 1435 |
| Correct, high conf | 4173 | 3067 |
| Incorrect, high conf | 704 | 217 |
| Incorrect, low conf | 76 | 403 |
| Mean conf correct | 0.936 | 0.833 |
| Mean conf incorrect | 0.787 | 0.615 |
| Mild-error mean conf | 0.822 | 0.680 |
| Mild errors with conf ≥ 0.80 | 299 / 484 | 123 / 418 |

ResNet-50 is more often **confident when wrong** (704 high-confidence errors).
On true mild, **incorrect** mean confidence (0.822) is **higher** than
correct (0.780): missed mild cases are not mostly low-confidence.

EfficientNet-B0 is less peaky; more errors sit below 0.50. Mild errors still
average 0.680 confidence. Example high-confidence mistakes include a
group-photo / banner image labelled `mild` in MEDIC but predicted
`little_or_none` at confidence ≈ 1.0 (possible label noise; not proven).

## Explainability

Grad-CAM on the last `Conv2d`, for the **predicted** class, on
EfficientNet-B0 and ResNet-50. Heatmaps are spatial activations weighted by
the class-score gradient. They are **not** proof of causal reasoning.

Panels: `reports/figures/efficientnet_b0_gradcam.png`,
`reports/figures/resnet50_gradcam.png`.

Observed on the generated panels: correct **severe** examples concentrate
activation on debris/collapse structure; a high-confidence **incorrect mild**
example is a posed group / “Hurricane Relief Week” banner with a diffuse map
and a `little_or_none` prediction. Overlay colour is a jet heatmap, not a
segmentation mask.

## Test set

Part 3 **did not** score the test set. Part 2 one-shot test numbers after
validation selection remain the only test report:

| Model | Test acc | Test macro-F1 |
|---|---:|---:|
| LinearBaseline | 0.5358 | 0.4391 |
| CustomCNN | 0.6294 | 0.5185 |
| ResNet-50 | 0.8070 | 0.6447 |
| EfficientNet-B0 | 0.7828 | 0.6418 |

Those figures were not used to choose a development model.

## Limitations

- Mild has 693 / 6128 validation images (~11%). Macro-F1 is sensitive to it.
- MEDIC labels are image-level and noisy for humanitarian / posed photos.
- Grad-CAM can highlight text overlays and faces; that is activation, not
  an explanation of the annotator’s label.
- Tiny fp32 vs AMP metric drift on transfer models (within 0.002).
- Inference timings are a short CUDA probe, not a production SLA.

## Files

- `ml/eval/` — predict, patterns, Grad-CAM, figures, benchmark
- `scripts/evaluate.py`
- `reports/evaluation/part3_validation.json`
- `reports/figures/*_cm_*.png`, `*_confidence.png`, `*_gradcam.png`,
  `efficientnet_b0_error_examples.png`
- `reports/error_analysis/efficientnet_b0_cases.json`
