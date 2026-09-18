"""Lightweight Part 3 evaluation tests. No full validation pass, no training."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ml.eval.gradcam import GradCAM, last_conv2d
from ml.eval.patterns import confidence_breakdown, mild_error_patterns, raw_confusion
from ml.eval.predict import CLASS_NAMES, PredictionBundle, compare_to_reference
from ml.models.custom_cnn import CustomCNN


def _bundle() -> PredictionBundle:
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 0, 2, 1, 2, 1], dtype=np.int64)
    probs = np.zeros((8, 3), dtype=np.float64)
    for i, p in enumerate(y_pred):
        probs[i] = 0.05
        probs[i, p] = 0.9
        probs[i] = probs[i] / probs[i].sum()
    return PredictionBundle(y_true=y_true, y_pred=y_pred, probs=probs, indices=np.arange(8))


def test_class_names_order():
    assert CLASS_NAMES == ("little_or_none", "mild", "severe")


def test_mild_error_pattern_counts():
    patterns = mild_error_patterns(_bundle())
    assert patterns["mild_support"] == 3
    assert patterns["mild_predicted_as_little_or_none"]["count"] == 1
    assert patterns["mild_predicted_as_severe"]["count"] == 1
    assert patterns["little_or_none_predicted_as_mild"]["count"] == 1
    assert patterns["severe_predicted_as_mild"]["count"] == 1
    assert patterns["false_negatives_mild"] == 2
    assert patterns["false_positives_mild"] == 2
    cm = np.asarray(patterns["confusion_matrix_raw"])
    assert cm.shape == (3, 3)
    np.testing.assert_array_equal(raw_confusion(_bundle()), cm)


def test_confidence_breakdown_counts():
    stats = confidence_breakdown(_bundle(), high=0.8, low=0.5)
    assert stats["n"] == 8
    assert stats["correct"] + stats["incorrect"] == 8
    assert stats["incorrect_high_confidence"] >= 1


def test_compare_to_reference_tolerance():
    summary = {
        "accuracy": 0.5,
        "macro_precision": 0.4,
        "macro_recall": 0.4,
        "macro_f1": 0.4,
        "per_class": {"mild": {"f1": 0.2}},
    }
    reference = {
        "accuracy": 0.5005,
        "macro_precision": 0.4,
        "macro_recall": 0.4,
        "macro_f1": 0.4,
        "per_class": {"mild": {"f1": 0.2}},
    }
    report = compare_to_reference(summary, reference, atol=0.002)
    assert report["accuracy"]["ok"]
    assert report["mild_f1"]["ok"]


def test_gradcam_shape_custom_cnn():
    model = CustomCNN(num_classes=3)
    model.eval()
    x = torch.randn(1, 3, 64, 64, requires_grad=True)
    layer = last_conv2d(model)
    assert isinstance(layer, nn.Conv2d)
    with GradCAM(model, layer) as cam:
        heat = cam(x, class_index=1)
    assert heat.ndim == 2
    assert heat.min() >= 0
    assert heat.max() <= 1 + 1e-6
