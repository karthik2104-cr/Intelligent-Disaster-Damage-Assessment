"""Class-wise error counts from actual predictions (no narrative substitution)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

from ml.eval.predict import CLASS_NAMES, PredictionBundle


def raw_confusion(bundle: PredictionBundle) -> np.ndarray:
    labels = list(range(len(bundle.class_names)))
    return confusion_matrix(bundle.y_true, bundle.y_pred, labels=labels)


def row_normalized_confusion(bundle: PredictionBundle) -> np.ndarray:
    cm = raw_confusion(bundle).astype(np.float64)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return cm / denom


def mild_error_patterns(bundle: PredictionBundle) -> dict:
    """Counts and percentages for the mild-class confusions requested in Part 3."""
    names = list(bundle.class_names)
    mild = names.index("mild")
    none = names.index("little_or_none")
    severe = names.index("severe")
    n = len(bundle)
    true_mild = bundle.y_true == mild
    pred_mild = bundle.y_pred == mild
    support_mild = int(true_mild.sum())
    n_mild_none = int((true_mild & (bundle.y_pred == none)).sum())
    n_mild_severe = int((true_mild & (bundle.y_pred == severe)).sum())
    n_none_mild = int(((bundle.y_true == none) & pred_mild).sum())
    n_severe_mild = int(((bundle.y_true == severe) & pred_mild).sum())
    fp_mild = int((pred_mild & ~true_mild).sum())
    fn_mild = int((true_mild & ~pred_mild).sum())
    tp_mild = int((true_mild & pred_mild).sum())
    cm = raw_confusion(bundle)

    def pct(count: int, denom: int) -> float:
        return float(count) / denom * 100.0 if denom else 0.0

    pairs = []
    for i, true_name in enumerate(names):
        for j, pred_name in enumerate(names):
            if i == j:
                continue
            count = int(cm[i, j])
            pairs.append(
                {
                    "true": true_name,
                    "predicted": pred_name,
                    "count": count,
                    "pct_of_true_class": pct(count, int((bundle.y_true == i).sum())),
                    "pct_of_validation": pct(count, n),
                }
            )
    pairs.sort(key=lambda row: row["count"], reverse=True)

    return {
        "n_validation": n,
        "mild_support": support_mild,
        "mild_true_positives": tp_mild,
        "false_positives_mild": fp_mild,
        "false_negatives_mild": fn_mild,
        "mild_predicted_as_little_or_none": {
            "count": n_mild_none,
            "pct_of_mild": pct(n_mild_none, support_mild),
            "pct_of_validation": pct(n_mild_none, n),
        },
        "mild_predicted_as_severe": {
            "count": n_mild_severe,
            "pct_of_mild": pct(n_mild_severe, support_mild),
            "pct_of_validation": pct(n_mild_severe, n),
        },
        "little_or_none_predicted_as_mild": {
            "count": n_none_mild,
            "pct_of_little_or_none": pct(n_none_mild, int((bundle.y_true == none).sum())),
            "pct_of_validation": pct(n_none_mild, n),
        },
        "severe_predicted_as_mild": {
            "count": n_severe_mild,
            "pct_of_severe": pct(n_severe_mild, int((bundle.y_true == severe).sum())),
            "pct_of_validation": pct(n_severe_mild, n),
        },
        "most_common_confusions": pairs[:8],
        "confusion_matrix_raw": cm.tolist(),
        "class_names": names,
    }


def confidence_breakdown(
    bundle: PredictionBundle,
    *,
    high: float = 0.80,
    low: float = 0.50,
) -> dict:
    conf = bundle.confidence
    correct = bundle.y_true == bundle.y_pred
    incorrect = ~correct
    names = list(bundle.class_names)
    by_class = {}
    for idx, name in enumerate(names):
        mask = bundle.y_true == idx
        by_class[name] = {
            "n": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if mask.any() else 0.0,
            "mean_confidence_correct": float(conf[mask & correct].mean())
            if (mask & correct).any()
            else None,
            "mean_confidence_incorrect": float(conf[mask & incorrect].mean())
            if (mask & incorrect).any()
            else None,
        }
    mild = names.index("mild")
    mild_err = (bundle.y_true == mild) & incorrect
    high_wrong = incorrect & (conf >= high)
    return {
        "thresholds": {"high": high, "low": low},
        "n": len(bundle),
        "correct": int(correct.sum()),
        "incorrect": int(incorrect.sum()),
        "correct_high_confidence": int((correct & (conf >= high)).sum()),
        "incorrect_high_confidence": int(high_wrong.sum()),
        "incorrect_low_confidence": int((incorrect & (conf < low)).sum()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[incorrect].mean()) if incorrect.any() else None,
        "by_true_class": by_class,
        "mild_errors": {
            "n": int(mild_err.sum()),
            "mean_confidence": float(conf[mild_err].mean()) if mild_err.any() else None,
            "high_confidence": int((mild_err & (conf >= high)).sum()),
            "low_confidence": int((mild_err & (conf < low)).sum()),
        },
    }


def select_indices(
    bundle: PredictionBundle,
    *,
    true_label: int | None = None,
    pred_label: int | None = None,
    correct: bool | None = None,
    highest_confidence: bool = True,
    limit: int = 6,
) -> list[int]:
    mask = np.ones(len(bundle), dtype=bool)
    if true_label is not None:
        mask &= bundle.y_true == true_label
    if pred_label is not None:
        mask &= bundle.y_pred == pred_label
    if correct is True:
        mask &= bundle.y_true == bundle.y_pred
    elif correct is False:
        mask &= bundle.y_true != bundle.y_pred
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return []
    order = bundle.confidence[idxs]
    ranked = idxs[np.argsort(-order if highest_confidence else order)]
    return [int(i) for i in ranked[:limit]]
