"""Part 3 evaluation: validation metrics, errors, confidence, Grad-CAM."""

from ml.eval.predict import PredictionBundle, collect_predictions, load_eval_model
from ml.eval.patterns import mild_error_patterns

__all__ = [
    "PredictionBundle",
    "collect_predictions",
    "load_eval_model",
    "mild_error_patterns",
]
