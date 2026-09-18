"""Singleton EfficientNet-B0 inference service. Model is loaded once."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from backend.services.preprocessing import decode_image, image_to_chw_tensor, validate_upload
from ml.config import load_config
from ml.data.augmentation import build_eval_pipeline
from ml.engine.checkpoints import load_checkpoint
from ml.engine.train_config import load_train_config
from ml.eval.predict import CLASS_NAMES, load_eval_model, parameter_count

MODEL_NAME = "efficientnet_b0"
MODEL_DISPLAY = "EfficientNet-B0"
EXPECTED_CLASSES = 3
EXPECTED_MAPPING = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class ModelNotReadyError(RuntimeError):
    pass


class InferenceService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ready = False
        self.device = torch.device("cpu")
        self.model = None
        self.pipeline = None
        self.checkpoint_path: Path | None = None
        self.image_size: tuple[int, int] = (224, 224)
        self.load_time_ms: float | None = None
        self.parameter_count: int | None = None
        self.class_to_index = dict(EXPECTED_MAPPING)

    def load(self) -> None:
        started = time.perf_counter()
        data_config = load_config()
        train_config = load_train_config()
        ckpt = train_config.models_dir / MODEL_NAME / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Required checkpoint not found: {ckpt}. Train EfficientNet-B0 before serving."
            )
        payload = load_checkpoint(ckpt, map_location="cpu")
        mapping = dict(payload.get("class_to_index") or {})
        if mapping and mapping != EXPECTED_MAPPING:
            raise ValueError(f"checkpoint class map {mapping} != {EXPECTED_MAPPING}")
        num_classes = EXPECTED_CLASSES
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_eval_model(MODEL_NAME, ckpt, device, num_classes=num_classes)
        classifier = model.backbone.classifier[-1]
        if int(classifier.out_features) != EXPECTED_CLASSES:
            raise ValueError(f"classifier out_features={classifier.out_features}, expected {EXPECTED_CLASSES}")
        self.device = device
        self.model = model
        self.pipeline = build_eval_pipeline(data_config)
        self.checkpoint_path = ckpt
        self.image_size = tuple(data_config.preprocessing.image_size)
        self.parameter_count = parameter_count(model)
        self.load_time_ms = (time.perf_counter() - started) * 1000.0
        self.ready = True

    def health(self) -> dict:
        return {
            "status": "ok" if self.ready else "loading",
            "model": MODEL_DISPLAY,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else "",
            "device": str(self.device),
            "num_classes": EXPECTED_CLASSES,
            "class_to_index": self.class_to_index,
            "image_size": list(self.image_size),
            "ready": self.ready,
            "parameter_count": self.parameter_count,
            "model_load_time_ms": self.load_time_ms,
        }

    def predict_bytes(self, *, filename: str | None, content_type: str | None, data: bytes) -> dict:
        validate_upload(filename=filename, content_type=content_type, data=data)
        image = decode_image(data)
        return self.predict_image(image)

    def predict_image(self, image) -> dict:
        if not self.ready or self.model is None or self.pipeline is None:
            raise ModelNotReadyError("model is not loaded")
        batch = image_to_chw_tensor(image, self.pipeline)
        with self._lock:
            started = time.perf_counter()
            probs, pred_idx, inference_ms = self._forward(batch)
        names = list(CLASS_NAMES)
        probabilities = {names[i]: float(probs[i]) for i in range(len(names))}
        return {
            "predicted_class": names[pred_idx],
            "predicted_class_index": int(pred_idx),
            "confidence": float(probs[pred_idx]),
            "probabilities": probabilities,
            "model": MODEL_DISPLAY,
            "image_size": list(self.image_size),
            "inference_time_ms": inference_ms,
            "device": str(self.device),
        }

    def _forward(self, batch: torch.Tensor) -> tuple[list[float], int, float]:
        batch = batch.to(self.device)
        started = time.perf_counter()
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(batch)
            probs = torch.softmax(logits.float(), dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pred = int(torch.argmax(probs).item())
        return probs.detach().cpu().tolist(), pred, elapsed_ms


service = InferenceService()
