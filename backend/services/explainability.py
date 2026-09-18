"""Grad-CAM overlay for the predicted class. Reuses Part 3 ``ml.eval.gradcam``."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from backend.services.inference import InferenceService
from backend.services.preprocessing import image_to_chw_tensor
from ml.eval.figures import overlay_heatmap
from ml.eval.gradcam import GradCAM, tensor_to_image

GRADCAM_DISCLAIMER = (
    "Grad-CAM visualizes spatial activations for the predicted class. "
    "It is not causal proof that those pixels determined the label."
)


def explain_image(service: InferenceService, image: Image.Image) -> tuple[bytes, dict]:
    if not service.ready or service.model is None or service.pipeline is None:
        raise RuntimeError("model is not loaded")
    batch = image_to_chw_tensor(image, service.pipeline)
    tensor = batch[0]
    with service._lock:
        batch_dev = batch.to(service.device)
        service.model.eval()
        with GradCAM(service.model) as cam:
            # Grad-CAM needs gradients; do not use inference_mode here.
            logits = service.model(batch_dev)
            probs = logits.float().softmax(dim=1)[0]
            pred = int(probs.argmax().item())
            heatmap = cam(batch_dev, pred)
        rgb = tensor_to_image(tensor, service.pipeline.preprocessor.denormalise)
        overlay = overlay_heatmap(rgb, heatmap)
    png = _to_png(overlay)
    ordered = [name for name, _ in sorted(service.class_to_index.items(), key=lambda kv: kv[1])]
    meta = {
        "predicted_class": ordered[pred],
        "predicted_class_index": pred,
        "confidence": float(probs[pred].detach().cpu()),
        "disclaimer": GRADCAM_DISCLAIMER,
        "model": "EfficientNet-B0",
    }
    return png, meta


def _to_png(array: np.ndarray) -> bytes:
    pixels = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()
