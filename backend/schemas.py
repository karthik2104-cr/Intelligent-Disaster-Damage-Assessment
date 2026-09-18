"""API request/response schemas for the damage-severity service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    checkpoint: str
    device: str
    num_classes: int
    class_to_index: dict[str, int]
    image_size: list[int]
    ready: bool
    parameter_count: int | None = None
    model_load_time_ms: float | None = None
    gradcam_note: str = (
        "Grad-CAM is an activation visualization of the predicted class, "
        "not causal proof that those pixels determined the label."
    )


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    model: str
    image_size: list[int]
    inference_time_ms: float
    device: str
