"""FastAPI tests. One session-scoped model load; no training."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.services.preprocessing import ImageValidationError, validate_upload


def _jpeg_bytes(size=(80, 64), color=(40, 80, 120)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_empty_file():
    with pytest.raises(ImageValidationError):
        validate_upload(filename="x.jpg", content_type="image/jpeg", data=b"")


def test_validate_text_file():
    with pytest.raises(ImageValidationError):
        validate_upload(filename="notes.txt", content_type="text/plain", data=b"not an image")


def test_validate_bad_magic():
    with pytest.raises(ImageValidationError):
        validate_upload(filename="x.jpg", content_type="image/jpeg", data=b"this is not jpeg")


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["model"] == "EfficientNet-B0"
    assert body["num_classes"] == 3
    assert body["class_to_index"] == {"little_or_none": 0, "mild": 1, "severe": 2}
    assert body["image_size"] == [224, 224]
    assert body["device"] in {"cpu", "cuda"}
    assert "best.pt" in body["checkpoint"].replace("\\", "/")
    assert "not causal" in body["gradcam_note"].lower() or "causal" in body["gradcam_note"]


def test_openapi(client: TestClient):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert "/predict" in spec["paths"]
    assert "/explain" in spec["paths"]
    assert "/health" in spec["paths"]


def test_predict_missing_file(client: TestClient):
    response = client.post("/predict")
    assert response.status_code in {400, 422}


def test_predict_invalid_file(client: TestClient):
    response = client.post(
        "/predict",
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_valid_jpeg(client: TestClient):
    payload = _jpeg_bytes()
    response = client.post(
        "/predict",
        files={"image": ("sample.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) >= {
        "predicted_class",
        "predicted_class_index",
        "confidence",
        "probabilities",
        "model",
        "image_size",
        "inference_time_ms",
    }
    assert body["model"] == "EfficientNet-B0"
    assert body["predicted_class"] in {"little_or_none", "mild", "severe"}
    assert body["predicted_class_index"] in {0, 1, 2}
    probs = body["probabilities"]
    assert set(probs) == {"little_or_none", "mild", "severe"}
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-5
    top = max(probs, key=probs.get)
    assert top == body["predicted_class"]
    assert abs(probs[top] - body["confidence"]) < 1e-6
    assert body["inference_time_ms"] > 0


def test_explain_png(client: TestClient):
    payload = _jpeg_bytes()
    response = client.post(
        "/explain",
        files={"image": ("sample.jpg", payload, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/png")
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert response.headers["x-predicted-class"] in {"little_or_none", "mild", "severe"}
