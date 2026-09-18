"""FastAPI app: EfficientNet-B0 damage-severity inference.

Run from the repository root:

    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.schemas import HealthResponse, PredictResponse
from backend.services.explainability import GRADCAM_DISCLAIMER, explain_image
from backend.services.inference import ModelNotReadyError, service
from backend.services.preprocessing import ImageValidationError, decode_image, validate_upload

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Disaster Damage Assessment API",
    description=(
        "Image-level 3-class damage severity classification "
        "(`little_or_none`, `mild`, `severe`) using EfficientNet-B0 "
        "(`models/efficientnet_b0/best.pt`). "
        "This is an AI-assisted classifier, not an operational disaster-response system. "
        "Grad-CAM (`POST /explain`) is an activation visualization, not causal proof."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _read_upload(upload: UploadFile | None) -> tuple[str | None, str | None, bytes]:
    if upload is None:
        raise HTTPException(status_code=400, detail="No image file was provided.")
    if not upload.filename and not upload.content_type:
        raise HTTPException(status_code=400, detail="No image file was provided.")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 8 MB upload limit.")
    return upload.filename, upload.content_type, data


@app.exception_handler(ImageValidationError)
async def _invalid_image(_request, exc: ImageValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelNotReadyError)
async def _not_ready(_request, exc: ModelNotReadyError):
    return JSONResponse(status_code=503, content={"detail": "Model is not ready."})


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    payload = service.health()
    payload["gradcam_note"] = GRADCAM_DISCLAIMER
    return payload


@app.get("/openapi.json", include_in_schema=False)
def openapi_alias():
    return app.openapi()


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(image: UploadFile = File(..., description="RGB disaster photograph")):
    filename, content_type, data = _read_upload(image)
    try:
        return service.predict_bytes(filename=filename, content_type=content_type, data=data)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed.") from None


@app.post("/explain", tags=["explainability"])
def explain(image: UploadFile = File(..., description="RGB disaster photograph")):
    filename, content_type, data = _read_upload(image)
    try:
        validate_upload(filename=filename, content_type=content_type, data=data)
        pil = decode_image(data)
        png, meta = explain_image(service, pil)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=500, detail="Explanation failed.") from None
    headers = {
        "X-Predicted-Class": meta["predicted_class"],
        "X-Predicted-Class-Index": str(meta["predicted_class_index"]),
        "X-Confidence": f"{meta['confidence']:.6f}",
        "X-GradCAM-Disclaimer": GRADCAM_DISCLAIMER,
        "Access-Control-Expose-Headers": "X-Predicted-Class,X-Predicted-Class-Index,X-Confidence,X-GradCAM-Disclaimer",
    }
    return Response(content=png, media_type="image/png", headers=headers)
