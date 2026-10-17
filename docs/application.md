# Application (Part 4)

Local FastAPI + React demo with the **Fieldnote / Damage Lab** (Field Evidence Lab) UI. Not a cloud deployment and not an operational disaster-response system.

## Model

- Architecture: EfficientNet-B0
- Checkpoint: `models/efficientnet_b0/best.pt` (never `last.pt`)
- Reason it is served: highest **validation** macro-F1 among the four Part 2/3 models
- Loaded once at API startup
- Preprocessing: Part 1 eval pipeline (RGB, short-side resize, center crop, 224, ImageNet norm) then HWC→CHW

## UI overview

The React app (`frontend/`) presents a single-page inspection workspace:

1. **Evidence intake** — drag-and-drop or file picker; analyze / Grad-CAM / reset
2. **Visual evidence** — source photograph and Grad-CAM activation overlay
3. **Assessment report** — predicted severity, confidence, inference time, model/device, class probabilities

Screenshots: [`docs/screenshots/`](screenshots/).

Grad-CAM highlights activations for the predicted class. It is not causal proof and not building-level segmentation.

## Run

Terminal 1 (repo root, `.venv` active):

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs and http://127.0.0.1:8000/openapi.json

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

API base URL: `frontend/.env` → `VITE_API_URL=http://127.0.0.1:8000`

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model, device, class map, ready flag |
| POST | `/predict` | Multipart `image` → JSON prediction |
| POST | `/explain` | Multipart `image` → PNG Grad-CAM overlay |

`POST /explain` is an activation visualization, not causal proof.

Upload limits: 8 MB; JPEG/PNG/WebP/BMP/GIF. Images are decoded in memory and not written to disk.

## Example `/predict` body

Field values are computed; this is the shape only:

```json
{
  "predicted_class": "little_or_none",
  "predicted_class_index": 0,
  "confidence": 0.87,
  "probabilities": {
    "little_or_none": 0.87,
    "mild": 0.08,
    "severe": 0.05
  },
  "model": "EfficientNet-B0",
  "image_size": [224, 224],
  "inference_time_ms": 12.3,
  "device": "cuda"
}
```

## Frontend build

```powershell
cd frontend
npm run build
```
