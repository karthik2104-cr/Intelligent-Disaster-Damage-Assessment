"""Turn an uploaded image into the same tensor the validation pipeline uses."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from ml.data.augmentation import ImagePipeline

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"RIFF", "webp"),  # WebP is RIFF....WEBP
    (b"BM", "bmp"),
)


class ImageValidationError(ValueError):
    """Raised for unusable uploads. The API maps this to HTTP 400."""


def validate_upload(*, filename: str | None, content_type: str | None, data: bytes) -> None:
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )
    suffix = ""
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
    type_ok = (content_type or "").lower() in ALLOWED_CONTENT_TYPES or content_type is None
    suffix_ok = suffix in ALLOWED_SUFFIXES or suffix == ""
    if filename and not suffix_ok:
        raise ImageValidationError(
            "Unsupported file extension. Use JPEG, PNG, WebP, BMP, or GIF."
        )
    if content_type and not type_ok and not suffix_ok:
        raise ImageValidationError(
            f"Unsupported content type {content_type!r}. Use JPEG, PNG, WebP, BMP, or GIF."
        )
    if not _looks_like_image(data):
        raise ImageValidationError("File content is not a recognised image format.")


def _looks_like_image(data: bytes) -> bool:
    for magic, kind in _SIGNATURES:
        if data.startswith(magic):
            if kind == "webp":
                return b"WEBP" in data[:16]
            return True
    return False


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("The file could not be decoded as an image.") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_chw_tensor(image: Image.Image, pipeline: ImagePipeline) -> torch.Tensor:
    """Eval preprocess (HWC) then permute to CHW — same boundary as Part 2/3."""
    array = pipeline.transform(image)
    if array.ndim != 3:
        raise ImageValidationError(f"expected HWC array, got shape {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)
