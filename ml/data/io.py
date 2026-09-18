"""Image reading and probing.

Everything that touches a file on disk goes through here so that failures are
reported as data rather than raised as exceptions in the middle of a scan. This
is what lets the validation stage count corrupted images instead of crashing on
the first one.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

# Social media images are frequently truncated in transit. We want to *detect*
# that rather than silently pad the missing bytes, so leave the tolerant loader
# off and let PIL raise.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# The dataset contains legitimately large photographs; raise the decompression
# bomb ceiling but keep it finite so a malformed header cannot exhaust memory.
Image.MAX_IMAGE_PIXELS = 200_000_000

_CHANNELS_BY_MODE = {
    "1": 1,
    "L": 1,
    "LA": 2,
    "I": 1,
    "I;16": 1,
    "F": 1,
    "P": 1,
    "PA": 2,
    "RGB": 3,
    "RGBA": 4,
    "CMYK": 4,
    "YCbCr": 3,
    "LAB": 3,
    "HSV": 3,
}


class ImageReadError(RuntimeError):
    """Raised by :func:`load_image` when an image cannot be decoded."""


@dataclass(frozen=True)
class ImageInfo:
    """Result of probing a single file. ``ok`` is False when ``error`` is set."""

    path: str
    exists: bool
    ok: bool
    error: str | None = None
    format: str | None = None
    mode: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    file_bytes: int | None = None
    is_animated: bool | None = None
    content_hash: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def channels_for_mode(mode: str | None) -> int | None:
    if mode is None:
        return None
    if mode in _CHANNELS_BY_MODE:
        return _CHANNELS_BY_MODE[mode]
    try:
        from PIL import ImageMode

        return len(ImageMode.getmode(mode).bands)
    except Exception:  # noqa: BLE001 - unknown modes are reported as None
        return None


def content_hash(path: str | Path, algorithm: str = "sha256", chunk_bytes: int = 1 << 20) -> str:
    """Hash of the raw file bytes, used for exact-duplicate detection."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_image(
    path: str | Path,
    *,
    hash_algorithm: str | None = "sha256",
    full_decode: bool = True,
) -> ImageInfo:
    """Inspect one image without raising.

    Args:
        path: File to inspect.
        hash_algorithm: Content hash to compute, or None to skip.
        full_decode: Decode the pixel data as well as the header. Header-only
            checks miss truncated files, which are exactly the failure mode we
            care about, so this defaults to True.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return ImageInfo(path=str(file_path), exists=False, ok=False, error="file_not_found")

    try:
        file_bytes = file_path.stat().st_size
    except OSError as exc:
        return ImageInfo(path=str(file_path), exists=True, ok=False, error=f"stat_failed: {exc}")

    if file_bytes == 0:
        return ImageInfo(
            path=str(file_path), exists=True, ok=False, error="empty_file", file_bytes=0
        )

    try:
        with Image.open(file_path) as image:
            fmt, mode = image.format, image.mode
            width, height = image.size
            is_animated = bool(getattr(image, "is_animated", False))
            if full_decode:
                # load() raises on truncated or otherwise broken pixel data.
                image.load()
    except UnidentifiedImageError:
        return ImageInfo(
            path=str(file_path),
            exists=True,
            ok=False,
            error="unidentified_image",
            file_bytes=file_bytes,
        )
    except Image.DecompressionBombError as exc:
        return ImageInfo(
            path=str(file_path),
            exists=True,
            ok=False,
            error=f"decompression_bomb: {exc}",
            file_bytes=file_bytes,
        )
    except (OSError, ValueError, SyntaxError) as exc:
        return ImageInfo(
            path=str(file_path),
            exists=True,
            ok=False,
            error=f"decode_failed: {type(exc).__name__}: {exc}",
            file_bytes=file_bytes,
        )

    digest = None
    if hash_algorithm:
        try:
            digest = content_hash(file_path, hash_algorithm)
        except OSError as exc:
            return ImageInfo(
                path=str(file_path),
                exists=True,
                ok=False,
                error=f"hash_failed: {exc}",
                file_bytes=file_bytes,
            )

    return ImageInfo(
        path=str(file_path),
        exists=True,
        ok=True,
        format=fmt,
        mode=mode,
        width=width,
        height=height,
        channels=channels_for_mode(mode),
        file_bytes=file_bytes,
        is_animated=is_animated,
        content_hash=digest,
    )


def load_image(path: str | Path, *, color_mode: str = "rgb") -> Image.Image:
    """Load an image and convert it to the requested colour mode.

    Raises:
        ImageReadError: If the file is missing or cannot be decoded. Callers that
            want to tolerate bad files should probe first with
            :func:`probe_image`.
    """
    target = {"rgb": "RGB", "rgba": "RGBA", "grayscale": "L", "l": "L"}.get(
        color_mode.lower()
    )
    if target is None:
        raise ValueError(f"unsupported color_mode {color_mode!r}")

    file_path = Path(path)
    try:
        with Image.open(file_path) as image:
            # Animated GIFs decode to their first frame; converting inside the
            # context manager also detaches the result from the file handle.
            image.load()
            if image.mode == target:
                return image.copy()
            # Palette images with transparency need an intermediate RGBA step,
            # otherwise PIL drops the alpha and can produce black backgrounds.
            if image.mode in {"P", "PA"} and "transparency" in image.info:
                image = image.convert("RGBA")
            if target == "RGB" and image.mode in {"RGBA", "LA"}:
                # Composite onto white: these are photographs, and a black
                # background would look like burnt/shadowed content.
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                return background
            return image.convert(target)
    except FileNotFoundError as exc:
        raise ImageReadError(f"image not found: {file_path}") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageReadError(f"cannot decode image {file_path}: {exc}") from exc
