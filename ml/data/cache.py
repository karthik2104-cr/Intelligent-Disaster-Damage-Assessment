"""Write resized copies of validated images for faster training I/O.

The cache is an optimisation, not a second source of truth. Raw files stay
untouched; cached files live under ``data/interim/cache`` and can be deleted
and regenerated at any time.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ml.config import DataConfig
from ml.data.io import load_image
from ml.data.preprocessing import resize


def cache_relative_path(image_path: str) -> str:
    """Stable cache path derived from the dataset-relative image path."""
    relative = Path(str(image_path).replace("\\", "/"))
    parts = relative.parts[1:] if relative.parts and relative.parts[0] == "data" else relative.parts
    return str(Path(*parts).with_suffix(".jpg"))


def write_cached_image(
    source: str | Path,
    destination: str | Path,
    config: DataConfig,
) -> Path:
    """Load ``source``, resize to ``cache.size``, and write a JPEG."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = load_image(source, color_mode=config.preprocessing.color_mode)
    resized = resize(
        image,
        config.cache.size,
        mode=config.preprocessing.resize_mode,
        interpolation=config.preprocessing.interpolation,
    )
    if resized.mode != "RGB":
        resized = resized.convert("RGB")
    resized.save(
        dest,
        format="JPEG",
        quality=config.cache.quality,
        optimize=True,
    )
    return dest
