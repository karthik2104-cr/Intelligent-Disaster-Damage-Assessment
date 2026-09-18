"""Preprocessing: raw image file to model-ready array.

Deliberately framework-free. The output is a NumPy array, and the channel order
is configurable, so Part 2 can wrap this for PyTorch or Keras without either
library being a dependency of the data pipeline or of the tests.

The only place image size is defined is ``preprocessing.image_size`` in the
config; nothing here or downstream should hard-code 224.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from ml.config import DataConfig, PreprocessingConfig
from ml.data.io import load_image

RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def resample_filter(name: str) -> Image.Resampling:
    try:
        return RESAMPLING[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported interpolation {name!r}; choose from {sorted(RESAMPLING)}"
        ) from exc


def resize_stretch(
    image: Image.Image, size: tuple[int, int], *, resample: Image.Resampling
) -> Image.Image:
    """Resize straight to ``(height, width)``, distorting the aspect ratio."""
    height, width = size
    return image.resize((width, height), resample=resample)


def resize_shorter_then_center_crop(
    image: Image.Image, size: tuple[int, int], *, resample: Image.Resampling
) -> Image.Image:
    """Scale so the short side matches the target, then crop the centre.

    Preserves aspect ratio at the cost of cropping the long edges. Chosen over a
    plain stretch because these are ordinary photographs where distorting
    proportions changes what damaged structures look like.
    """
    target_height, target_width = size
    original_width, original_height = image.size
    scale = max(target_width / original_width, target_height / original_height)
    scaled = (
        max(target_width, round(original_width * scale)),
        max(target_height, round(original_height * scale)),
    )
    resized = image.resize(scaled, resample=resample)
    left = (scaled[0] - target_width) // 2
    top = (scaled[1] - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


RESIZE_MODES: dict[str, Callable[..., Image.Image]] = {
    "stretch": resize_stretch,
    "resize_shorter_then_center_crop": resize_shorter_then_center_crop,
}


def resize(
    image: Image.Image,
    size: tuple[int, int],
    *,
    mode: str = "resize_shorter_then_center_crop",
    interpolation: str = "bilinear",
) -> Image.Image:
    try:
        resizer = RESIZE_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported resize mode {mode!r}") from exc
    return resizer(image, size, resample=resample_filter(interpolation))


def to_array(
    image: Image.Image,
    *,
    scale: float = 1.0 / 255.0,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    dtype: str = "float32",
    channel_order: str = "hwc",
) -> np.ndarray:
    """Convert a PIL image to a scaled, normalised array.

    Applies ``value * scale`` then ``(value - mean) / std``. With the ImageNet
    statistics in the config this yields the input distribution the pretrained
    backbones in Part 2 expect.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, :, None]
    array = array * float(scale)
    if mean is not None:
        array = array - np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    if std is not None:
        array = array / np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    array = array.astype(dtype, copy=False)
    if channel_order == "chw":
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
    elif channel_order != "hwc":
        raise ValueError(f"unsupported channel_order {channel_order!r}")
    return array


@dataclass(frozen=True)
class Preprocessor:
    """Callable preprocessing pipeline built from the configuration.

    Two entry points:

    * :meth:`prepare_image` - PIL in, PIL out, at the target size. Used by the
      cache writer and by the augmentation stage.
    * :meth:`__call__` - path or PIL in, model-ready array out.
    """

    config: PreprocessingConfig

    @classmethod
    def from_config(cls, config: DataConfig) -> "Preprocessor":
        return cls(config=config.preprocessing)

    @property
    def image_size(self) -> tuple[int, int]:
        return self.config.image_size

    @property
    def output_shape(self) -> tuple[int, int, int]:
        height, width = self.config.image_size
        channels = {"rgb": 3, "rgba": 4, "grayscale": 1, "l": 1}[self.config.color_mode.lower()]
        return (
            (channels, height, width)
            if self.config.channel_order == "chw"
            else (height, width, channels)
        )

    def prepare_image(self, image: Image.Image) -> Image.Image:
        return resize(
            image,
            self.config.image_size,
            mode=self.config.resize_mode,
            interpolation=self.config.interpolation,
        )

    def load(self, path: str | Path) -> Image.Image:
        return load_image(path, color_mode=self.config.color_mode)

    def __call__(self, source: str | Path | Image.Image) -> np.ndarray:
        image = source if isinstance(source, Image.Image) else self.load(source)
        if image.size != (self.config.image_size[1], self.config.image_size[0]):
            image = self.prepare_image(image)
        return to_array(
            image,
            scale=self.config.scale,
            mean=self.config.normalize.mean,
            std=self.config.normalize.std,
            dtype=self.config.output_dtype,
            channel_order=self.config.channel_order,
        )

    def denormalise(self, array: np.ndarray) -> np.ndarray:
        """Undo scaling and normalisation, for plotting augmented samples."""
        working = array.astype(np.float32, copy=True)
        if self.config.channel_order == "chw":
            working = working.transpose(1, 2, 0)
        working = working * np.asarray(self.config.normalize.std, dtype=np.float32)
        working = working + np.asarray(self.config.normalize.mean, dtype=np.float32)
        return np.clip(working / max(self.config.scale, 1e-12) / 255.0, 0.0, 1.0)
