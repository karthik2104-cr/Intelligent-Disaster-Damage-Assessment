"""Training-only data augmentation.

Which transforms are enabled is a data decision, not a default. MEDIC images are
ordinary photographs taken by people on the ground, so the transform set is
restricted to variations that could plausibly have produced a different real
photograph of the same scene:

enabled
    horizontal flip      - a scene photographed from the opposite side is equally
                           plausible; damage has no left/right handedness.
    small rotation (15 deg) - hand-held camera tilt.
    random resized crop  - the photographer standing nearer or framing wider.
    brightness/contrast  - time of day, weather, exposure. Disaster photos in the
                           dataset span bright daylight to smoke-darkened scenes.
    mild saturation      - camera and white-balance differences between devices.

deliberately disabled
    vertical flip        - upside-down photographs do not occur, and gravity is a
                           real cue for damage (collapse, debris, water level).
    grayscale            - colour carries damage signal: fire, smoke, floodwater.
    heavy colour jitter  - could turn muddy floodwater into clear water, or smoke
                           haze into clear sky, changing the true label.
    cutout / erasing     - damage evidence is often a small part of the frame, so
                           occluding a patch risks removing the only evidence
                           while keeping the label.

Every transform is applied through a caller-supplied
:class:`numpy.random.Generator`, so an augmented epoch is reproducible from a
seed. :func:`build_eval_pipeline` returns a pipeline with augmentation off - it is
what validation and test data must use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageEnhance

from ml.config import DataConfig
from ml.data.preprocessing import Preprocessor, resample_filter, to_array


@dataclass(frozen=True)
class AugmentationConfig:
    """Resolved augmentation parameters. Probabilities are per-sample."""

    enabled: bool = False
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0
    grayscale: float = 0.0
    rotation_probability: float = 0.0
    rotation_max_degrees: float = 0.0
    crop_probability: float = 0.0
    crop_scale: tuple[float, float] = (1.0, 1.0)
    crop_ratio: tuple[float, float] = (1.0, 1.0)
    brightness_probability: float = 0.0
    brightness_delta: float = 0.0
    contrast_probability: float = 0.0
    contrast_delta: float = 0.0
    saturation_probability: float = 0.0
    saturation_delta: float = 0.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "AugmentationConfig":
        def block(name: str) -> Mapping[str, object]:
            value = mapping.get(name)
            return value if isinstance(value, Mapping) else {}

        rotation, crop = block("rotation"), block("random_resized_crop")
        brightness, contrast = block("brightness"), block("contrast")
        saturation = block("saturation")
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            horizontal_flip=float(mapping.get("horizontal_flip", 0.0) or 0.0),
            vertical_flip=float(mapping.get("vertical_flip", 0.0) or 0.0),
            grayscale=float(mapping.get("grayscale", 0.0) or 0.0),
            rotation_probability=float(rotation.get("probability", 0.0) or 0.0),
            rotation_max_degrees=float(rotation.get("max_degrees", 0.0) or 0.0),
            crop_probability=float(crop.get("probability", 0.0) or 0.0),
            crop_scale=_as_pair(crop.get("scale", (1.0, 1.0))),
            crop_ratio=_as_pair(crop.get("ratio", (1.0, 1.0))),
            brightness_probability=float(brightness.get("probability", 0.0) or 0.0),
            brightness_delta=float(brightness.get("delta", 0.0) or 0.0),
            contrast_probability=float(contrast.get("probability", 0.0) or 0.0),
            contrast_delta=float(contrast.get("delta", 0.0) or 0.0),
            saturation_probability=float(saturation.get("probability", 0.0) or 0.0),
            saturation_delta=float(saturation.get("delta", 0.0) or 0.0),
        )

    def active_transforms(self) -> list[str]:
        """Names of the transforms that can actually fire. Used in the docs/EDA."""
        if not self.enabled:
            return []
        active = []
        if self.horizontal_flip > 0:
            active.append("horizontal_flip")
        if self.vertical_flip > 0:
            active.append("vertical_flip")
        if self.rotation_probability > 0 and self.rotation_max_degrees > 0:
            active.append("rotation")
        if self.crop_probability > 0:
            active.append("random_resized_crop")
        if self.brightness_probability > 0 and self.brightness_delta > 0:
            active.append("brightness")
        if self.contrast_probability > 0 and self.contrast_delta > 0:
            active.append("contrast")
        if self.saturation_probability > 0 and self.saturation_delta > 0:
            active.append("saturation")
        if self.grayscale > 0:
            active.append("grayscale")
        return active


def _as_pair(value: object) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"expected a pair of numbers, got {value!r}")


def random_resized_crop(
    image: Image.Image,
    size: tuple[int, int],
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    rng: np.random.Generator,
    interpolation: str = "bilinear",
    attempts: int = 10,
) -> Image.Image:
    """Crop a random area/aspect region, then resize it to ``(height, width)``."""
    target_height, target_width = size
    width, height = image.size
    area = width * height
    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))

    for _ in range(attempts):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect = float(np.exp(rng.uniform(*log_ratio)))
        crop_width = int(round(np.sqrt(target_area * aspect)))
        crop_height = int(round(np.sqrt(target_area / aspect)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            region = image.crop((left, top, left + crop_width, top + crop_height))
            return region.resize(
                (target_width, target_height), resample=resample_filter(interpolation)
            )

    # Degenerate aspect ratios: fall back to the largest centred square-ish crop.
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    region = image.crop((left, top, left + side, top + side))
    return region.resize((target_width, target_height), resample=resample_filter(interpolation))


def apply_augmentations(
    image: Image.Image,
    config: AugmentationConfig,
    size: tuple[int, int],
    rng: np.random.Generator,
    *,
    interpolation: str = "bilinear",
) -> Image.Image:
    """Apply the configured augmentations and return an image at ``size``.

    Geometry first, then photometry. The crop (or a final resize when the crop
    does not fire) is what guarantees the output size.
    """
    result = image

    if config.rotation_probability and rng.random() < config.rotation_probability:
        angle = float(rng.uniform(-config.rotation_max_degrees, config.rotation_max_degrees))
        # expand=True keeps the whole rotated frame so the subsequent crop cannot
        # be dominated by the black corners rotation introduces.
        result = result.rotate(
            angle, resample=resample_filter(interpolation), expand=True, fillcolor=(0, 0, 0)
        )

    if config.horizontal_flip and rng.random() < config.horizontal_flip:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if config.vertical_flip and rng.random() < config.vertical_flip:
        result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if config.crop_probability and rng.random() < config.crop_probability:
        result = random_resized_crop(
            result,
            size,
            scale=config.crop_scale,
            ratio=config.crop_ratio,
            rng=rng,
            interpolation=interpolation,
        )

    if config.brightness_probability and rng.random() < config.brightness_probability:
        factor = 1.0 + float(rng.uniform(-config.brightness_delta, config.brightness_delta))
        result = ImageEnhance.Brightness(result).enhance(factor)
    if config.contrast_probability and rng.random() < config.contrast_probability:
        factor = 1.0 + float(rng.uniform(-config.contrast_delta, config.contrast_delta))
        result = ImageEnhance.Contrast(result).enhance(factor)
    if config.saturation_probability and rng.random() < config.saturation_probability:
        factor = 1.0 + float(rng.uniform(-config.saturation_delta, config.saturation_delta))
        result = ImageEnhance.Color(result).enhance(factor)
    if config.grayscale and rng.random() < config.grayscale:
        result = result.convert("L").convert(image.mode)

    return result


@dataclass(frozen=True)
class ImagePipeline:
    """Preprocessing plus, for training only, augmentation.

    ``train=False`` pipelines ignore the augmentation config entirely, which is
    the mechanism that keeps validation and test data un-augmented.
    """

    preprocessor: Preprocessor
    augmentation: AugmentationConfig
    train: bool

    @classmethod
    def from_config(cls, config: DataConfig, *, train: bool) -> "ImagePipeline":
        return cls(
            preprocessor=Preprocessor.from_config(config),
            augmentation=AugmentationConfig.from_mapping(config.augmentation),
            train=train,
        )

    @property
    def augments(self) -> bool:
        return self.train and self.augmentation.enabled

    def transform(
        self, image: Image.Image, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        preprocessing = self.preprocessor.config
        if self.augments:
            if rng is None:
                raise ValueError("a random generator is required when augmenting")
            image = apply_augmentations(
                image,
                self.augmentation,
                preprocessing.image_size,
                rng,
                interpolation=preprocessing.interpolation,
            )
        image = self.preprocessor.prepare_image(image)
        return to_array(
            image,
            scale=preprocessing.scale,
            mean=preprocessing.normalize.mean,
            std=preprocessing.normalize.std,
            dtype=preprocessing.output_dtype,
            channel_order=preprocessing.channel_order,
        )

    def __call__(
        self, source, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        image = source if isinstance(source, Image.Image) else self.preprocessor.load(source)
        return self.transform(image, rng)


def build_train_pipeline(config: DataConfig) -> ImagePipeline:
    return ImagePipeline.from_config(config, train=True)


def build_eval_pipeline(config: DataConfig) -> ImagePipeline:
    """Pipeline for validation and test data: preprocessing only, no augmentation."""
    return ImagePipeline.from_config(config, train=False)
