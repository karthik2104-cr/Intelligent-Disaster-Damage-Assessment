"""Report-ready evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from ml.eval.patterns import raw_confusion, row_normalized_confusion
from ml.eval.predict import PredictionBundle


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrices(bundle: PredictionBundle, stem: Path) -> dict[str, str]:
    names = [n.replace("_", "\n") for n in bundle.class_names]
    raw = raw_confusion(bundle)
    norm = row_normalized_confusion(bundle)
    paths = {}

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(raw, cmap="Blues")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (counts) — {stem.name}")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            color = "white" if raw[i, j] > raw.max() / 2 else "black"
            ax.text(j, i, str(int(raw[i, j])), ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    paths["raw"] = str(_save(fig, stem.parent / f"{stem.name}_cm_counts.png"))

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized) — {stem.name}")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]*100:.1f}%", ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    paths["normalized"] = str(_save(fig, stem.parent / f"{stem.name}_cm_normalized.png"))
    return paths


def plot_confidence(bundle: PredictionBundle, destination: Path, title: str) -> Path:
    conf = bundle.confidence
    correct = bundle.y_true == bundle.y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(conf[correct], bins=20, alpha=0.75, label="correct", color="#2a6f97")
    axes[0].hist(conf[~correct], bins=20, alpha=0.75, label="incorrect", color="#c1121f")
    axes[0].set_xlabel("softmax confidence (max class)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Confidence: correct vs incorrect")
    axes[0].legend()
    for idx, name in enumerate(bundle.class_names):
        mask = bundle.y_true == idx
        axes[1].hist(conf[mask], bins=15, alpha=0.45, label=name)
    axes[1].set_xlabel("softmax confidence")
    axes[1].set_ylabel("count")
    axes[1].set_title("Confidence by true class")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, destination)


def plot_example_grid(
    images: list[np.ndarray],
    captions: list[str],
    destination: Path,
    *,
    title: str,
    ncols: int = 3,
) -> Path:
    n = len(images)
    if n == 0:
        raise ValueError("no images to plot")
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, image, caption in zip(axes, images, captions):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(caption, fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, destination)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_resized = _resize_cam(cam, image.shape[:2])
    cmap = plt.get_cmap("jet")
    heat = cmap(Normalize(vmin=0, vmax=1)(cam_resized))[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def _resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    height, width = size
    img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
