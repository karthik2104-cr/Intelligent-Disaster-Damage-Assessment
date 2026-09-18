"""Learning-curve plots from a real training history."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_history(history: Sequence[dict], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train acc")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="val acc")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="val macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Accuracy / macro-F1")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination
