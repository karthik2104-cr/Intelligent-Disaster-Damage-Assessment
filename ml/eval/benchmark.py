"""Small inference-time benchmark. No training."""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader


def benchmark_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int = 8,
    timed_batches: int = 20,
) -> dict:
    model.eval()
    batches = []
    for i, (images, _) in enumerate(loader):
        batches.append(images)
        if i + 1 >= warmup_batches + timed_batches:
            break
    if len(batches) < 2:
        raise ValueError("not enough batches to benchmark")
    warm, timed = batches[:warmup_batches], batches[warmup_batches:]
    if not timed:
        timed = warm[-1:]
        warm = warm[:-1] or timed

    with torch.no_grad():
        for images in warm:
            images = images.to(device, non_blocking=True)
            _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        n_images = 0
        for images in timed:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            n_images += int(images.shape[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return {
        "device": str(device),
        "batch_size": int(timed[0].shape[0]),
        "timed_batches": len(timed),
        "timed_images": n_images,
        "seconds": elapsed,
        "ms_per_image": (elapsed / n_images) * 1000.0 if n_images else None,
        "images_per_second": n_images / elapsed if elapsed else None,
    }
