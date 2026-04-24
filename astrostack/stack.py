"""Stacking algorithms."""

from __future__ import annotations

import numpy as np


def stack_mean(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0).astype(np.float32)


def stack_median(frames: np.ndarray) -> np.ndarray:
    return np.median(frames, axis=0).astype(np.float32)


def stack_sigma_clip(
    frames: np.ndarray,
    sigma: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """Sigma-clipped mean stack along axis 0.

    frames: (N, H, W) or (N, H, W, C), float32 in [0, 1].
    """
    data = frames.astype(np.float32, copy=True)
    mask = np.ones_like(data, dtype=bool)
    for _ in range(iters):
        masked = np.where(mask, data, np.nan)
        mean = np.nanmean(masked, axis=0, keepdims=True)
        std = np.nanstd(masked, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1e-6, std)
        deviation = np.abs(data - mean)
        new_mask = deviation <= sigma * std
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    masked = np.where(mask, data, np.nan)
    out = np.nanmean(masked, axis=0)
    # Replace any remaining NaN (pixels rejected in every frame) with median.
    if np.isnan(out).any():
        med = np.nanmedian(data, axis=0)
        out = np.where(np.isnan(out), med, out)
    return out.astype(np.float32)


METHODS = {
    "mean": stack_mean,
    "median": stack_median,
    "sigma": stack_sigma_clip,
}


def stack(frames: np.ndarray, method: str = "sigma", **kwargs) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"Unknown stack method: {method}. Choose from {list(METHODS)}")
    fn = METHODS[method]
    if method == "sigma":
        return fn(frames, **kwargs)
    return fn(frames)
