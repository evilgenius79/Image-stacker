"""Optional calibration: dark / bias subtraction and flat-field division."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .io import load_image
from .stack import stack_median


def _load_master(paths: Iterable[Path] | None) -> np.ndarray | None:
    if not paths:
        return None
    arrays = [load_image(p) for p in paths]
    if not arrays:
        return None
    if len(arrays) == 1:
        return arrays[0]
    return stack_median(np.stack(arrays, axis=0))


def calibrate_frame(
    light: np.ndarray,
    master_dark: np.ndarray | None = None,
    master_bias: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
) -> np.ndarray:
    out = light.astype(np.float32, copy=True)
    if master_bias is not None:
        out = out - master_bias
    if master_dark is not None:
        out = out - master_dark
    if master_flat is not None:
        flat = master_flat.astype(np.float32)
        flat_norm = flat / max(float(flat.mean()), 1e-9)
        out = out / np.where(flat_norm < 1e-3, 1e-3, flat_norm)
    return np.clip(out, 0.0, 1.0)


def build_masters(
    darks: Iterable[Path] | None,
    bias: Iterable[Path] | None,
    flats: Iterable[Path] | None,
) -> dict:
    return {
        "dark": _load_master(darks),
        "bias": _load_master(bias),
        "flat": _load_master(flats),
    }
