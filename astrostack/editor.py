"""Post-processing image adjustments for the stacked result.

All operations take and return float32 in [0, 1]. Color images are (H, W, 3),
mono images are (H, W). Adjustments are applied in this order:

    levels (black/white point) -> asinh stretch -> gamma ->
    brightness -> contrast -> saturation -> sharpen
"""

from __future__ import annotations

import numpy as np


def _is_color(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3


def apply_levels(img: np.ndarray, black: float, white: float) -> np.ndarray:
    if black <= 0.0 and white >= 1.0:
        return img
    white = max(white, black + 1e-4)
    return np.clip((img - black) / (white - black), 0.0, 1.0)


def apply_asinh(img: np.ndarray, strength: float) -> np.ndarray:
    """Astro-friendly nonlinear stretch.

    strength in [0, 1]: 0 = identity, 1 = strong faint-detail boost.
    """
    if strength <= 0.0:
        return img
    a = 10.0 ** (strength * 3.0)  # 1 .. 1000
    return np.arcsinh(img * a) / np.arcsinh(a)


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-4:
        return img
    return np.clip(img, 0.0, 1.0) ** (1.0 / max(gamma, 1e-3))


def apply_brightness_contrast(
    img: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    out = img
    if abs(contrast - 1.0) >= 1e-4:
        out = (out - 0.5) * contrast + 0.5
    if abs(brightness) >= 1e-4:
        out = out + brightness
    return np.clip(out, 0.0, 1.0)


def apply_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    if not _is_color(img) or abs(saturation - 1.0) < 1e-4:
        return img
    luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])
    luma = luma[..., None]
    return np.clip(luma + (img - luma) * saturation, 0.0, 1.0)


def apply_sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img
    from skimage.filters import unsharp_mask
    channel_axis = -1 if _is_color(img) else None
    out = unsharp_mask(
        img.astype(np.float32),
        radius=1.5,
        amount=float(amount),
        channel_axis=channel_axis,
    )
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def adjust(
    img: np.ndarray,
    *,
    black_point: float = 0.0,
    white_point: float = 1.0,
    asinh: float = 0.0,
    gamma: float = 1.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpen: float = 0.0,
) -> np.ndarray:
    """Apply the full adjustment chain."""
    out = img.astype(np.float32, copy=False)
    out = apply_levels(out, black_point, white_point)
    out = apply_asinh(out, asinh)
    out = apply_gamma(out, gamma)
    out = apply_brightness_contrast(out, brightness, contrast)
    out = apply_saturation(out, saturation)
    out = apply_sharpen(out, sharpen)
    return out


DEFAULTS = dict(
    black_point=0.0,
    white_point=1.0,
    asinh=0.0,
    gamma=1.0,
    brightness=0.0,
    contrast=1.0,
    saturation=1.0,
    sharpen=0.0,
)
