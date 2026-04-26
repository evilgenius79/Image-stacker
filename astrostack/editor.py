"""Post-processing image adjustments for the stacked result.

All operations take and return float32 in [0, 1]. Color images are (H, W, 3),
mono images are (H, W).

Adjustment chain order:
    per-channel levels (black/white) -> asinh stretch -> gamma ->
    brightness -> contrast -> saturation -> star reduction -> sharpen -> crop

Background neutralization is computed on demand by ``compute_background_offsets``
and applied through the per-channel black points.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_color(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3


def _per_channel(img: np.ndarray, vals: list[float] | tuple[float, ...] | float):
    """Broadcast a scalar or per-channel list/tuple to (1, 1, C) for color or scalar for mono."""
    if not _is_color(img):
        if hasattr(vals, "__len__"):
            return float(np.mean(vals))
        return float(vals)
    if hasattr(vals, "__len__"):
        arr = np.asarray(vals[: img.shape[2]], dtype=np.float32)
    else:
        arr = np.full(img.shape[2], float(vals), dtype=np.float32)
    return arr.reshape(1, 1, -1)


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------

def apply_levels(
    img: np.ndarray,
    black: float = 0.0,
    white: float = 1.0,
    black_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    white_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply global + per-channel black/white levels.

    Per-channel offsets are added on top of the global value, then clamped.
    """
    if _is_color(img):
        c = img.shape[2]
        b = np.clip(black + np.asarray(black_rgb[:c], dtype=np.float32), 0.0, 1.0)
        w = np.clip(white + np.asarray(white_rgb[:c], dtype=np.float32), 0.0, 1.0)
        b = b.reshape(1, 1, -1)
        w = np.maximum(w.reshape(1, 1, -1), b + 1e-4)
    else:
        b = max(min(black, 1.0), 0.0)
        w = max(white, b + 1e-4)
    return np.clip((img - b) / (w - b), 0.0, 1.0)


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


def apply_star_reduction(img: np.ndarray, amount: float, threshold: float = 0.6) -> np.ndarray:
    """Shrink small bright objects (stars) while preserving extended structure.

    amount in [0, 1]: blend strength toward an eroded version.
    """
    if amount <= 0.0:
        return img
    from scipy.ndimage import grey_erosion, gaussian_filter

    if _is_color(img):
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    else:
        luma = img

    mask = (luma > threshold).astype(np.float32)
    mask = gaussian_filter(mask, sigma=1.0)
    mask = np.clip(mask, 0.0, 1.0) * float(amount)

    if _is_color(img):
        eroded = np.empty_like(img)
        for c in range(img.shape[2]):
            eroded[..., c] = grey_erosion(img[..., c], size=2)
        m = mask[..., None]
    else:
        eroded = grey_erosion(img, size=2)
        m = mask
    return np.clip(img * (1.0 - m) + eroded * m, 0.0, 1.0)


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


def apply_crop(
    img: np.ndarray,
    left_pct: float = 0.0,
    top_pct: float = 0.0,
    right_pct: float = 0.0,
    bottom_pct: float = 0.0,
) -> np.ndarray:
    """Crop margins as percentages of the source dimensions."""
    if all(v <= 0.0 for v in (left_pct, top_pct, right_pct, bottom_pct)):
        return img
    h, w = img.shape[:2]
    x0 = int(round(left_pct / 100.0 * w))
    y0 = int(round(top_pct / 100.0 * h))
    x1 = w - int(round(right_pct / 100.0 * w))
    y1 = h - int(round(bottom_pct / 100.0 * h))
    x1 = max(x1, x0 + 1)
    y1 = max(y1, y0 + 1)
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Background neutralization
# ---------------------------------------------------------------------------

def compute_background_offsets(
    img: np.ndarray, percentile: float = 1.0
) -> tuple[float, float, float]:
    """Return per-channel offsets that, subtracted from black point,
    neutralize background to a common dark grey.

    For mono returns (v, v, v).
    """
    if _is_color(img):
        c = img.shape[2]
        offsets = []
        for ch in range(min(c, 3)):
            offsets.append(float(np.percentile(img[..., ch], percentile)))
        # Pad to 3
        while len(offsets) < 3:
            offsets.append(offsets[-1])
        # Subtract the lightest of the three so we keep at least one channel
        # at 0 offset; this avoids globally darkening the image.
        baseline = min(offsets)
        return tuple(o - baseline for o in offsets[:3])  # type: ignore[return-value]
    v = float(np.percentile(img, percentile))
    return (0.0, 0.0, 0.0)  # nothing to neutralize on mono


# ---------------------------------------------------------------------------
# Master adjustment chain
# ---------------------------------------------------------------------------

DEFAULTS: dict = dict(
    black_point=0.0,
    white_point=1.0,
    black_r=0.0, black_g=0.0, black_b=0.0,
    white_r=0.0, white_g=0.0, white_b=0.0,
    asinh=0.0,
    gamma=1.0,
    brightness=0.0,
    contrast=1.0,
    saturation=1.0,
    star_reduce=0.0,
    sharpen=0.0,
    crop_left=0.0, crop_top=0.0, crop_right=0.0, crop_bottom=0.0,
)


def adjust(
    img: np.ndarray,
    *,
    black_point: float = 0.0,
    white_point: float = 1.0,
    black_r: float = 0.0, black_g: float = 0.0, black_b: float = 0.0,
    white_r: float = 0.0, white_g: float = 0.0, white_b: float = 0.0,
    asinh: float = 0.0,
    gamma: float = 1.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    star_reduce: float = 0.0,
    sharpen: float = 0.0,
    crop_left: float = 0.0, crop_top: float = 0.0,
    crop_right: float = 0.0, crop_bottom: float = 0.0,
) -> np.ndarray:
    """Apply the full adjustment chain. See module docstring for order."""
    out = img.astype(np.float32, copy=False)
    out = apply_levels(
        out, black_point, white_point,
        black_rgb=(black_r, black_g, black_b),
        white_rgb=(white_r, white_g, white_b),
    )
    out = apply_asinh(out, asinh)
    out = apply_gamma(out, gamma)
    out = apply_brightness_contrast(out, brightness, contrast)
    out = apply_saturation(out, saturation)
    out = apply_star_reduction(out, star_reduce)
    out = apply_sharpen(out, sharpen)
    out = apply_crop(out, crop_left, crop_top, crop_right, crop_bottom)
    return out


# ---------------------------------------------------------------------------
# Auto-stretch and presets
# ---------------------------------------------------------------------------

def auto_stretch(img: np.ndarray, low: float = 0.5, high: float = 99.7) -> dict:
    """Compute black/white points from histogram percentiles.

    Returns a dict slice of DEFAULTS suitable for merging.
    """
    flat = img.reshape(-1) if not _is_color(img) else (
        0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    ).reshape(-1)
    b = float(np.percentile(flat, low))
    w = float(np.percentile(flat, high))
    if w <= b:
        w = min(b + 0.05, 1.0)
    return {"black_point": b, "white_point": w, "asinh": 0.4, "gamma": 1.0}


PRESETS: dict[str, dict] = {
    "None": {},
    "Gentle": {"black_point": 0.02, "white_point": 0.95,
               "asinh": 0.25, "gamma": 1.1, "saturation": 1.1},
    "Standard": {"black_point": 0.05, "white_point": 0.92,
                 "asinh": 0.45, "gamma": 1.2, "saturation": 1.25,
                 "contrast": 1.1},
    "Aggressive": {"black_point": 0.08, "white_point": 0.88,
                   "asinh": 0.65, "gamma": 1.3, "saturation": 1.4,
                   "contrast": 1.2, "sharpen": 0.5},
    "Extreme": {"black_point": 0.12, "white_point": 0.85,
                "asinh": 0.85, "gamma": 1.4, "saturation": 1.6,
                "contrast": 1.3, "sharpen": 0.8, "star_reduce": 0.2},
}


def apply_preset(name: str) -> dict:
    """Return a full settings dict (DEFAULTS + preset overrides)."""
    out = dict(DEFAULTS)
    out.update(PRESETS.get(name, {}))
    return out
