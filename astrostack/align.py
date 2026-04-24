"""Star-pattern image alignment.

Uses astroalign for affine registration based on detected source triangles.
Falls back to phase-correlation translation if astroalign fails.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _to_mono(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    # Luminance for source detection only
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)


def _phase_translation(ref_mono: np.ndarray, mov_mono: np.ndarray) -> tuple[float, float]:
    from skimage.registration import phase_cross_correlation
    shift, _, _ = phase_cross_correlation(ref_mono, mov_mono, upsample_factor=10)
    return float(shift[0]), float(shift[1])


def _apply_translation(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    from scipy.ndimage import shift as ndi_shift
    if img.ndim == 2:
        return ndi_shift(img, (dy, dx), order=1, mode="constant", cval=0.0)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = ndi_shift(img[..., c], (dy, dx), order=1,
                                mode="constant", cval=0.0)
    return out


def align_to_reference(
    reference: np.ndarray,
    target: np.ndarray,
    detection_sigma: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Align `target` to `reference`. Returns (aligned, info)."""
    import astroalign as aa

    ref_mono = _to_mono(reference)
    mov_mono = _to_mono(target)

    try:
        if target.ndim == 3:
            # Estimate transform on mono, then apply per channel.
            transf, _ = aa.find_transform(
                mov_mono, ref_mono, detection_sigma=detection_sigma
            )
            aligned = np.empty_like(target)
            for c in range(target.shape[2]):
                aligned[..., c], _ = aa.apply_transform(
                    transf, target[..., c], reference[..., c]
                )
            return np.clip(aligned, 0.0, 1.0), {"method": "astroalign", "transform": transf.params}
        else:
            aligned, _ = aa.register(target, reference, detection_sigma=detection_sigma)
            return np.clip(aligned.astype(np.float32), 0.0, 1.0), {"method": "astroalign"}
    except Exception as e:
        log.warning("astroalign failed (%s); falling back to phase correlation.", e)
        dy, dx = _phase_translation(ref_mono, mov_mono)
        aligned = _apply_translation(target, dy, dx)
        return np.clip(aligned, 0.0, 1.0), {"method": "phase", "shift": (dy, dx)}
