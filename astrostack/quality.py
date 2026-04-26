"""Frame quality analysis: star count and FWHM estimation.

Uses photutils' DAOStarFinder to detect stars, then fits a 2D Gaussian to
a small sample to estimate seeing (FWHM in pixels). All values are robust
to mono or RGB input — color frames are converted to luminance first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import load_image

log = logging.getLogger(__name__)


@dataclass
class FrameStats:
    name: str
    width: int
    height: int
    n_stars: int
    fwhm_px: float
    median_bg: float
    score: float  # higher = better


def _to_luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)


def _estimate_fwhm(luma: np.ndarray, sources, n_sample: int = 12) -> float:
    """Fit a 2D Gaussian to up to n_sample brightest non-saturated stars."""
    if sources is None or len(sources) == 0:
        return float("nan")
    try:
        from astropy.modeling import models, fitting
    except Exception:
        return float("nan")

    # Pick brightest sources but skip near-saturated peaks.
    flux_col = "flux" if "flux" in sources.colnames else "peak"
    src = sources.copy()
    src.sort(flux_col)
    src.reverse()

    fwhms = []
    fitter = fitting.LevMarLSQFitter()
    half = 6
    for row in src[:max(n_sample * 3, n_sample)]:
        x = int(round(float(row["xcentroid"])))
        y = int(round(float(row["ycentroid"])))
        if (x - half < 0 or y - half < 0
                or x + half >= luma.shape[1] or y + half >= luma.shape[0]):
            continue
        patch = luma[y - half:y + half + 1, x - half:x + half + 1]
        peak = float(patch.max())
        if peak >= 0.99 or peak < 0.05:
            continue
        yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
        g_init = models.Gaussian2D(
            amplitude=peak, x_mean=half, y_mean=half,
            x_stddev=2.0, y_stddev=2.0,
        )
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fitter(g_init, xx, yy, patch)
        except Exception:
            continue
        sx = abs(float(fit.x_stddev.value))
        sy = abs(float(fit.y_stddev.value))
        if 0.4 < sx < 8 and 0.4 < sy < 8:
            fwhms.append(2.355 * 0.5 * (sx + sy))
        if len(fwhms) >= n_sample:
            break
    if not fwhms:
        return float("nan")
    return float(np.median(fwhms))


def analyze_frame(
    img: np.ndarray,
    name: str,
    detection_sigma: float = 5.0,
) -> FrameStats:
    """Run star detection + FWHM estimation on a single frame."""
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    luma = _to_luma(img)
    h, w = luma.shape

    mean, median, std = sigma_clipped_stats(luma, sigma=3.0, maxiters=3)
    finder = DAOStarFinder(fwhm=3.0, threshold=detection_sigma * std)
    sources = finder(luma - median)

    n = 0 if sources is None else len(sources)
    fwhm = _estimate_fwhm(luma, sources)

    # Score: more stars and tighter (lower) FWHM = better.
    if n == 0 or not np.isfinite(fwhm):
        score = 0.0
    else:
        score = float(n) / max(fwhm, 1.0)

    return FrameStats(
        name=name, width=w, height=h,
        n_stars=int(n), fwhm_px=float(fwhm) if np.isfinite(fwhm) else float("nan"),
        median_bg=float(median), score=score,
    )


def analyze_paths(paths: list[Path]) -> list[FrameStats]:
    """Run analyze_frame on each file. Returns list ordered as input."""
    out: list[FrameStats] = []
    for p in paths:
        try:
            img = load_image(p)
            out.append(analyze_frame(img, p.name))
        except Exception as e:
            log.warning("Quality analysis failed for %s: %s", p.name, e)
            out.append(FrameStats(
                name=p.name, width=0, height=0, n_stars=0,
                fwhm_px=float("nan"), median_bg=0.0, score=0.0,
            ))
    return out


def stats_to_table(stats: list[FrameStats]) -> list[list]:
    """Format stats as a list-of-lists for gr.Dataframe."""
    rows = []
    for s in stats:
        rows.append([
            s.name, s.width, s.height,
            s.n_stars,
            f"{s.fwhm_px:.2f}" if np.isfinite(s.fwhm_px) else "—",
            f"{s.median_bg:.4f}",
            f"{s.score:.2f}",
        ])
    return rows


TABLE_HEADERS = ["File", "W", "H", "Stars", "FWHM (px)", "Background", "Score"]
