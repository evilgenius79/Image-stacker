"""AI enhancement of stacked images.

Primary backend: ``spandrel`` (pure-Python model loader from chaiNNer) loads
Real-ESRGAN weights downloaded from the official xinntao/Real-ESRGAN GitHub
releases. We do tiled inference ourselves to keep memory bounded.

Fallback (no torch / no model / model fails): scikit-image wavelet denoise
+ unsharp mask. Always available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .device import select_device

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path.home() / ".cache" / "astrostack" / "weights"

# (download_url, cached_filename, scale)
MODEL_REGISTRY = {
    "realesrgan-x2": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x2plus.pth",
        2,
    ),
    "realesrgan-x4": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRGAN_x4plus.pth",
        4,
    ),
}


def _classical_enhance(img: np.ndarray, denoise_strength: float = 0.05) -> np.ndarray:
    """CPU-only fallback: wavelet denoise + unsharp mask."""
    from skimage.restoration import denoise_wavelet
    from skimage.filters import unsharp_mask

    is_color = img.ndim == 3
    work = img.astype(np.float32)
    work = denoise_wavelet(
        work,
        sigma=denoise_strength,
        channel_axis=-1 if is_color else None,
        rescale_sigma=True,
    )
    work = unsharp_mask(
        work,
        radius=1.5,
        amount=1.0,
        channel_axis=-1 if is_color else None,
    )
    return np.clip(work.astype(np.float32), 0.0, 1.0)


def _download_weights(url: str, filename: str) -> Path:
    import torch
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / filename
    if not dest.exists():
        log.info("Downloading %s -> %s", url, dest)
        torch.hub.download_url_to_file(url, str(dest), progress=True)
    return dest


def _pick_tile(img_shape, device: str) -> int:
    """Choose a sensible tile size. 0 disables tiling."""
    h, w = img_shape[:2]
    if device == "cpu":
        return 256 if max(h, w) > 512 else 0
    if max(h, w) <= 1024:
        return 0
    if max(h, w) <= 2048:
        return 512
    return 400


_MODEL_CACHE: dict = {}


def _load_model(weights_path: Path, device: str):
    """Load weights via spandrel; returns a callable wrapper on (B,C,H,W) tensors.

    Caches by (weights_path, device) so back-to-back stacks reuse the loaded
    descriptor instead of paying disk + deserialize cost each time.
    """
    cache_key = (str(weights_path), device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import torch
    from spandrel import ModelLoader

    loader = ModelLoader(device=torch.device(device))
    descriptor = loader.load_from_file(str(weights_path))
    descriptor.eval()
    if device == "cuda":
        descriptor.model.half()

    _MODEL_CACHE[cache_key] = descriptor
    return descriptor


def _to_dtype(t, half: bool):
    return t.half() if half else t.float()


def _tiled_infer(descriptor, img_chw: "np.ndarray", scale: int,
                 tile: int, pad: int, device: str) -> "np.ndarray":
    """Run inference, tiling spatially when ``tile > 0``.

    img_chw: (3, H, W) float32 in [0, 1]. Returns (3, H*scale, W*scale).
    """
    import torch

    half = (device == "cuda")
    dev = torch.device(device)
    _, h, w = img_chw.shape

    if tile <= 0:
        with torch.no_grad():
            t = torch.from_numpy(img_chw).unsqueeze(0).to(dev)
            t = _to_dtype(t, half)
            out = descriptor(t)
            return out.clamp(0, 1).float().squeeze(0).cpu().numpy()

    out_h, out_w = h * scale, w * scale
    out = np.zeros((3, out_h, out_w), dtype=np.float32)

    n_y = (h + tile - 1) // tile
    n_x = (w + tile - 1) // tile
    for iy in range(n_y):
        for ix in range(n_x):
            y0 = iy * tile
            x0 = ix * tile
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)

            # padded region (input space)
            py0 = max(y0 - pad, 0)
            px0 = max(x0 - pad, 0)
            py1 = min(y1 + pad, h)
            px1 = min(x1 + pad, w)

            patch = img_chw[:, py0:py1, px0:px1]
            with torch.no_grad():
                t = torch.from_numpy(patch).unsqueeze(0).to(dev)
                t = _to_dtype(t, half)
                up = descriptor(t).clamp(0, 1).float().squeeze(0).cpu().numpy()

            # crop the padded region back to the requested tile (in output space)
            cy0 = (y0 - py0) * scale
            cx0 = (x0 - px0) * scale
            cy1 = cy0 + (y1 - y0) * scale
            cx1 = cx0 + (x1 - x0) * scale
            out[:, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                up[:, cy0:cy1, cx0:cx1]
    return out


def enhance(
    img: np.ndarray,
    model: str = "realesrgan-x2",
    device: str = "auto",
    fallback: bool = True,
) -> np.ndarray:
    """Run AI enhancement. Returns float32 [0, 1]; spatial dims may be upscaled."""
    dev = select_device(device)
    log.info("Enhance device: %s", dev)

    is_color = img.ndim == 3
    if not is_color:
        img_in = np.stack([img] * 3, axis=-1)
    else:
        img_in = img

    try:
        url, filename, scale = MODEL_REGISTRY[model]
        weights_path = _download_weights(url, filename)
        descriptor = _load_model(weights_path, dev)

        # spandrel tells us the actual scale; trust it over our registry hint.
        actual_scale = getattr(descriptor, "scale", scale) or scale

        chw = np.transpose(img_in.astype(np.float32), (2, 0, 1))
        tile = _pick_tile(img_in.shape, dev)
        log.info("Real-ESRGAN: scale=%d, device=%s, tile=%d",
                 actual_scale, dev, tile)
        out_chw = _tiled_infer(descriptor, chw, actual_scale, tile,
                               pad=16, device=dev)
        out = np.transpose(out_chw, (1, 2, 0))

        if not is_color:
            out = out.mean(axis=-1)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        if not fallback:
            raise
        log.warning("AI enhancement failed (%s); using classical fallback.", e)
        return _classical_enhance(img)
