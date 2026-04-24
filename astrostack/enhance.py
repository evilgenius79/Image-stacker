"""AI enhancement of stacked images.

Primary: Real-ESRGAN (RealESRGAN_x2plus or x4plus) with weights downloaded
from HuggingFace on first run. Tiled inference keeps memory bounded.

Fallback (no torch / no model / model fails): scikit-image wavelet denoise +
unsharp mask. Always available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .device import select_device

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path.home() / ".cache" / "astrostack" / "weights"

# (HuggingFace repo, filename, scale, model factory key)
MODEL_REGISTRY = {
    "realesrgan-x2": (
        "ai-forever/Real-ESRGAN",
        "RealESRGAN_x2.pth",
        2,
    ),
    "realesrgan-x4": (
        "ai-forever/Real-ESRGAN",
        "RealESRGAN_x4.pth",
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


def _download_weights(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s/%s ...", repo_id, filename)
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(WEIGHTS_DIR))
    return Path(path)


def _load_realesrgan(model_key: str, device: str):
    """Load Real-ESRGAN, returning (upsampler, scale) or raising."""
    repo_id, filename, scale = MODEL_REGISTRY[model_key]
    weights = _download_weights(repo_id, filename)

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale,
    )
    half = (device == "cuda")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weights),
        model=model,
        tile=512,
        tile_pad=16,
        pre_pad=0,
        half=half,
        device=torch.device(device),
    )
    return upsampler, scale


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
        upsampler, scale = _load_realesrgan(model, dev)
        bgr = (img_in[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
        out = out_bgr[..., ::-1].astype(np.float32) / 255.0
        if not is_color:
            out = out.mean(axis=-1)
        return np.clip(out, 0.0, 1.0)
    except Exception as e:
        if not fallback:
            raise
        log.warning("AI enhancement failed (%s); using classical fallback.", e)
        return _classical_enhance(img)
