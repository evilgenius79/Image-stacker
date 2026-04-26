"""End-to-end pipeline: load -> calibrate -> align -> stack -> enhance -> save."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tqdm import tqdm

from .align import align_to_reference
from .calibrate import build_masters, calibrate_frame
from .enhance import enhance as enhance_image
from .io import load_image, save_image
from .stack import stack as stack_frames

log = logging.getLogger(__name__)


def _ensure_same_shape(ref: np.ndarray, arr: np.ndarray) -> np.ndarray:
    if arr.shape == ref.shape:
        return arr
    ref_hw = ref.shape[:2]
    arr_hw = arr.shape[:2]
    if ref_hw != arr_hw:
        raise ValueError(
            f"Frame resolution {arr_hw} does not match reference {ref_hw}. "
            "All light frames must share the same dimensions."
        )
    if arr.ndim == 2 and ref.ndim == 3:
        return np.stack([arr] * ref.shape[2], axis=-1)
    if arr.ndim == 3 and ref.ndim == 2:
        return arr.mean(axis=-1)
    raise ValueError(f"Frame shape {arr.shape} incompatible with reference {ref.shape}")


def run_pipeline(
    light_paths: Sequence[Path],
    output: Path,
    *,
    darks: Iterable[Path] | None = None,
    bias: Iterable[Path] | None = None,
    flats: Iterable[Path] | None = None,
    stack_method: str = "sigma",
    sigma: float = 3.0,
    sigma_iters: int = 3,
    align: bool = True,
    enhance: bool = True,
    enhance_model: str = "realesrgan-x2",
    device: str = "auto",
    bit_depth: int = 16,
    save_intermediate: bool = False,
    return_arrays: bool = False,
):
    """Run the full pipeline.

    Returns the output Path by default. If ``return_arrays`` is True,
    returns a dict ``{"output": Path, "stacked": ndarray, "enhanced": ndarray|None}``
    with both the pre-enhancement stack and the final enhanced result so the
    caller can preview before/after.
    """
    if len(light_paths) < 1:
        raise ValueError("Need at least one light frame.")

    log.info("Building calibration masters ...")
    masters = build_masters(darks, bias, flats)

    log.info("Loading + calibrating %d light frames ...", len(light_paths))
    frames: list[np.ndarray] = []
    for p in tqdm(light_paths, desc="load"):
        img = load_image(p)
        img = calibrate_frame(img, masters["dark"], masters["bias"], masters["flat"])
        frames.append(img)

    reference = frames[0]
    aligned: list[np.ndarray] = [reference]

    if align and len(frames) > 1:
        log.info("Aligning frames to reference (%s) ...", light_paths[0].name)
        for i, frame in enumerate(tqdm(frames[1:], desc="align"), start=1):
            try:
                frame = _ensure_same_shape(reference, frame)
                aligned_frame, info = align_to_reference(reference, frame)
                log.debug("Frame %d aligned via %s", i, info.get("method"))
                aligned.append(aligned_frame)
            except Exception as e:
                log.warning("Skipping frame %d (%s): alignment failed: %s",
                            i, light_paths[i].name, e)
    else:
        aligned = [_ensure_same_shape(reference, f) for f in frames]

    log.info("Stacking %d frames with %s ...", len(aligned), stack_method)
    cube = np.stack(aligned, axis=0)
    if stack_method == "sigma":
        stacked = stack_frames(cube, method="sigma", sigma=sigma, iters=sigma_iters)
    else:
        stacked = stack_frames(cube, method=stack_method)

    if save_intermediate:
        intermediate = output.with_name(output.stem + "_stacked" + output.suffix)
        save_image(intermediate, stacked, bit_depth=bit_depth)
        log.info("Wrote pre-enhancement stack: %s", intermediate)

    enhanced = None
    if enhance:
        log.info("Running AI enhancement (%s) ...", enhance_model)
        enhanced = enhance_image(stacked, model=enhance_model, device=device)
        result = enhanced
    else:
        result = stacked

    save_image(output, result, bit_depth=bit_depth)
    log.info("Wrote: %s", output)

    if return_arrays:
        return {"output": output, "stacked": stacked, "enhanced": enhanced}
    return output
