"""Help content rendered in the web UI Guide tab."""

from __future__ import annotations

QUICK_HELP_STACK = """\
**Quick help — stacking**

1. Drop **light frames** (your sky images) into the top uploader. 5–50+ is typical.
2. (Optional) expand **Calibration frames** and add matching darks, flats, bias.
3. Leave defaults (`sigma` method, `3.0`, align ON, Real-ESRGAN x2 ON) unless you know better.
4. Click **Stack & Enhance**. First run downloads the model (~65 MB).
5. Preview + download appear on the right. Click **Open in Editor →** to post-process.

See the **Guide** tab for a full walkthrough.
"""

QUICK_HELP_EDITOR = """\
**Quick help — editor**

- Sliders apply to a downsampled preview for speed. **Export full-resolution** re-renders at full size.
- **Asinh stretch** is the astro-magic slider — reveals faint nebulosity.
- **Black/White point** clips and rescales the histogram; use together to boost contrast on faint data.
- **Reset adjustments** restores identity.

See the **Guide** tab for what each control does.
"""

GUIDE_MARKDOWN = r"""
# astrostack — User Guide

A local pipeline for stacking and AI-enhancing astrophotography. No cloud, no
upload to strangers — everything runs on your machine.

---

## 1. What the pipeline does

For each run:

1. **Load** every input as float32 in `[0, 1]`. RAW files are demosaiced with
   camera white balance and linear gamma so the stacker sees linear light.
2. **Calibrate** (optional): subtract master bias, subtract master dark,
   divide by a normalised master flat. Any of the three can be omitted.
3. **Align**: the first frame is the reference. Each other frame is matched
   to it using detected star triangles (`astroalign`). If star detection
   fails (cloudy, under-exposed, too few stars), it falls back to sub-pixel
   phase correlation (pure translation).
4. **Stack**: combine aligned frames pixel-by-pixel. Default is
   **sigma-clipped mean** — it throws away satellite trails, plane lights,
   cosmic ray hits, and hot pixels that only appear in a minority of frames.
5. **Enhance** (optional): Real-ESRGAN runs a deep network over the stacked
   result for denoise + detail recovery + upscale. The model weights are
   downloaded on first run into `~/.cache/astrostack/weights/`.
6. **Save** to the format chosen by the output extension (TIFF, PNG, JPG,
   FITS).

If torch is missing or the AI model fails for any reason, a classical
wavelet-denoise + unsharp-mask step runs instead so you always get a result.

---

## 2. Supported input formats

| Category | Extensions |
| --- | --- |
| Camera RAW | `.cr2`, `.cr3`, `.nef`, `.arw`, `.dng`, `.raf`, `.orf`, `.rw2`, `.pef`, `.srw`, `.kdc`, `.3fr` |
| Standard | `.jpg`/`.jpeg`, `.png`, `.bmp`, `.webp` |
| High bit depth | `.tif`/`.tiff` (8/16-bit mono or RGB) |
| Astronomy | `.fits`, `.fit`, `.fts` |

**All frames in one stack must share the same pixel dimensions.** Mixing
mono and color is OK — mono gets promoted to RGB. Mixing resolutions is not.

---

## 3. Tab 1 — Stack & Enhance

### Light frames
The images of your target. Quality beats quantity, but more frames = lower
noise (signal-to-noise grows with √N).

### Calibration frames (optional)
- **Darks**: same exposure/ISO/temperature as lights, lens cap on. Removes
  thermal noise + hot pixels.
- **Flats**: even illumination (dawn sky, white screen). Corrects vignetting
  and dust shadows.
- **Bias**: shortest possible exposure, lens cap on. Removes readout
  pattern. If you only have one of these, flats give the biggest visible
  improvement.

Multiple files of each kind are automatically median-combined into a master.

### Stacking panel
- **Algorithm**
  - `sigma` (recommended) — sigma-clipped mean. Best rejection of outliers
    (planes, satellites, cosmic rays) while keeping most of the SNR benefit
    of averaging.
  - `median` — most outlier rejection but ~25% worse SNR than mean.
  - `mean` — fastest, no rejection. Use only if your frames are already
    clean.
- **Sigma threshold** — how many standard deviations from the per-pixel
  mean a value must be before it's dropped. Lower = more aggressive. `3.0`
  is a safe default; drop to `2.5` if you see residual satellite trails.
- **Sigma iterations** — how many times the clip is re-applied. `3` is
  plenty; more just costs CPU.
- **Star-align frames** — turn OFF only if your frames are already aligned
  (e.g. a motorised mount with perfect tracking) or have no stars at all.

### AI enhancement panel
- **Run Real-ESRGAN** — off for a pure classical stack. On for denoise +
  detail recovery. Expect moderate CPU time (minutes) or fast GPU time
  (seconds).
- **Model**
  - `realesrgan-x2` — 2× upscale, ~65 MB weights, lighter.
  - `realesrgan-x4` — 4× upscale, ~65 MB weights, stronger detail.
  Both were trained on natural images, not astro specifically, so they can
  occasionally invent plausible-but-false detail. Inspect before sharing.
- **Device**
  - `auto` (default) picks CUDA → Apple MPS → CPU.
  - Force `cpu` if your GPU is short on VRAM.

### Output panel
- **Format**: TIF preserves most detail; PNG is lossless; FITS keeps float
  precision (best for further scientific processing); JPG is the smallest
  and lossy.
- **Bit depth**: 16-bit is honoured for PNG (mono only) and TIFF. JPG and
  BMP are 8-bit by format.

### After stacking
- Preview on the right shows the result. For FITS output, a PNG preview is
  auto-generated since browsers can't display FITS.
- **Download full-resolution result** gives you the file.
- **Open in Editor →** pushes the stacked image to the Editor tab.

---

## 4. Tab 2 — Editor

Non-destructive post-processing. Sliders affect a downsampled live preview
for responsiveness; the final **Export full-resolution** button applies the
identical chain at full size.

Pipeline order (all stages skip cleanly at their identity value):

1. **Levels — black point / white point** — clips and rescales the
   histogram. Pull black up to kill background fog; pull white down to let
   bright cores bloom.
2. **Asinh stretch** — inverse-hyperbolic-sine compression. `0` = identity,
   `1` = aggressive. This is the single most useful astro control: it
   brightens faint nebulosity without blowing out stars. Try `0.4` first.
3. **Gamma** — power curve. `<1` brightens shadows, `>1` darkens them.
4. **Brightness** — additive offset in `[-0.5, 0.5]`.
5. **Contrast** — multiplier around mid-grey. `1` = identity.
6. **Saturation** — `0` = grayscale, `1` = identity, `>1` boosts colour.
   Ignored on mono images.
7. **Sharpen** — unsharp mask amount. `0` = off. Above `1.5` starts showing
   ringing on bright stars.

**Reset adjustments** puts all sliders back to identity.

You can also upload any image directly (including one you saved earlier)
and edit it without re-running the stacker.

### Export
- **Format / bit depth** match the stacker tab.
- The file appears under **Download edited image**.

---

## 5. Command line

The web UI is a thin wrapper over the same pipeline. Equivalent CLI:

```bash
astrostack lights/ \
    --darks darks/ --flats flats/ \
    --method sigma --sigma 3.0 \
    --enhance-model realesrgan-x2 --device auto \
    -o result.tif
```

Full help: `astrostack --help`.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Frame resolution ... does not match reference` | Mixed image sizes | Keep one resolution per run. |
| Alignment warning "astroalign failed" | Too few detectable stars in a frame | Drop that frame, or turn off align if tracking is accurate. |
| AI enhancement fell back to classical | Torch/CUDA unavailable or model download failed | Check network; or install torch with CUDA; or just use the result — the fallback is reasonable. |
| Out of memory on GPU | Large image + x4 model | Switch to x2, or set `--device cpu`. |
| Result looks washed out | Linear stack before stretch | Use the Editor: raise black point, apply asinh stretch. |
| Colour tint on stars | Raw white balance off | Adjust `Saturation` down or the black point per-channel isn't supported yet (todo). |
| Stars look like short streaks | Rotation between frames exceeded what translation-fallback can fix | Ensure star-align is ON, and enough stars are visible. |

Logs: launch with `astrostack-ui -v` for INFO, `-vv` for DEBUG.

---

## 7. Privacy

Every step runs locally. The only network call is the one-time model weight
download from the official `xinntao/Real-ESRGAN` GitHub release (~65 MB,
cached at `~/.cache/astrostack/weights/`). Turn off AI enhancement to run
fully offline after install.

The Real-ESRGAN inference uses [`spandrel`](https://github.com/chaiNNer-org/spandrel)
to load the weights — pure-Python, no `basicsr` build step.
"""
