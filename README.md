# astrostack

A local AI astronomy image stacker. Calibrate, star-align, sigma-clip stack,
AI-enhance, and post-process your light frames — all on your machine, with
GPU acceleration when available and a classical CPU fallback when not.

- **CLI** for scripted / batch use.
- **Web UI** (Gradio) for interactive work, including a built-in photo
  editor and an in-page user guide.
- Everything runs locally. The only network call is a one-time download of
  the AI model weights (~65 MB) from the official Real-ESRGAN GitHub
  release.

---

## Contents

- [Features](#features)
- [Install](#install)
- [Web UI](#web-ui)
- [CLI quick start](#cli-quick-start)
- [CLI reference](#cli-reference)
- [How the pipeline works](#how-the-pipeline-works)
- [Supported formats](#supported-formats)
- [Capturing good data](#capturing-good-data)
- [The Editor tab](#the-editor-tab)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [License](#license)

---

## Features

- **Multi-format I/O**: JPEG, PNG, BMP, WebP, TIFF (8/16-bit), FITS, and
  camera RAW (CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, PEF, SRW, KDC, 3FR).
- **Optional calibration** with master darks, bias, and flats (multiple
  frames are auto median-combined into a master).
- **Star-pattern alignment** via `astroalign`, with sub-pixel phase
  correlation as a fallback when star detection fails.
- **Stacking**: mean, median, or sigma-clipped mean (default; best outlier
  rejection).
- **AI enhancement**: Real-ESRGAN x2 / x4 with weights downloaded on first
  use. Automatic CPU fallback to wavelet denoise + unsharp mask.
- **Device selection**: CUDA, Apple MPS, or CPU (`auto` picks the best).
- **Adaptive tiling** so big stacks don't blow out GPU VRAM.
- **Built-in editor**: brightness, contrast, saturation, gamma, black/white
  levels, asinh stretch (astro-friendly faint-detail boost), and sharpen —
  all with a live preview and full-resolution export.
- **In-page Guide tab** documenting every control.

---

## Install

```bash
git clone https://github.com/evilgenius79/image-stacker
cd image-stacker
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**For GPU users:** install the matching PyTorch wheel *before* `pip install
-e .` so the correct CUDA build gets pulled in. See
<https://pytorch.org/get-started/locally/>.

Python 3.10+ required.

---

## Web UI

```bash
astrostack-ui              # http://127.0.0.1:7860
astrostack-ui --port 8000  # custom port
astrostack-ui --share      # public Gradio share link
astrostack-ui -vv          # DEBUG logs
```

Three tabs:

1. **Stack & Enhance** — upload light frames (and optional darks/flats/bias),
   pick a stacking method and AI model, run. Preview + download. A
   **Quick help** accordion sits at the top of the tab.
2. **Editor** — non-destructive post-processing on the stacked result (or
   any uploaded image) with a live preview and full-resolution export.
3. **Guide** — the full user guide, rendered right in the page.

---

## CLI quick start

```bash
# Stack every supported file in lights/, save 16-bit TIFF
astrostack lights/ -o stacked.tif

# Full pipeline with calibration + 4x AI enhancement
astrostack lights/ \
    --darks darks/ --flats flats/ --bias bias/ \
    --enhance-model realesrgan-x4 \
    -o m31.tif

# Skip the AI step (pure classical stack)
astrostack lights/ --no-enhance -o stacked.fits

# Mix files and directories, save as FITS
astrostack img1.cr2 img2.cr2 lights/extra/ -o out.fits

# Force CPU
astrostack lights/ --device cpu -o out.png

# Also keep the pre-enhancement stack next to the output
astrostack lights/ --save-intermediate -o final.tif
```

---

## CLI reference

```
astrostack INPUTS... -o OUTPUT [options]

INPUTS: one or more files or directories. Directories are scanned for
        supported extensions.

-o, --output PATH          Output path. Format inferred from extension.
    --darks PATH           Dark frame file(s) or directory. Repeatable.
    --bias PATH            Bias frame file(s) or directory. Repeatable.
    --flats PATH           Flat frame file(s) or directory. Repeatable.
    --method [mean|median|sigma]  Stacking algorithm. Default: sigma.
    --sigma FLOAT          Sigma threshold for sigma-clip. Default: 3.0.
    --sigma-iters INT      Sigma-clip iterations. Default: 3.
    --no-align             Skip star alignment.
    --no-enhance           Skip AI enhancement.
    --enhance-model [realesrgan-x2|realesrgan-x4]   Default: x2.
    --device [auto|cuda|mps|cpu]    Default: auto.
    --bit-depth [8|16]     Output bit depth for PNG/TIFF. Default: 16.
    --save-intermediate    Also save the pre-enhancement stack.
-v, --verbose              Increase log verbosity (-v INFO, -vv DEBUG).
-h, --help                 Show help.
```

---

## How the pipeline works

Given N light frames:

1. **Load** every frame as float32 in `[0, 1]`. RAW files are demosaiced
   with camera white balance and linear gamma so the stacker sees linear
   light.
2. **Calibrate** (if any master frames provided):
   `calibrated = light − bias − dark`, then divide by normalised flat.
3. **Align** frames 2..N to frame 1 using triangle-matched star patterns.
   Failure falls back to sub-pixel phase correlation (translation only).
   Frames whose alignment fails are dropped from the stack with a warning.
4. **Stack** with sigma-clipped mean (default) — for each pixel, reject
   values more than `σ` standard deviations from the per-pixel mean, repeat
   `sigma_iters` times, then average what's left.
5. **Enhance** with Real-ESRGAN. The 3-channel network runs on RGB; mono
   images are triple-stacked and collapsed back after. Images are tiled if
   they're big enough to risk VRAM / RAM pressure.
6. **Save** in the format the output extension implies.

If anything in step 5 fails (no torch, no network, model load error, OOM),
a classical wavelet-denoise + unsharp-mask pass runs instead so you still
get a finished image.

---

## Supported formats

| Category | Extensions |
| --- | --- |
| Camera RAW | `.cr2` `.cr3` `.nef` `.arw` `.dng` `.raf` `.orf` `.rw2` `.pef` `.srw` `.kdc` `.3fr` |
| Standard | `.jpg`/`.jpeg` `.png` `.bmp` `.webp` |
| High bit depth | `.tif`/`.tiff` (8/16-bit, mono or RGB) |
| Astronomy | `.fits` `.fit` `.fts` |

All light frames in one stack must share the same pixel dimensions. Mono
and color can be mixed (mono is promoted to pseudo-RGB).

---

## Capturing good data

A few quick rules of thumb:

- **Quantity + quality**. More frames help, but throw out obvious junk
  (clouds, plane trails, badly tracked). Sigma-clip handles the rest.
- **Match calibration**. Darks need the *same* exposure, ISO, and sensor
  temperature as your lights. Flats need the *exact same* optical setup
  (focus, rotation, filter, dust on the sensor) as the lights.
- **Bias optional**. If your darks already "eat" the bias (same exposure
  length gives roughly the same readout baseline), you can skip master
  bias.
- **Expose for the noise**. Don't sub-expose — bring up the histogram so
  the background is a few percent above zero. `astrostack` does not
  magically invent signal that wasn't captured.

---

## The Editor tab

Applied in this order; every stage is an identity op at its default value:

1. **Black / white point** — histogram clip + rescale. Raise black to kill
   background fog; drop white to control blown highlights.
2. **Asinh stretch** — inverse-hyperbolic-sine compression. `0` off; `1`
   aggressive. The single most useful astro slider: brightens faint
   nebulosity without clipping bright stars. Try `0.4` first.
3. **Gamma** — power curve. `<1` brightens shadows, `>1` darkens.
4. **Brightness** — additive offset in `[-0.5, 0.5]`.
5. **Contrast** — multiplier around mid-grey. `1` = identity.
6. **Saturation** — `0` = grayscale, `1` = identity, `>1` boosts colour.
   Ignored on mono.
7. **Sharpen** — unsharp-mask amount. `0` off. Above `~1.5` expect ringing.

The live preview is downsampled (longest edge ≤ 1280 px) for slider
responsiveness. **Export full-resolution** re-runs the identical chain on
the original-size image and gives you a download.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Frame resolution ... does not match reference` | Mixed image sizes | Keep one resolution per run. |
| `astroalign failed; falling back to phase correlation` | Too few detectable stars | Try a different reference frame (reorder inputs), or turn off align. |
| AI enhancement fell back to classical | Torch/CUDA missing or model download blocked | Check network; or install torch; or accept the classical result. |
| `CUDA out of memory` | Big image + x4 model | Use `realesrgan-x2`, or `--device cpu`. |
| Result looks washed out | Linear stack before stretch | Use the Editor: raise black point, apply asinh stretch. |
| Tiny stars look square / grid-like | Raw demosaic artefact on under-exposed frames | Expose longer or disable enhance for that run. |
| FITS preview missing | Browsers can't render FITS | The UI auto-writes a PNG preview alongside. |

Model weights cache: `~/.cache/astrostack/weights/`. Delete to force
re-download.

---

## Project layout

```
astrostack/
├── __init__.py
├── align.py       # star-triangle + phase-correlation alignment
├── calibrate.py   # master dark/bias/flat construction + application
├── cli.py         # click-based CLI entry point
├── device.py      # CUDA / MPS / CPU selection
├── editor.py      # post-processing adjustments (pure numpy)
├── enhance.py     # Real-ESRGAN with classical fallback
├── guide.py       # help content shown in the web UI
├── io.py          # multi-format load/save
├── pipeline.py    # end-to-end orchestration
├── stack.py       # mean / median / sigma-clipped stacking
└── webui.py       # Gradio app (3 tabs: Stack, Editor, Guide)
```

---

## License

MIT
