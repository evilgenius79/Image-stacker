# astrostack

A local AI astronomy image stacker. Calibrates, star-aligns, sigma-clip
stacks, and AI-enhances your light frames — all on your machine, with GPU
acceleration when available and a CPU fallback when not.

## Features

- **Multi-format I/O**: JPEG, PNG, BMP, WebP, TIFF (8/16-bit), FITS, and
  camera RAW (CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, ...).
- **Optional calibration** with master darks, bias, and flats (median-combined
  from multiple frames automatically).
- **Star-pattern alignment** via `astroalign` with phase-correlation fallback.
- **Stacking**: mean, median, or sigma-clipped mean.
- **AI enhancement**: Real-ESRGAN x2 / x4 with weights downloaded from
  HuggingFace on first use. Falls back to wavelet denoise + unsharp mask if
  PyTorch / weights are unavailable.
- **Device selection**: CUDA, Apple MPS, or CPU (`--device auto` picks the
  best available).
- **Tiled inference** so big stacks don't blow up VRAM.

## Install

```bash
git clone https://github.com/evilgenius79/image-stacker
cd image-stacker
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

If you have a CUDA GPU, install the matching PyTorch build first
(see https://pytorch.org/) before `pip install -e .` so the right
CUDA wheels get pulled in.

## Web UI

```bash
astrostack-ui          # opens http://127.0.0.1:7860
astrostack-ui --share  # public Gradio link
```

Two tabs:

- **Stack & Enhance** — drop in light frames (and optional darks/flats/bias),
  pick a stacking method and AI model, hit run, preview and download.
- **Editor** — post-process the stacked result (or any uploaded image) with
  brightness, contrast, saturation, gamma, black/white levels, asinh
  (astro-friendly faint-detail) stretch, and sharpen. Live preview is
  downsampled for responsiveness; **Export** applies the edits at full
  resolution. Use **Open in Editor →** on the Stack tab to send the result
  straight over.

## Quick start (CLI)

```bash
# Stack every supported file in lights/, save 16-bit TIFF
astrostack lights/ -o stacked.tif

# With calibration frames and AI x4 upscale + denoise
astrostack lights/ \
    --darks darks/ --flats flats/ --bias bias/ \
    --enhance-model realesrgan-x4 \
    -o m31.tif

# Skip the AI step (pure classical stack)
astrostack lights/ --no-enhance -o stacked.fits

# Mix files and directories, save as FITS
astrostack img1.cr2 img2.cr2 lights/extra/ -o out.fits

# Force CPU-only (also used as fallback automatically)
astrostack lights/ --device cpu -o out.png
```

Run `astrostack --help` for the full option list.

## Pipeline

1. **Load** every input as float32 in `[0, 1]` (RAW images are demosaiced
   with camera white-balance and linear gamma).
2. **Calibrate** each light: subtract bias, subtract dark, divide by
   normalised flat (each step optional).
3. **Align** to the first frame using triangle-matched stars; if star
   detection fails, fall back to sub-pixel phase correlation.
4. **Stack** using sigma-clipped mean (default), median, or simple mean.
5. **Enhance** the stacked image with Real-ESRGAN. On first run the model
   weights are downloaded into `~/.cache/astrostack/weights/`. If torch or
   the model fails for any reason, a classical wavelet denoise +
   unsharp-mask pass runs instead.
6. **Save** to the format inferred from the output extension. PNG / TIFF
   default to 16-bit.

## Notes

- Real-ESRGAN upscales by the model's scale factor (x2 or x4). If you want
  the output at the original resolution, downsample after, or stick with
  `--no-enhance`.
- FITS output is written as float32, channel-first for color.
- Frames whose alignment fails are dropped from the stack with a warning.

## License

MIT
