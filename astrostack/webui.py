"""Gradio web UI: stacker + post-processing editor.

Launch with: ``astrostack-ui`` (after install) or ``python -m astrostack.webui``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np

from .device import select_device
from .editor import DEFAULTS, adjust
from .guide import GUIDE_MARKDOWN, QUICK_HELP_EDITOR, QUICK_HELP_STACK
from .io import is_supported, load_image, save_image
from .pipeline import run_pipeline

log = logging.getLogger(__name__)

PREVIEW_MAX = 1280  # max dimension for live preview to keep slider response snappy


def _collect(files) -> list[Path]:
    if not files:
        return []
    out: list[Path] = []
    for f in files:
        name = f.name if hasattr(f, "name") else f
        p = Path(name)
        if p.is_file() and is_supported(p):
            out.append(p)
    return out


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _downsample(arr: np.ndarray, max_dim: int = PREVIEW_MAX) -> np.ndarray:
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return arr
    step = int(np.ceil(longest / max_dim))
    return arr[::step, ::step]


# ---------------------------------------------------------------------------
# Stack tab
# ---------------------------------------------------------------------------

def _run_stack(
    lights, darks, flats, bias,
    method, sigma, sigma_iters,
    do_align, do_enhance, enhance_model, device,
    bit_depth, output_format,
    progress=gr.Progress(track_tqdm=True),
):
    light_paths = _collect(lights)
    if not light_paths:
        raise gr.Error("Please upload at least one supported light frame.")

    workdir = Path(tempfile.mkdtemp(prefix="astrostack_"))
    out_path = workdir / f"stacked.{output_format.lower()}"

    progress(0, desc="Running pipeline ...")
    run_pipeline(
        light_paths=light_paths,
        output=out_path,
        darks=_collect(darks) or None,
        bias=_collect(bias) or None,
        flats=_collect(flats) or None,
        stack_method=method,
        sigma=float(sigma),
        sigma_iters=int(sigma_iters),
        align=bool(do_align),
        enhance=bool(do_enhance),
        enhance_model=enhance_model,
        device=device,
        bit_depth=int(bit_depth),
    )

    preview_path = out_path
    if out_path.suffix.lower() in {".fits", ".fit", ".fts"}:
        preview_path = workdir / "preview.png"
        save_image(preview_path, load_image(out_path), bit_depth=8)

    return str(preview_path), str(out_path), f"Saved to `{out_path}`", str(out_path)


# ---------------------------------------------------------------------------
# Editor tab
# ---------------------------------------------------------------------------

def _load_into_editor(path: str | None):
    """Load an image into the editor; returns full-res array, preview array, status."""
    if not path:
        return None, None, "No image loaded."
    p = Path(path)
    if not p.exists():
        return None, None, f"File not found: {p}"
    full = load_image(p)
    preview = _downsample(full)
    return full, _to_uint8(preview), f"Loaded `{p.name}` ({full.shape[1]}x{full.shape[0]})"


def _send_from_stack(stack_result_path: str | None):
    return _load_into_editor(stack_result_path)


def _upload_to_editor(file):
    if file is None:
        return None, None, "No file uploaded."
    name = file.name if hasattr(file, "name") else file
    return _load_into_editor(name)


def _live_preview(preview_arr, black, white, asinh_, gamma, brightness,
                  contrast, saturation, sharpen):
    """Apply adjustments to the downsampled preview only."""
    if preview_arr is None:
        return None
    img = preview_arr.astype(np.float32) / 255.0
    out = adjust(
        img,
        black_point=float(black),
        white_point=float(white),
        asinh=float(asinh_),
        gamma=float(gamma),
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        sharpen=float(sharpen),
    )
    return _to_uint8(out)


def _export_full(full_arr, black, white, asinh_, gamma, brightness,
                 contrast, saturation, sharpen, output_format, bit_depth):
    if full_arr is None:
        raise gr.Error("Load an image into the editor first.")
    out = adjust(
        full_arr,
        black_point=float(black),
        white_point=float(white),
        asinh=float(asinh_),
        gamma=float(gamma),
        brightness=float(brightness),
        contrast=float(contrast),
        saturation=float(saturation),
        sharpen=float(sharpen),
    )
    workdir = Path(tempfile.mkdtemp(prefix="astrostack_edit_"))
    out_path = workdir / f"edited.{output_format.lower()}"
    save_image(out_path, out, bit_depth=int(bit_depth))
    return str(out_path), f"Exported `{out_path}`"


def _reset_sliders():
    d = DEFAULTS
    return (d["black_point"], d["white_point"], d["asinh"], d["gamma"],
            d["brightness"], d["contrast"], d["saturation"], d["sharpen"])


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    device_default = select_device("auto")

    with gr.Blocks(title="astrostack", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# astrostack\n"
            "Local AI astronomy image stacker — calibrate, align, stack, "
            "enhance, and edit.\n"
            f"**Detected device:** `{device_default}` · "
            "**New here?** See the **Guide** tab."
        )

        # Shared state: path of latest stacker output, + arrays for the editor.
        last_stack_path = gr.State(None)
        editor_full = gr.State(None)     # full-res float32 array
        editor_preview = gr.State(None)  # downsampled uint8 preview source

        with gr.Tabs():
            # ---------------- Stack tab ----------------
            with gr.Tab("Stack & Enhance"):
                with gr.Accordion("ℹ Quick help", open=False):
                    gr.Markdown(QUICK_HELP_STACK)
                with gr.Row():
                    with gr.Column(scale=1):
                        lights = gr.File(label="Light frames",
                                         file_count="multiple")
                        with gr.Accordion("Calibration frames (optional)",
                                          open=False):
                            darks = gr.File(label="Darks", file_count="multiple")
                            flats = gr.File(label="Flats", file_count="multiple")
                            bias = gr.File(label="Bias", file_count="multiple")

                        with gr.Accordion("Stacking", open=True):
                            method = gr.Radio(
                                choices=["sigma", "median", "mean"],
                                value="sigma", label="Algorithm")
                            sigma = gr.Slider(1.0, 5.0, value=3.0, step=0.1,
                                              label="Sigma threshold")
                            sigma_iters = gr.Slider(1, 6, value=3, step=1,
                                                    label="Sigma iterations")
                            do_align = gr.Checkbox(value=True,
                                                   label="Star-align frames")

                        with gr.Accordion("AI enhancement", open=True):
                            do_enhance = gr.Checkbox(value=True,
                                                     label="Run Real-ESRGAN")
                            enhance_model = gr.Radio(
                                choices=["realesrgan-x2", "realesrgan-x4"],
                                value="realesrgan-x2",
                                label="Model (downloaded on first use)")
                            device = gr.Radio(
                                choices=["auto", "cuda", "mps", "cpu"],
                                value="auto", label="Device")

                        with gr.Accordion("Output", open=False):
                            output_format = gr.Radio(
                                choices=["TIF", "PNG", "FITS", "JPG"],
                                value="TIF", label="Format")
                            bit_depth = gr.Radio(
                                choices=["16", "8"], value="16",
                                label="Bit depth (PNG/TIFF only)")

                        run_btn = gr.Button("Stack & Enhance", variant="primary")

                    with gr.Column(scale=2):
                        stack_preview = gr.Image(label="Stacker result",
                                                 type="filepath", height=520)
                        stack_download = gr.File(
                            label="Download full-resolution result")
                        stack_status = gr.Markdown("")
                        send_to_editor_btn = gr.Button(
                            "Open in Editor →", variant="secondary")

                run_btn.click(
                    _run_stack,
                    inputs=[lights, darks, flats, bias,
                            method, sigma, sigma_iters,
                            do_align, do_enhance, enhance_model, device,
                            bit_depth, output_format],
                    outputs=[stack_preview, stack_download,
                             stack_status, last_stack_path],
                )

            # ---------------- Editor tab ----------------
            with gr.Tab("Editor") as editor_tab:
                with gr.Accordion("ℹ Quick help", open=False):
                    gr.Markdown(QUICK_HELP_EDITOR)
                gr.Markdown(
                    "Adjust the stacked result with classic photo controls "
                    "plus astro-specific stretch. Live preview is downsampled "
                    "for responsiveness; **Export** applies edits at full "
                    "resolution."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        editor_upload = gr.File(
                            label="Or upload an image to edit",
                            file_count="single")

                        with gr.Accordion("Levels & stretch", open=True):
                            black = gr.Slider(0.0, 0.5, value=0.0, step=0.005,
                                              label="Black point")
                            white = gr.Slider(0.5, 1.0, value=1.0, step=0.005,
                                              label="White point")
                            asinh_ = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                               label="Asinh stretch (faint detail)")
                            gamma = gr.Slider(0.2, 3.0, value=1.0, step=0.01,
                                              label="Gamma")

                        with gr.Accordion("Tone & color", open=True):
                            brightness = gr.Slider(-0.5, 0.5, value=0.0,
                                                   step=0.01, label="Brightness")
                            contrast = gr.Slider(0.2, 3.0, value=1.0, step=0.01,
                                                 label="Contrast")
                            saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.01,
                                                   label="Saturation")

                        with gr.Accordion("Detail", open=False):
                            sharpen = gr.Slider(0.0, 3.0, value=0.0, step=0.05,
                                                label="Sharpen amount")

                        reset_btn = gr.Button("Reset adjustments")

                        with gr.Accordion("Export", open=True):
                            export_format = gr.Radio(
                                choices=["TIF", "PNG", "JPG", "FITS"],
                                value="TIF", label="Format")
                            export_bits = gr.Radio(
                                choices=["16", "8"], value="16",
                                label="Bit depth (PNG/TIFF)")
                            export_btn = gr.Button("Export full-resolution",
                                                   variant="primary")

                    with gr.Column(scale=2):
                        editor_preview_img = gr.Image(
                            label="Live preview (downsampled)",
                            type="numpy", height=520, interactive=False)
                        editor_status = gr.Markdown("No image loaded.")
                        editor_download = gr.File(label="Download edited image")

                # --- editor wiring ---
                slider_inputs = [black, white, asinh_, gamma,
                                 brightness, contrast, saturation, sharpen]

                def _on_load_full(full, preview, status):
                    rendered = _live_preview(preview, *[DEFAULTS[k] for k in
                        ("black_point", "white_point", "asinh", "gamma",
                         "brightness", "contrast", "saturation", "sharpen")])
                    return (full, preview, rendered, status,
                            *_reset_sliders())

                # Send from stack tab.
                send_to_editor_btn.click(
                    _send_from_stack,
                    inputs=[last_stack_path],
                    outputs=[editor_full, editor_preview, editor_status],
                ).then(
                    _on_load_full,
                    inputs=[editor_full, editor_preview, editor_status],
                    outputs=[editor_full, editor_preview, editor_preview_img,
                             editor_status, *slider_inputs],
                )

                # Upload directly into editor.
                editor_upload.upload(
                    _upload_to_editor,
                    inputs=[editor_upload],
                    outputs=[editor_full, editor_preview, editor_status],
                ).then(
                    _on_load_full,
                    inputs=[editor_full, editor_preview, editor_status],
                    outputs=[editor_full, editor_preview, editor_preview_img,
                             editor_status, *slider_inputs],
                )

                # Live preview updates when any slider changes.
                for s in slider_inputs:
                    s.release(
                        _live_preview,
                        inputs=[editor_preview, *slider_inputs],
                        outputs=[editor_preview_img],
                    )

                reset_btn.click(
                    _reset_sliders, inputs=None, outputs=slider_inputs,
                ).then(
                    _live_preview,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                )

                export_btn.click(
                    _export_full,
                    inputs=[editor_full, *slider_inputs,
                            export_format, export_bits],
                    outputs=[editor_download, editor_status],
                )

            # ---------------- Guide tab ----------------
            with gr.Tab("Guide"):
                gr.Markdown(GUIDE_MARKDOWN)

    return ui


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Launch the astrostack web UI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link.")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    ui = build_ui()
    ui.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
