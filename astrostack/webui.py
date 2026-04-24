"""Gradio web UI for the stacker.

Launch with: ``astrostack-ui`` (after install) or ``python -m astrostack.webui``.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from .device import select_device
from .io import is_supported, load_image
from .pipeline import run_pipeline

log = logging.getLogger(__name__)


def _collect(files) -> list[Path]:
    if not files:
        return []
    out: list[Path] = []
    for f in files:
        # Gradio File component returns NamedString or dict-like with .name
        name = f.name if hasattr(f, "name") else f
        p = Path(name)
        if p.is_file() and is_supported(p):
            out.append(p)
    return out


def _run(
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

    progress(0, desc="Starting pipeline ...")
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
        save_intermediate=False,
    )

    # FITS won't preview; return a PNG preview alongside the file download.
    preview_path = out_path
    if out_path.suffix.lower() in {".fits", ".fit", ".fts"}:
        from .io import save_image
        preview_path = workdir / "preview.png"
        save_image(preview_path, load_image(out_path), bit_depth=8)

    return str(preview_path), str(out_path), f"Saved to {out_path}"


def build_ui() -> gr.Blocks:
    device_default = select_device("auto")

    with gr.Blocks(title="astrostack", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# astrostack\n"
            "Local AI astronomy image stacker — calibrate, align, stack, and "
            "enhance with Real-ESRGAN.\n"
            f"**Detected device:** `{device_default}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                lights = gr.File(
                    label="Light frames",
                    file_count="multiple",
                    file_types=None,  # accept anything; we filter by extension
                )
                with gr.Accordion("Calibration frames (optional)", open=False):
                    darks = gr.File(label="Darks", file_count="multiple")
                    flats = gr.File(label="Flats", file_count="multiple")
                    bias = gr.File(label="Bias", file_count="multiple")

                with gr.Accordion("Stacking", open=True):
                    method = gr.Radio(
                        choices=["sigma", "median", "mean"],
                        value="sigma",
                        label="Algorithm",
                    )
                    sigma = gr.Slider(1.0, 5.0, value=3.0, step=0.1,
                                      label="Sigma threshold")
                    sigma_iters = gr.Slider(1, 6, value=3, step=1,
                                            label="Sigma iterations")
                    do_align = gr.Checkbox(value=True, label="Star-align frames")

                with gr.Accordion("AI enhancement", open=True):
                    do_enhance = gr.Checkbox(value=True, label="Run Real-ESRGAN")
                    enhance_model = gr.Radio(
                        choices=["realesrgan-x2", "realesrgan-x4"],
                        value="realesrgan-x2",
                        label="Model (downloaded on first use)",
                    )
                    device = gr.Radio(
                        choices=["auto", "cuda", "mps", "cpu"],
                        value="auto",
                        label="Device",
                    )

                with gr.Accordion("Output", open=False):
                    output_format = gr.Radio(
                        choices=["TIF", "PNG", "FITS", "JPG"],
                        value="TIF",
                        label="Format",
                    )
                    bit_depth = gr.Radio(
                        choices=["16", "8"],
                        value="16",
                        label="Bit depth (PNG/TIFF only)",
                    )

                run_btn = gr.Button("Stack & Enhance", variant="primary")

            with gr.Column(scale=2):
                preview = gr.Image(label="Result preview", type="filepath",
                                   height=520)
                download = gr.File(label="Download full-resolution result")
                status = gr.Markdown("")

        run_btn.click(
            _run,
            inputs=[lights, darks, flats, bias,
                    method, sigma, sigma_iters,
                    do_align, do_enhance, enhance_model, device,
                    bit_depth, output_format],
            outputs=[preview, download, status],
        )

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
