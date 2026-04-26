"""Gradio web UI: stacker + post-processing editor.

Launch with: ``astrostack-ui`` (after install) or ``python -m astrostack.webui``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")  # must run before any pyplot import anywhere
import numpy as np

from .device import select_device
from .editor import (DEFAULTS, PRESETS, adjust, apply_preset, auto_stretch,
                     compute_background_offsets)
from .guide import GUIDE_MARKDOWN, QUICK_HELP_EDITOR, QUICK_HELP_STACK
from .io import is_supported, is_thumbnail, load_image, save_image
from .pipeline import run_pipeline
from .quality import TABLE_HEADERS, analyze_paths, stats_to_table
from .settings import (delete_user_preset, load_settings, load_user_presets,
                       save_settings, save_user_preset)

log = logging.getLogger(__name__)

PREVIEW_MAX = 1280  # max dimension for live preview to keep slider response snappy


def _analyze_frames(files):
    paths = _collect(files)
    if not paths:
        return [], "No frames to analyze."
    stats = analyze_paths(paths)
    return stats_to_table(stats), f"Analyzed {len(stats)} frame(s)."


def _collect(files) -> list[Path]:
    if not files:
        return []
    out: list[Path] = []
    skipped_thumbs = 0
    for f in files:
        name = f.name if hasattr(f, "name") else f
        p = Path(name)
        if not (p.is_file() and is_supported(p)):
            continue
        if is_thumbnail(p):
            skipped_thumbs += 1
            continue
        out.append(p)
    if skipped_thumbs:
        log.info("Skipped %d thumbnail file(s).", skipped_thumbs)
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
    result = run_pipeline(
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
        return_arrays=True,
    )
    pre_arr = result.get("stacked")

    save_settings({
        "stack_method": method, "sigma": float(sigma),
        "sigma_iters": int(sigma_iters), "do_align": bool(do_align),
        "do_enhance": bool(do_enhance), "enhance_model": enhance_model,
        "device": device, "bit_depth": str(bit_depth),
        "output_format": output_format,
    })

    preview_path = out_path
    if out_path.suffix.lower() in {".fits", ".fit", ".fts"}:
        preview_path = workdir / "preview.png"
        save_image(preview_path, load_image(out_path), bit_depth=8)

    pre_preview_path = None
    if pre_arr is not None and bool(do_enhance):
        pre_preview_path = workdir / "pre_enhance.png"
        save_image(pre_preview_path, pre_arr, bit_depth=8)

    return (str(preview_path),
            str(pre_preview_path) if pre_preview_path else None,
            str(out_path),
            f"Saved to `{out_path}`",
            str(out_path))


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


SLIDER_KEYS = (
    "black_point", "white_point",
    "black_r", "black_g", "black_b",
    "white_r", "white_g", "white_b",
    "asinh", "gamma", "brightness", "contrast", "saturation",
    "star_reduce", "sharpen",
    "crop_left", "crop_top", "crop_right", "crop_bottom",
)


def _slider_kwargs(values):
    """Expand a positional slider tuple into the kwargs adjust() expects."""
    return {k: float(v) for k, v in zip(SLIDER_KEYS, values)}


def _live_preview(preview_arr, *slider_values):
    if preview_arr is None:
        return None
    img = preview_arr.astype(np.float32) / 255.0
    return _to_uint8(adjust(img, **_slider_kwargs(slider_values)))


def _export_full(full_arr, *args):
    *slider_values, output_format, bit_depth = args
    if full_arr is None:
        raise gr.Error("Load an image into the editor first.")
    out = adjust(full_arr, **_slider_kwargs(slider_values))
    workdir = Path(tempfile.mkdtemp(prefix="astrostack_edit_"))
    out_path = workdir / f"edited.{output_format.lower()}"
    save_image(out_path, out, bit_depth=int(bit_depth))
    return str(out_path), f"Exported `{out_path}`"


def _reset_sliders():
    return tuple(DEFAULTS[k] for k in SLIDER_KEYS)


def _values_from_dict(d: dict) -> tuple:
    """Project a settings dict onto the SLIDER_KEYS tuple with defaults."""
    return tuple(float(d.get(k, DEFAULTS[k])) for k in SLIDER_KEYS)


def _apply_builtin_preset(name: str):
    return _values_from_dict(apply_preset(name))


def _apply_user_preset(name: str, *current_values):
    """Load a user preset; fall back to current values if name unknown."""
    presets = load_user_presets()
    if name and name in presets:
        merged = dict(DEFAULTS)
        merged.update(presets[name])
        return _values_from_dict(merged)
    return tuple(current_values)


def _save_user_preset_handler(name, *slider_values):
    if not name or not name.strip():
        choices = sorted(load_user_presets().keys())
        return gr.update(choices=choices), "Enter a preset name first."
    payload = dict(zip(SLIDER_KEYS, slider_values))
    save_user_preset(name.strip(), payload)
    choices = sorted(load_user_presets().keys())
    return (gr.update(choices=choices, value=name.strip()),
            f"Saved preset `{name.strip()}`.")


def _delete_user_preset_handler(name):
    if not name:
        choices = sorted(load_user_presets().keys())
        return gr.update(choices=choices, value=None), "No preset selected."
    delete_user_preset(name)
    choices = sorted(load_user_presets().keys())
    return gr.update(choices=choices, value=None), f"Deleted preset `{name}`."


def _do_neutralize(preview_arr, *slider_values):
    """Compute per-channel offsets from preview and update black_r/g/b."""
    if preview_arr is None:
        return tuple(slider_values)
    img = preview_arr.astype(np.float32) / 255.0
    rgb = compute_background_offsets(img)
    values = list(slider_values)
    # Indices of black_r/g/b in SLIDER_KEYS
    idx_r = SLIDER_KEYS.index("black_r")
    idx_g = SLIDER_KEYS.index("black_g")
    idx_b = SLIDER_KEYS.index("black_b")
    values[idx_r] = float(rgb[0])
    values[idx_g] = float(rgb[1])
    values[idx_b] = float(rgb[2])
    return tuple(values)


def _do_auto_stretch(preview_arr, *slider_values):
    """Set black_point, white_point, asinh from histogram percentiles."""
    if preview_arr is None:
        return tuple(slider_values)
    img = preview_arr.astype(np.float32) / 255.0
    new_vals = auto_stretch(img)
    values = list(slider_values)
    for k, v in new_vals.items():
        if k in SLIDER_KEYS:
            values[SLIDER_KEYS.index(k)] = float(v)
    return tuple(values)


def _render_histogram(preview_arr, black, white, b_r, b_g, b_b, w_r, w_g, w_b):
    import matplotlib.pyplot as plt
    # Close any prior figures from earlier calls to bound pyplot's global
    # registry. The new figure we return below is owned by Gradio.
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 2.4))
    if preview_arr is None:
        ax.set_title("No image loaded")
        ax.set_facecolor("#0e1116")
        fig.patch.set_facecolor("#0b0d12")
        return fig

    img = preview_arr.astype(np.float32) / 255.0
    bins = 128
    if img.ndim == 3:
        colors = ("#ef4444", "#22c55e", "#3b82f6")
        for c, color in enumerate(colors[:img.shape[2]]):
            h, edges = np.histogram(img[..., c], bins=bins, range=(0, 1))
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, h, color=color, alpha=0.85, linewidth=1.0)
    else:
        h, edges = np.histogram(img, bins=bins, range=(0, 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, h, color="#cbd5e1", linewidth=1.0)

    ax.axvline(float(black), color="#94a3b8", linestyle="--",
               alpha=0.6, linewidth=1.0)
    ax.axvline(float(white), color="#94a3b8", linestyle="--",
               alpha=0.6, linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_facecolor("#0e1116")
    fig.patch.set_facecolor("#0b0d12")
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#9ca3af")
    fig.tight_layout()
    return fig


def _toggle_before_after(show_before, before_arr, after_arr):
    return before_arr if bool(show_before) and before_arr is not None else after_arr


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------

FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.toString();
  }
}
"""

DARK_CSS = """
:root, body, .gradio-container {
  --background-fill-primary: #0e1116;
  --background-fill-secondary: #161b22;
  --color-accent-soft: #1f2733;
  --border-color-primary: #2a3441;
}
.gradio-container { background: #0b0d12 !important; }
.dark .gradio-container { background: #0b0d12 !important; }
"""


def build_ui(dark: bool = True) -> gr.Blocks:
    device_default = select_device("auto")
    s = load_settings()

    blocks_kwargs = dict(title="astrostack", css=DARK_CSS if dark else None)
    if dark:
        blocks_kwargs["js"] = FORCE_DARK_JS

    with gr.Blocks(**blocks_kwargs) as ui:
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
                        with gr.Accordion("Frame quality analysis", open=False):
                            analyze_btn = gr.Button("Analyze loaded frames",
                                                    variant="secondary")
                            quality_status = gr.Markdown("")
                            quality_table = gr.Dataframe(
                                headers=TABLE_HEADERS,
                                interactive=False, wrap=True,
                                label="Per-frame stars / FWHM / score")
                        with gr.Accordion("Calibration frames (optional)",
                                          open=False):
                            darks = gr.File(label="Darks", file_count="multiple")
                            flats = gr.File(label="Flats", file_count="multiple")
                            bias = gr.File(label="Bias", file_count="multiple")

                        with gr.Accordion("Stacking", open=True):
                            method = gr.Radio(
                                choices=["sigma", "median", "mean"],
                                value=s["stack_method"], label="Algorithm")
                            sigma = gr.Slider(1.0, 5.0, value=float(s["sigma"]),
                                              step=0.1, label="Sigma threshold")
                            sigma_iters = gr.Slider(1, 6, value=int(s["sigma_iters"]),
                                                    step=1, label="Sigma iterations")
                            do_align = gr.Checkbox(value=bool(s["do_align"]),
                                                   label="Star-align frames")

                        with gr.Accordion("AI enhancement", open=True):
                            do_enhance = gr.Checkbox(value=bool(s["do_enhance"]),
                                                     label="Run Real-ESRGAN")
                            enhance_model = gr.Radio(
                                choices=["realesrgan-x2", "realesrgan-x4"],
                                value=s["enhance_model"],
                                label="Model (downloaded on first use)")
                            device = gr.Radio(
                                choices=["auto", "cuda", "mps", "cpu"],
                                value=s["device"], label="Device")

                        with gr.Accordion("Output", open=False):
                            output_format = gr.Radio(
                                choices=["TIF", "PNG", "FITS", "JPG"],
                                value=s["output_format"], label="Format")
                            bit_depth = gr.Radio(
                                choices=["16", "8"], value=str(s["bit_depth"]),
                                label="Bit depth (PNG/TIFF only)")

                        with gr.Row():
                            run_btn = gr.Button("Stack & Enhance",
                                                variant="primary", scale=3)
                            cancel_btn = gr.Button("Cancel", variant="stop",
                                                   scale=1)

                    with gr.Column(scale=2):
                        with gr.Row():
                            stack_preview_pre = gr.Image(
                                label="Pre-enhance (stacked only)",
                                type="filepath", height=420)
                            stack_preview = gr.Image(
                                label="Final (post-enhance)",
                                type="filepath", height=420)
                        stack_download = gr.File(
                            label="Download full-resolution result")
                        stack_status = gr.Markdown("")
                        send_to_editor_btn = gr.Button(
                            "Open in Editor →", variant="secondary")

                run_event = run_btn.click(
                    _run_stack,
                    inputs=[lights, darks, flats, bias,
                            method, sigma, sigma_iters,
                            do_align, do_enhance, enhance_model, device,
                            bit_depth, output_format],
                    outputs=[stack_preview, stack_preview_pre,
                             stack_download, stack_status, last_stack_path],
                )
                cancel_btn.click(fn=None, inputs=None, outputs=None,
                                 cancels=[run_event])

                analyze_btn.click(
                    _analyze_frames,
                    inputs=[lights],
                    outputs=[quality_table, quality_status],
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
                user_preset_choices = sorted(load_user_presets().keys())
                with gr.Row():
                    with gr.Column(scale=1):
                        editor_upload = gr.File(
                            label="Or upload an image to edit",
                            file_count="single")

                        with gr.Accordion("Presets", open=True):
                            builtin_preset = gr.Dropdown(
                                choices=list(PRESETS.keys()), value="None",
                                label="Built-in stretch preset",
                                interactive=True)
                            with gr.Row():
                                user_preset = gr.Dropdown(
                                    choices=user_preset_choices, value=None,
                                    label="Your saved presets",
                                    interactive=True, allow_custom_value=False)
                                load_preset_btn = gr.Button("Load", scale=0)
                                delete_preset_btn = gr.Button("Delete", scale=0)
                            with gr.Row():
                                preset_name = gr.Textbox(
                                    label="Save current as",
                                    placeholder="My preset",
                                    scale=3)
                                save_preset_btn = gr.Button("Save", scale=1)

                        with gr.Accordion("Levels & stretch", open=True):
                            with gr.Row():
                                auto_stretch_btn = gr.Button(
                                    "Auto stretch", variant="secondary")
                                neutralize_btn = gr.Button(
                                    "Neutralize background", variant="secondary")
                            black = gr.Slider(0.0, 0.5, value=0.0, step=0.005,
                                              label="Black point")
                            white = gr.Slider(0.5, 1.0, value=1.0, step=0.005,
                                              label="White point")
                            asinh_ = gr.Slider(0.0, 1.0, value=0.0, step=0.01,
                                               label="Asinh stretch (faint detail)")
                            gamma = gr.Slider(0.2, 3.0, value=1.0, step=0.01,
                                              label="Gamma")
                            with gr.Accordion("Per-channel levels (RGB)",
                                              open=False):
                                black_r = gr.Slider(-0.5, 0.5, value=0.0,
                                                    step=0.005, label="Red black")
                                black_g = gr.Slider(-0.5, 0.5, value=0.0,
                                                    step=0.005, label="Green black")
                                black_b = gr.Slider(-0.5, 0.5, value=0.0,
                                                    step=0.005, label="Blue black")
                                white_r = gr.Slider(-0.5, 0.5, value=0.0,
                                                    step=0.005, label="Red white")
                                white_g = gr.Slider(-0.5, 0.5, value=0.0,
                                                    step=0.005, label="Green white")
                                white_b = gr.Slider(-0.5, 0.5, value=0.0,
                                                    step=0.005, label="Blue white")

                        with gr.Accordion("Tone & color", open=True):
                            brightness = gr.Slider(-0.5, 0.5, value=0.0,
                                                   step=0.01, label="Brightness")
                            contrast = gr.Slider(0.2, 3.0, value=1.0, step=0.01,
                                                 label="Contrast")
                            saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.01,
                                                   label="Saturation")

                        with gr.Accordion("Detail", open=False):
                            star_reduce = gr.Slider(0.0, 1.0, value=0.0,
                                                    step=0.02,
                                                    label="Star size reduction")
                            sharpen = gr.Slider(0.0, 3.0, value=0.0, step=0.05,
                                                label="Sharpen amount")

                        with gr.Accordion("Crop (margins as %)", open=False):
                            crop_left = gr.Slider(0, 45, value=0, step=0.5,
                                                  label="Left")
                            crop_top = gr.Slider(0, 45, value=0, step=0.5,
                                                 label="Top")
                            crop_right = gr.Slider(0, 45, value=0, step=0.5,
                                                   label="Right")
                            crop_bottom = gr.Slider(0, 45, value=0, step=0.5,
                                                    label="Bottom")

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
                        with gr.Row():
                            show_before = gr.Checkbox(
                                value=False, label="Show original (Before)")
                        editor_preview_img = gr.Image(
                            label="Live preview (downsampled)",
                            type="numpy", height=520, interactive=False)
                        histogram_plot = gr.Plot(label="Histogram")
                        editor_status = gr.Markdown("No image loaded.")
                        editor_download = gr.File(label="Download edited image")

                # Slider list must match SLIDER_KEYS order for _live_preview /
                # _export_full to work.
                slider_inputs = [black, white,
                                 black_r, black_g, black_b,
                                 white_r, white_g, white_b,
                                 asinh_, gamma, brightness, contrast,
                                 saturation, star_reduce, sharpen,
                                 crop_left, crop_top, crop_right, crop_bottom]

                def _on_load_full(full, preview, status):
                    defaults_t = _reset_sliders()
                    rendered = _live_preview(preview, *defaults_t)
                    hist = _render_histogram(
                        preview, defaults_t[0], defaults_t[1],
                        defaults_t[2], defaults_t[3], defaults_t[4],
                        defaults_t[5], defaults_t[6], defaults_t[7])
                    return (full, preview, rendered, hist, status, *defaults_t)

                send_to_editor_btn.click(
                    _send_from_stack,
                    inputs=[last_stack_path],
                    outputs=[editor_full, editor_preview, editor_status],
                ).then(
                    _on_load_full,
                    inputs=[editor_full, editor_preview, editor_status],
                    outputs=[editor_full, editor_preview, editor_preview_img,
                             histogram_plot, editor_status, *slider_inputs],
                )

                editor_upload.upload(
                    _upload_to_editor,
                    inputs=[editor_upload],
                    outputs=[editor_full, editor_preview, editor_status],
                ).then(
                    _on_load_full,
                    inputs=[editor_full, editor_preview, editor_status],
                    outputs=[editor_full, editor_preview, editor_preview_img,
                             histogram_plot, editor_status, *slider_inputs],
                )

                # Live preview + histogram update when any slider releases.
                for s in slider_inputs:
                    s.release(
                        _live_preview,
                        inputs=[editor_preview, *slider_inputs],
                        outputs=[editor_preview_img],
                    )
                    s.release(
                        _render_histogram,
                        inputs=[editor_preview, black, white,
                                black_r, black_g, black_b,
                                white_r, white_g, white_b],
                        outputs=[histogram_plot],
                    )

                # Before/after toggle.
                def _on_toggle_before(show, preview_arr, *slider_values):
                    if not show:
                        return _live_preview(preview_arr, *slider_values)
                    if preview_arr is None:
                        return None
                    return preview_arr  # the unedited downsampled source
                show_before.change(
                    _on_toggle_before,
                    inputs=[show_before, editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                )

                # Built-in preset.
                builtin_preset.change(
                    _apply_builtin_preset,
                    inputs=[builtin_preset],
                    outputs=slider_inputs,
                ).then(
                    _live_preview,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                ).then(
                    _render_histogram,
                    inputs=[editor_preview, black, white,
                            black_r, black_g, black_b,
                            white_r, white_g, white_b],
                    outputs=[histogram_plot],
                )

                # User preset load.
                load_preset_btn.click(
                    _apply_user_preset,
                    inputs=[user_preset, *slider_inputs],
                    outputs=slider_inputs,
                ).then(
                    _live_preview,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                ).then(
                    _render_histogram,
                    inputs=[editor_preview, black, white,
                            black_r, black_g, black_b,
                            white_r, white_g, white_b],
                    outputs=[histogram_plot],
                )

                # User preset save / delete.
                save_preset_btn.click(
                    _save_user_preset_handler,
                    inputs=[preset_name, *slider_inputs],
                    outputs=[user_preset, editor_status],
                )
                delete_preset_btn.click(
                    _delete_user_preset_handler,
                    inputs=[user_preset],
                    outputs=[user_preset, editor_status],
                )

                # Auto-stretch.
                auto_stretch_btn.click(
                    _do_auto_stretch,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=slider_inputs,
                ).then(
                    _live_preview,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                ).then(
                    _render_histogram,
                    inputs=[editor_preview, black, white,
                            black_r, black_g, black_b,
                            white_r, white_g, white_b],
                    outputs=[histogram_plot],
                )

                # Background neutralize.
                neutralize_btn.click(
                    _do_neutralize,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=slider_inputs,
                ).then(
                    _live_preview,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                ).then(
                    _render_histogram,
                    inputs=[editor_preview, black, white,
                            black_r, black_g, black_b,
                            white_r, white_g, white_b],
                    outputs=[histogram_plot],
                )

                reset_btn.click(
                    _reset_sliders, inputs=None, outputs=slider_inputs,
                ).then(
                    _live_preview,
                    inputs=[editor_preview, *slider_inputs],
                    outputs=[editor_preview_img],
                ).then(
                    _render_histogram,
                    inputs=[editor_preview, black, white,
                            black_r, black_g, black_b,
                            white_r, white_g, white_b],
                    outputs=[histogram_plot],
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
    ap.add_argument("--light", action="store_true",
                    help="Use light theme (default is dark).")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    ui = build_ui(dark=not args.light)
    launch_kwargs = dict(server_name=args.host, server_port=args.port,
                         share=args.share)
    # Gradio 6.x accepts `theme` on launch(); older versions accepted it on
    # Blocks(). Try launch() first, fall back gracefully.
    try:
        ui.queue().launch(theme=gr.themes.Soft(), **launch_kwargs)
    except TypeError:
        ui.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
