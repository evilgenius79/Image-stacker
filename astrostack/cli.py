"""Command-line interface."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from .io import SUPPORTED_EXTS, is_supported
from .pipeline import run_pipeline


def _expand(paths: tuple[str, ...]) -> list[Path]:
    """Expand a list of file or directory paths into supported image files."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and is_supported(child):
                    out.append(child)
        elif p.is_file() and is_supported(p):
            out.append(p)
        elif p.is_file():
            click.echo(f"Skipping unsupported file: {p}", err=True)
        else:
            click.echo(f"Path not found: {p}", err=True)
    return out


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, required=True, type=click.Path())
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output image path (extension picks format).")
@click.option("--darks", multiple=True, type=click.Path(),
              help="Dark frame file(s) or directory.")
@click.option("--bias", multiple=True, type=click.Path(),
              help="Bias frame file(s) or directory.")
@click.option("--flats", multiple=True, type=click.Path(),
              help="Flat frame file(s) or directory.")
@click.option("--method", "stack_method",
              type=click.Choice(["mean", "median", "sigma"]),
              default="sigma", show_default=True,
              help="Stacking algorithm.")
@click.option("--sigma", type=float, default=3.0, show_default=True,
              help="Sigma threshold for sigma-clip stacking.")
@click.option("--sigma-iters", type=int, default=3, show_default=True,
              help="Iterations for sigma-clip stacking.")
@click.option("--no-align", is_flag=True, help="Skip star alignment.")
@click.option("--no-enhance", is_flag=True, help="Skip AI enhancement.")
@click.option("--enhance-model",
              type=click.Choice(["realesrgan-x2", "realesrgan-x4"]),
              default="realesrgan-x2", show_default=True,
              help="AI enhancement model (downloaded on first use).")
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "cpu"]),
              default="auto", show_default=True,
              help="Compute device for AI enhancement.")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="16",
              show_default=True, help="Output bit depth (PNG/TIFF).")
@click.option("--save-intermediate", is_flag=True,
              help="Also save the pre-enhancement stack.")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity.")
def main(
    inputs, output, darks, bias, flats,
    stack_method, sigma, sigma_iters,
    no_align, no_enhance, enhance_model, device,
    bit_depth, save_intermediate, verbose,
):
    """Stack and AI-enhance astronomy images.

    INPUTS may be individual files or directories. Supported formats:
    JPEG, PNG, BMP, WebP, TIFF, FITS, and camera RAW (CR2/NEF/ARW/DNG/...).
    """
    level = logging.WARNING - 10 * min(verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    light_paths = _expand(inputs)
    if not light_paths:
        click.echo("No supported input images found. Supported extensions: "
                   + ", ".join(sorted(SUPPORTED_EXTS)), err=True)
        sys.exit(1)

    dark_paths = _expand(darks) or None
    bias_paths = _expand(bias) or None
    flat_paths = _expand(flats) or None

    click.echo(f"Lights: {len(light_paths)} | Darks: {len(dark_paths or [])} | "
               f"Bias: {len(bias_paths or [])} | Flats: {len(flat_paths or [])}")

    run_pipeline(
        light_paths=light_paths,
        output=Path(output),
        darks=dark_paths,
        bias=bias_paths,
        flats=flat_paths,
        stack_method=stack_method,
        sigma=sigma,
        sigma_iters=sigma_iters,
        align=not no_align,
        enhance=not no_enhance,
        enhance_model=enhance_model,
        device=device,
        bit_depth=int(bit_depth),
        save_intermediate=save_intermediate,
    )


if __name__ == "__main__":
    main()
