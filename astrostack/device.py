"""Device selection with GPU detection and CPU fallback."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def select_device(prefer: str = "auto") -> str:
    """Return the torch device string to use.

    prefer: "auto" | "cuda" | "mps" | "cpu"
    """
    try:
        import torch
    except ImportError:
        log.warning("PyTorch not installed; falling back to CPU-only mode.")
        return "cpu"

    if prefer == "cpu":
        return "cpu"

    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"

    if prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return "mps"

    if prefer not in ("auto", "cpu"):
        log.warning("Requested device %r unavailable; falling back to CPU.", prefer)
    return "cpu"
