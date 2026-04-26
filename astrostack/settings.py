"""Persistent settings for the web UI.

Stores the last-used run options at ``~/.config/astrostack/settings.json``
and the saved adjustment presets at ``~/.config/astrostack/presets.json``.
Both files are pure JSON; safe to edit by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "astrostack"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
PRESETS_FILE = CONFIG_DIR / "presets.json"

DEFAULT_SETTINGS: dict = {
    "stack_method": "sigma",
    "sigma": 3.0,
    "sigma_iters": 3,
    "do_align": True,
    "do_enhance": True,
    "enhance_model": "realesrgan-x2",
    "device": "auto",
    "bit_depth": "16",
    "output_format": "TIF",
}


def _load_json(path: Path, default: dict) -> dict:
    try:
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                merged = dict(default)
                merged.update(data)
                return merged
    except Exception as e:
        log.warning("Failed to load %s: %s — using defaults.", path, e)
    return dict(default)


def _save_json(path: Path, data: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception as e:
        log.warning("Failed to save %s: %s", path, e)


def load_settings() -> dict:
    return _load_json(SETTINGS_FILE, DEFAULT_SETTINGS)


def save_settings(values: dict) -> None:
    """Save only the keys we know about to avoid junk accumulation."""
    out = {k: values.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
    _save_json(SETTINGS_FILE, out)


def load_user_presets() -> dict[str, dict]:
    """User-saved editor presets keyed by name.

    Defends against hand-edited JSON: drops any preset whose values aren't
    all numeric, since they're applied via float() to slider state.
    """
    data = _load_json(PRESETS_FILE, {})
    out: dict[str, dict] = {}
    for name, payload in data.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        clean: dict = {}
        ok = True
        for key, val in payload.items():
            if not isinstance(key, str):
                ok = False
                break
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                # bool is a subclass of int — exclude it explicitly.
                ok = False
                break
            clean[key] = float(val)
        if ok:
            out[name] = clean
        else:
            log.warning("Skipping malformed preset %r in %s", name, PRESETS_FILE)
    return out


def save_user_preset(name: str, values: dict) -> dict[str, dict]:
    """Add or overwrite a preset; returns the updated dict."""
    if not name or not name.strip():
        return load_user_presets()
    presets = load_user_presets()
    presets[name.strip()] = {k: float(v) for k, v in values.items()
                             if isinstance(v, (int, float))}
    _save_json(PRESETS_FILE, presets)
    return presets


def delete_user_preset(name: str) -> dict[str, dict]:
    presets = load_user_presets()
    presets.pop(name, None)
    _save_json(PRESETS_FILE, presets)
    return presets
