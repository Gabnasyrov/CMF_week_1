"""Load per-symbol strategy config."""

from __future__ import annotations

import json
from pathlib import Path

from .config import CONFIG_DIR


def load_symbol_config(sym: str) -> dict:
    path = CONFIG_DIR / f"{sym}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())
