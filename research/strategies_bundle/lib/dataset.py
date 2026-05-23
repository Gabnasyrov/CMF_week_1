"""Panel load, split, subsample."""

from __future__ import annotations

import pandas as pd

from .config import MAX_PANEL_ROWS, TRAIN_FRAC
from .feature_builder import build_panel
from .io_config import load_symbol_config
from .targets import add_mu_direction_labels


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def subsample_panel(df: pd.DataFrame, max_rows: int = MAX_PANEL_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    stride = max(1, len(df) // max_rows)
    return df.iloc[::stride].reset_index(drop=True)


def build_labeled_panel(sym: str, max_days: int | None = None) -> pd.DataFrame:
    cfg = load_symbol_config(sym)
    k = cfg["kalman"]
    panel = build_panel(sym, max_days=max_days or cfg.get("max_days"), **k)
    panel = add_mu_direction_labels(panel)
    return subsample_panel(panel)
