"""Shared train/test split and panel prep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_FEATURES, FEATURES_NO_KALMAN, MAX_PANEL_ROWS, TRAIN_FRAC
from .feature_builder import build_panel
from .io_config import load_symbol_config
from .normalization import ZScoreNormalizer
from .targets import add_mu_direction_labels


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def subsample_panel(df: pd.DataFrame, max_rows: int = MAX_PANEL_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    stride = max(1, len(df) // max_rows)
    return df.iloc[::stride].reset_index(drop=True)


def prepare_panel(
    sym: str,
    max_days: int | None = None,
    subsample: bool = True,
    skip_hawkes: bool = False,
) -> pd.DataFrame:
    cfg = load_symbol_config(sym)
    k = cfg["kalman"]
    panel = build_panel(
        sym,
        max_days=max_days or cfg.get("max_days"),
        skip_hawkes=skip_hawkes,
        **k,
    )
    panel = add_mu_direction_labels(panel)
    if subsample:
        panel = subsample_panel(panel)
    return panel


def fit_normalizers(train: pd.DataFrame) -> tuple[ZScoreNormalizer, ZScoreNormalizer]:
    norm_full = ZScoreNormalizer(ALL_FEATURES).fit(train)
    norm_nk = ZScoreNormalizer(FEATURES_NO_KALMAN).fit(train)
    return norm_full, norm_nk


def transform_panel(df: pd.DataFrame, norm_full: ZScoreNormalizer, norm_nk: ZScoreNormalizer):
    zf = norm_full.transform(df)
    zn = norm_nk.transform(df)
    return zf, zn
