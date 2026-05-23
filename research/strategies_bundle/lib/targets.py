"""Targets: sign(mu[t+h] - mu[t]) from Kalman level."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS_SEC


def add_mu_direction_labels(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS_SEC) -> pd.DataFrame:
    mu = df["kalman_mu"].values.astype(np.float64)
    out = df.copy()
    for h in horizons:
        delta = np.full(len(mu), np.nan)
        if h < len(mu):
            delta[:-h] = mu[h:] - mu[:-h]
        col = f"y_mu_sign_{h}s"
        out[col] = np.sign(delta)
        out.loc[out[col] == 0, col] = np.nan
        out[f"y_mu_delta_{h}s"] = delta
    return out


def lgbm_train_mask(
    panel: pd.DataFrame,
    train_mask: np.ndarray,
    ycol: str,
    horizon_sec: int,
) -> np.ndarray:
    """Boolean mask over panel: train rows with label μ[t+h] still inside train."""
    train_idx = np.flatnonzero(train_mask)
    n_tr = len(train_idx)
    m = np.zeros(len(panel), dtype=bool)
    if n_tr == 0:
        return m
    if horizon_sec > 0:
        use_idx = train_idx[: max(0, n_tr - horizon_sec)]
    else:
        use_idx = train_idx
    y = panel[ycol].values
    ok = np.isfinite(y[use_idx]) & (y[use_idx] != 0)
    m[use_idx[ok]] = True
    return m
