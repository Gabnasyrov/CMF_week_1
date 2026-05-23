"""Strategy signal functions (Kalman, ensemble)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_direction_sign(df: pd.DataFrame, horizon_sec: int) -> np.ndarray:
    """Sign(mu[t+h]-mu[t]) — for **backtest** vs label (uses future μ)."""
    mu = df["kalman_mu"].values.astype(np.float64)
    h = horizon_sec
    delta = np.full(len(mu), np.nan)
    if h < len(mu):
        delta[:-h] = mu[h:] - mu[:-h]
    return np.sign(delta)


def kalman_causal_sign(row: pd.Series) -> float:
    """Live signal: sign(velocity) from Kalman state (no lookahead)."""
    v = float(row.get("kalman_vel", 0.0))
    if not np.isfinite(v) or v == 0:
        return 0.0
    return float(np.sign(v))


def ensemble_sign(kalman_delta: np.ndarray, model_sign: np.ndarray, weights: tuple[float, float] = (0.4, 0.6)) -> np.ndarray:
    wk, wm = weights
    s = wk * np.sign(kalman_delta) + wm * model_sign
    return np.sign(s)
