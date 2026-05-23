"""Extended burst features: Hawkes, cross-venue, co-burst, Bayes trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

HAWKES_KAPPA = 0.08  # 1/sec decay
EXTRA_FEATURES = [
    "hawkes_binance_120s",
    "hawkes_bybit_120s",
    "binance_L_30s_pre",
    "bybit_L_30s_pre",
    "cross_signed_liq_60s",
    "concurrent_other_venue",
    "time_since_last_burst_sec",
    "trend_regime_5s",
    "trend_regime_60s",
    "trend_slope_5s",
    "trend_slope_60s",
    "cp_cred_5s",
    "cp_cred_60s",
    "signed_notional_ratio",
]


def _hawkes(ts_us: int, events_us: np.ndarray, window_sec: int = 120) -> float:
    if len(events_us) == 0:
        return 0.0
    t0 = ts_us - window_sec * 1_000_000
    m = (events_us >= t0) & (events_us < ts_us)
    if not m.any():
        return 0.0
    dt = (ts_us - events_us[m]) / 1_000_000.0
    return float(np.exp(-HAWKES_KAPPA * dt).sum())


def _liq_window(
    ts_us: int, liq_ts: np.ndarray, liq_signed: np.ndarray, liq_abs: np.ndarray, window_sec: int
) -> tuple[float, float]:
    t0 = ts_us - window_sec * 1_000_000
    m = (liq_ts >= t0) & (liq_ts < ts_us)
    if not m.any():
        return 0.0, 0.0
    return float(liq_signed[m].sum()), float(liq_abs[m].sum())


def _concurrent_burst(t0: int, t1: int, other: pd.DataFrame) -> int:
    if other.empty:
        return 0
    o0 = other["t_start"].values
    o1 = other["t_end"].values
    return int(np.any((o0 <= t1) & (o1 >= t0)))


def _trend_at(trend_df: pd.DataFrame, ts_us: int, prefix: str) -> dict:
    if trend_df is None or trend_df.empty:
        return {
            f"trend_regime_{prefix}": 0,
            f"trend_slope_{prefix}": 0.0,
            f"cp_cred_{prefix}": 0.5,
        }
    bts = trend_df["ts_us"].values
    idx = int(np.searchsorted(bts, ts_us, side="right") - 1)
    if idx < 0:
        idx = 0
    row = trend_df.iloc[idx]
    return {
        f"trend_regime_{prefix}": int(row.get(f"trend_regime_{prefix}", 0)),
        f"trend_slope_{prefix}": float(row.get(f"trend_slope_{prefix}", 0.0)),
        f"cp_cred_{prefix}": float(row.get(f"cp_cred_{prefix}", 0.5)),
    }


def enrich_burst_row(
    t0: int,
    t1: int,
    signed_notional: float,
    total_notional: float,
    n_events: int,
    venue: str,
    bursts_other: pd.DataFrame,
    liq_b_ts: np.ndarray,
    liq_b_signed: np.ndarray,
    liq_b_abs: np.ndarray,
    liq_y_ts: np.ndarray,
    liq_y_signed: np.ndarray,
    liq_y_abs: np.ndarray,
    last_burst_end: int | None,
    trend_5s: pd.DataFrame | None,
    trend_60s: pd.DataFrame | None,
) -> dict:
    sn_b, ab_b = _liq_window(t0, liq_b_ts, liq_b_signed, liq_b_abs, 60)
    sn_y, ab_y = _liq_window(t0, liq_y_ts, liq_y_signed, liq_y_abs, 60)
    row = {
        "hawkes_binance_120s": _hawkes(t0, liq_b_ts, 120),
        "hawkes_bybit_120s": _hawkes(t0, liq_y_ts, 120),
        "binance_L_30s_pre": _liq_window(t0, liq_b_ts, liq_b_signed, liq_b_abs, 30)[1],
        "bybit_L_30s_pre": _liq_window(t0, liq_y_ts, liq_y_signed, liq_y_abs, 30)[1],
        "cross_signed_liq_60s": sn_b + sn_y,
        "concurrent_other_venue": _concurrent_burst(t0, t1, bursts_other),
        "time_since_last_burst_sec": (t0 - last_burst_end) / 1_000_000 if last_burst_end else 999.0,
        "signed_notional_ratio": signed_notional / (total_notional + 1e-9),
        "n_events_log": float(np.log1p(n_events)),
    }
    row.update(_trend_at(trend_5s, t0, "5s"))
    row.update(_trend_at(trend_60s, t0, "60s"))
    return row


def prepare_liq_arrays(liq: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    liq = liq.sort_values("timestamp")
    notional = liq["price"].values * liq["amount"].values
    signed = np.where(liq["side"].values == "buy", notional, -notional)
    return liq["timestamp"].values.astype(np.int64), signed, notional
