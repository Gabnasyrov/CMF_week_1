"""Markout PnL and aggregate metrics per description.md."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAKER_REBATE_BPS, MIN_TURNOVER_PER_DAY


def maker_pnl_bps(price: np.ndarray, mid_tau: np.ndarray, taker_side: np.ndarray) -> np.ndarray:
    if taker_side.dtype.kind in ("i", "u", "f"):
        s = taker_side.astype(np.float64)
    else:
        s = np.where(taker_side == "buy", 1.0, -1.0)
    pnl = np.full(len(price), np.nan)
    m = np.isfinite(mid_tau) & np.isfinite(price)
    pnl[m] = -s[m] * (mid_tau[m] - price[m]) / price[m] * 10_000 + MAKER_REBATE_BPS
    return pnl


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def win_rate(pnl: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.isfinite(pnl) & (w > 0)
    if mask is not None:
        m &= mask
    if not m.any():
        return np.nan
    return float((pnl[m] > 0).mean())


def max_drawdown_bps(pnl: np.ndarray, w: np.ndarray, ts: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.isfinite(pnl) & (w > 0) & np.isfinite(ts)
    if mask is not None:
        m &= mask
    if m.sum() < 10:
        return np.nan
    order = np.argsort(ts[m])
    p = pnl[m][order]
    ww = w[m][order]
    cw = np.cumsum(ww)
    cwp = np.cumsum(p * ww)
    avg = cwp / np.maximum(cw, 1e-9)
    peak = np.maximum.accumulate(avg)
    return float(np.max(peak - avg))


def n_days(ts_us: np.ndarray) -> float:
    if len(ts_us) == 0:
        return 1.0
    span = (ts_us.max() - ts_us.min()) / 86_400_000_000
    return max(span, 1.0)


def evaluate_filter(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    ts: np.ndarray,
) -> dict:
    """PnL_all, PnL_kept, PnL_filtered, Score, turnover, constraint flag."""
    m = np.isfinite(pnl) & (w > 0)
    days = n_days(ts[m])

    pnl_all = weighted_mean(pnl, w)
    kept = m & (f == 0)
    filt = m & (f == 1)

    pnl_kept = weighted_mean(pnl, w * kept.astype(float))
    pnl_filtered = weighted_mean(pnl, w * filt.astype(float))
    turnover_kept_day = float(w[kept].sum() / days) if kept.any() else 0.0
    turnover_all_day = float(w[m].sum() / days) if m.any() else 0.0

    return {
        "pnl_all_bps": pnl_all,
        "pnl_kept_bps": pnl_kept,
        "pnl_filtered_bps": pnl_filtered,
        "score_bps": pnl_kept - pnl_all if np.isfinite(pnl_kept) and np.isfinite(pnl_all) else np.nan,
        "winrate_kept": win_rate(pnl, w, kept),
        "winrate_all": win_rate(pnl, w, m),
        "maxdd_kept_bps": max_drawdown_bps(pnl, w, ts, kept),
        "turnover_kept_usd_day": turnover_kept_day,
        "turnover_all_usd_day": turnover_all_day,
        "n_trades": int(m.sum()),
        "n_kept": int(kept.sum()),
        "n_filtered": int(filt.sum()),
        "kept_frac": float(kept.sum() / max(m.sum(), 1)),
        "meets_turnover_constraint": turnover_kept_day >= MIN_TURNOVER_PER_DAY,
        "n_days": days,
    }


def mid_at_horizon(trade_ts: np.ndarray, bbo_ts: np.ndarray, bbo_mid: np.ndarray, tau_us: int) -> np.ndarray:
    target = trade_ts + tau_us
    idx = np.searchsorted(bbo_ts, target, side="right") - 1
    out = np.full(len(trade_ts), np.nan)
    valid = idx >= 0
    out[valid] = bbo_mid[idx[valid]]
    if len(bbo_ts) > 0:
        out[target > bbo_ts[-1]] = np.nan
    return out
