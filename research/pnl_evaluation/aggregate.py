"""Online aggregation of markout metrics across chunks."""

from __future__ import annotations

import numpy as np


def empty_state() -> dict:
    return {
        "sum_w": 0.0,
        "sum_wp": 0.0,
        "sum_w_kept": 0.0,
        "sum_wp_kept": 0.0,
        "sum_w_filt": 0.0,
        "sum_wp_filt": 0.0,
        "n": 0,
        "n_kept": 0,
        "n_filt": 0,
        "n_win_kept": 0,
        "day_pnl_kept": [],
        "prev_w_kept": 0.0,
        "prev_wp_kept": 0.0,
        "ts_min": None,
        "ts_max": None,
    }


def update_state(state: dict, pnl: np.ndarray, w: np.ndarray, f: np.ndarray, ts: np.ndarray) -> dict:
    m = np.isfinite(pnl) & (w > 0)
    if not m.any():
        return state

    p, ww, ff, tt = pnl[m], w[m], f[m].astype(np.int8), ts[m]
    kept = ff == 0
    filt = ff == 1

    state["sum_w"] += float(ww.sum())
    state["sum_wp"] += float((ww * p).sum())
    state["n"] += int(len(p))

    if kept.any():
        wk, pk = ww[kept], p[kept]
        state["sum_w_kept"] += float(wk.sum())
        state["sum_wp_kept"] += float((wk * pk).sum())
        state["n_kept"] += int(kept.sum())
        state["n_win_kept"] += int((pk > 0).sum())

    if filt.any():
        wf, pf = ww[filt], p[filt]
        state["sum_w_filt"] += float(wf.sum())
        state["sum_wp_filt"] += float((wf * pf).sum())
        state["n_filt"] += int(filt.sum())

    tmin, tmax = int(tt.min()), int(tt.max())
    state["ts_min"] = tmin if state["ts_min"] is None else min(state["ts_min"], tmin)
    state["ts_max"] = tmax if state["ts_max"] is None else max(state["ts_max"], tmax)
    return state


def flush_day(state: dict) -> dict:
    """Close a calendar day: append daily weighted PnL for maxDD."""
    dw = state["sum_w_kept"] - state["prev_w_kept"]
    dwp = state["sum_wp_kept"] - state["prev_wp_kept"]
    if dw > 0:
        state["day_pnl_kept"].append(float(dwp / dw))
    state["prev_w_kept"] = state["sum_w_kept"]
    state["prev_wp_kept"] = state["sum_wp_kept"]
    return state


def _maxdd_from_series(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    cum = np.cumsum(vals)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def state_to_metrics(state: dict) -> dict:
    from .markout_metrics import MIN_TURNOVER_PER_DAY

    def wm(sw, swp):
        return float(swp / sw) if sw > 0 else np.nan

    pnl_all = wm(state["sum_w"], state["sum_wp"])
    pnl_kept = wm(state["sum_w_kept"], state["sum_wp_kept"])
    pnl_filt = wm(state["sum_w_filt"], state["sum_wp_filt"]) if state["sum_w_filt"] > 0 else np.nan

    if state["ts_min"] is not None and state["ts_max"] is not None:
        days = max((state["ts_max"] - state["ts_min"]) / 86_400_000_000, 1.0)
    else:
        days = 1.0

    turnover_kept = state["sum_w_kept"] / days
    return {
        "pnl_all_bps": pnl_all,
        "pnl_kept_bps": pnl_kept,
        "pnl_filtered_bps": pnl_filt,
        "score_bps": pnl_kept - pnl_all if np.isfinite(pnl_kept) and np.isfinite(pnl_all) else np.nan,
        "winrate_kept": state["n_win_kept"] / state["n_kept"] if state["n_kept"] > 0 else np.nan,
        "winrate_all": np.nan,
        "maxdd_kept_bps": _maxdd_from_series(state["day_pnl_kept"]),
        "turnover_kept_usd_day": turnover_kept,
        "turnover_all_usd_day": state["sum_w"] / days,
        "n_trades": state["n"],
        "n_kept": state["n_kept"],
        "n_filtered": state["n_filt"],
        "kept_frac": state["n_kept"] / max(state["n"], 1),
        "meets_turnover_constraint": turnover_kept >= MIN_TURNOVER_PER_DAY,
        "n_days": days,
    }
