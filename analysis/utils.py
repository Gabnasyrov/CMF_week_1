"""Helpers for liquidation_task EDA (chunk-friendly)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import BYBIT_LATENCY_US, MAKER_REBATE_BPS, WEIGHT_CAP_USD


def load_parquet_window(path, t0_us: int, t1_us: int, columns=None) -> pd.DataFrame:
    """Load rows with timestamp in [t0_us, t1_us] via row-group filtering."""
    pf = pq.ParquetFile(path)
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=columns)
        if "timestamp" not in table.column_names:
            chunks.append(table.to_pandas())
            continue
        ts = table.column("timestamp")
        mn = pc.min(ts).as_py()
        mx = pc.max(ts).as_py()
        if mx < t0_us or mn > t1_us:
            continue
        mask = pc.and_(pc.greater_equal(ts, t0_us), pc.less_equal(ts, t1_us))
        chunks.append(table.filter(mask).to_pandas())
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_liquidations(path, shift_us: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if shift_us:
        df = df.copy()
        df["timestamp"] = df["timestamp"] + shift_us
    df["notional"] = df["price"] * df["amount"]
    df["signed_usd"] = np.where(df["side"] == "buy", df["notional"], -df["notional"])
    return df.sort_values("timestamp").reset_index(drop=True)


def liq_to_bars(liq: pd.DataFrame, freq_ms: int) -> pd.DataFrame:
    """Resample liquidations to regular bars (sum signed/abs USD)."""
    if liq.empty:
        return pd.DataFrame()
    x = liq.copy()
    x.index = pd.to_datetime(x["timestamp"], unit="us", utc=True)
    rule = f"{freq_ms}ms"
    g = x.resample(rule, label="right", closed="right")
    out = pd.DataFrame(
        {
            "L_net": g["signed_usd"].sum(),
            "L_abs": g["notional"].sum(),
            "n_events": g["timestamp"].count(),
        }
    )
    buy = x.loc[x["side"] == "buy", "notional"].resample(rule, label="right", closed="right").sum()
    out["L_buy"] = buy.reindex(out.index).fillna(0.0)
    out["L_sell"] = out["L_abs"] - out["L_buy"]
    # DatetimeIndex from unit='us' stores asi8 in microseconds
    out["ts_us"] = out.index.asi8
    return out.reset_index(drop=True)


def bbo_mid_spread(bbo: pd.DataFrame) -> pd.DataFrame:
    bbo = bbo.sort_values("timestamp").reset_index(drop=True)
    mid = (bbo["bid_price"] + bbo["ask_price"]) / 2.0
    spread_bps = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
    return pd.DataFrame(
        {"timestamp": bbo["timestamp"].values, "mid": mid.values, "spread_bps": spread_bps.values}
    )


def mid_at_horizon(trade_ts: np.ndarray, bbo_ts: np.ndarray, bbo_mid: np.ndarray, tau_us: int) -> np.ndarray:
    """Forward-fill mid at trade_ts + tau_us."""
    target = trade_ts + tau_us
    idx = np.searchsorted(bbo_ts, target, side="right") - 1
    out = np.full(len(trade_ts), np.nan)
    valid = idx >= 0
    out[valid] = bbo_mid[idx[valid]]
    return out


def trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["notional"] = t["price"] * t["amount"]
    t["w"] = t["notional"].clip(upper=WEIGHT_CAP_USD)
    t["s"] = np.where(t["side"] == "buy", 1.0, -1.0)
    t["maker_side"] = np.where(t["side"] == "buy", "sell", "buy")
    return t


def compute_pnl(trades: pd.DataFrame, mid_tau: np.ndarray) -> np.ndarray:
    valid = np.isfinite(mid_tau)
    pnl = np.full(len(trades), np.nan)
    p = trades["price"].values
    s = np.where(trades["side"] == "buy", 1.0, -1.0)
    pnl[valid] = -s[valid] * (mid_tau[valid] - p[valid]) / p[valid] * 10_000 + MAKER_REBATE_BPS
    return pnl


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def decile_table(signal: np.ndarray, pnl: np.ndarray, w: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    m = np.isfinite(signal) & np.isfinite(pnl) & (w > 0)
    s, p, ww = signal[m], pnl[m], w[m]
    if len(s) < n_bins * 10:
        return pd.DataFrame()
    ranks = pd.qcut(s, n_bins, labels=False, duplicates="drop")
    rows = []
    valid_bins = ranks[np.isfinite(ranks)]
    for d in sorted(np.unique(valid_bins)):
        mask = ranks == d
        rows.append(
            {
                "decile": int(d),
                "pnl_bps": weighted_mean(p[mask], ww[mask]),
                "weight_sum": float(ww[mask].sum()),
                "n": int(mask.sum()),
                "signal_mean": float(s[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def lag_correlation(x_ts: np.ndarray, x_val: np.ndarray, y_ts: np.ndarray, y_val: np.ndarray, lags_sec: tuple) -> pd.DataFrame:
    """Corr(x[t], y[t+lag]) with 1s bins (x,y pre-aggregated to same grid)."""
    rows = []
    for lag in lags_sec:
        shift_us = lag * 1_000_000
        y_shifted_ts = y_ts + shift_us
        # align: interpolate y onto x_ts + shift
        yi = np.interp(x_ts, y_shifted_ts, y_val, left=np.nan, right=np.nan)
        m = np.isfinite(x_val) & np.isfinite(yi)
        if m.sum() < 30:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x_val[m], yi[m])[0, 1])
        rows.append({"lag_sec": lag, "corr": corr})
    return pd.DataFrame(rows)


def asof_join_signal(trades: pd.DataFrame, bars: pd.DataFrame, col: str) -> np.ndarray:
    """Last bar value at or before trade timestamp."""
    if bars.empty or trades.empty:
        return np.full(len(trades), np.nan)
    b_ts = bars["ts_us"].values.astype(np.int64)
    b_val = bars[col].values.astype(np.float64)
    t_ts = trades["timestamp"].values
    idx = np.searchsorted(b_ts, t_ts, side="right") - 1
    out = np.full(len(t_ts), np.nan)
    ok = idx >= 0
    out[ok] = b_val[idx[ok]]
    return out
