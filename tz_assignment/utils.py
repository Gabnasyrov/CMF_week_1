"""Chunk-friendly parquet helpers for TZ EDA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import BYBIT_LATENCY_US, DAY_US, SAMPLE_DAYS, dt_to_us


def load_parquet_window(path: Path, t0_us: int, t1_us: int, columns=None) -> pd.DataFrame:
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
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def sample_parquet(path: Path, n: int, seed: int = 0, columns=None) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    parts = []
    remaining = n
    rgs = list(range(pf.metadata.num_row_groups))
    rng.shuffle(rgs)
    for rg in rgs:
        if remaining <= 0:
            break
        table = pf.read_row_group(rg, columns=columns)
        df = table.to_pandas()
        if len(df) <= remaining:
            parts.append(df)
            remaining -= len(df)
        else:
            idx = rng.choice(len(df), size=remaining, replace=False)
            parts.append(df.iloc[np.sort(idx)])
            remaining = 0
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def parquet_quality_row(path: Path, source: str, symbol: str) -> dict:
    if not path.exists():
        return {"source": source, "symbol": symbol, "exists": False}
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    n = pf.metadata.num_rows
    t0 = pc.min(pf.read(columns=["timestamp"]).column("timestamp")).as_py()
    t1 = pc.max(pf.read(columns=["timestamp"]).column("timestamp")).as_py()
    mono_violations = 0
    dup_ts = 0
    null_counts = {c: 0 for c in cols}
    prev = -1
    for rg in range(min(pf.metadata.num_row_groups, 80)):
        table = pf.read_row_group(rg)
        if "timestamp" in table.column_names:
            ts = table.column("timestamp").to_numpy()
            if len(ts):
                mono_violations += int(np.sum(ts[1:] < ts[:-1]))
                _, cnt = np.unique(ts, return_counts=True)
                dup_ts += int((cnt > 1).sum())
                if ts[0] < prev:
                    mono_violations += 1
                prev = int(ts[-1])
        for c in cols:
            col = table.column(c)
            null_counts[c] += int(col.null_count) if hasattr(col, "null_count") else 0
    days = max(1, (t1 - t0) // DAY_US + 1)
    return {
        "source": source,
        "symbol": symbol,
        "exists": True,
        "rows": n,
        "t0_us": t0,
        "t1_us": t1,
        "days_span": days,
        "rows_per_day": n / days,
        "mono_violations_rg": mono_violations,
        "duplicate_ts_buckets_rg": dup_ts,
        **{f"null_{c}": null_counts[c] for c in cols},
    }


def load_liq(sym: str, venue: str, paths: dict) -> pd.DataFrame:
    if venue == "binance":
        path = paths["liq_binance"]
        shift = 0
    else:
        path = paths["liq_bybit"]
        shift = BYBIT_LATENCY_US
    df = pd.read_parquet(path)
    if shift:
        df = df.copy()
        df["timestamp"] = df["timestamp"] + shift
    df["notional"] = df["price"] * df["amount"]
    df["venue"] = venue
    df["hour_utc"] = pd.to_datetime(df["timestamp"], unit="us", utc=True).dt.hour
    df["dow"] = pd.to_datetime(df["timestamp"], unit="us", utc=True).dt.dayofweek
    return df


def sample_day_bounds(day) -> tuple[int, int]:
    t0 = dt_to_us(day.replace(hour=0, minute=0, second=0, microsecond=0))
    return t0, t0 + DAY_US - 1


def bbo_mid_frame(bbo: pd.DataFrame) -> pd.DataFrame:
    bbo = bbo.sort_values("timestamp")
    mid = (bbo["bid_price"] + bbo["ask_price"]) / 2.0
    spread_bps = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
    return pd.DataFrame(
        {
            "timestamp": bbo["timestamp"].values,
            "mid": mid.values,
            "spread_bps": spread_bps.values,
        }
    )


def resample_mid_1s(bbo_mid: pd.DataFrame) -> pd.Series:
    x = bbo_mid.copy()
    x.index = pd.to_datetime(x["timestamp"], unit="us", utc=True)
    return x["mid"].resample("1s", label="right", closed="right").last().dropna()


def lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> pd.DataFrame:
    """Pearson corr(a[t], b[t+lag]) for integer second lags."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[: n - lag], b[lag:n]
        else:
            lag = -lag
            x, y = a[lag:n], b[: n - lag]
        if len(x) < 50:
            continue
        c = np.corrcoef(x, y)[0, 1]
        rows.append({"lag_sec": lag if lag >= 0 else -lag, "corr": c})
    return pd.DataFrame(rows)
