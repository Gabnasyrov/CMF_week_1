"""Load trades / liq for evaluation window."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from config import BYBIT_LATENCY_US, WEIGHT_CAP_USD, bybit_market_paths, paths_for_symbol  # noqa: E402
from utils import load_liquidations, load_parquet_window  # noqa: E402

US_PER_DAY = 86_400_000_000


def day_ranges_us(t0: int, t1: int) -> list[tuple[int, int]]:
    """Inclusive day buckets [d0, d1] in microseconds."""
    d0 = (t0 // US_PER_DAY) * US_PER_DAY
    out = []
    while d0 <= t1:
        d1 = min(d0 + US_PER_DAY - 1, t1)
        if d1 >= t0:
            out.append((max(d0, t0), d1))
        d0 += US_PER_DAY
    return out


def load_binance_trades(sym: str, t0: int, t1: int, subsample: int = 1) -> pd.DataFrame:
    p = paths_for_symbol(sym)
    cols = ["timestamp", "side", "price", "amount"]
    df = load_parquet_window(p["trades"], t0, t1, columns=cols)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    if subsample > 1:
        df = df.iloc[::subsample].reset_index(drop=True)
    df["notional"] = df["price"] * df["amount"]
    df["w"] = df["notional"].clip(upper=WEIGHT_CAP_USD)
    df["side_code"] = np.where(df["side"].values == "buy", np.int8(1), np.int8(-1))
    df["venue"] = "binance"
    return df[["timestamp", "side_code", "price", "amount", "notional", "w", "venue"]]


def load_bybit_liq_as_trades(sym: str, t0: int, t1: int, subsample: int = 1) -> pd.DataFrame:
    p = paths_for_symbol(sym)
    df = load_parquet_window(
        p["liq_bybit"],
        t0 - BYBIT_LATENCY_US,
        t1 - BYBIT_LATENCY_US,
        columns=["timestamp", "side", "price", "amount"],
    )
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = df["timestamp"] + BYBIT_LATENCY_US
    df = df.sort_values("timestamp").reset_index(drop=True)
    if subsample > 1:
        df = df.iloc[::subsample].reset_index(drop=True)
    df["notional"] = df["price"] * df["amount"]
    side_code = np.where(df["side"].values == "buy", np.int8(1), np.int8(-1))
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"].values,
            "side_code": side_code,
            "price": df["price"].values,
            "amount": df["amount"].values,
            "notional": df["notional"].values,
            "w": df["notional"].clip(upper=WEIGHT_CAP_USD).values,
            "venue": "bybit",
        }
    )


def load_bybit_trades(sym: str, t0: int, t1: int, subsample: int = 1) -> pd.DataFrame:
    p = bybit_market_paths(sym)
    cols = ["timestamp", "side", "price", "amount"]
    df = load_parquet_window(p["trades"], t0, t1, columns=cols)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    if subsample > 1:
        df = df.iloc[::subsample].reset_index(drop=True)
    df["notional"] = df["price"] * df["amount"]
    df["w"] = df["notional"].clip(upper=WEIGHT_CAP_USD)
    df["side_code"] = np.where(df["side"].values == "buy", np.int8(1), np.int8(-1))
    df["venue"] = "bybit"
    return df[["timestamp", "side_code", "price", "amount", "notional", "w", "venue"]]


def load_bbo_mid(sym: str, t0: int, t1: int, tau_pad_sec: int = 300) -> pd.DataFrame:
    p = paths_for_symbol(sym)
    bbo = load_parquet_window(
        p["bbo"],
        t0,
        t1 + tau_pad_sec * 1_000_000,
        columns=["timestamp", "bid_price", "ask_price"],
    )
    if bbo.empty:
        return bbo
    bbo = bbo.sort_values("timestamp").reset_index(drop=True)
    mid = (bbo["bid_price"] + bbo["ask_price"]) / 2.0
    return pd.DataFrame({"timestamp": bbo["timestamp"].values, "mid": mid.values})


def load_bybit_bbo_mid(sym: str, t0: int, t1: int, tau_pad_sec: int = 300) -> pd.DataFrame:
    p = bybit_market_paths(sym)
    bbo = load_parquet_window(
        p["bbo"],
        t0,
        t1 + tau_pad_sec * 1_000_000,
        columns=["timestamp", "bid_price", "ask_price"],
    )
    if bbo.empty:
        return bbo
    bbo = bbo.sort_values("timestamp").reset_index(drop=True)
    mid = (bbo["bid_price"] + bbo["ask_price"]) / 2.0
    return pd.DataFrame({"timestamp": bbo["timestamp"].values, "mid": mid.values})
