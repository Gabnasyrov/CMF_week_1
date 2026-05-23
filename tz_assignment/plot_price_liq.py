#!/usr/bin/env python3
"""Price (Binance mid) with Binance + Bybit liquidations overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BYBIT_LATENCY_US, DAY_US, FIGURES, SAMPLE_DAYS, dt_to_us, paths_for_symbol
from utils import load_parquet_window


def plot_day(sym: str, day, out_name: str | None = None) -> Path:
    p = paths_for_symbol(sym)
    t0 = dt_to_us(day)
    t1 = t0 + DAY_US - 1

    bbo = load_parquet_window(
        p["bbo"], t0, t1, columns=["timestamp", "bid_price", "ask_price"]
    )
    if bbo.empty:
        raise RuntimeError(f"No BBO in window {day} for {sym}")

    bbo["mid"] = (bbo["bid_price"] + bbo["ask_price"]) / 2
    bbo["ts"] = pd.to_datetime(bbo["timestamp"], unit="us", utc=True)
    mid_1s = bbo.set_index("ts")["mid"].resample("1s").last().ffill()

    liq_b = load_parquet_window(
        p["liq_binance"], t0, t1, columns=["timestamp", "side", "price", "amount"]
    )
    liq_y = load_parquet_window(
        p["liq_bybit"], t0, t1, columns=["timestamp", "side", "price", "amount"]
    )
    if not liq_y.empty:
        liq_y = liq_y.copy()
        liq_y["timestamp"] = liq_y["timestamp"] + BYBIT_LATENCY_US

    fig, ax = plt.subplots(figsize=(14, 5))
    mid_1s.plot(ax=ax, color="#444444", lw=0.7, label="Binance mid (1s)")

    for liq, prefix, marker in [
        (liq_b, "Binance", "o"),
        (liq_y, "Bybit (+200ms)", "x"),
    ]:
        if liq.empty:
            continue
        liq = liq.copy()
        liq["ts"] = pd.to_datetime(liq["timestamp"], unit="us", utc=True)
        liq["notional"] = liq["price"] * liq["amount"]
        sz = np.clip(liq["notional"].values / 5000.0, 8, 150)
        buy = liq["side"] == "buy"
        sell = liq["side"] == "sell"
        ax.scatter(
            liq.loc[buy, "ts"].to_numpy(),
            liq.loc[buy, "price"].to_numpy(),
            s=sz[buy],
            c="#2ca02c",
            marker=marker,
            alpha=0.55,
            linewidths=0.4,
            label=f"{prefix} liq buy",
        )
        ax.scatter(
            liq.loc[sell, "ts"].to_numpy(),
            liq.loc[sell, "price"].to_numpy(),
            s=sz[sell],
            c="#d62728",
            marker=marker,
            alpha=0.55,
            linewidths=0.4,
            label=f"{prefix} liq sell",
        )

    ax.set_title(f"{sym.upper()} — price & liquidations ({day.date()} UTC)")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("price (USDT)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    name = out_name or f"{sym}_price_liq_{day.date()}.png"
    out = FIGURES / name
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} | BBO pts={len(bbo):,} liq B={len(liq_b):,} Y={len(liq_y):,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ethusdt", choices=["btcusdt", "ethusdt"])
    ap.add_argument("--day", default=None, help="YYYY-MM-DD (default: first SAMPLE_DAYS)")
    args = ap.parse_args()
    day = pd.Timestamp(args.day, tz="UTC").to_pydatetime() if args.day else SAMPLE_DAYS[0]
    plot_day(args.symbol, day)


if __name__ == "__main__":
    main()
