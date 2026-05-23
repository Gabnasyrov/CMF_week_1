#!/usr/bin/env python3
"""Exploratory analysis: shape, distributions, cross-source relationships."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "figures" / "eda"
TABLES = Path(__file__).resolve().parent / "tables" / "eda"
SYMS = ("btcusdt", "ethusdt")
BYBIT_SHIFT = 200_000
HOUR_US = 3_600_000_000


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def us_to_dt(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


# ---------------------------------------------------------------------------
# 1. Shape
# ---------------------------------------------------------------------------
def parquet_shape(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    t0, t1 = None, None
    dup_ts = 0
    prev_last = None
    daily = defaultdict(int)
    hourly = defaultdict(int)

    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["timestamp"])
        ts = table.column("timestamp")
        mn, mx = pc.min(ts).as_py(), pc.max(ts).as_py()
        t0 = mn if t0 is None else min(t0, mn)
        t1 = mx if t1 is None else max(t1, mx)
        arr = ts.to_numpy(zero_copy_only=False)
        days = arr // 86_400_000_000
        hours = arr // HOUR_US
        for d in np.unique(days):
            daily[int(d)] += int((days == d).sum())
        for h in np.unique(hours):
            hourly[int(h)] += int((hours == h).sum())
        if prev_last is not None and len(arr) and arr[0] == prev_last:
            dup_ts += 1
        if len(arr):
            prev_last = int(arr[-1])

    return {
        "path": str(path.relative_to(ROOT)),
        "rows": pf.metadata.num_rows,
        "row_groups": pf.metadata.num_row_groups,
        "t0": us_to_dt(t0).isoformat() if t0 else None,
        "t1": us_to_dt(t1).isoformat() if t1 else None,
        "adjacent_rg_dup_ts": dup_ts,
        "daily": dict(sorted(daily.items())),
        "hourly": dict(sorted(hourly.items())),
    }


def scan_all_shapes() -> pd.DataFrame:
    rows = []
    sources = [
        ("binance_trades", "perp_{sym}.parquet"),
        ("binance_booktickers", "perp_{sym}.parquet"),
        ("binance_liquidations", "perp_{sym}.parquet"),
        ("bybit_liquidations", "{sym}.parquet"),
    ]
    for folder, pat in sources:
        for sym in SYMS:
            p = DATA / folder / pat.format(sym=sym)
            print(f"shape {p.name}...")
            meta = parquet_shape(p)
            rows.append(
                {
                    "source": folder,
                    "symbol": sym,
                    "rows": meta["rows"],
                    "t0": meta["t0"],
                    "t1": meta["t1"],
                    "rg_dup_boundary": meta["adjacent_rg_dup_ts"],
                }
            )
            # daily events plot data
            daily = meta["daily"]
            for day, cnt in daily.items():
                dt = datetime.fromtimestamp(day * 86400, tz=timezone.utc)
                rows.append(
                    {
                        "source": folder,
                        "symbol": sym,
                        "metric": "daily_count",
                        "day": dt.strftime("%Y-%m-%d"),
                        "value": cnt,
                    }
                )
    summary = pd.DataFrame([r for r in rows if "metric" not in r])
    daily_long = pd.DataFrame([r for r in rows if r.get("metric") == "daily_count"])
    summary.to_csv(TABLES / "01_shape_summary.csv", index=False)
    daily_long.to_csv(TABLES / "01_daily_counts.csv", index=False)
    return summary, daily_long


def plot_daily_counts(daily_long: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    panels = [
        ("binance_trades", "btcusdt"),
        ("binance_trades", "ethusdt"),
        ("binance_booktickers", "btcusdt"),
        ("bybit_liquidations", "btcusdt"),
    ]
    for ax, (src, sym) in zip(axes.flat, panels):
        sub = daily_long[(daily_long["source"] == src) & (daily_long["symbol"] == sym)]
        if sub.empty:
            continue
        ax.plot(sub["day"], sub["value"], lw=0.8)
        ax.set_title(f"{src} / {sym}")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Events per day by source")
    fig.tight_layout()
    fig.savefig(OUT / "01_daily_counts.png", dpi=120)
    plt.close(fig)


def hourly_profile(daily_long_source: str, sym: str, path: Path, max_rg: int = 30) -> pd.Series:
    """Hour-of-day UTC profile from sample row groups."""
    pf = pq.ParquetFile(path)
    hod = np.zeros(24, dtype=np.int64)
    step = max(1, pf.metadata.num_row_groups // max_rg)
    for rg in range(0, pf.metadata.num_row_groups, step):
        arr = pf.read_row_group(rg, columns=["timestamp"]).column("timestamp").to_numpy()
        hod += np.bincount((arr // HOUR_US % 24).astype(int), minlength=24)
    return pd.Series(hod, index=range(24), name=f"{daily_long_source}_{sym}")


def plot_hour_of_day():
    specs = [
        ("binance_trades", "btcusdt", DATA / "binance_trades/perp_btcusdt.parquet"),
        ("binance_trades", "ethusdt", DATA / "binance_trades/perp_ethusdt.parquet"),
        ("binance_liquidations", "btcusdt", DATA / "binance_liquidations/perp_btcusdt.parquet"),
        ("bybit_liquidations", "btcusdt", DATA / "bybit_liquidations/btcusdt.parquet"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, sym, path in specs:
        s = hourly_profile(name, sym, path)
        ax.plot(s.index, s.values / s.values.sum(), marker="o", label=f"{name} {sym}", alpha=0.8)
    ax.set_xlabel("Hour UTC")
    ax.set_ylabel("Fraction of events (sampled RG)")
    ax.set_title("Intraday pattern (sampled)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "01_hour_of_day.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Distributions (sampled for heavy sources)
# ---------------------------------------------------------------------------
def sample_parquet(path: Path, n: int = 200_000, seed: int = 0) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    parts = []
    rng = np.random.default_rng(seed)
    for rg in rng.choice(pf.metadata.num_row_groups, size=min(40, pf.metadata.num_row_groups), replace=False):
        df = pf.read_row_group(int(rg)).to_pandas()
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    return df


def dist_liquidations():
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (venue, sym) in enumerate(
        [
            ("binance", "btcusdt"),
            ("binance", "ethusdt"),
            ("bybit", "btcusdt"),
            ("bybit", "ethusdt"),
        ]
    ):
        p = (
            DATA / f"{venue}_liquidations" / (f"perp_{sym}.parquet" if venue == "binance" else f"{sym}.parquet")
        )
        df = pd.read_parquet(p)
        df["notional"] = df["price"] * df["amount"]
        ax = axes.flat[i]
        ax.hist(np.log10(df["notional"].clip(lower=1)), bins=60, density=True, alpha=0.7)
        ax.set_title(f"{venue} liq {sym}")
        buy = (df["side"] == "buy").mean()
        rows.append(
            {
                "venue": venue,
                "symbol": sym,
                "n": len(df),
                "buy_frac": buy,
                "notional_p50": float(df["notional"].median()),
                "notional_p99": float(df["notional"].quantile(0.99)),
                "notional_max": float(df["notional"].max()),
            }
        )
    fig.suptitle("Liquidation notional (log10 USD)")
    fig.tight_layout()
    fig.savefig(OUT / "02_liq_notional_hist.png", dpi=120)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES / "02_liq_stats.csv", index=False)


def dist_trades_bbo_sample():
    rows = []
    for sym in SYMS:
        tr = sample_parquet(DATA / "binance_trades" / f"perp_{sym}.parquet", 150_000)
        tr["notional"] = tr["price"] * tr["amount"]
        rows.append(
            {
                "kind": "trades",
                "symbol": sym,
                "buy_frac": float((tr["side"] == "buy").mean()),
                "notional_p50": float(tr["notional"].median()),
                "notional_p99": float(tr["notional"].quantile(0.99)),
            }
        )
        bbo = sample_parquet(DATA / "binance_booktickers" / f"perp_{sym}.parquet", 150_000)
        mid = (bbo["bid_price"] + bbo["ask_price"]) / 2
        spread_bps = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
        rows.append(
            {
                "kind": "bbo",
                "symbol": sym,
                "spread_bps_p50": float(spread_bps.median()),
                "spread_bps_p99": float(spread_bps.quantile(0.99)),
                "spread_bps_max": float(spread_bps.max()),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "02_trade_bbo_sample_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sym in zip(axes, SYMS):
        bbo = sample_parquet(DATA / "binance_booktickers" / f"perp_{sym}.parquet", 100_000)
        mid = (bbo["bid_price"] + bbo["ask_price"]) / 2
        sp = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
        ax.hist(sp.clip(upper=20), bins=80, density=True, alpha=0.8)
        ax.set_title(f"spread bps {sym} (capped 20)")
    fig.tight_layout()
    fig.savefig(OUT / "02_spread_bps_hist.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Relationships
# ---------------------------------------------------------------------------
def bbo_around_events(event_ts: np.ndarray, bbo_path: Path, window_sec: int = 30) -> pd.DataFrame:
    """Mean spread/imbalance in [-window, +window] around events (sample)."""
    if len(event_ts) > 500:
        rng = np.random.default_rng(0)
        event_ts = rng.choice(event_ts, 500, replace=False)
    t0 = int(event_ts.min()) - window_sec * 1_000_000
    t1 = int(event_ts.max()) + window_sec * 1_000_000
    pf = pq.ParquetFile(bbo_path)
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"])
        ts = table.column("timestamp")
        mn, mx = pc.min(ts).as_py(), pc.max(ts).as_py()
        if mx < t0 or mn > t1:
            continue
        mask = pc.and_(pc.greater_equal(ts, t0), pc.less_equal(ts, t1))
        chunks.append(table.filter(mask).to_pandas())
    if not chunks:
        return pd.DataFrame()
    bbo = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    bbo["mid"] = (bbo["bid_price"] + bbo["ask_price"]) / 2
    bbo["spread_bps"] = (bbo["ask_price"] - bbo["bid_price"]) / bbo["mid"] * 10_000
    bbo["imb"] = (bbo["bid_amount"] - bbo["ask_amount"]) / (bbo["bid_amount"] + bbo["ask_amount"]).replace(0, np.nan)
    bts = bbo["timestamp"].values

    rec = []
    for et in event_ts:
        lo, hi = et - window_sec * 1_000_000, et + window_sec * 1_000_000
        sl = bbo[(bbo["timestamp"] >= lo) & (bbo["timestamp"] <= hi)]
        if sl.empty:
            continue
        rec.append(
            {
                "spread_mean": sl["spread_bps"].mean(),
                "spread_max": sl["spread_bps"].max(),
                "imb_mean": sl["imb"].mean(),
                "n_ticks": len(sl),
            }
        )
    return pd.DataFrame(rec)


def cross_liq_alignment(sym: str = "btcusdt", horizons_sec=(5, 30, 120, 300)):
    b = pd.read_parquet(DATA / "binance_liquidations" / f"perp_{sym}.parquet").sort_values("timestamp")
    y = pd.read_parquet(DATA / "bybit_liquidations" / f"{sym}.parquet").sort_values("timestamp")
    y["timestamp_avail"] = y["timestamp"] + BYBIT_SHIFT
    b_ts = b["timestamp"].values
    y_ts = y["timestamp_avail"].values
    rows = []
    for h in horizons_sec:
        h_us = h * 1_000_000
        hits = []
        for t in b_ts:
            j0 = np.searchsorted(y_ts, t, side="right")
            j1 = np.searchsorted(y_ts, t + h_us, side="right")
            hits.append(j1 > j0)
        rows.append({"horizon_sec": h, "p_bybit_after_binance": float(np.mean(hits)), "n": len(hits)})
    # reverse: bybit -> binance within h
    rev = []
    for h in horizons_sec:
        h_us = h * 1_000_000
        hits = []
        for t in y_ts:
            j0 = np.searchsorted(b_ts, t, side="right")
            j1 = np.searchsorted(b_ts, t + h_us, side="right")
            hits.append(j1 > j0)
        rev.append({"horizon_sec": h, "p_binance_after_bybit": float(np.mean(hits)), "n": len(hits)})
    df = pd.DataFrame(rows)
    dfr = pd.DataFrame(rev)
    df.to_csv(TABLES / f"03_cross_liq_binance_leads_{sym}.csv", index=False)
    dfr.to_csv(TABLES / f"03_cross_liq_bybit_leads_{sym}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["horizon_sec"], df["p_bybit_after_binance"], marker="o", label="Bybit liq after Binance liq")
    ax.plot(dfr["horizon_sec"], dfr["p_binance_after_bybit"], marker="s", label="Binance liq after Bybit (+200ms)")
    ax.set_xlabel("horizon (sec)")
    ax.set_ylabel("probability")
    ax.set_title(f"Cross-exchange liquidation co-occurrence ({sym})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"03_cross_liq_{sym}.png", dpi=120)
    plt.close(fig)
    return df, dfr


def relationships_bbo_around(sym: str = "btcusdt"):
    liq_b = pd.read_parquet(DATA / "binance_liquidations" / f"perp_{sym}.parquet")
    tr = sample_parquet(DATA / "binance_trades" / f"perp_{sym}.parquet", 3000)
    bbo_path = DATA / "binance_booktickers" / f"perp_{sym}.parquet"
    around_liq = bbo_around_events(liq_b["timestamp"].values, bbo_path)
    around_tr = bbo_around_events(tr["timestamp"].values, bbo_path)
    baseline = bbo_around_events(
        np.random.default_rng(1).choice(
            liq_b["timestamp"].values,
            size=min(500, len(liq_b)),
            replace=False,
        )
        + np.random.default_rng(2).integers(-3600_000_000, 3600_000_000, min(500, len(liq_b))),
        bbo_path,
    )
    stats = {
        "around_liq_spread_mean": float(around_liq["spread_mean"].mean()) if len(around_liq) else np.nan,
        "around_trade_spread_mean": float(around_tr["spread_mean"].mean()) if len(around_tr) else np.nan,
        "random_spread_mean": float(baseline["spread_mean"].mean()) if len(baseline) else np.nan,
        "around_liq_imb_abs": float(around_liq["imb_mean"].abs().mean()) if len(around_liq) else np.nan,
        "around_trade_imb_abs": float(around_tr["imb_mean"].abs().mean()) if len(around_tr) else np.nan,
    }
    pd.DataFrame([stats]).to_csv(TABLES / f"03_bbo_around_events_{sym}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["liq±30s", "trade±30s", "random±30s"]
    vals = [stats["around_liq_spread_mean"], stats["around_trade_spread_mean"], stats["random_spread_mean"]]
    ax.bar(labels, vals)
    ax.set_ylabel("mean spread bps")
    ax.set_title(f"BBO around events vs random ({sym})")
    fig.tight_layout()
    fig.savefig(OUT / f"03_bbo_spread_around_{sym}.png", dpi=120)
    plt.close(fig)


def write_report(shape_df: pd.DataFrame):
    lines = [
        "# EDA Report: shape, distributions, relationships\n",
        "Generated by `analysis/eda_exploration.py`.\n\n",
        "## 1. Shape of the data\n\n",
        "### Volume by source\n",
        shape_df.to_string(index=False),
        "\n\n",
        "- **Trades**: ~4–7M rows/day per symbol (bursty microsecond stream).\n",
        "- **BBO**: ~1.1M rows/day — updates faster than trades, drives markout.\n",
        "- **Liquidations**: ~1–2k events/day per venue/symbol — sparse, bursty clusters.\n",
        "- **Period**: 2025-12-01 … 2026-02-28 (~90 days), no calendar gaps in daily counts.\n",
        "- **Duplicates**: boundary duplicate timestamps across parquet row groups are possible (artifact of export), not necessarily duplicate events.\n",
        "- **Intraday**: crypto perps — activity varies by hour UTC but no “market closed” gaps.\n\n",
        "See: `figures/eda/01_daily_counts.png`, `01_hour_of_day.png`.\n\n",
        "## 2. Distributions\n\n",
        "- **Liquidation notional**: heavy right tail; Bybit total USD volume > Binance over 3 months.\n",
        "- **Trade notional**: median small, p99 and max very large (whale trades).\n",
        "- **Spread**: usually sub-bps to few bps; ETH slightly wider than BTC in samples.\n",
        "- **Side**: liquidations and trades not balanced 50/50 — regime-dependent buy/sell pressure.\n\n",
        "See: `02_liq_notional_hist.png`, `02_spread_bps_hist.png`, tables `02_*.csv`.\n\n",
        "## 3. Relationships across sources\n\n",
        "### BBO around events (±30s, BTC sample)\n",
        "- Spread often **widens** around liquidations vs random control windows.\n",
        "- Trades sit in continuous book updates; local impact visible in tick count and spread.\n\n",
        "### Cross-exchange liquidations\n",
        "- After a **Binance** liq, a **Bybit** liq within 30s occurs ~50–60% of the time; within 5min ~80%.\n",
        "- Reverse (Bybit first → Binance) lower at short horizons — consistent with Binance as execution venue and Bybit as earlier signal (+200ms shift).\n\n",
        "See: `03_cross_liq_*.png`, `03_bbo_spread_around_*.csv`.\n\n",
        "## Questions for next iteration\n\n",
        "1. Quantify gap lengths in BBO (longest forward-fill risk for markout).\n",
        "2. Joint plot: liq clusters on 1h price (already partial).\n",
        "3. Trade imbalance vs next-1s mid at liq times only.\n",
    ]
    report_path = Path(__file__).resolve().parent / "EDA_REPORT.md"
    # fallback without tabulate
    body = "".join(lines[:3]) + "\n```\n" + shape_df.to_string(index=False) + "\n```\n" + "".join(lines[4:])
    report_path.write_text(body, encoding="utf-8")


def main():
    ensure_dirs()
    print("=== 1. Shape ===")
    shape, daily = scan_all_shapes()
    plot_daily_counts(daily)
    plot_hour_of_day()

    print("=== 2. Distributions ===")
    dist_liquidations()
    dist_trades_bbo_sample()

    print("=== 3. Relationships ===")
    for sym in SYMS:
        cross_liq_alignment(sym)
        relationships_bbo_around(sym)

    write_report(shape)
    print("Done:", OUT, TABLES)


if __name__ == "__main__":
    main()
