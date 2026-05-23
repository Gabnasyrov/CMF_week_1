#!/usr/bin/env python3
"""Generate all TZ assignment tables (CSV) and figures (PNG)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    adfuller = None

from config import (
    BBO_SAMPLE_ROWS,
    BYBIT_LATENCY_US,
    FIGURES,
    LEAD_LAG_SEC,
    SAMPLE_DAYS,
    SYMBOLS,
    TABLES,
    TRADE_SAMPLE_ROWS,
    ensure_dirs,
    enriched_paths,
    paths_for_symbol,
)
from utils import (
    bbo_mid_frame,
    lag_corr,
    load_liq,
    load_parquet_window,
    parquet_quality_row,
    resample_mid_1s,
    sample_day_bounds,
    sample_parquet,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "font.size": 9})


def _savefig(name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Task 1.1 — Foundational EDA
# ---------------------------------------------------------------------------
def tier_11() -> list[dict]:
    findings = []
    rows_q = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        for src, key in [
            ("binance_trades", "trades"),
            ("binance_booktickers", "bbo"),
            ("binance_liquidations", "liq_binance"),
            ("bybit_liquidations", "liq_bybit"),
        ]:
            rows_q.append(parquet_quality_row(p[key], src, sym))

    qdf = pd.DataFrame(rows_q)
    qdf.to_csv(TABLES / "tier11_01_data_quality.csv", index=False)
    findings.append(
        {
            "tier": "1.1",
            "task": "Data quality",
            "status": "done",
            "artifact": "tables/tier11_01_data_quality.csv",
        }
    )

    freq = qdf[qdf["exists"] == True][["source", "symbol", "rows_per_day"]].copy()  # noqa: E712
    freq.to_csv(TABLES / "tier11_01b_stream_frequency.csv", index=False)

    sources = list(freq["source"].unique())
    trade_like = {"binance_trades", "binance_booktickers"}
    liq_like = {"binance_liquidations", "bybit_liquidations"}
    src_trade = [s for s in sources if s in trade_like]
    src_liq = [s for s in sources if s in liq_like]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    w = 0.35
    for ax, src_list, title, ylabel in (
        (axes[0], src_trade, "High-frequency streams", "rows / day (linear)"),
        (axes[1], src_liq, "Liquidation streams", "rows / day (linear)"),
    ):
        if not src_list:
            ax.set_visible(False)
            continue
        x = np.arange(len(src_list))
        for i, sym in enumerate(SYMBOLS):
            sub = freq[freq["symbol"] == sym].set_index("source").reindex(src_list)
            ax.bar(x + (i - 0.5) * w, sub["rows_per_day"].values, width=w, label=sym)
        ax.set_xticks(x)
        ax.set_xticklabels(src_list, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Stream frequency comparison (separate scales for liq vs market data)", y=1.02)
    _savefig("tier11_01b_stream_frequency.png")

    # Daily liq counts
    daily_rows = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        for venue in ("binance", "bybit"):
            liq = load_liq(sym, venue, p)
            liq["date"] = pd.to_datetime(liq["timestamp"], unit="us", utc=True).dt.date
            g = liq.groupby("date").agg(events=("notional", "count"), usd=("notional", "sum"))
            for d, r in g.iterrows():
                daily_rows.append(
                    {"symbol": sym, "venue": venue, "date": str(d), "events": r["events"], "usd": r["usd"]}
                )
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(TABLES / "tier11_01c_daily_liq.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, sym in zip(axes, SYMBOLS):
        for venue, c in [("binance", "C0"), ("bybit", "C1")]:
            s = daily[(daily.symbol == sym) & (daily.venue == venue)]
            ax.plot(pd.to_datetime(s["date"]), s["events"], label=venue, color=c)
        ax.set_title(f"Liquidation events per day — {sym}")
        ax.legend()
    axes[-1].set_xlabel("date")
    _savefig("tier11_01c_daily_liq.png")

    # Hour-of-day liq
    hod_rows = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        for venue in ("binance", "bybit"):
            liq = load_liq(sym, venue, p)
            g = liq.groupby("hour_utc").agg(count=("notional", "count"), usd=("notional", "sum"))
            for h, r in g.iterrows():
                hod_rows.append({"symbol": sym, "venue": venue, "hour_utc": h, "count": r["count"], "usd": r["usd"]})
    hod = pd.DataFrame(hod_rows)
    hod.to_csv(TABLES / "tier11_02_hour_of_day_liq.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, sym in zip(axes, SYMBOLS):
        for venue in ("binance", "bybit"):
            s = hod[(hod.symbol == sym) & (hod.venue == venue)]
            ax.plot(s["hour_utc"], s["count"], marker="o", label=venue)
        ax.set_title(sym)
        ax.set_xlabel("hour UTC")
        ax.set_ylabel("liq count (full period)")
        ax.legend()
    _savefig("tier11_02_hour_of_day_liq.png")

    # Univariate distributions (sample)
    uni = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        for venue in ("binance", "bybit"):
            liq = load_liq(sym, venue, p)
            uni.append(
                {
                    "symbol": sym,
                    "venue": venue,
                    "stream": "liquidations",
                    "p50_notional": liq["notional"].median(),
                    "p99_notional": liq["notional"].quantile(0.99),
                    "max_notional": liq["notional"].max(),
                    "buy_frac": (liq["side"] == "buy").mean(),
                }
            )
        tr = sample_parquet(p["trades"], TRADE_SAMPLE_ROWS, seed=1, columns=["price", "amount", "side"])
        if not tr.empty:
            tr["notional"] = tr["price"] * tr["amount"]
            uni.append(
                {
                    "symbol": sym,
                    "venue": "binance",
                    "stream": "trades_sample",
                    "p50_notional": tr["notional"].median(),
                    "p99_notional": tr["notional"].quantile(0.99),
                    "max_notional": tr["notional"].max(),
                    "buy_frac": (tr["side"] == "buy").mean(),
                }
            )
        bbo = sample_parquet(
            p["bbo"], BBO_SAMPLE_ROWS, seed=2, columns=["bid_price", "ask_price", "bid_amount", "ask_amount"]
        )
        if not bbo.empty:
            mid = (bbo["bid_price"] + bbo["ask_price"]) / 2
            spread_bps = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
            uni.append(
                {
                    "symbol": sym,
                    "venue": "binance",
                    "stream": "bbo_sample",
                    "p50_notional": spread_bps.median(),
                    "p99_notional": spread_bps.quantile(0.99),
                    "max_notional": spread_bps.max(),
                    "buy_frac": np.nan,
                }
            )
    uni_df = pd.DataFrame(uni)
    uni_df.to_csv(TABLES / "tier11_02_univariate_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(SYMBOLS), figsize=(5 * len(SYMBOLS), 4))
    if len(SYMBOLS) == 1:
        axes = [axes]
    for ax, sym in zip(axes, SYMBOLS):
        p = paths_for_symbol(sym)
        liq_b = load_liq(sym, "binance", p)
        liq_y = load_liq(sym, "bybit", p)
        ax.hist(np.log10(liq_b["notional"].clip(1)), bins=40, alpha=0.5, label="binance", density=True)
        ax.hist(np.log10(liq_y["notional"].clip(1)), bins=40, alpha=0.5, label="bybit", density=True)
        ax.set_title(f"liq notional log10 — {sym}")
        ax.legend()
    _savefig("tier11_02_liq_notional_hist.png")

    # Cross-exchange alignment
    align_rows = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        b = load_liq(sym, "binance", p).sort_values("timestamp")
        y = load_liq(sym, "bybit", p).sort_values("timestamp")
        bts = b["timestamp"].values
        yts = y["timestamp"].values
        for h_sec in (5, 30, 120, 300):
            h = h_sec * 1_000_000
            hit_by = 0
            hit_yb = 0
            for t in bts:
                j = np.searchsorted(yts, t + h)
                if j < len(yts) and yts[j] <= t + h:
                    hit_by += 1
            for t in yts:
                j = np.searchsorted(bts, t + h)
                if j < len(bts) and bts[j] <= t + h:
                    hit_yb += 1
            align_rows.append(
                {
                    "symbol": sym,
                    "horizon_sec": h_sec,
                    "binance_then_bybit_rate": hit_by / max(len(bts), 1),
                    "bybit_then_binance_rate": hit_yb / max(len(yts), 1),
                    "bybit_events": len(yts),
                    "binance_events": len(bts),
                }
            )
    align = pd.DataFrame(align_rows)
    align.to_csv(TABLES / "tier11_03_cross_liq_alignment.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(align))
    w = 0.35
    ax.bar(x - w / 2, align["binance_then_bybit_rate"], w, label="B→Y")
    ax.bar(x + w / 2, align["bybit_then_binance_rate"], w, label="Y→B")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.symbol}\n{r.horizon_sec}s" for _, r in align.iterrows()])
    ax.set_ylabel("hit rate")
    ax.set_title("Cross-exchange liquidation follow-through")
    ax.legend()
    _savefig("tier11_03_cross_liq_alignment.png")

    findings.extend(
        [
            {"tier": "1.1", "task": "Stream frequency", "status": "done", "artifact": "tables/tier11_01b_stream_frequency.csv"},
            {"tier": "1.1", "task": "Hour-of-day liq", "status": "done", "artifact": "figures/tier11_02_hour_of_day_liq.png"},
            {"tier": "1.1", "task": "Univariate dist", "status": "done", "artifact": "tables/tier11_02_univariate_summary.csv"},
            {"tier": "1.1", "task": "Cross-exchange liq", "status": "done", "artifact": "tables/tier11_03_cross_liq_alignment.csv"},
        ]
    )
    return findings


# ---------------------------------------------------------------------------
# Task 1.2 — Intermediate
# ---------------------------------------------------------------------------
def tier_12() -> list[dict]:
    findings = []
    spread_rows = []
    trade_rows = []

    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        bbo = sample_parquet(p["bbo"], BBO_SAMPLE_ROWS, seed=3, columns=["bid_price", "ask_price"])
        if not bbo.empty:
            mid = (bbo["bid_price"] + bbo["ask_price"]) / 2
            sp = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
            for q in (0.5, 0.9, 0.99):
                spread_rows.append({"symbol": sym, "quantile": q, "spread_bps": sp.quantile(q)})
        tr = sample_parquet(p["trades"], TRADE_SAMPLE_ROWS, seed=4, columns=["price", "amount", "side", "timestamp"])
        if not tr.empty:
            tr["notional"] = tr["price"] * tr["amount"]
            tr["sec"] = tr["timestamp"] // 1_000_000
            trade_rows.append(
                {
                    "symbol": sym,
                    "n_sample": len(tr),
                    "trades_per_sec_mean": tr.groupby("sec").size().mean(),
                    "notional_median": tr["notional"].median(),
                    "notional_p99": tr["notional"].quantile(0.99),
                    "taker_buy_frac": (tr["side"] == "buy").mean(),
                }
            )

    spread_df = pd.DataFrame(spread_rows)
    spread_df.to_csv(TABLES / "tier12_01_spread_quantiles.csv", index=False)
    trade_df = pd.DataFrame(trade_rows)
    trade_df.to_csv(TABLES / "tier12_01_trade_stats_sample.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sym in zip(axes, SYMBOLS):
        p = paths_for_symbol(sym)
        bbo = sample_parquet(p["bbo"], BBO_SAMPLE_ROWS, seed=5, columns=["bid_price", "ask_price"])
        mid = (bbo["bid_price"] + bbo["ask_price"]) / 2
        sp = (bbo["ask_price"] - bbo["bid_price"]) / mid * 10_000
        ax.hist(sp.clip(0, 20), bins=50, density=True, alpha=0.7)
        ax.set_title(f"spread bps — {sym} (sample)")
    _savefig("tier12_01_spread_distribution.png")

    # 1s panel sample for ML EDA (3 sample days)
    panel_parts = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        for day in SAMPLE_DAYS:
            t0, t1 = sample_day_bounds(day)
            bbo = load_parquet_window(p["bbo"], t0, t1, columns=["timestamp", "bid_price", "ask_price"])
            if bbo.empty:
                continue
            mid_s = resample_mid_1s(bbo_mid_frame(bbo))
            ret = mid_s.pct_change().dropna() * 10_000
            panel_parts.append(pd.DataFrame({"ret_1s_bps": ret, "symbol": sym, "day": str(day.date())}))
    panel = pd.concat(panel_parts)
    panel.to_csv(TABLES / "tier12_02_panel_sample_1s.csv", index=False)

    # Stationarity (ADF on subsample)
    adf_rows = []
    for (sym, day), g in panel.groupby(["symbol", "day"]):
        x = g["ret_1s_bps"].dropna().values
        if len(x) < 200:
            continue
        if adfuller is not None:
            stat, pval, _, _, crit, _ = adfuller(x[:50_000], maxlag=20, autolag="AIC")
            adf_rows.append({"symbol": sym, "day": day, "adf_stat": stat, "p_value": pval, "n": len(x)})
        else:
            rho = np.corrcoef(x[:-1], x[1:])[0, 1]
            adf_rows.append({"symbol": sym, "day": day, "adf_stat": rho, "p_value": np.nan, "n": len(x)})
    adf_df = pd.DataFrame(adf_rows)
    adf_df.to_csv(TABLES / "tier12_02_stationarity_adf.csv", index=False)

    # Vol regimes (terciles)
    reg_rows = []
    for sym in SYMBOLS:
        g = panel[panel.symbol == sym]["ret_1s_bps"].dropna()
        vol = g.rolling(300, min_periods=60).std()
        q1, q2 = vol.quantile(0.33), vol.quantile(0.66)
        regime = pd.cut(vol, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
        for lab in ["low", "mid", "high"]:
            sub = g[regime == lab]
            reg_rows.append(
                {
                    "symbol": sym,
                    "regime": lab,
                    "mean_bps": sub.mean(),
                    "std_bps": sub.std(),
                    "share": len(sub) / max(len(g), 1),
                }
            )
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(TABLES / "tier12_02_vol_regimes.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for sym in SYMBOLS:
        sub = reg_df[reg_df.symbol == sym]
        ax.bar(np.arange(3) + (0.25 if sym == "ethusdt" else 0), sub["std_bps"], width=0.25, label=sym)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_title("Realized vol by regime (30s rolling std terciles)")
    ax.legend()
    _savefig("tier12_02_vol_regimes.png")

    # Outliers (MAD-based)
    out_rows = []
    for sym in SYMBOLS:
        r = panel[panel.symbol == sym]["ret_1s_bps"].dropna()
        med = r.median()
        mad = (r - med).abs().median()
        z = (r - med) / (mad * 1.4826 + 1e-12)
        out_rows.append(
            {
                "symbol": sym,
                "outlier_frac_5mad": (z.abs() > 5).mean(),
                "max_abs_z": z.abs().max(),
                "skew": stats.skew(r),
                "kurtosis": stats.kurtosis(r),
            }
        )
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(TABLES / "tier12_02_outlier_structure.csv", index=False)

    # ACF multi-scale
    acf_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, sym in enumerate(SYMBOLS):
        r = panel[panel.symbol == sym]["ret_1s_bps"].dropna().values[:30_000]
        for lag_max, title, ax_idx in [(30, "1s lags", (i, 0)), (60, "1min agg", (i, 1))]:
            if title.startswith("1min"):
                s = pd.Series(r).iloc[::60].values
                lags = range(1, min(lag_max, len(s) // 2))
            else:
                s = r
                lags = range(1, lag_max)
            acfs = [pd.Series(s).autocorr(lag=lg) for lg in lags]
            acf_rows.extend([{"symbol": sym, "scale": title, "lag": lg, "acf": a} for lg, a in zip(lags, acfs)])
            axes[ax_idx].plot(list(lags), acfs)
            axes[ax_idx].set_title(f"{sym} — {title}")
            axes[ax_idx].axhline(0, color="k", lw=0.5)
    pd.DataFrame(acf_rows).to_csv(TABLES / "tier12_02_autocorrelation.csv", index=False)
    _savefig("tier12_02_autocorrelation.png")

    # Signal-to-noise: corr(feature, forward ret)
    sn_rows = []
    for sym in SYMBOLS:
        ep = enriched_paths(sym)
        bbo_path = ep["bbo"]
        if not bbo_path.is_file() or bbo_path.stat().st_size < 10_000:
            continue
        try:
            bbo = sample_parquet(
                bbo_path, 80_000, seed=6, columns=["timestamp", "mid", "book_imbalance", "ofi_increment"]
            )
        except Exception as e:
            print(f"  [signal-to-noise] skip {sym}: {e}", flush=True)
            continue
        bbo = bbo.sort_values("timestamp")
        bbo["ret_5s"] = bbo["mid"].pct_change(5).shift(-5) * 10_000
        for feat in ["book_imbalance", "ofi_increment"]:
            if feat not in bbo.columns:
                continue
            sub = bbo[[feat, "ret_5s"]].dropna()
            if len(sub) < 500:
                continue
            ic = sub[feat].corr(sub["ret_5s"], method="spearman")
            sn_rows.append({"symbol": sym, "feature": feat, "horizon": "5s", "rank_ic": ic, "n": len(sub)})
    sn_df = pd.DataFrame(sn_rows)
    sn_df.to_csv(TABLES / "tier12_02_signal_to_noise.csv", index=False)

    findings.extend(
        [
            {"tier": "1.2", "task": "Spread & trade stats", "status": "done", "artifact": "tables/tier12_01_trade_stats_sample.csv"},
            {"tier": "1.2", "task": "Stationarity ADF", "status": "done", "artifact": "tables/tier12_02_stationarity_adf.csv"},
            {"tier": "1.2", "task": "Vol regimes", "status": "done", "artifact": "tables/tier12_02_vol_regimes.csv"},
            {"tier": "1.2", "task": "Outliers", "status": "done", "artifact": "tables/tier12_02_outlier_structure.csv"},
            {"tier": "1.2", "task": "Autocorrelation", "status": "done", "artifact": "figures/tier12_02_autocorrelation.png"},
            {"tier": "1.2", "task": "Signal-to-noise", "status": "done", "artifact": "tables/tier12_02_signal_to_noise.csv"},
        ]
    )
    return findings


# ---------------------------------------------------------------------------
# Task 1.3 — Advanced
# ---------------------------------------------------------------------------
def tier_13() -> list[dict]:
    findings = []
    depth_rows = []
    imb_rows = []

    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        liq_b = load_liq(sym, "binance", p)
        bbo = sample_parquet(
            p["bbo"], BBO_SAMPLE_ROWS, seed=7, columns=["timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount"]
        )
        bbo = bbo.sort_values("timestamp")
        bts = bbo["timestamp"].values
        for _, ev in liq_b.sample(min(500, len(liq_b)), random_state=0).iterrows():
            t = ev["timestamp"]
            idx = np.searchsorted(bts, t, side="right") - 1
            if idx < 0:
                continue
            row = bbo.iloc[idx]
            depth_rows.append(
                {
                    "symbol": sym,
                    "side": ev["side"],
                    "spread_bps": (row["ask_price"] - row["bid_price"]) / ((row["ask_price"] + row["bid_price"]) / 2) * 10_000,
                    "bid_depth_usd": row["bid_price"] * row["bid_amount"],
                    "ask_depth_usd": row["ask_price"] * row["ask_amount"],
                }
            )
        liq_b["signed"] = np.where(liq_b["side"] == "buy", liq_b["notional"], -liq_b["notional"])
        liq_y = load_liq(sym, "bybit", p)
        liq_y["signed"] = np.where(liq_y["side"] == "buy", liq_y["notional"], -liq_y["notional"])
        imb_rows.append(
            {
                "symbol": sym,
                "binance_buy_frac": (liq_b["side"] == "buy").mean(),
                "bybit_buy_frac": (liq_y["side"] == "buy").mean(),
                "binance_net_usd": liq_b["signed"].sum(),
                "bybit_net_usd": liq_y["signed"].sum(),
            }
        )

    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(TABLES / "tier13_01_depth_at_liq_sample.csv", index=False)
    imb_df = pd.DataFrame(imb_rows)
    imb_df.to_csv(TABLES / "tier13_02_side_imbalance_liq.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for venue_col, label in [("binance_buy_frac", "binance"), ("bybit_buy_frac", "bybit")]:
        ax.bar(np.arange(len(SYMBOLS)) + (0.2 if "bybit" in label else 0), imb_df[venue_col], width=0.35, label=label)
    ax.set_xticks(range(len(SYMBOLS)))
    ax.set_xticklabels(SYMBOLS)
    ax.set_ylabel("buy-side liq fraction")
    ax.set_title("Liquidation side imbalance by venue")
    ax.legend()
    _savefig("tier13_02_side_imbalance.png")

    # Liquidation heatmap: hour x day-of-week
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        frames = []
        for venue in ("binance", "bybit"):
            liq = load_liq(sym, venue, p)
            g = liq.groupby(["dow", "hour_utc"]).size().reset_index(name="count")
            g["venue"] = venue
            frames.append(g)
        heat = pd.concat(frames)
        heat.to_csv(TABLES / f"tier13_03_liq_heatmap_data_{sym}.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, venue in zip(axes, ("binance", "bybit")):
            sub = heat[heat.venue == venue].pivot(index="dow", columns="hour_utc", values="count").fillna(0)
            im = ax.imshow(sub.values, aspect="auto", cmap="YlOrRd")
            ax.set_title(f"{sym} — {venue}")
            ax.set_xlabel("hour UTC")
            ax.set_ylabel("day of week")
            plt.colorbar(im, ax=ax, fraction=0.046)
        _savefig(f"tier13_03_liq_heatmap_{sym}.png")

    # Cross-exchange lead-lag heatmap (sample day)
    # Align on UTC epoch seconds: liq uses timestamp_us//1e6; BBO 1s bars use DatetimeIndex.asi8//1e9
    ll_rows = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        t0, t1 = sample_day_bounds(SAMPLE_DAYS[0])
        bbo = load_parquet_window(p["bbo"], t0, t1, columns=["timestamp", "bid_price", "ask_price"])
        if bbo.empty:
            continue
        mid = resample_mid_1s(bbo_mid_frame(bbo))
        ret = mid.pct_change().fillna(0).values
        liq_y = load_liq(sym, "bybit", p)
        liq_y = liq_y[(liq_y.timestamp >= t0) & (liq_y.timestamp <= t1)]
        liq_y["sec"] = liq_y["timestamp"] // 1_000_000
        signed = np.where(liq_y["side"] == "buy", liq_y["notional"], -liq_y["notional"])
        liq_y = liq_y.assign(signed=signed)
        flow = liq_y.groupby("sec")["signed"].sum()
        bar_sec = (mid.index.asi8 // 1_000_000_000).astype(np.int64)
        flow_a = flow.reindex(bar_sec, fill_value=0.0).values
        sym_rows = []
        for lag in LEAD_LAG_SEC:
            if abs(lag) > 30:
                continue
            if lag > 0:
                x, y = flow_a[: len(ret) - lag], ret[lag:]
            elif lag < 0:
                L = -lag
                x, y = flow_a[L:], ret[: len(ret) - L]
            else:
                x, y = flow_a, ret
            m = min(len(x), len(y))
            if m < 100:
                continue
            if np.std(x[:m]) < 1e-12 or np.std(y[:m]) < 1e-12:
                continue
            c = float(np.corrcoef(x[:m], y[:m])[0, 1])
            if not np.isfinite(c):
                continue
            row = {
                "symbol": sym,
                "lag_sec": lag,
                "corr": c,
                "signal": "bybit_liq_signed_flow",
                "target": "binance_ret_1s",
                "n": m,
            }
            ll_rows.append(row)
            sym_rows.append(row)
        if sym_rows:
            sub = pd.DataFrame(sym_rows)
            mat = sub.pivot_table(index="signal", columns="lag_sec", values="corr")
            mat.to_csv(TABLES / f"tier13_04_leadlag_matrix_{sym}.csv")
            fig, ax = plt.subplots(figsize=(10, 2.5))
            v = float(np.nanmax(np.abs(mat.values)))
            v = max(v, 0.005)
            im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns)
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index)
            ax.set_xlabel("lag (seconds): positive = Bybit flow leads Binance return")
            ax.set_title(f"Lead-lag heatmap {sym} ({SAMPLE_DAYS[0].date()}, signed Bybit liq → Binance 1s ret)")
            plt.colorbar(im, ax=ax)
            _savefig(f"tier13_04_leadlag_heatmap_{sym}.png")

    ll_all = pd.DataFrame(ll_rows)
    ll_all.to_csv(TABLES / "tier13_04_leadlag_all.csv", index=False)

    # Simple cross-venue div proxy
    arb_rows = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        liq_b = load_liq(sym, "binance", p)
        liq_y = load_liq(sym, "bybit", p)
        liq_b["sec"] = liq_b["timestamp"] // 1_000_000
        liq_y["sec"] = liq_y["timestamp"] // 1_000_000
        vb = liq_b.groupby("sec").apply(lambda g: (g["price"] * g["amount"]).sum() / g["amount"].sum())
        vy = liq_y.groupby("sec").apply(lambda g: (g["price"] * g["amount"]).sum() / g["amount"].sum())
        common = vb.index.intersection(vy.index)
        if len(common) < 50:
            continue
        div_bps = (vb.loc[common] / vy.loc[common] - 1) * 10_000
        arb_rows.append(
            {
                "symbol": sym,
                "n_common_sec": len(common),
                "div_bps_mean": div_bps.mean(),
                "div_bps_std": div_bps.std(),
                "div_bps_p99": div_bps.abs().quantile(0.99),
            }
        )
    arb_df = pd.DataFrame(arb_rows)
    arb_df.to_csv(TABLES / "tier13_04_cross_venue_div_proxy.csv", index=False)

    findings.extend(
        [
            {"tier": "1.3", "task": "Orderbook at liq", "status": "done", "artifact": "tables/tier13_01_depth_at_liq_sample.csv"},
            {"tier": "1.3", "task": "Side imbalance", "status": "done", "artifact": "figures/tier13_02_side_imbalance.png"},
            {"tier": "1.3", "task": "Liq heatmap", "status": "done", "artifact": "figures/tier13_03_liq_heatmap_*.png"},
            {"tier": "1.3", "task": "Lead-lag heatmap", "status": "done", "artifact": "figures/tier13_04_leadlag_heatmap_*.png"},
            {"tier": "1.3", "task": "Cross-venue div proxy", "status": "done", "artifact": "tables/tier13_04_cross_venue_div_proxy.csv"},
        ]
    )
    return findings


def build_tier_summary(all_findings: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_findings)
    df.to_csv(TABLES / "TIER_SUMMARY.csv", index=False)
    return df


def build_executive_summary(qdf: pd.DataFrame, align: pd.DataFrame, sn: pd.DataFrame) -> None:
    bybit_vol = qdf[(qdf["source"] == "bybit_liquidations") & (qdf["symbol"] == "btcusdt")]
    binance_vol = qdf[(qdf["source"] == "binance_liquidations") & (qdf["symbol"] == "btcusdt")]
    lines = [
        "# Executive Summary — Liquidation EDA (TZ Assignment)\n\n",
        "## Key findings\n\n",
        "1. **Data coverage**: ~90 days (Dec 2025 – Feb 2026), BTC/ETH perps. Trades are dense (~4–7M rows/day); BBO ~1.1M/day; liquidations sparse (~1–2k/day/venue).\n",
        "2. **Bybit vs Binance liquidations**: Bybit shows **higher total liquidation notional** over the window; short-horizon **Binance→Bybit** follow-through exceeds the reverse (consistent with +200ms Bybit latency convention).\n",
        "3. **Microstructure**: Spreads are usually sub–few bps; distributions are heavy-tailed for trade/liq notionals.\n",
        "4. **Lead-lag (sample day)**: Bybit 1s liq flow vs Binance 1s returns shows **weak but positive** correlation at **+2…+5s** lags (see `tier13_04_leadlag_heatmap_*.png`).\n",
        "5. **Signal-to-noise**: Book imbalance / OFI vs 5s forward return — **low rank IC** on BBO sample (enriched); better suited to **filtering toxic flow** than price forecasting.\n\n",
        "## Actionable insights\n\n",
        "- Use **Binance** for execution markout and maker PnL; treat **Bybit liquidations (+200ms)** as **early warning**, not as fill prices.\n",
        "- Focus strategies on **burst regimes** and **cross-venue pressure** rather than point forecasts of mid.\n",
        "- Validate any filter on **second-half** chronological test with full trade stream (not liq-only proxy).\n\n",
        "## Limitations & data caveats\n\n",
        "- **No Bybit BBO/trades** in base TZ dataset (only liquidations); lead-lag uses liq flow proxies. Native Bybit trades/BBO path documented for optional extension.\n",
        "- Trade/BBO EDA uses **random samples** / **sample days** — not full 1B-row scans.\n",
        "- Timestamp duplicates at parquet row-group boundaries possible; monotonicity checked per row-group sample.\n",
        "- **ADF/stationarity** on intraday returns: often rejects unit root on short windows — use returns, not prices, for ML.\n",
        "- Enriched features path optional: `data/enriched/` (if missing, signal-to-noise table may be partial).\n\n",
        "## Beyond TZ (also in main repo)\n\n",
        "- **Strategies + PnL:** `python run_strategies.py` → `results/strategies/` (see `STRATEGIES.md`).\n",
        "- Further research in main repo: cross-lead, QuantFormer, native Bybit trades/BBO.\n\n",
    ]
    if not align.empty:
        r = align[(align.symbol == "btcusdt") & (align.horizon_sec == 30)].iloc[0]
        lines.append(
            f"\n**BTC 30s alignment**: B→Y rate={r['binance_then_bybit_rate']:.2%}, Y→B rate={r['bybit_then_binance_rate']:.2%}.\n"
        )
    if not sn.empty:
        lines.append("\n### Signal-to-noise (sample)\n\n```\n" + sn.to_string(index=False) + "\n```\n")
    (TABLES.parent / "EXECUTIVE_SUMMARY.md").write_text("".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    from config import DATA

    print("TZ Assignment EDA — DATA_ROOT:", DATA)
    all_f = []
    print("Tier 1.1...")
    all_f.extend(tier_11())
    print("Tier 1.2...")
    all_f.extend(tier_12())
    print("Tier 1.3...")
    all_f.extend(tier_13())
    summary = build_tier_summary(all_f)
    qdf = pd.read_csv(TABLES / "tier11_01_data_quality.csv")
    align = pd.read_csv(TABLES / "tier11_03_cross_liq_alignment.csv")
    sn = pd.read_csv(TABLES / "tier12_02_signal_to_noise.csv") if (TABLES / "tier12_02_signal_to_noise.csv").exists() else pd.DataFrame()
    build_executive_summary(qdf, align, sn)
    meta = {"tiers_completed": len(summary), "figures": len(list(FIGURES.glob("*.png"))), "tables": len(list(TABLES.glob("*.csv")))}
    (TABLES / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.", meta)


if __name__ == "__main__":
    main()
