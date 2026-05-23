#!/usr/bin/env python3
"""Run liquidation_task EDA and write figures/tables to analysis/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BYBIT_LATENCY_US,
    FIGURES,
    LEAD_LAG_SEC,
    SAMPLE_DAYS,
    SYMBOLS,
    TAUS_SEC,
    TABLES,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    dt_to_us,
    paths_for_symbol,
)
from utils import (
    asof_join_signal,
    bbo_mid_spread,
    compute_pnl,
    decile_table,
    lag_correlation,
    liq_to_bars,
    load_liquidations,
    load_parquet_window,
    mid_at_horizon,
    trade_features,
    weighted_mean,
)

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True})
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def day_bounds(day) -> tuple[int, int]:
    t0 = dt_to_us(day.replace(hour=0, minute=0, second=0, microsecond=0))
    t1 = t0 + 86_400 * 1_000_000 - 1
    return t0, t1


# ---------------------------------------------------------------------------
# Phase 0–1: liquidations sanity & distributions (full period)
# ---------------------------------------------------------------------------
def phase_liq_overview():
    rows = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        for name, path, shift in [
            ("binance", p["liq_binance"], 0),
            ("bybit", p["liq_bybit"], BYBIT_LATENCY_US),
        ]:
            liq = load_liquidations(path, shift_us=shift)
            liq["date"] = pd.to_datetime(liq["timestamp"], unit="us", utc=True).dt.date
            daily = liq.groupby("date").agg(events=("notional", "count"), usd=("notional", "sum"))
            rows.append({"symbol": sym, "venue": name, "events": len(liq), "usd_total": liq["notional"].sum()})
            daily.to_csv(TABLES / f"daily_liq_{name}_{sym}.csv")

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "liq_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, sym in zip(axes.flat, SYMBOLS):
        p = paths_for_symbol(sym)
        liq_b = load_liquidations(p["liq_binance"])
        liq_y = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
        for liq, label, color in [(liq_b, "binance", "C0"), (liq_y, "bybit+200ms", "C1")]:
            ax.hist(np.log10(liq["notional"].clip(lower=1)), bins=50, alpha=0.5, label=label, color=color, density=True)
        ax.set_title(sym)
        ax.set_xlabel("log10(notional USD)")
        ax.legend()
    fig.suptitle("Liquidation notional distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_liq_notional_hist.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        liq = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
        daily = liq.groupby(pd.to_datetime(liq["timestamp"], unit="us", utc=True).dt.date)["notional"].sum()
        ax.plot(daily.index, daily.values / 1e6, label=f"bybit {sym}", alpha=0.8)
    ax.set_ylabel("USD millions / day")
    ax.set_title("Bybit liquidation volume (latency-adjusted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "02_liq_daily_volume.png", dpi=120)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Phase 2: lead-lag (1s bars, full liq + sample-day mid returns)
# ---------------------------------------------------------------------------
def phase_lead_lag():
    results = []
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        liq_b = load_liquidations(p["liq_binance"])
        liq_y = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
        bars_b = liq_to_bars(liq_b, 1000)
        bars_y = liq_to_bars(liq_y, 1000)

        # mid returns from one sample day BBO
        day = SAMPLE_DAYS[0]
        t0, t1 = day_bounds(day)
        bbo = load_parquet_window(p["bbo"], t0, t1 - 300_000_000)
        ms = bbo_mid_spread(bbo)
        ms1 = ms.set_index(pd.to_datetime(ms["timestamp"], unit="us", utc=True)).resample("1s").last().dropna()
        ret = (ms1["mid"] / ms1["mid"].shift(1) - 1) * 10_000
        ret = ret.dropna()
        ret_us = ret.index.asi8
        ret_val = ret.values

        for src_name, bars in [("binance_liq", bars_b), ("bybit_liq", bars_y)]:
            if bars.empty:
                continue
            x_ts = bars["ts_us"].values.astype(np.float64)
            x_val = bars["L_net"].values.astype(np.float64)
            ll = lag_correlation(x_ts, x_val, ret_us.astype(np.float64), ret_val, LEAD_LAG_SEC)
            ll["symbol"] = sym
            ll["source"] = src_name
            results.append(ll)

    ll_df = pd.concat(results, ignore_index=True)
    ll_df.to_csv(TABLES / "lead_lag_corr.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for (sym, src), g in ll_df.groupby(["symbol", "source"]):
        ax.plot(g["lag_sec"], g["corr"], marker="o", label=f"{sym} {src}")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("lag sec (L_net[t] vs mid return[t+lag])")
    ax.set_ylabel("correlation")
    ax.set_title("Lead-lag: liquidation pressure vs Binance mid return (1s, sample day)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "03_lead_lag.png", dpi=120)
    plt.close(fig)
    return ll_df


# ---------------------------------------------------------------------------
# Phase 3–5: markout + deciles on sample days (subsampled trades)
# ---------------------------------------------------------------------------
def load_trades_subsample(path, t0: int, t1: int, max_rows: int = 150_000, seed: int = 42) -> pd.DataFrame:
    df = load_parquet_window(path, t0, t1)
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_values("timestamp").reset_index(drop=True)


def phase_markout_deciles():
    all_deciles = []
    pnl_by_tau = []

    for day in SAMPLE_DAYS:
        t0, t1 = day_bounds(day)
        t1_ext = t1 + max(TAUS_SEC) * 1_000_000

        for sym in SYMBOLS:
            p = paths_for_symbol(sym)
            trades = load_trades_subsample(p["trades"], t0, t1)
            if trades.empty:
                continue
            trades = trade_features(trades)

            bbo = load_parquet_window(p["bbo"], t0, t1_ext)
            ms = bbo_mid_spread(bbo)
            bbo_ts = ms["timestamp"].values
            bbo_mid = ms["mid"].values

            liq_b = load_liquidations(p["liq_binance"])
            liq_y = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
            liq_b = liq_b[(liq_b["timestamp"] >= t0 - 60_000_000) & (liq_b["timestamp"] <= t1)]
            liq_y = liq_y[(liq_y["timestamp"] >= t0 - 60_000_000) & (liq_y["timestamp"] <= t1)]
            bars_b = liq_to_bars(liq_b, 1000)
            bars_y = liq_to_bars(liq_y, 1000)
            for bars in (bars_b, bars_y):
                if not bars.empty:
                    bars.sort_values("ts_us", inplace=True)
                    bars["L_net_30s"] = bars["L_net"].rolling(30, min_periods=1).sum()
            sig_b30 = asof_join_signal(trades, bars_b, "L_net_30s") if not bars_b.empty else np.full(len(trades), np.nan)
            sig_y30 = asof_join_signal(trades, bars_y, "L_net_30s") if not bars_y.empty else np.full(len(trades), np.nan)
            sig_y = asof_join_signal(trades, bars_y, "L_net") if not bars_y.empty else sig_y30

            for tau in TAUS_SEC:
                mid_t = mid_at_horizon(trades["timestamp"].values, bbo_ts, bbo_mid, tau * 1_000_000)
                pnl = compute_pnl(trades, mid_t)
                w = trades["w"].values
                base = weighted_mean(pnl, w)
                pnl_by_tau.append(
                    {
                        "day": str(day.date()),
                        "symbol": sym,
                        "tau": tau,
                        "pnl_all_bps": base,
                        "n": int(np.isfinite(pnl).sum()),
                    }
                )

                for sig, sig_name in [
                    (sig_y30, "bybit_L_net_30s"),
                    (sig_b30, "binance_L_net_30s"),
                    (sig_y, "bybit_L_net_1s"),
                ]:
                    dt = decile_table(sig, pnl, w)
                    if dt.empty:
                        continue
                    dt["day"] = str(day.date())
                    dt["symbol"] = sym
                    dt["tau"] = tau
                    dt["signal"] = sig_name
                    all_deciles.append(dt)

    dec_df = pd.concat(all_deciles, ignore_index=True) if all_deciles else pd.DataFrame()
    dec_df.to_csv(TABLES / "decile_pnl.csv", index=False)
    pnl_df = pd.DataFrame(pnl_by_tau)
    pnl_df.to_csv(TABLES / "baseline_pnl_sample.csv", index=False)

    if not dec_df.empty:
        for sig_name in dec_df["signal"].unique():
            sub = dec_df[dec_df["signal"] == sig_name]
            fig, axes = plt.subplots(1, len(TAUS_SEC), figsize=(4 * len(TAUS_SEC), 4), sharey=True)
            if len(TAUS_SEC) == 1:
                axes = [axes]
            for ax, tau in zip(axes, TAUS_SEC):
                g = sub[sub["tau"] == tau].groupby("decile")["pnl_bps"].mean()
                ax.plot(g.index, g.values, marker="o")
                ax.axhline(0, color="gray", ls="--", lw=0.8)
                ax.set_title(f"τ={tau}s")
                ax.set_xlabel("decile (low→high signal)")
            fig.suptitle(f"Maker PnL by signal decile — {sig_name}")
            fig.tight_layout()
            fig.savefig(FIGURES / f"04_deciles_{sig_name}.png", dpi=120)
            plt.close(fig)

    # conditional by maker side (BTC, last sample day)
    day = SAMPLE_DAYS[-1]
    t0, t1 = day_bounds(day)
    sym = "btcusdt"
    p = paths_for_symbol(sym)
    trades = load_trades_subsample(p["trades"], t0, t1, max_rows=200_000)
    if not trades.empty:
        trades = trade_features(trades)
        bbo = load_parquet_window(p["bbo"], t0, t1 + max(TAUS_SEC) * 1_000_000)
        ms = bbo_mid_spread(bbo)
        liq_y = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
        liq_y = liq_y[(liq_y["timestamp"] >= t0 - 60_000_000) & (liq_y["timestamp"] <= t1)]
        bars_y = liq_to_bars(liq_y, 1000)
        if not bars_y.empty:
            bars_y = bars_y.sort_values("ts_us")
            bars_y["L_net_30s"] = bars_y["L_net"].rolling(30, min_periods=1).sum()
        sig = asof_join_signal(trades, bars_y, "L_net_30s")
        mid_t = mid_at_horizon(trades["timestamp"].values, ms["timestamp"].values, ms["mid"].values, 30 * 1_000_000)
        pnl = compute_pnl(trades, mid_t)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, maker in zip(axes, ["sell", "buy"]):
            mask = (trades["maker_side"] == maker).values
            dt = decile_table(sig[mask], pnl[mask], trades.loc[mask, "w"].values)
            if not dt.empty:
                ax.plot(dt["decile"], dt["pnl_bps"], marker="o")
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.set_title(f"maker {maker}, τ=30s")
            ax.set_xlabel("bybit L_net_30s decile")
            ax.set_ylabel("PnL bps")
        fig.suptitle("Conditional markout by maker side (BTC sample day)")
        fig.tight_layout()
        fig.savefig(FIGURES / "05_conditional_markout_side.png", dpi=120)
        plt.close(fig)

    return dec_df, pnl_df


# ---------------------------------------------------------------------------
# Phase 6: latency sweep on bybit shift
# ---------------------------------------------------------------------------
def phase_latency_sweep():
    day = SAMPLE_DAYS[1]
    t0, t1 = day_bounds(day)
    sym = "btcusdt"
    p = paths_for_symbol(sym)
    trades = load_trades_subsample(p["trades"], t0, t1, max_rows=100_000)
    trades = trade_features(trades)
    bbo = load_parquet_window(p["bbo"], t0, t1 + 300_000_000)
    ms = bbo_mid_spread(bbo)
    mid30 = mid_at_horizon(trades["timestamp"].values, ms["timestamp"].values, ms["mid"].values, 30_000_000)
    pnl = compute_pnl(trades, mid30)
    w = trades["w"].values

    shifts_ms = [0, 100, 200, 300, 500, 1000]
    rows = []
    for sh in shifts_ms:
        liq = load_liquidations(p["liq_bybit"], shift_us=sh * 1000)
        liq = liq[(liq["timestamp"] >= t0 - 120_000_000) & (liq["timestamp"] <= t1)]
        bars = liq_to_bars(liq, 1000)
        bars["L_net_30s"] = bars["L_net"].rolling(30, min_periods=1).sum()
        sig = asof_join_signal(trades, bars, "L_net_30s")
        dt = decile_table(sig, pnl, w)
        if dt.empty:
            spread = np.nan
        else:
            spread = float(dt["pnl_bps"].iloc[-1] - dt["pnl_bps"].iloc[0])
        rows.append({"shift_ms": sh, "decile_spread_bps": spread})

    sw = pd.DataFrame(rows)
    sw.to_csv(TABLES / "latency_sweep.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sw["shift_ms"], sw["decile_spread_bps"], marker="o")
    ax.axvline(200, color="red", ls="--", label="spec +200ms")
    ax.set_xlabel("Bybit liq timestamp shift (ms)")
    ax.set_ylabel("PnL spread: top - bottom decile (30s τ)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_latency_sweep.png", dpi=120)
    plt.close(fig)
    return sw


# ---------------------------------------------------------------------------
# Phase 7: regime — liq burst vs quiet
# ---------------------------------------------------------------------------
def phase_regimes():
    rows = []
    day = SAMPLE_DAYS[2]
    t0, t1 = day_bounds(day)
    for sym in SYMBOLS:
        p = paths_for_symbol(sym)
        trades = load_trades_subsample(p["trades"], t0, t1, max_rows=120_000)
        trades = trade_features(trades)
        bbo = load_parquet_window(p["bbo"], t0, t1 + 300_000_000)
        ms = bbo_mid_spread(bbo)
        pnl = compute_pnl(
            trades,
            mid_at_horizon(trades["timestamp"].values, ms["timestamp"].values, ms["mid"].values, 30_000_000),
        )
        liq = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
        liq = liq[(liq["timestamp"] >= t0 - 60_000_000) & (liq["timestamp"] <= t1)]
        bars = liq_to_bars(liq, 1000)
        bars["L_abs_60s"] = bars["L_abs"].rolling(60, min_periods=1).sum()
        sig = asof_join_signal(trades, bars, "L_abs_60s")
        finite = np.isfinite(sig)
        if finite.sum() < 100:
            continue
        thresh = np.nanpercentile(sig[finite], 95)
        regime = np.where(finite & (sig >= thresh), "burst", "quiet")
        for reg in ("burst", "quiet"):
            m = regime == reg
            rows.append(
                {
                    "symbol": sym,
                    "regime": reg,
                    "pnl30_bps": weighted_mean(pnl[m], trades["w"].values[m]),
                    "n": int(m.sum()),
                }
            )
    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(TABLES / "regime_pnl.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(reg_df))
    ax.bar(x, reg_df["pnl30_bps"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['symbol']}\n{r['regime']}" for _, r in reg_df.iterrows()], fontsize=8)
    ax.set_ylabel("weighted PnL bps (30s)")
    ax.set_title("Maker PnL: liq burst (p95 L_abs_60s) vs quiet")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_regime_burst.png", dpi=120)
    plt.close(fig)
    return reg_df


# ---------------------------------------------------------------------------
# Phase 9: simple filter sweep (rule-based proxy)
# ---------------------------------------------------------------------------
def phase_filter_sweep():
    """Score vs kept turnover for |bybit L_net_30s| filter."""
    day = SAMPLE_DAYS[1]
    t0, t1 = day_bounds(day)
    sym = "btcusdt"
    p = paths_for_symbol(sym)
    trades = load_trades_subsample(p["trades"], t0, t1, max_rows=200_000)
    trades = trade_features(trades)
    bbo = load_parquet_window(p["bbo"], t0, t1 + max(TAUS_SEC) * 1_000_000)
    ms = bbo_mid_spread(bbo)

    liq = load_liquidations(p["liq_bybit"], shift_us=BYBIT_LATENCY_US)
    liq = liq[(liq["timestamp"] >= t0 - 120_000_000) & (liq["timestamp"] <= t1)]
    bars = liq_to_bars(liq, 1000).sort_values("ts_us")
    bars["L_net_30s"] = bars["L_net"].rolling(30, min_periods=1).sum()
    trades = trades.reset_index(drop=True)
    sig_all = np.abs(asof_join_signal(trades, bars, "L_net_30s"))

    rows = []
    for tau in TAUS_SEC:
        pnl = compute_pnl(
            trades,
            mid_at_horizon(trades["timestamp"].values, ms["timestamp"].values, ms["mid"].values, tau * 1_000_000),
        )
        w = trades["w"].values
        m = np.isfinite(pnl) & np.isfinite(sig_all)
        pnl, w, sig = pnl[m], w[m], sig_all[m]
        if len(pnl) < 100:
            continue
        pnl_all = weighted_mean(pnl, w)
        for pct in [50, 70, 80, 90, 95, 99]:
            thr = np.nanpercentile(sig[np.isfinite(sig)], pct)
            if not np.isfinite(thr):
                continue
            kept = sig < thr
            if kept.sum() == 0:
                continue
            pnl_kept = weighted_mean(pnl[kept], w[kept])
            turnover_day = w[kept].sum()  # one day sample
            rows.append(
                {
                    "tau": tau,
                    "filter_pct": pct,
                    "pnl_all": pnl_all,
                    "pnl_kept": pnl_kept,
                    "score": pnl_kept - pnl_all,
                    "kept_turnover_usd": turnover_day,
                    "kept_frac": float(kept.mean()),
                }
            )
    sw = pd.DataFrame(rows)
    sw.to_csv(TABLES / "filter_sweep_sample_day.csv", index=False)

    if not sw.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for tau in TAUS_SEC:
            s = sw[sw["tau"] == tau]
            if s.empty:
                continue
            ax.plot(s["kept_turnover_usd"] / 1e6, s["score"], marker="o", label=f"τ={tau}s")
        ax.set_xlabel("kept clipped turnover (USD, one sample day)")
        ax.set_ylabel("Score = PnL_kept - PnL_all (bps)")
        ax.legend()
        ax.set_title("Rule filter: drop top |bybit L_net_30s| trades")
        fig.tight_layout()
        fig.savefig(FIGURES / "08_score_vs_turnover.png", dpi=120)
        plt.close(fig)
    return sw


def write_notes(summary, ll_df, pnl_df, reg_df):
    lines = [
        "# EDA notes (auto-generated)\n",
        "## Data overview\n",
        summary.to_string(index=False),
        "\n## Lead-lag (peak |corr|)\n",
    ]
    if not ll_df.empty:
        for (sym, src), g in ll_df.groupby(["symbol", "source"]):
            i = g["corr"].abs().idxmax()
            lines.append(f"- {sym} {src}: best lag={g.loc[i,'lag_sec']}s corr={g.loc[i,'corr']:.4f}\n")
    lines.append("\n## Baseline PnL (sample days, subsampled trades)\n")
    if pnl_df is not None and not pnl_df.empty:
        lines.append(pnl_df.groupby(["symbol", "tau"])["pnl_all_bps"].mean().to_string())
        lines.append("\n")
    lines.append("\n## Regime (30s PnL)\n")
    if reg_df is not None and not reg_df.empty:
        lines.append(reg_df.to_string(index=False))
        lines.append("\n")
    lines.append(
        "\n## Next steps\n"
        "- Scale markout to full validation month (chunked).\n"
        "- Train ML filter on liq features + trade context; tune threshold for 500k/day.\n"
        "- Verify decile monotonicity stable train→val.\n"
    )
    (Path(__file__).parent / "notes.md").write_text("".join(lines), encoding="utf-8")


def main():
    print("Phase: liq overview")
    summary = phase_liq_overview()
    print("Phase: lead-lag")
    ll_df = phase_lead_lag()
    print("Phase: markout deciles")
    dec_df, pnl_df = phase_markout_deciles()
    print("Phase: latency sweep")
    phase_latency_sweep()
    print("Phase: regimes")
    reg_df = phase_regimes()
    print("Phase: filter sweep")
    phase_filter_sweep()
    write_notes(summary, ll_df, pnl_df, reg_df)
    print(f"Done. Figures -> {FIGURES}, tables -> {TABLES}")


if __name__ == "__main__":
    main()
