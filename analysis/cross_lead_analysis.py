#!/usr/bin/env python3
"""
1. Lag: Binance OFI shift -> Bybit reaction (liq flow / vwap ret proxy)
2. Binance microprice lead -> Bybit metrics; time advantage
3. Liquidation burst duration vs book state (imb, spread, microprice, OFI)
4. Signed volume & L1 depth by side (1% depth needs L2+ — L1 proxy documented)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from bayes_trend import compute_trend_on_bbo  # noqa: E402
from burst_survival import run_burst_survival  # noqa: E402
from config import BYBIT_LATENCY_US, SYMBOLS, enriched_paths  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "cross_lead"
TAB = Path(__file__).resolve().parent / "tables" / "cross_lead"
TRAIN_FRAC = 0.6
LAGS_SEC = np.array([-30, -10, -5, -2, -1, 0, 1, 2, 5, 10, 30, 60, 120], dtype=float)
MAX_BBO_DAYS = None  # full sample (~90 days)
HOUR_US = 3_600_000_000


def ensure():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def build_bbo_1s(sym: str) -> pd.DataFrame:
    path = enriched_paths(sym)["bbo"]
    pf = pq.ParquetFile(path)
    cols = [
        "timestamp",
        "mid",
        "microprice",
        "microprice_g",
        "book_imbalance",
        "spread_bps",
        "ofi_increment",
        "bid_price",
        "ask_price",
        "bid_amount",
        "ask_amount",
    ]
    chunks = []
    seen = set()
    for rg in range(pf.metadata.num_row_groups):
        day = int(pc.min(pf.read_row_group(rg, columns=["timestamp"]).column("timestamp")).as_py() // 86_400_000_000)
        if day in seen:
            continue
        seen.add(day)
        if MAX_BBO_DAYS is not None and len(seen) > MAX_BBO_DAYS:
            break
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df.index = pd.to_datetime(df["timestamp"], unit="us", utc=True)
        s = df.resample("1s").last()
        s["bid_depth_usd"] = s["bid_price"] * s["bid_amount"]
        s["ask_depth_usd"] = s["ask_price"] * s["ask_amount"]
        s["signed_depth_l1"] = s["bid_depth_usd"] - s["ask_depth_usd"]
        s["depth_l1_total"] = s["bid_depth_usd"] + s["ask_depth_usd"]
        s["ret_micro_bps"] = (s["microprice"] / s["microprice"].shift(1) - 1) * 10_000
        s["ofi_5s"] = s["ofi_increment"].rolling(5, min_periods=1).sum()
        s = s.dropna(subset=["timestamp"])
        s["ts_us"] = s["timestamp"].to_numpy(dtype=np.int64)
        chunks.append(s.reset_index(drop=True))
    return pd.concat(chunks, ignore_index=True).dropna(subset=["mid"])


def build_bybit_1s(sym: str) -> pd.DataFrame:
    y = pd.read_parquet(enriched_paths(sym)["liq_bybit"]).sort_values("timestamp")
    y["notional"] = y["price"] * y["amount"]
    y["signed"] = np.where(y["side"] == "buy", y["notional"], -y["notional"])
    y["px_amt"] = y["price"] * y["amount"]
    y["sec"] = (y["timestamp"] // 1_000_000).astype(np.int64)
    g = y.groupby("sec", sort=True)
    bars = pd.DataFrame(
        {
            "ts_us": g["timestamp"].max(),
            "bybit_TFI": g["signed"].sum(),
            "bybit_L_abs": g["notional"].sum(),
            "px_amt": g["px_amt"].sum(),
            "amt": g["amount"].sum(),
            "bybit_n": g.size(),
        }
    ).reset_index(drop=True)
    bars["bybit_vwap"] = bars["px_amt"] / bars["amt"].replace(0, np.nan)
    bars = bars.drop(columns=["px_amt", "amt"])
    bars["bybit_vwap_ret_bps"] = (bars["bybit_vwap"] / bars["bybit_vwap"].shift(1) - 1) * 10_000
    bars["bybit_TFI_30s"] = bars["bybit_TFI"].rolling(30, min_periods=1).sum()
    return bars


def attach_trend(panel: pd.DataFrame, trend: pd.DataFrame, freq: str) -> pd.DataFrame:
    cols = [f"trend_regime_{freq}", f"trend_slope_{freq}", f"cp_cred_{freq}"]
    t = trend[["ts_us"] + [c for c in cols if c in trend.columns]].sort_values("ts_us")
    bts = panel["ts_us"].values
    tts = t["ts_us"].values
    out = panel.copy()
    idx = np.searchsorted(tts, bts, side="right") - 1
    idx = np.clip(idx, 0, max(len(tts) - 1, 0))
    for c in cols:
        if c in t.columns:
            out[c] = t[c].values[idx]
    return out


def lead_lag_by_trend(
    panel: pd.DataFrame, signal: str, target: str, trend_col: str, sym: str
) -> pd.DataFrame:
    cut = int(len(panel) * TRAIN_FRAC)
    rows = []
    for regime, label in [(-1, "down"), (0, "flat"), (1, "up")]:
        if trend_col not in panel.columns:
            break
        for split_name, sub in [("train", panel.iloc[:cut]), ("test", panel.iloc[cut:])]:
            mask = (sub[trend_col] == regime).values
            ll = lead_lag_table(sub, signal, target, split_name, mask)
            if ll.empty:
                continue
            ll["regime"] = label
            ll["trend_col"] = trend_col
            rows.append(ll)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(TAB / f"06_leadlag_{signal}_{target}_by_{trend_col}_{sym}.csv", index=False)
    return df


def plot_trend_regimes(bbo: pd.DataFrame, trend: pd.DataFrame, freq: str, sym: str, hours: int = 24):
    """Sample window: microprice colored by Bayes trend regime."""
    if trend.empty or len(bbo) < 1000:
        return
    t0 = int(bbo["ts_us"].iloc[len(bbo) // 2])
    t1 = t0 + hours * HOUR_US
    sl = bbo[(bbo["ts_us"] >= t0) & (bbo["ts_us"] <= t1)]
    if len(sl) < 100:
        return
    tr = trend[(trend["ts_us"] >= t0) & (trend["ts_us"] <= t1)]
    col = f"trend_regime_{freq}"
    if col not in tr.columns:
        return
    bts = sl["ts_us"].values
    tts = tr["ts_us"].values
    idx = np.clip(np.searchsorted(tts, bts, side="right") - 1, 0, len(tts) - 1)
    regimes = tr[col].values[idx]
    ts_h = (sl["ts_us"] - t0) / HOUR_US
    mp = sl["microprice"].values
    colors = {1: "green", 0: "gray", -1: "red"}
    fig, ax = plt.subplots(figsize=(12, 3))
    for r, c in colors.items():
        m = regimes == r
        if m.any():
            ax.scatter(ts_h[m], mp[m], s=1, c=c, label={1: "up", 0: "flat", -1: "down"}[r], alpha=0.5)
    ax.set_xlabel("hours")
    ax.set_ylabel("microprice")
    ax.set_title(f"Bayes trend {freq} — {sym} ({hours}h sample)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"06_trend_{freq}_{sym}.png", dpi=120)
    plt.close(fig)


def merge_panel(bbo: pd.DataFrame, bybit: pd.DataFrame) -> pd.DataFrame:
    m = bbo.merge(bybit, on="ts_us", how="left")
    for col in ("bybit_TFI", "bybit_TFI_30s", "bybit_L_abs", "bybit_vwap_ret_bps"):
        if col in m.columns:
            m[col] = m[col].fillna(0)
    return m.sort_values("ts_us").reset_index(drop=True)


def lead_lag_table(panel: pd.DataFrame, signal: str, target: str, split_name: str, mask: np.ndarray) -> pd.DataFrame:
    sub = panel.loc[mask].dropna(subset=[signal, target])
    if len(sub) < 500:
        return pd.DataFrame()
    ts = sub["ts_us"].values.astype(np.int64)
    sig = sub[signal].values.astype(np.float64)
    tgt = sub[target].values.astype(np.float64)
    rows = []
    for lag in LAGS_SEC:
        lag_us = int(lag * 1_000_000)
        aligned = np.full(len(sig), np.nan)
        idx = np.searchsorted(ts, ts - lag_us, side="right") - 1
        ok = idx >= 0
        aligned[ok] = sig[idx[ok]]
        m = np.isfinite(aligned) & np.isfinite(tgt)
        if m.sum() < 100:
            c = ic = np.nan
        else:
            c = float(np.corrcoef(aligned[m], tgt[m])[0, 1])
            ic = float(pd.Series(aligned[m]).corr(pd.Series(tgt[m]), method="spearman"))
        rows.append({"lag_sec": lag, "corr": c, "rank_ic": ic, "n": int(m.sum()), "split": split_name})
    return pd.DataFrame(rows)


def stability_best_lag(train_df: pd.DataFrame, test_df: pd.DataFrame, signal: str, target: str) -> dict:
    tr = lead_lag_table(train_df, signal, target, "train", np.ones(len(train_df), bool))
    te = lead_lag_table(test_df, signal, target, "test", np.ones(len(test_df), bool))
    if tr.empty or te.empty or "split" not in tr.columns:
        return {}
    i_tr = tr["rank_ic"].abs().idxmax()
    i_te = te["rank_ic"].abs().idxmax()
    return {
        "best_lag_train": float(tr.loc[i_tr, "lag_sec"]),
        "rank_ic_train": float(tr.loc[i_tr, "rank_ic"]),
        "best_lag_test": float(te.loc[i_te, "lag_sec"]),
        "rank_ic_test": float(te.loc[i_te, "rank_ic"]),
        "lag_stable": abs(tr.loc[i_tr, "lag_sec"] - te.loc[i_te, "lag_sec"]) <= 5,
    }


def plot_leadlag(df: pd.DataFrame, title: str, fname: str):
    if df.empty or "split" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for split, style in [("train", "-"), ("test", "--")]:
        s = df[df["split"] == split]
        if s.empty:
            continue
        ax.plot(s["lag_sec"], s["rank_ic"], style + "o", label=f"{split} rank IC")
        ax.plot(s["lag_sec"], s["corr"], style + "x", alpha=0.6, label=f"{split} corr")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("lag sec (Binance signal leads Bybit target)")
    ax.set_ylabel("association")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)


def microprice_lead_advantage(panel: pd.DataFrame) -> pd.DataFrame:
    """Peak lag where Binance microprice ret predicts Bybit TFI_30s / vwap ret."""
    rows = []
    for target in ("bybit_TFI_30s", "bybit_vwap_ret_bps", "bybit_L_abs"):
        cut = int(len(panel) * TRAIN_FRAC)
        tr, te = panel.iloc[:cut], panel.iloc[cut:]
        for name, sub in [("train", tr), ("test", te)]:
            ll = lead_lag_table(sub, "ret_micro_bps", target, name, np.ones(len(sub), bool))
            if ll.empty:
                continue
            i = ll["rank_ic"].abs().idxmax()
            rows.append(
                {
                    "target": target,
                    "split": name,
                    "best_lag_sec": float(ll.loc[i, "lag_sec"]),
                    "rank_ic": float(ll.loc[i, "rank_ic"]),
                    "corr": float(ll.loc[i, "corr"]),
                }
            )
    return pd.DataFrame(rows)


def detect_bursts(liq: pd.DataFrame, gap_sec: int = 30, min_events: int = 3) -> pd.DataFrame:
    liq = liq.sort_values("timestamp").reset_index(drop=True)
    if "notional" not in liq.columns:
        liq = liq.copy()
        liq["notional"] = liq["price"] * liq["amount"]
    ts = liq["timestamp"].values
    bursts = []
    start = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] > gap_sec * 1_000_000:
            if i - start >= min_events:
                bursts.append((start, i - 1))
            start = i
    if len(ts) - start >= min_events:
        bursts.append((start, len(ts) - 1))
    rows = []
    for a, b in bursts:
        sl = liq.iloc[a : b + 1]
        t0, t1 = int(sl["timestamp"].iloc[0]), int(sl["timestamp"].iloc[-1])
        rows.append(
            {
                "t_start": t0,
                "t_end": t1,
                "duration_sec": (t1 - t0) / 1_000_000,
                "n_events": len(sl),
                "total_notional": float((sl["price"] * sl["amount"]).sum()),
                "signed_notional": float(
                    np.where(sl["side"] == "buy", sl["notional"], -sl["notional"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def burst_book_correlations(sym: str, venue: str, bbo: pd.DataFrame) -> pd.DataFrame:
    if venue == "binance":
        liq = pd.read_parquet(enriched_paths(sym)["liq_binance"])
    else:
        liq = pd.read_parquet(enriched_paths(sym)["liq_bybit"])
    bursts = detect_bursts(liq)
    if bursts.empty:
        return pd.DataFrame()
    bts = bbo["ts_us"].values
    rec = []
    for _, b in bursts.iterrows():
        idx = np.searchsorted(bts, b["t_start"], side="right") - 1
        if idx < 0:
            continue
        row = bbo.iloc[idx]
        rec.append(
            {
                "duration_sec": b["duration_sec"],
                "total_notional": b["total_notional"],
                "n_events": b["n_events"],
                "spread_bps": row["spread_bps"],
                "book_imbalance": row["book_imbalance"],
                "microprice_g": row["microprice_g"],
                "ofi_5s": row["ofi_5s"],
                "bid_depth_usd": row["bid_depth_usd"],
                "ask_depth_usd": row["ask_depth_usd"],
            }
        )
    df = pd.DataFrame(rec)
    if len(df) < 20:
        return df
    corr_rows = []
    for feat in ["spread_bps", "book_imbalance", "microprice_g", "ofi_5s", "bid_depth_usd", "ask_depth_usd"]:
        for tgt in ["duration_sec", "total_notional", "n_events"]:
            c = float(df[feat].corr(df[tgt], method="spearman"))
            corr_rows.append({"venue": venue, "symbol": sym, "feature": feat, "target": tgt, "spearman": c})
    return pd.DataFrame(corr_rows)


def run_symbol(sym: str) -> dict:
    print(f"  {sym}: bbo 1s...")
    bbo = build_bbo_1s(sym)
    print(f"  {sym}: bayes trend 5s / 60s...")
    trend_5s = compute_trend_on_bbo(bbo, "5s")
    trend_60s = compute_trend_on_bbo(bbo, "60s")
    trend_5s.to_csv(TAB / f"06_bayes_trend_5s_{sym}.csv", index=False)
    trend_60s.to_csv(TAB / f"06_bayes_trend_60s_{sym}.csv", index=False)
    plot_trend_regimes(bbo, trend_5s, "5s", sym)
    plot_trend_regimes(bbo, trend_60s, "60s", sym)

    print(f"  {sym}: bybit 1s...")
    bybit = build_bybit_1s(sym)
    panel = merge_panel(bbo, bybit)
    panel = attach_trend(panel, trend_5s, "5s")
    panel = attach_trend(panel, trend_60s, "60s")
    cut = int(len(panel) * TRAIN_FRAC)
    tr, te = panel.iloc[:cut], panel.iloc[cut:]

    # Q1 OFI -> Bybit reaction
    targets = {
        "bybit_vwap_ret_bps": "Bybit liq VWAP 1s return (proxy, no Bybit BBO)",
        "bybit_TFI_30s": "Bybit signed liq USD 30s",
        "bybit_L_abs": "Bybit liq intensity 1s",
    }
    q1 = {}
    for tgt, desc in targets.items():
        ll_tr = lead_lag_table(tr, "ofi_5s", tgt, "train", np.ones(len(tr), bool))
        ll_te = lead_lag_table(te, "ofi_5s", tgt, "test", np.ones(len(te), bool))
        ll = pd.concat([ll_tr, ll_te], ignore_index=True)
        ll.to_csv(TAB / f"01_ofi_to_{tgt}_{sym}.csv", index=False)
        plot_leadlag(ll, f"Binance OFI_5s -> {tgt} ({sym})", f"01_ofi_{tgt}_{sym}.png")
        stab = stability_best_lag(tr, te, "ofi_5s", tgt)
        q1[tgt] = stab

    # Q2 microprice lead
    mp = microprice_lead_advantage(panel)
    mp.to_csv(TAB / f"02_microprice_lead_{sym}.csv", index=False)
    ll_mp = lead_lag_table(panel.iloc[:cut], "ret_micro_bps", "bybit_TFI_30s", "train", np.ones(cut, bool))
    ll_mp_te = lead_lag_table(panel.iloc[cut:], "ret_micro_bps", "bybit_TFI_30s", "test", np.ones(len(te), bool))
    plot_leadlag(pd.concat([ll_mp, ll_mp_te]), f"Binance microprice ret -> Bybit TFI_30s ({sym})", f"02_microprice_{sym}.png")

    # Q6 lead-lag stratified by Bayes trend
    trend_ll = {}
    for freq in ("5s", "60s"):
        col = f"trend_regime_{freq}"
        ll = lead_lag_by_trend(panel, "ofi_5s", "bybit_TFI_30s", col, sym)
        if not ll.empty:
            def _peak_ic(s: pd.Series) -> float:
                s = s.dropna()
                if s.empty:
                    return float("nan")
                return float(s.iloc[s.abs().argmax()])

            i = ll.groupby(["regime", "split"])["rank_ic"].apply(_peak_ic)
            trend_ll[freq] = {f"{reg}_{spl}": float(v) for (reg, spl), v in i.items()}

    liq_b = pd.read_parquet(enriched_paths(sym)["liq_binance"])
    liq_y = pd.read_parquet(enriched_paths(sym)["liq_bybit"])
    bursts_b = detect_bursts(liq_b)
    bursts_y = detect_bursts(liq_y)

    # Q3 burst correlations + Q5 survival
    c_b = burst_book_correlations(sym, "binance", bbo)
    c_y = burst_book_correlations(sym, "bybit", bbo)
    c_all = pd.concat([c_b, c_y], ignore_index=True)
    c_all.to_csv(TAB / f"03_burst_corr_{sym}.csv", index=False)

    print(f"  {sym}: burst survival (extended features)...")
    surv = {}
    for venue in ("binance", "bybit"):
        bursts = bursts_b if venue == "binance" else bursts_y
        other = bursts_y if venue == "binance" else bursts_b
        surv[venue] = run_burst_survival(
            sym,
            venue,
            bursts,
            bbo,
            bybit,
            OUT,
            TAB,
            bursts_other=other,
            liq_binance=liq_b,
            liq_bybit=liq_y,
            trend_5s=trend_5s,
            trend_60s=trend_60s,
        )

    # Q4 depth at liq
    liq_enr = pd.read_parquet(enriched_paths(sym)["liq_binance"])
    bts = bbo["ts_us"].values
    ts = liq_enr["timestamp"].values
    idx = np.searchsorted(bts, ts, side="right") - 1
    ok = idx >= 0
    depth_df = liq_enr.loc[ok, ["timestamp", "side", "book_imbalance", "microprice_g"]].copy()
    depth_df["spread_bps"] = bbo["spread_bps"].values[idx[ok]]
    depth_df["bid_depth_usd"] = bbo["bid_depth_usd"].values[idx[ok]]
    depth_df["ask_depth_usd"] = bbo["ask_depth_usd"].values[idx[ok]]
    depth_df["signed_depth_l1"] = depth_df["bid_depth_usd"] - depth_df["ask_depth_usd"]
    if "order_flow_imbalance_1s" in liq_enr.columns:
        depth_df["signed_vol_1s"] = liq_enr.loc[ok, "order_flow_imbalance_1s"].values
    depth_df.to_csv(TAB / f"04_depth_at_liq_{sym}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, title in zip(
        axes,
        ["bid_depth_usd", "ask_depth_usd", "signed_depth_l1"],
        ["bid L1 depth $", "ask L1 depth $", "signed L1 depth"],
    ):
        ax.hist(np.log10(depth_df[col].clip(lower=1)), bins=40, alpha=0.8)
        ax.set_title(title)
    fig.suptitle(f"Depth at Binance liq events ({sym}) — L1 proxy, not full 1% book")
    fig.tight_layout()
    fig.savefig(OUT / f"04_depth_liq_{sym}.png", dpi=120)
    plt.close(fig)

    regime_share = {}
    for freq in ("5s", "60s"):
        col = f"trend_regime_{freq}"
        if col in panel.columns:
            regime_share[freq] = panel[col].value_counts(normalize=True).to_dict()

    return {
        "q1": q1,
        "microprice_lead": mp.to_dict("records"),
        "burst_corr": c_all.to_dict("records"),
        "survival": surv,
        "trend_leadlag": trend_ll,
        "regime_share": regime_share,
        "panel_rows": len(panel),
        "bbo_days": int(bbo["ts_us"].max() - bbo["ts_us"].min()) // 86_400_000_000 + 1,
    }


def write_report(all_res: dict, note: dict, path: Path) -> None:
    lines = [
        "# Cross-venue lead / liquidation burst report",
        "",
        "## Data limits",
        f"- {note['data_limit']}",
        f"- {note['depth_1pct']}",
        f"- {note['bybit_timestamp']}",
        "",
        f"Panels: Binance BBO 1s (full sample, MAX_BBO_DAYS={MAX_BBO_DAYS}) merged with Bybit liq 1s bars.",
        "",
    ]
    for sym, res in all_res.items():
        lines.append(f"## {sym.upper()}")
        lines.append("")
        lines.append("### 1. Binance OFI_5s → Bybit reaction (lag stability)")
        lines.append("")
        lines.append("| target | best lag train (s) | rank IC train | best lag test (s) | rank IC test | stable (±5s) |")
        lines.append("|--------|-------------------|---------------|-------------------|--------------|--------------|")
        for tgt, stab in res.get("q1", {}).items():
            if not stab:
                lines.append(f"| {tgt} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {tgt} | {stab.get('best_lag_train', '—')} | {stab.get('rank_ic_train', '—'):.4f} "
                f"| {stab.get('best_lag_test', '—')} | {stab.get('rank_ic_test', '—'):.4f} "
                f"| {stab.get('lag_stable', '—')} |"
            )
        lines.append("")
        lines.append("Figures: `figures/cross_lead/01_ofi_*_{sym}.png`")
        lines.append("")
        lines.append("### 2. Binance microprice return → Bybit (time lead)")
        lines.append("")
        lines.append("| target | split | best lag (s) | rank IC | corr |")
        lines.append("|--------|-------|--------------|---------|------|")
        for row in res.get("microprice_lead", []):
            lines.append(
                f"| {row['target']} | {row['split']} | {row['best_lag_sec']} "
                f"| {row['rank_ic']:.4f} | {row['corr']:.4f} |"
            )
        lines.append("")
        lines.append("Positive lag = Binance microprice move precedes Bybit metric.")
        lines.append("")
        lines.append("### 3. Burst duration vs book state (Spearman at burst start)")
        lines.append("")
        corr = res.get("burst_corr", [])
        if corr:
            lines.append("| venue | feature | target | spearman |")
            lines.append("|-------|---------|--------|----------|")
            for row in sorted(corr, key=lambda r: abs(r["spearman"]), reverse=True)[:12]:
                lines.append(
                    f"| {row['venue']} | {row['feature']} | {row['target']} | {row['spearman']:.3f} |"
                )
        lines.append("")
        lines.append("### 4. Depth / signed flow at liq")
        lines.append("")
        lines.append(f"- Table: `tables/cross_lead/04_depth_at_liq_{sym}.csv`")
        lines.append(f"- Hist: `figures/cross_lead/04_depth_liq_{sym}.png` (L1 touch depth, not 1% book)")
        lines.append("")
        lines.append("### 5. Burst duration / depletion (Cox + LGBM)")
        lines.append("")
        lines.append(f"Panel rows: {res.get('panel_rows', '—')}, BBO span ~{res.get('bbo_days', '—')} days")
        lines.append("")
        for venue, s in res.get("survival", {}).items():
            if not s.get("cox", {}).get("ok") and not s.get("lgbm", {}).get("ok"):
                lines.append(f"- **{venue}**: insufficient bursts (n={s.get('n', 0)})")
                continue
            cox = s.get("cox", {})
            lgb = s.get("lgbm", {})
            lines.append(f"#### {venue}")
            lines.append(f"- bursts: {s.get('n_bursts')}, median duration: {s.get('median_duration_sec', 0):.1f}s")
            if cox.get("ok"):
                lines.append(
                    f"- Cox PH C-index train/test: {cox['c_index_train']:.3f} / {cox['c_index_test']:.3f}"
                )
            if lgb.get("ok"):
                lines.append(
                    f"- LGBM log-duration MAE test: {lgb['mae_sec_test']:.1f}s "
                    f"(baseline median MAE {lgb['median_baseline_mae']:.1f}s), "
                    f"R²={lgb['r2_test']:.3f}, C-index={lgb['c_index_test']:.3f}"
                )
            lines.append(f"- KM: `figures/cross_lead/05_km_spread_{venue}_{sym}.png`")
            lines.append(f"- Pred: `figures/cross_lead/05_lgbm_duration_{venue}_{sym}.png`")
            cb = s.get("cox_base", {})
            lb = s.get("lgbm_base", {})
            if cb.get("ok") and cox.get("ok"):
                lines.append(
                    f"- Cox C-index ext vs base (test): {cox['c_index_test']:.3f} vs {cb['c_index_test']:.3f}"
                )
            if lgb.get("ok") and lb.get("ok"):
                lines.append(
                    f"- LGBM C-index ext vs base (test): {lgb['c_index_test']:.3f} vs {lb['c_index_test']:.3f}"
                )
        lines.append("")
        lines.append("### 6. Bayes trend regimes (Schütz & Holschneider)")
        lines.append("")
        rs = res.get("regime_share", {})
        for freq, shares in rs.items():
            lines.append(f"- **{freq}** regime share: {shares}")
        lines.append(f"- Trend CSV: `tables/cross_lead/06_bayes_trend_{{5s,60s}}_{sym}.csv`")
        lines.append(f"- Plots: `figures/cross_lead/06_trend_{{5s,60s}}_{sym}.png`")
        tll = res.get("trend_leadlag", {})
        if tll:
            lines.append("- OFI→Bybit TFI peak rank IC by regime (see `06_leadlag_*_by_trend_regime_*`):")
            for freq, d in tll.items():
                lines.append(f"  - {freq}: {d}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure()
    all_res = {}
    for sym in SYMBOLS:
        print(sym)
        all_res[sym] = run_symbol(sym)

    note = {
        "data_limit": "No Bybit BBO/trades — Bybit 'return' = liq VWAP 1s change or signed liq flow",
        "depth_1pct": "Only L1 in dataset; bid/ask_depth_usd = touch notional; true 1% depth needs deeper book",
        "bybit_timestamp": f"+{BYBIT_LATENCY_US} us applied in enriched bybit liq",
    }
    with open(TAB / "summary.json", "w") as f:
        json.dump({"note": note, "results": all_res}, f, indent=2, default=float)

    report = Path(__file__).resolve().parent / "CROSS_LEAD_REPORT.md"
    write_report(all_res, note, report)
    print("Done", OUT, TAB)


if __name__ == "__main__":
    main()
