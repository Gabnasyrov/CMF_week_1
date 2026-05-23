#!/usr/bin/env python3
"""
1. Bayes trend as standalone forward-return predictor (5s, 60s).
2. Parametric Hawkes MLE (μ, α, κ) per venue/symbol.
3. Lead-lag inside up/down regimes: microprice vs OFI → Bybit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from bayes_trend import compute_trend_on_bbo  # noqa: E402
from bayes_trend_forecast import evaluate_trend_forecast  # noqa: E402
from config import SYMBOLS, enriched_paths  # noqa: E402
from cross_lead_analysis import (  # noqa: E402
    TRAIN_FRAC,
    attach_trend,
    build_bbo_1s,
    build_bybit_1s,
    lead_lag_table,
    merge_panel,
)
from hawkes_liq import fit_hawkes_mle, hawkes_intensity_at  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "trend_hawkes"
TAB = Path(__file__).resolve().parent / "tables" / "trend_hawkes"
CROSS_TAB = Path(__file__).resolve().parent / "tables" / "cross_lead"


def ensure():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def load_or_build_trend(sym: str, bbo: pd.DataFrame | None, freq: str) -> pd.DataFrame:
    cached = CROSS_TAB / f"06_bayes_trend_{freq}_{sym}.csv"
    if cached.exists():
        return pd.read_csv(cached)
    if bbo is None:
        bbo = build_bbo_1s(sym)
    return compute_trend_on_bbo(bbo, freq)


def lead_lag_regime_panel(
    panel: pd.DataFrame,
    signal: str,
    target: str,
    trend_col: str,
    sym: str,
    label: str,
) -> pd.DataFrame:
    cut = int(len(panel) * TRAIN_FRAC)
    rows = []
    for regime, rname in [(-1, "down"), (1, "up")]:
        for split_name, sub in [("train", panel.iloc[:cut]), ("test", panel.iloc[cut:])]:
            mask = (sub[trend_col] == regime).values
            if mask.sum() < 500:
                continue
            ll = lead_lag_table(sub, signal, target, split_name, mask)
            if ll.empty:
                continue
            i = ll["rank_ic"].abs().idxmax()
            rows.append(
                {
                    "symbol": sym,
                    "signal": signal,
                    "target": target,
                    "trend_col": trend_col,
                    "regime": rname,
                    "split": split_name,
                    "best_lag_sec": float(ll.loc[i, "lag_sec"]),
                    "rank_ic": float(ll.loc[i, "rank_ic"]),
                    "corr": float(ll.loc[i, "corr"]),
                    "n": int(ll.loc[i, "n"]),
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def plot_forecast_hit(results: list, sym: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, hits = [], []
    for r in results:
        if not r.get("ok"):
            continue
        for k in ("regime_test", "slope_sign_test"):
            m = r.get(k, {})
            if m:
                labels.append(f"{r['freq']}_{k}")
                hits.append(m.get("hit_rate", 0))
    if not labels:
        return
    ax.barh(labels, hits)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("hit rate (sign match)")
    ax.set_title(f"Bayes trend forecast — {sym}")
    fig.tight_layout()
    fig.savefig(OUT / f"07_trend_forecast_{sym}.png", dpi=120)
    plt.close(fig)


def run_symbol(sym: str, bbo: pd.DataFrame | None = None) -> dict:
    print(sym)
    trend_res = []
    for freq in ("5s", "60s"):
        td = load_or_build_trend(sym, bbo, freq)
        ev = evaluate_trend_forecast(td, freq)
        trend_res.append(ev)
        pd.DataFrame([ev]).to_csv(TAB / f"07_trend_forecast_{freq}_{sym}.csv", index=False)

    hawkes_res = {}
    for venue, key in [("binance", "liq_binance"), ("bybit", "liq_bybit")]:
        liq = pd.read_parquet(enriched_paths(sym)[key])
        fit = fit_hawkes_mle(liq["timestamp"].values)
        hawkes_res[venue] = fit
        if fit.get("ok"):
            ts_us = liq["timestamp"].values.astype(np.int64)
            ev_sec = np.sort(ts_us) / 1_000_000.0
            t_end = ev_sec[-1]
            lam_end = hawkes_intensity_at(ev_sec, t_end, fit["mu"], fit["alpha"], fit["kappa"])
            fit["lambda_end"] = lam_end
        pd.DataFrame([{**fit, "venue": venue, "symbol": sym}]).to_csv(
            TAB / f"07_hawkes_{venue}_{sym}.csv", index=False
        )

    print(f"  {sym}: panel + regime lead-lag...")
    if bbo is None:
        bbo = build_bbo_1s(sym)
    bybit = build_bybit_1s(sym)
    t5 = load_or_build_trend(sym, bbo, "5s")
    t60 = load_or_build_trend(sym, bbo, "60s")
    panel = merge_panel(bbo, bybit)
    panel = attach_trend(panel, t5, "5s")
    panel = attach_trend(panel, t60, "60s")

    ll_rows = []
    for freq in ("5s", "60s"):
        col = f"trend_regime_{freq}"
        for signal, sig_label in [("ret_micro_bps", "microprice"), ("ofi_5s", "ofi_5s")]:
            df = lead_lag_regime_panel(
                panel, signal, "bybit_TFI_30s", col, sym, sig_label
            )
            if not df.empty:
                df["horizon"] = freq
                ll_rows.append(df)
    ll_all = pd.concat(ll_rows, ignore_index=True) if ll_rows else pd.DataFrame()
    if not ll_all.empty:
        ll_all.to_csv(TAB / f"07_leadlag_regime_{sym}.csv", index=False)

    plot_forecast_hit(trend_res, sym)
    return {"trend_forecast": trend_res, "hawkes": hawkes_res, "leadlag_regime": ll_all.to_dict("records")}


def write_report(all_res: dict) -> None:
    lines = [
        "# Bayes trend forecast & Hawkes / regime lead-lag",
        "",
        "## 0. Что значит C-index на ETH (survival)",
        "",
        "**C-index (concordance)** — доля пар burst, где модель правильно упорядочивает",
        "короткие/длинные `duration_sec`. 0.5 = случайное угадывание, 1.0 = идеал.",
        "",
        "Фраза «ETH Cox ~0.56–0.61, LGBM ~0.65–0.69, прирост скромнее» означала:",
        "на **ETH** extended-фичи поднимают C-index с ~0.53 (base) до ~0.60 (Cox) / ~0.69 (LGBM) —",
        "прирост **есть**, но **меньше**, чем на BTC (Cox ~0.60, LGBM ~0.68).",
        "Это метрики **прогноза длительности burst**, не lead-lag и не trend forecast.",
        "",
        "---",
        "",
    ]
    for sym, res in all_res.items():
        lines.append(f"## {sym.upper()}")
        lines.append("")
        lines.append("### 1. Bayes trend → следующий бар (standalone)")
        lines.append("")
        lines.append("| freq | split | method | hit_rate | rank_IC | slope×fwd_IC |")
        lines.append("|------|-------|--------|----------|---------|--------------|")
        for ev in res.get("trend_forecast", []):
            if not ev.get("ok"):
                continue
            for split, key in [("test", "regime_test"), ("test", "slope_sign_test")]:
                m = ev.get(key, {})
                if m:
                    lines.append(
                        f"| {ev['freq']} | {split} | {key} | {m.get('hit_rate', 0):.3f} "
                        f"| {m.get('rank_ic', 0):.4f} | {ev.get('slope_fwd_rank_ic_test', 0):.4f} |"
                    )
            lines.append(
                f"| {ev['freq']} | test | mean |fwd| bps | — | — | {ev.get('mean_abs_fwd_bps_test', 0):.2f} |"
            )
        lines.append("")
        lines.append(f"Figure: `figures/trend_hawkes/07_trend_forecast_{sym}.png`")
        lines.append("")
        lines.append("### 2. Hawkes MLE (μ, α, κ)")
        lines.append("")
        lines.append("| venue | μ | α | κ | branching α/κ | loglik |")
        lines.append("|-------|---|---|---|---------------|--------|")
        for venue, h in res.get("hawkes", {}).items():
            if not h.get("ok"):
                lines.append(f"| {venue} | — | — | — | — | {h.get('reason', 'fail')} |")
                continue
            lines.append(
                f"| {venue} | {h['mu']:.4g} | {h['alpha']:.4g} | {h['kappa']:.4g} | "
                f"{h['branching_ratio']:.3f} | {h['loglik']:.1f} |"
            )
        lines.append("")
        lines.append("### 3. Lead-lag внутри up/down (→ Bybit TFI_30s)")
        lines.append("")
        lines.append("| horizon | signal | regime | split | best_lag_s | rank_IC | n |")
        lines.append("|---------|--------|--------|-------|------------|---------|---|")
        for row in res.get("leadlag_regime", []):
            lines.append(
                f"| {row.get('horizon')} | {row.get('label')} | {row.get('regime')} | {row.get('split')} "
                f"| {row.get('best_lag_sec')} | {row.get('rank_ic', 0):.4f} | {row.get('n')} |"
            )
        lines.append("")
    Path(__file__).resolve().parent.joinpath("TREND_HAWKES_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    ensure()
    all_res = {}
    for sym in SYMBOLS:
        all_res[sym] = run_symbol(sym)
    with open(TAB / "summary.json", "w") as f:
        json.dump(all_res, f, indent=2, default=float)
    write_report(all_res)
    print("Done", OUT, TAB)


if __name__ == "__main__":
    main()
