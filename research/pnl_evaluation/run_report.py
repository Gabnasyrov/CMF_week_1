#!/usr/bin/env python3
"""
PnL report: train 50% / test 50% chronological, all trades, chunked by day (OOM-safe).
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "strategies_bundle"))

from research.pnl_evaluation.aggregate import state_to_metrics  # noqa: E402
from research.pnl_evaluation.config import (  # noqa: E402
    OUT,
    STRATEGIES,
    SYMBOLS,
    TAUS_SEC,
    TAB,
    TRAIN_FRAC,
    VENUES,
)

TRADE_BATCH = 1_000_000
SIG_COLS = ["ts_us"] + [sid for sid, _, _ in STRATEGIES]
from research.pnl_evaluation.data_loaders import day_ranges_us  # noqa: E402
from research.pnl_evaluation.strategy_signals import build_signal_table, test_window_us  # noqa: E402


def _signal_arrays(sig_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Numpy-only signals (float32) to avoid pandas overhead in day loop."""
    bts = sig_df["ts_us"].values.astype(np.int64)
    cols = [c for c in sig_df.columns if c not in ("ts_us", "mid")]
    return bts, {c: sig_df[c].values.astype(np.float32) for c in cols}


def process_symbol(
    sym: str,
    subsample: int,
    cache_signals: bool,
    max_days: int | None,
    train_frac: float,
    start_day: int = 1,
) -> pd.DataFrame:
    sig_cache = TAB / f"signals_{sym}_train{int(train_frac * 100)}_causal.parquet"
    meta_cache = TAB / f"split_meta_{sym}.json"

    if cache_signals and sig_cache.exists() and meta_cache.exists():
        sig_df = pd.read_parquet(sig_cache, columns=SIG_COLS)
        meta = json.loads(meta_cache.read_text())
        t0, t1 = meta["test_t0_us"], meta["test_t1_us"]
        print(f"  [{sym}] cached signals, panel_rows={meta.get('panel_rows')}", flush=True)
    else:
        print(f"  [{sym}] building panel + signals (train {train_frac:.0%} / test {1 - train_frac:.0%})...", flush=True)
        sig_df = build_signal_table(sym, max_days=max_days, train_frac=train_frac)
        TAB.mkdir(parents=True, exist_ok=True)
        sig_df.to_parquet(sig_cache, index=False)
        t0, t1 = test_window_us(sig_df, train_frac)
        meta = {
            "train_frac": train_frac,
            "test_t0_us": t0,
            "test_t1_us": t1,
            "panel_rows": len(sig_df),
            "train_rows": int(len(sig_df) * train_frac),
        }
        meta_cache.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    del sig_df
    gc.collect()

    days = day_ranges_us(t0, t1)
    print(f"  [{sym}] test window: {len(days)} days, subsample={subsample}", flush=True)

    venue_script = Path(__file__).resolve().parent / "run_venue_pnl.py"
    rows = []
    for venue in VENUES:
        print(f"  [{sym}] venue={venue}", flush=True)
        ckpt_prev = None
        venue_start = 1
        if start_day > 1:
            prev = TAB / f"ckpt_{sym}_{venue}_train{int(train_frac * 100)}_d{start_day - 1}.pkl"
            if prev.exists():
                venue_start = start_day
                ckpt_prev = prev
        for di, _ in enumerate(days):
            day_n = di + 1
            if day_n < venue_start:
                continue
            ckpt = TAB / f"ckpt_{sym}_{venue}_train{int(train_frac * 100)}_d{day_n}.pkl"
            cmd = [
                sys.executable,
                str(venue_script),
                "--symbol",
                sym,
                "--venue",
                venue,
                "--subsample",
                str(subsample),
                "--start-day",
                str(day_n),
                "--end-day",
                str(day_n),
                "--signals",
                str(sig_cache),
                "--meta",
                str(meta_cache),
                "--ckpt-out",
                str(ckpt),
            ]
            if ckpt_prev is not None:
                cmd.extend(["--ckpt-in", str(ckpt_prev)])
            subprocess.run(cmd, check=True)
            ckpt_prev = ckpt
        states = pickle.loads(ckpt_prev.read_bytes())

        for sid, kind, _fh in STRATEGIES:
            for tau in TAUS_SEC:
                m = state_to_metrics(states[sid][tau])
                rows.append(
                    {
                        "symbol": sym,
                        "venue": venue,
                        "strategy": sid,
                        "tau_sec": tau,
                        "forecast_kind": kind,
                        "split": f"train_{int(train_frac * 100)}_test_{int((1 - train_frac) * 100)}",
                        **m,
                    }
                )
        print(f"    done venue={venue} n_trades={states['baseline'][TAUS_SEC[0]]['n']:,}", flush=True)

    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, path: Path, train_frac: float) -> None:
    lines = [
        f"# PnL evaluation report (train {train_frac:.0%} / test {1 - train_frac:.0%}, all trades)",
        "",
        "Signals: **Binance** 1s panel; LGBM fit on **first half**; PnL on **second half** (chunked by day).",
        "Bybit: liquidations **+200ms**; markout mid: **Binance BBO**.",
        "Filter: `f=1` when `signal × s_taker > 0`.",
        "",
        "| symbol | venue | strategy | τ (s) | score | pnl_kept | pnl_filt | pnl_all | win_kept | maxdd_kept | turnover/day | ok |",
        "|--------|-------|----------|-------|-------|----------|----------|---------|----------|------------|--------------|-----|",
    ]
    for _, r in df.sort_values(["symbol", "venue", "strategy", "tau_sec"]).iterrows():
        ok = "✓" if r["meets_turnover_constraint"] else "✗"
        pf = r["pnl_filtered_bps"]
        pf_s = f"{pf:.3f}" if np.isfinite(pf) else "nan"
        lines.append(
            f"| {r['symbol']} | {r['venue']} | {r['strategy']} | {int(r['tau_sec'])} | "
            f"{r['score_bps']:.3f} | {r['pnl_kept_bps']:.3f} | {pf_s} | {r['pnl_all_bps']:.3f} | "
            f"{r['winrate_kept']:.3f} | {r['maxdd_kept_bps']:.2f} | {r['turnover_kept_usd_day']:.0f} | {ok} |"
        )
    lines.append("")
    lines.append(f"Full table: `{TAB.name}/pnl_report.csv`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="all", choices=["all", "btcusdt", "ethusdt"])
    ap.add_argument("--subsample", type=int, default=1, help="1 = all trades")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--max-days", type=int, default=0, help="0 = all available days in enriched BBO")
    ap.add_argument("--no-cache", action="store_true", help="Rebuild signal panel")
    ap.add_argument("--reuse-signals", action="store_true", help="Use cached signals even with --no-cache")
    ap.add_argument("--start-day", type=int, default=1, help="1-based day index to resume PnL loop")
    args = ap.parse_args()

    max_days = None if args.max_days <= 0 else args.max_days
    syms = SYMBOLS if args.symbol == "all" else (args.symbol,)
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    cache_sig = not args.no_cache or args.reuse_signals

    all_df = []
    for sym in syms:
        print(f"\n=== {sym} ===", flush=True)
        all_df.append(
            process_symbol(
                sym,
                subsample=args.subsample,
                cache_signals=cache_sig,
                max_days=max_days,
                train_frac=args.train_frac,
                start_day=args.start_day,
            )
        )

    df = pd.concat(all_df, ignore_index=True)
    csv_path = TAB / "pnl_report.csv"
    df.to_csv(csv_path, index=False)
    write_markdown(df, OUT / "PNL_REPORT.md", args.train_frac)
    print(f"\nDone: {csv_path}", flush=True)
    print(f"Report: {OUT / 'PNL_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
