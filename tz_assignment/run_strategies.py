#!/usr/bin/env python3
"""
Train direction models, evaluate WR, run maker PnL (research pipelines).

Uses research/strategies_bundle and research/pnl_evaluation with LIQUIDATION_DATA_ROOT
from tz_assignment/config.py. Copies artifacts into results/strategies/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from config import DATA, SYMBOLS, ensure_dirs

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "research" / "strategies_bundle"
PNL_SCRIPT = REPO_ROOT / "research" / "pnl_evaluation" / "run_report.py"
STRAT_OUT = Path(__file__).resolve().parent / "results" / "strategies"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["LIQUIDATION_DATA_ROOT"] = str(DATA)
    return env


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=_env(), check=True)


def _enriched_ok(sym: str) -> bool:
    bbo = DATA / "enriched" / "binance_bbo" / f"perp_{sym}.parquet"
    return bbo.is_file()


def train_symbol(sym: str, max_days: int | None) -> None:
    cmd = [sys.executable, "scripts/train_models.py", "--symbol", sym]
    if max_days is not None:
        cmd += ["--max-days", str(max_days)]
    _run(cmd, BUNDLE)


def evaluate_symbol(sym: str, max_days: int | None) -> Path:
    cmd = [sys.executable, "scripts/evaluate_models.py", "--symbol", sym]
    if max_days is not None:
        cmd += ["--max-days", str(max_days)]
    _run(cmd, BUNDLE)
    src = BUNDLE / "artifacts" / "models" / sym / "eval_results.json"
    if not src.is_file():
        raise FileNotFoundError(f"Missing {src} after evaluate")
    STRAT_OUT.mkdir(parents=True, exist_ok=True)
    dst = STRAT_OUT / f"direction_eval_{sym}.json"
    shutil.copy2(src, dst)
    return dst


def eval_to_csv(sym: str, eval_path: Path) -> None:
    import pandas as pd

    raw = json.loads(eval_path.read_text())
    rows = []
    for h, block in raw.items():
        for model, m in block.items():
            rows.append(
                {
                    "symbol": sym,
                    "horizon_sec": int(h),
                    "model": model,
                    "hit_rate": m.get("hit_rate"),
                    "n": m.get("n"),
                }
            )
    pd.DataFrame(rows).to_csv(STRAT_OUT / f"direction_metrics_{sym}.csv", index=False)


def run_pnl(
    symbols: tuple[str, ...],
    max_days: int | None,
    subsample: int,
    train_frac: float,
    no_cache: bool,
) -> None:
    if not PNL_SCRIPT.is_file():
        raise FileNotFoundError(PNL_SCRIPT)
    STRAT_OUT.mkdir(parents=True, exist_ok=True)
    sym_arg = "all" if len(symbols) >= 2 else symbols[0]
    cmd = [
        sys.executable,
        str(PNL_SCRIPT),
        "--symbol",
        sym_arg,
        "--subsample",
        str(subsample),
        "--train-frac",
        str(train_frac),
    ]
    if max_days is not None:
        cmd += ["--max-days", str(max_days)]
    if no_cache:
        cmd.append("--no-cache")
    _run(cmd, REPO_ROOT)

    pnl_src = REPO_ROOT / "research" / "pnl_evaluation" / "results"
    tab_src = pnl_src / "tables" / "pnl_report.csv"
    if tab_src.is_file():
        shutil.copy2(tab_src, STRAT_OUT / "pnl_report.csv")
        print(f"Copied {tab_src} -> {STRAT_OUT / 'pnl_report.csv'}", flush=True)
    md_src = pnl_src / "PNL_REPORT.md"
    if md_src.is_file():
        shutil.copy2(md_src, STRAT_OUT / "PNL_REPORT.md")
        print(f"Copied {md_src} -> {STRAT_OUT / 'PNL_REPORT.md'}", flush=True)
    export_bybit_filter_summary()


def export_bybit_filter_summary() -> None:
    """Highlight venue=bybit rows for reviewers (τ=30s)."""
    import pandas as pd

    pnl_path = STRAT_OUT / "pnl_report.csv"
    if not pnl_path.is_file():
        return
    pnl = pd.read_csv(pnl_path)
    sub = pnl[(pnl["venue"] == "bybit") & (pnl["tau_sec"] == 30)].copy()
    cols = [
        "symbol",
        "strategy",
        "score_bps",
        "pnl_kept_bps",
        "pnl_all_bps",
        "winrate_kept",
        "n_kept",
        "n_all",
        "kept_frac",
        "turnover_kept_usd_day",
        "meets_turnover_constraint",
    ]
    if "kept_frac" not in sub.columns and "n_kept" in sub.columns:
        sub["kept_frac"] = sub["n_kept"] / sub["n_all"].replace(0, np.nan)
    out_cols = [c for c in cols if c in sub.columns]
    sub = sub[out_cols].sort_values(["symbol", "score_bps"], ascending=[True, False])
    out = STRAT_OUT / "bybit_filter_summary.csv"
    sub.to_csv(out, index=False)
    print(f"Wrote {out} ({len(sub)} rows)", flush=True)


def write_run_meta(
    symbols: tuple[str, ...],
    max_days: int | None,
    quick: bool,
    ran_pnl: bool,
) -> None:
    meta = {
        "data_root": str(DATA),
        "symbols": list(symbols),
        "max_days": max_days,
        "quick": quick,
        "pnl_ran": ran_pnl,
        "bundle_models": str(BUNDLE / "artifacts" / "models"),
        "bybit_filter_csv": "results/strategies/bybit_filter_summary.csv",
        "note": "PnL venue=bybit: Bybit liquidations as proxy events, Binance signal, Binance markout",
    }
    (STRAT_OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train strategies + PnL (TZ wrapper)")
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    ap.add_argument("--skip-train", action="store_true", help="Use existing bundle artifacts")
    ap.add_argument("--skip-pnl", action="store_true", help="Direction WR only (faster)")
    ap.add_argument("--quick", action="store_true", help="max-days=14, subsample=10 for PnL")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--no-cache", action="store_true", help="Rebuild PnL signal panel")
    args = ap.parse_args(argv)

    ensure_dirs()
    STRAT_OUT.mkdir(parents=True, exist_ok=True)
    symbols = tuple(args.symbols)

    for sym in symbols:
        if not _enriched_ok(sym):
            print(f"Missing enriched BBO for {sym}. Run: python build_features.py", file=sys.stderr)
            return 1

    max_days = 14 if args.quick and args.max_days is None else args.max_days
    subsample = 10 if args.quick and args.subsample == 1 else args.subsample

    for sym in symbols:
        if not args.skip_train:
            train_symbol(sym, max_days)
        ev = evaluate_symbol(sym, max_days)
        eval_to_csv(sym, ev)

    if not args.skip_pnl:
        run_pnl(symbols, max_days, subsample, args.train_frac, args.no_cache)

    write_run_meta(symbols, max_days, args.quick, not args.skip_pnl)
    if (STRAT_OUT / "pnl_report.csv").is_file():
        export_bybit_filter_summary()
    print(f"\nStrategy outputs: {STRAT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
