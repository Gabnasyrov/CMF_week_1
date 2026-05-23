#!/usr/bin/env python3
"""
Full TZ pipeline: enriched features → strategies → EDA.

  python run_all.py              # all steps (when data present)
  python run_all.py --view-only  # no data: show bundled results paths
  python run_all.py --quick      # 14-day strategy smoke + EDA
  python run_all.py --eda-only   # skip features & strategies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DATA, FIGURES, NOTEBOOKS, RESULTS, TABLES, ensure_dirs, has_enriched, has_raw_data


def _print_view_only() -> None:
    print("=== TZ Assignment — bundled results (no raw data required) ===\n")
    print("DATA_ROOT:", DATA, "| raw_data:", has_raw_data(), "| enriched:", has_enriched())
    print("\nOpen:")
    print(" ", NOTEBOOKS / "Liquidation_EDA_TZ.ipynb")
    print(" ", RESULTS / "EXECUTIVE_SUMMARY.md")
    print(" ", RESULTS / "strategies" / "PNL_REPORT.md")
    print(" ", RESULTS / "strategies" / "bybit_filter_summary.csv", "  ← Bybit liq filter")
    print(" ", RESULTS / "figures" / "ethusdt_price_liq_2025-12-15.png")
    print("\nTo reproduce with parquet:")
    print("  1. Copy data per data/README.md into", DATA)
    print("  2. pip install -r requirements.txt -r requirements-strategies.txt")
    print("  3. python run_all.py")
    print("\nSee SUBMISSION.md for details.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TZ assignment: features + strategies + EDA")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-strategies", action="store_true")
    ap.add_argument("--eda-only", action="store_true", help="Only run tier 1.1–1.3 analysis")
    ap.add_argument("--view-only", action="store_true", help="Print bundled results paths; no compute")
    ap.add_argument("--quick", action="store_true", help="Strategies: max-days=14, PnL subsample=10")
    ap.add_argument("--skip-pnl", action="store_true", help="Strategies: direction WR only")
    ap.add_argument("--skip-train", action="store_true", help="Use existing strategy artifacts")
    ap.add_argument("--no-pnl-cache", action="store_true", help="Rebuild PnL signal panel")
    ap.add_argument("--force-features", action="store_true", help="Rebuild enriched even if present")
    args = ap.parse_args(argv)

    if args.view_only:
        _print_view_only()
        return 0

    if args.eda_only:
        args.skip_features = True
        args.skip_strategies = True

    ensure_dirs()
    print(f"DATA_ROOT={DATA}")
    print(f"raw_data={has_raw_data()} enriched={has_enriched()}\n")

    if not has_raw_data():
        print("ERROR: Raw parquet missing. See data/README.md", file=sys.stderr)
        print("Review without data: python run_all.py --view-only", file=sys.stderr)
        _print_view_only()
        return 1

    if not args.skip_features and (args.force_features or not has_enriched()):
        print("=== Build enriched features ===")
        from build_features import main as build_main

        code = build_main([])
        if code != 0:
            return code
    elif not args.skip_features:
        print("=== Enriched features present — skip build (use --force-features to rebuild) ===\n")

    if not args.skip_strategies:
        if not has_enriched():
            print("ERROR: Need enriched/ before strategies. Run build_features.py", file=sys.stderr)
            return 1
        print("=== Strategies (train / evaluate / PnL incl. venue=bybit) ===")
        from run_strategies import main as strat_main

        strat_argv = []
        if args.quick:
            strat_argv.append("--quick")
        if args.skip_pnl:
            strat_argv.append("--skip-pnl")
        if args.skip_train:
            strat_argv.append("--skip-train")
        if args.no_pnl_cache:
            strat_argv.append("--no-cache")
        code = strat_main(strat_argv)
        if code != 0:
            return code

    print("=== EDA tiers 1.1–1.3 ===")
    from run_analysis import main as eda_main

    eda_main()
    print(f"\nDone. Figures: {len(list(FIGURES.glob('*.png')))} | Tables: {len(list(TABLES.glob('*.csv')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
