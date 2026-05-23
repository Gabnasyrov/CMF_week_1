#!/usr/bin/env python3
"""Build enriched microstructure parquet (wrapper around analysis/build_features.py)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import DATA, SYMBOLS, ensure_dirs

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "analysis" / "build_features.py"


def _required_raw(sym: str) -> list[Path]:
    from config import paths_for_symbol

    p = paths_for_symbol(sym)
    return [p["bbo"], p["trades"], p["liq_binance"], p["liq_bybit"]]


def check_raw_data(symbols: tuple[str, ...]) -> list[str]:
    missing = []
    for sym in symbols:
        for path in _required_raw(sym):
            if not path.exists():
                missing.append(str(path))
    return missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build data/enriched/ from raw parquet")
    ap.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    ap.add_argument("--skip-bbo", action="store_true")
    ap.add_argument("--skip-trades", action="store_true")
    ap.add_argument("--bbo-batch", type=int, default=4)
    ap.add_argument("--trades-batch", type=int, default=2)
    args = ap.parse_args(argv)

    ensure_dirs()
    missing = check_raw_data(tuple(args.symbols))
    if missing:
        print("Missing raw parquet (see data/README.md):", file=sys.stderr)
        for m in missing[:8]:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) > 8:
            print(f"  ... and {len(missing) - 8} more", file=sys.stderr)
        return 1

    if not BUILD_SCRIPT.is_file():
        print(f"Not found: {BUILD_SCRIPT}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["LIQUIDATION_DATA_ROOT"] = str(DATA)
    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--symbols",
        *args.symbols,
        "--bbo-batch",
        str(args.bbo_batch),
        "--trades-batch",
        str(args.trades_batch),
    ]
    if args.skip_bbo:
        cmd.append("--skip-bbo")
    if args.skip_trades:
        cmd.append("--skip-trades")

    print(f"DATA_ROOT={DATA}")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT / "analysis", env=env, check=True)
    print("Enriched output:", DATA / "enriched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
