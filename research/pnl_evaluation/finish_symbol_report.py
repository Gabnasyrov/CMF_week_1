#!/usr/bin/env python3
"""Build PnL report rows from final per-venue day checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.pnl_evaluation.aggregate import state_to_metrics  # noqa: E402
from research.pnl_evaluation.config import OUT, STRATEGIES, TAUS_SEC, TAB, VENUES  # noqa: E402


def rows_from_ckpt(sym: str, venue: str, train_frac: float, day_n: int) -> list[dict]:
    ckpt = TAB / f"ckpt_{sym}_{venue}_train{int(train_frac * 100)}_d{day_n}.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    states = pickle.loads(ckpt.read_bytes())
    split = f"train_{int(train_frac * 100)}_test_{int((1 - train_frac) * 100)}"
    rows = []
    for sid, kind, _ in STRATEGIES:
        for tau in TAUS_SEC:
            m = state_to_metrics(states[sid][tau])
            rows.append(
                {
                    "symbol": sym,
                    "venue": venue,
                    "strategy": sid,
                    "tau_sec": tau,
                    "forecast_kind": kind,
                    "split": split,
                    **m,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--last-day", type=int, default=45)
    ap.add_argument("--append-csv", action="store_true")
    args = ap.parse_args()

    meta = json.loads((TAB / f"split_meta_{args.symbol}.json").read_text())
    n_days = len(
        __import__("research.pnl_evaluation.data_loaders", fromlist=["day_ranges_us"]).day_ranges_us(
            meta["test_t0_us"], meta["test_t1_us"]
        )
    )
    day_n = args.last_day if args.last_day > 0 else n_days
    if day_n > n_days:
        day_n = n_days

    rows = []
    for venue in VENUES:
        rows.extend(rows_from_ckpt(args.symbol, venue, args.train_frac, day_n))

    df = pd.DataFrame(rows)
    csv_path = TAB / "pnl_report.csv"
    if args.append_csv and csv_path.exists():
        old = pd.read_csv(csv_path)
        old = old[old["symbol"] != args.symbol]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
