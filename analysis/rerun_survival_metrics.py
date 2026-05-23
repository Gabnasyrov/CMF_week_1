#!/usr/bin/env python3
"""Refit survival metrics from saved burst feature tables (no BBO rebuild)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from burst_survival import (  # noqa: E402
    FEATURES,
    FEATURES_BASE,
    _temporal_split,
    fit_cox,
    fit_lgbm_duration,
)
from config import SYMBOLS  # noqa: E402

TAB = Path(__file__).resolve().parent / "tables" / "cross_lead"


def refit(sym: str, venue: str) -> dict:
    path = TAB / f"05_burst_features_{venue}_{sym}.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=["n_events_log"], errors="ignore")
    tr, te = _temporal_split(df)
    return {
        "symbol": sym,
        "venue": venue,
        "n_bursts": len(df),
        "median_duration_sec": float(df["duration_sec"].median()),
        "cox": fit_cox(tr, te, FEATURES),
        "cox_base": fit_cox(tr, te, FEATURES_BASE),
        "lgbm": {k: v for k, v in fit_lgbm_duration(tr, te, FEATURES).items() if k not in ("pred_test", "actual_test")},
        "lgbm_base": {
            k: v
            for k, v in fit_lgbm_duration(tr, te, FEATURES_BASE).items()
            if k not in ("pred_test", "actual_test")
        },
    }


def main():
    for sym in SYMBOLS:
        for venue in ("binance", "bybit"):
            row = refit(sym, venue)
            out = TAB / f"05_survival_metrics_{venue}_{sym}.json"
            pd.DataFrame([row]).to_json(out, indent=2)
            print(sym, venue, "cox_test", row["cox"].get("c_index_test"), "lgbm_test", row["lgbm"].get("c_index_test"))


if __name__ == "__main__":
    main()
