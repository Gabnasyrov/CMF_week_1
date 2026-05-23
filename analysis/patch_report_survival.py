#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "tables" / "cross_lead"
summary_path = TAB / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"note": {}, "results": {}}

for sym in ("btcusdt", "ethusdt"):
    surv = {}
    for venue in ("binance", "bybit"):
        p = TAB / f"05_survival_metrics_{venue}_{sym}.json"
        raw = pd.read_json(p).iloc[0].to_dict()
        surv[venue] = {
            "n_bursts": int(raw["n_bursts"]),
            "median_duration_sec": float(raw["median_duration_sec"]),
            "cox": raw["cox"],
            "cox_base": raw["cox_base"],
            "lgbm": raw["lgbm"],
            "lgbm_base": raw["lgbm_base"],
        }
    if sym in summary.get("results", {}):
        summary["results"][sym]["survival"] = surv

summary_path.write_text(json.dumps(summary, indent=2, default=float))
from cross_lead_analysis import write_report  # noqa: E402

write_report(summary["results"], summary.get("note", {}), ROOT / "CROSS_LEAD_REPORT.md")
print("patched", ROOT / "CROSS_LEAD_REPORT.md")
