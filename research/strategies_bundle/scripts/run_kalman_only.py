#!/usr/bin/env python3
"""Kalman-only strategy — no ML weights required."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import HORIZONS_SEC  # noqa: E402
from lib.evaluate import mu_direction_metrics  # noqa: E402
from lib.io_config import load_symbol_config  # noqa: E402
from lib.pipeline import chronological_split, prepare_panel  # noqa: E402
from lib.signals import kalman_direction_sign  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["btcusdt", "ethusdt"])
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()

    cfg = load_symbol_config(args.symbol)
    panel = prepare_panel(args.symbol, max_days=args.max_days)
    tr, te = chronological_split(panel, cfg.get("train_frac", 0.4))
    ref = cfg.get("reference_wr_test", {}).get("kalman_only", {})

    print(f"Kalman-only {args.symbol} test n={len(te)}\n")
    for h in HORIZONS_SEC:
        y = te[f"y_mu_sign_{h}s"].values
        pred = kalman_direction_sign(te, h)
        m = mu_direction_metrics(y, pred)
        r = ref.get(str(h), float("nan"))
        flag = " *" if m["hit_rate"] > 0.53 else ""
        print(f"  h={h:3d}s WR={m['hit_rate']:.3f} (ref {r}) n={m['n']}{flag}")


if __name__ == "__main__":
    main()
