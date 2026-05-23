#!/usr/bin/env python3
"""
Emit direction signal (-1 / +1) for the latest row of the panel.
Requires trained artifacts from train_models.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import MODELS_DIR  # noqa: E402
from lib.io_config import load_symbol_config  # noqa: E402
from lib.models import LGBMDirectionModel  # noqa: E402
from lib.normalization import ZScoreNormalizer  # noqa: E402
from lib.pipeline import prepare_panel, transform_panel  # noqa: E402
from lib.signals import ensemble_sign, kalman_causal_sign, kalman_direction_sign  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["btcusdt", "ethusdt"])
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument(
        "--strategy",
        default="lgbm_nk",
        choices=["kalman", "lgbm_nk", "lgbm_full", "ensemble"],
    )
    ap.add_argument("--max-days", type=int, default=30, help="Days of data for live window (default 30)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sym = args.symbol
    mdir = MODELS_DIR / sym
    panel = prepare_panel(sym, max_days=args.max_days)
    if panel.empty:
        raise RuntimeError("Empty panel — check data paths")

    row = panel.iloc[-1]
    norm_nk = ZScoreNormalizer.load(mdir / "normalizer_nk.json")
    zn = norm_nk.transform(panel.tail(1))

    out = {
        "symbol": sym,
        "horizon_sec": args.horizon,
        "strategy": args.strategy,
        "ts_us": int(row["ts_us"]),
        "mid": float(row["mid"]),
    }

    if args.strategy == "kalman":
        sign = float(kalman_direction_sign(panel, args.horizon)[-1])
    elif args.strategy == "lgbm_nk":
        m = LGBMDirectionModel.load(mdir / f"lgbm_nk_{args.horizon}s.joblib")
        sign = float(m.predict_sign(zn[norm_nk.z_columns()])[-1])
    elif args.strategy == "lgbm_full":
        norm_f = ZScoreNormalizer.load(mdir / "normalizer_full.json")
        zf_row = norm_f.transform(panel.tail(1))
        m = LGBMDirectionModel.load(mdir / f"lgbm_full_{args.horizon}s.joblib")
        sign = float(m.predict_sign(zf_row[norm_f.z_columns()])[-1])
    else:
        m = LGBMDirectionModel.load(mdir / f"lgbm_nk_{args.horizon}s.joblib")
        pred = m.predict_sign(zn[norm_nk.z_columns()])[-1]
        mu = panel["kalman_mu"].values
        h = args.horizon
        kd = 0.0 if len(mu) <= h else mu[-1] - mu[-1 - h]
        sign = float(ensemble_sign(np.array([kd]), np.array([pred]))[-1])

    out["signal"] = int(sign) if sign in (-1, 1) else 0

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"symbol={sym} h={args.horizon}s strategy={args.strategy} signal={out['signal']} mid={out['mid']:.2f}")


if __name__ == "__main__":
    main()
