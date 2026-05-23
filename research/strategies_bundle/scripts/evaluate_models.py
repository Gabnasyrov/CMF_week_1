#!/usr/bin/env python3
"""Evaluate saved models on chronological test split; compare to reference WR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import HORIZONS_SEC, MODELS_DIR, PRIMARY_HORIZONS  # noqa: E402
from lib.evaluate import mu_direction_metrics  # noqa: E402
from lib.io_config import load_symbol_config  # noqa: E402
from lib.models import LGBMDirectionModel  # noqa: E402
from lib.normalization import ZScoreNormalizer  # noqa: E402
from lib.pipeline import chronological_split, prepare_panel, transform_panel  # noqa: E402
from lib.signals import ensemble_sign, kalman_direction_sign  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["btcusdt", "ethusdt"])
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--all-horizons", action="store_true")
    args = ap.parse_args()

    sym = args.symbol
    cfg = load_symbol_config(sym)
    mdir = MODELS_DIR / sym
    if not (mdir / "normalizer_nk.json").exists():
        raise FileNotFoundError(f"Run train_models.py first: {mdir}")

    ref = cfg.get("reference_wr_test", {})
    panel = prepare_panel(sym, max_days=args.max_days)
    tr, te = chronological_split(panel, cfg.get("train_frac", 0.4))
    norm_nk = ZScoreNormalizer.load(mdir / "normalizer_nk.json")
    _, zn_te = transform_panel(te, norm_nk, norm_nk)
    zcols_nk = norm_nk.z_columns()

    horizons = HORIZONS_SEC if args.all_horizons else PRIMARY_HORIZONS
    results = {}

    print(f"\n=== {sym} test evaluation (n={len(te)}) ===\n")
    print(f"{'h':>4} | {'Kalman':>7} | {'LGBM nk':>7} | {'Ensemble':>8} | ref nk")
    print("-" * 50)

    for h in horizons:
        y_te = te[f"y_mu_sign_{h}s"].values
        k_sign = kalman_direction_sign(te, h)
        m_k = mu_direction_metrics(y_te, k_sign)

        model_path = mdir / f"lgbm_nk_{h}s.joblib"
        if model_path.exists():
            m = LGBMDirectionModel.load(model_path)
            pred = m.predict_sign(zn_te[zcols_nk])
            m_nk = mu_direction_metrics(y_te, pred)
        else:
            m_nk = {"hit_rate": np.nan}
            pred = np.zeros(len(te))

        mu = te["kalman_mu"].values
        kd = np.zeros(len(mu))
        if h < len(mu):
            kd[:-h] = mu[h:] - mu[:-h]
        m_ens = mu_direction_metrics(y_te, ensemble_sign(kd, pred))

        ref_nk = ref.get("lgbm_no_kalman", {}).get(str(h), np.nan)
        print(
            f"{h:4d} | {m_k['hit_rate']:7.3f} | {m_nk['hit_rate']:7.3f} | "
            f"{m_ens['hit_rate']:8.3f} | {ref_nk}"
        )
        results[str(h)] = {"kalman": m_k, "lgbm_nk": m_nk, "ensemble": m_ens}

    out_path = mdir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
