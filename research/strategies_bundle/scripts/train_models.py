#!/usr/bin/env python3
"""
Train and save strategy artifacts for one symbol.
Saves: normalizers, LGBM no-K (primary), optional LGBM full per horizon.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.config import HORIZONS_SEC, MODELS_DIR, PRIMARY_HORIZONS  # noqa: E402
from lib.io_config import load_symbol_config  # noqa: E402
from lib.models import LGBMDirectionModel  # noqa: E402
from lib.feature_builder import add_hawkes_features  # noqa: E402
from lib.pipeline import (  # noqa: E402
    chronological_split,
    fit_normalizers,
    prepare_panel,
    transform_panel,
)
from lib.targets import lgbm_train_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["btcusdt", "ethusdt"])
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--all-horizons", action="store_true", help="Train all horizons, not only 30+300")
    ap.add_argument("--with-lgbm-full", action="store_true", help="Also train LGBM with kalman features")
    args = ap.parse_args()

    sym = args.symbol
    cfg = load_symbol_config(sym)
    out_dir = MODELS_DIR / sym
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building panel {sym}...")
    panel = prepare_panel(sym, max_days=args.max_days, skip_hawkes=True)
    tr, te = chronological_split(panel, cfg.get("train_frac", 0.4))
    train_end_us = int(tr["ts_us"].iloc[-1]) if len(tr) else int(panel["ts_us"].iloc[0])
    panel = add_hawkes_features(panel, sym, train_end_us)
    tr, te = chronological_split(panel, cfg.get("train_frac", 0.4))
    print(f"  rows={len(panel)}")
    mtr = np.zeros(len(panel), dtype=bool)
    mtr[: len(tr)] = True
    norm_full, norm_nk = fit_normalizers(tr)
    zf_tr, zn_tr = transform_panel(tr, norm_full, norm_nk)
    zf_te, zn_te = transform_panel(te, norm_full, norm_nk)

    norm_full.save(out_dir / "normalizer_full.json")
    norm_nk.save(out_dir / "normalizer_nk.json")

    horizons = HORIZONS_SEC if args.all_horizons else PRIMARY_HORIZONS
    lgbm_params = cfg["lgbm"]
    meta = {"symbol": sym, "panel_rows": len(panel), "train_rows": len(tr), "test_rows": len(te), "horizons": list(horizons)}

    for h in horizons:
        ycol = f"y_mu_sign_{h}s"
        y_tr, y_te = tr[ycol].values, te[ycol].values
        zcols_nk = norm_nk.z_columns()

        m_fit = lgbm_train_mask(panel, mtr, ycol, h)
        m_fit_tr = m_fit[mtr]
        print(f"  Training LGBM no-K h={h}s...")
        m_nk = LGBMDirectionModel(**lgbm_params)
        m_nk.fit(zn_tr.loc[m_fit_tr, zcols_nk], panel.loc[m_fit, ycol].values, zn_te[zcols_nk], y_te)
        m_nk.save(out_dir / f"lgbm_nk_{h}s.joblib")

        if args.with_lgbm_full:
            zcols_f = norm_full.z_columns()
            print(f"  Training LGBM full h={h}s...")
            m_f = LGBMDirectionModel(**lgbm_params)
            m_f.fit(zf_tr.loc[m_fit_tr, zcols_f], panel.loc[m_fit, ycol].values, zf_te[zcols_f], y_te)
            m_f.save(out_dir / f"lgbm_full_{h}s.joblib")

    meta_path = out_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
