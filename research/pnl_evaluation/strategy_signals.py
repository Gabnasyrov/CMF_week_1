"""Build Binance strategy signals on 1s grid (chronological train/test split)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "research" / "strategies_bundle"
sys.path.insert(0, str(BUNDLE))

from lib.feature_builder import add_hawkes_features  # noqa: E402
from lib.io_config import load_symbol_config  # noqa: E402
from lib.models import LGBMDirectionModel  # noqa: E402
from lib.pipeline import fit_normalizers, prepare_panel, transform_panel  # noqa: E402
from lib.signals import ensemble_sign  # noqa: E402
from lib.targets import lgbm_train_mask  # noqa: E402

from .config import STRATEGIES, TRAIN_FRAC  # noqa: E402


def kalman_causal_sign(panel: pd.DataFrame, horizon_sec: int) -> np.ndarray:
    """Causal direction at t: sign(mu[t] - mu[t-h]), no future leak."""
    mu = panel["kalman_mu"].values.astype(np.float64)
    h = horizon_sec
    out = np.zeros(len(mu))
    if h < len(mu):
        out[h:] = np.sign(mu[h:] - mu[:-h])
    return out


def chronological_train_mask(ts_us: np.ndarray, train_frac: float = TRAIN_FRAC) -> np.ndarray:
    cut = int(len(ts_us) * train_frac)
    m = np.zeros(len(ts_us), dtype=bool)
    m[:cut] = True
    return m


def test_window_us(panel: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> tuple[int, int]:
    ts = panel["ts_us"].values.astype(np.int64)
    cut = int(len(ts) * train_frac)
    if cut >= len(ts):
        raise RuntimeError("Empty test split")
    return int(ts[cut]), int(ts[-1])


def build_signal_table(sym: str, max_days: int | None = None, train_frac: float = TRAIN_FRAC) -> pd.DataFrame:
    panel = prepare_panel(sym, max_days=max_days, subsample=False, skip_hawkes=True)
    if panel.empty:
        raise RuntimeError(f"Empty panel for {sym}")

    cut = int(len(panel) * train_frac)
    train_end_us = int(panel["ts_us"].iloc[cut - 1]) if cut > 0 else int(panel["ts_us"].iloc[0])
    panel = add_hawkes_features(panel, sym, train_end_us)

    mtr = chronological_train_mask(panel["ts_us"].values, train_frac)
    tr = panel.loc[mtr]
    norm_full, norm_nk = fit_normalizers(tr)
    zf, zn = transform_panel(panel, norm_full, norm_nk)
    zcols_f = norm_full.z_columns()
    zcols_nk = norm_nk.z_columns()
    lgbm_params = load_symbol_config(sym)["lgbm"]

    out = panel[["ts_us", "mid"]].copy()

    for sid, kind, h in STRATEGIES:
        if kind == "baseline":
            out[sid] = 0.0
        elif kind == "kalman":
            out[sid] = kalman_causal_sign(panel, h)
        elif kind == "lgbm_nk":
            ycol = f"y_mu_sign_{h}s"
            m_fit = lgbm_train_mask(panel, mtr, ycol, h)
            model = LGBMDirectionModel(**lgbm_params)
            model.fit(
                zn.loc[m_fit, zcols_nk],
                panel.loc[m_fit, ycol].values,
                zn.loc[m_fit, zcols_nk],
                panel.loc[m_fit, ycol].values,
            )
            out[sid] = model.predict_sign(zn[zcols_nk])
        elif kind == "lgbm_full":
            ycol = f"y_mu_sign_{h}s"
            m_fit = lgbm_train_mask(panel, mtr, ycol, h)
            model = LGBMDirectionModel(**lgbm_params)
            model.fit(
                zf.loc[m_fit, zcols_f],
                panel.loc[m_fit, ycol].values,
                zf.loc[m_fit, zcols_f],
                panel.loc[m_fit, ycol].values,
            )
            out[sid] = model.predict_sign(zf[zcols_f])
        elif kind == "ensemble":
            pass
        else:
            raise ValueError(kind)

    h = 30
    mu = panel["kalman_mu"].values
    delta = np.zeros(len(mu))
    if h < len(mu):
        delta[h:] = mu[h:] - mu[:-h]
    out["ensemble_30s"] = ensemble_sign(delta, out["lgbm_nk_30s"].values)

    return out


def asof_signal_to_events(sig_ts: np.ndarray, sig_val: np.ndarray, event_ts: np.ndarray) -> np.ndarray:
    """Backward asof: latest signal at or before each event timestamp."""
    idx = np.searchsorted(sig_ts, event_ts, side="right") - 1
    out = np.zeros(len(event_ts), dtype=np.float64)
    ok = idx >= 0
    out[ok] = sig_val[idx[ok]].astype(np.float64)
    return out
