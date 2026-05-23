"""Evaluate Bayes trend as a standalone forward-return predictor."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.6


def _forward_log_ret(df: pd.DataFrame, freq: str) -> pd.Series:
    lp = np.log(df["microprice"].astype(float))
    return lp.shift(-1) - lp


def evaluate_trend_forecast(trend_df: pd.DataFrame, freq: str) -> dict:
    """
    Predict sign of next-bar log return from Bayes regime / slope.
    Baselines: zero (flat), persistence (same regime).
    """
    col_r = f"trend_regime_{freq}"
    col_s = f"trend_slope_{freq}"
    df = trend_df.dropna(subset=["microprice"]).copy()
    df["fwd_ret"] = _forward_log_ret(df, freq)
    df = df.dropna(subset=["fwd_ret", col_r, col_s])
    if len(df) < 500:
        return {"ok": False}

    cut = int(len(df) * TRAIN_FRAC)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    y_te = te["fwd_ret"].values
    y_tr = tr["fwd_ret"].values

    def _metrics(pred_sign: np.ndarray, y: np.ndarray) -> dict:
        m = np.isfinite(y) & np.isfinite(pred_sign)
        if m.sum() < 100:
            return {}
        hit = float((np.sign(y[m]) == pred_sign[m]).mean())
        ic = float(pd.Series(pred_sign[m]).corr(pd.Series(y[m]), method="spearman"))
        return {"hit_rate": hit, "rank_ic": ic, "n": int(m.sum())}

    pred_regime_te = te[col_r].astype(int).values.astype(float)
    pred_regime_tr = tr[col_r].astype(int).values.astype(float)
    pred_slope_te = np.sign(te[col_s].values)
    pred_slope_tr = np.sign(tr[col_s].values)
    pred_slope_te[pred_slope_te == 0] = np.nan
    pred_regime_te = pred_regime_te.copy()
    pred_regime_te[pred_regime_te == 0] = np.nan

    # Weighted by CP credibility
    cred = te[f"cp_cred_{freq}"].values if f"cp_cred_{freq}" in te.columns else np.ones(len(te))
    w = cred / (cred.sum() + 1e-9)
    weighted_hit = float(
        ((np.sign(y_te) == np.sign(te[col_s].values)) & (te[col_r].values != 0)).astype(float) @ w
        / (w.sum() + 1e-9)
    )

    out = {
        "ok": True,
        "freq": freq,
        "regime_train": _metrics(pred_regime_tr, y_tr),
        "regime_test": _metrics(pred_regime_te, y_te),
        "slope_sign_train": _metrics(pred_slope_tr, y_tr),
        "slope_sign_test": _metrics(pred_slope_te, y_te),
        "slope_fwd_rank_ic_train": float(tr[col_s].corr(tr["fwd_ret"], method="spearman")),
        "slope_fwd_rank_ic_test": float(te[col_s].corr(te["fwd_ret"], method="spearman")),
        "weighted_hit_test": weighted_hit,
        "mean_abs_fwd_bps_test": float((te["fwd_ret"].abs().mean() * 10_000)),
    }
    return out
