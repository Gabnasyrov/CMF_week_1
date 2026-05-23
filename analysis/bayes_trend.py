"""
Bayesian trend / change-point detection (Schütz & Holschneider, arXiv:1104.3448).

Sliding-window local posterior with hockey-stick mean (Eq. 1–7) and
linear vs change-point Bayes factor (Eq. 30). Homoscedastic profile (s1=s2=0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# window length in bars, stride in bars, inner prior width fraction for θ
HORIZON_CONFIG = {
    "5s": {"window": 48, "stride": 6, "n_theta": 21, "inner_a": 0.7, "slope_flat_bps": 0.05},
    "60s": {"window": 60, "stride": 1, "n_theta": 25, "inner_a": 0.7, "slope_flat_bps": 0.15},
}


def _design_cp(theta: float, t: np.ndarray) -> np.ndarray:
    """Hockey-stick columns per Eq. (2–3)."""
    zm = np.where(t <= theta, theta - t, 0.0)
    zp = np.where(t >= theta, theta - t, 0.0)
    return np.column_stack([np.ones_like(t), zm, zp])


def _design_linear(t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(t), t])


def _profile_cp(y: np.ndarray, t: np.ndarray, theta: float) -> tuple[float, np.ndarray, float]:
    F = _design_cp(theta, t)
    try:
        FtF = F.T @ F
        det = np.linalg.det(FtF)
        if det <= 1e-30:
            return -np.inf, np.zeros(3), np.inf
        beta = np.linalg.solve(FtF, F.T @ y)
        resid = y - F @ beta
        r2 = float(resid @ resid)
        if r2 <= 1e-30:
            r2 = 1e-30
        n = len(y)
        log_ev = -0.5 * (n - 2) * np.log(r2) - 0.5 * np.log(det)
        return log_ev, beta, r2
    except np.linalg.LinAlgError:
        return -np.inf, np.zeros(3), np.inf


def _profile_linear(y: np.ndarray, t: np.ndarray) -> tuple[float, np.ndarray, float]:
    F = _design_linear(t)
    FtF = F.T @ F
    det = np.linalg.det(FtF)
    if det <= 1e-30:
        return -np.inf, np.zeros(2), np.inf
    beta = np.linalg.solve(FtF, F.T @ y)
    resid = y - F @ beta
    r2 = float(resid @ resid)
    if r2 <= 1e-30:
        r2 = 1e-30
    n = len(y)
    log_ev = -0.5 * (n - 2) * np.log(r2) - 0.5 * np.log(det)
    return log_ev, beta, r2


def _window_inference(y: np.ndarray, cfg: dict) -> dict:
    n = len(y)
    t = np.linspace(0.0, 1.0, n)
    ic = n // 2
    center = t[ic]
    a = cfg["inner_a"] * (t[-1] - t[0] + 1e-9)
    thetas = np.linspace(center - a / 2, center + a / 2, cfg["n_theta"])

    best_cp = -np.inf
    best_beta = np.zeros(3)
    best_theta = center
    for th in thetas:
        le, beta, _ = _profile_cp(y, t, th)
        if le > best_cp:
            best_cp = le
            best_beta = beta
            best_theta = th

    le_lin, beta_lin, _ = _profile_linear(y, t)
    log_bf = le_lin - best_cp  # >0 → linear preferred
    cp_cred = float(1.0 / (1.0 + np.exp(log_bf)))

    if log_bf > 0.5:
        slope = float(beta_lin[1])
        model = "linear"
    else:
        slope = float(-best_beta[2])
        model = "changepoint"

    thr = cfg["slope_flat_bps"] / 10_000.0
    if abs(slope) < thr:
        regime = 0
    else:
        regime = 1 if slope > 0 else -1

    return {
        "slope": slope,
        "regime": regime,
        "cp_credibility": cp_cred,
        "log_bf_linear": float(log_bf),
        "model": model,
        "beta_pre": float(-best_beta[1]),
        "beta_post": float(-best_beta[2]),
        "theta_hat": float(best_theta),
        "log_ev_cp": float(best_cp),
        "log_ev_lin": float(le_lin),
    }


def sliding_bayes_trend(log_price: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Compute trend labels on log-price series (NaNs forward-filled)."""
    n = len(log_price)
    w = cfg["window"]
    stride = cfg["stride"]
    slope = np.full(n, np.nan)
    regime = np.zeros(n, dtype=np.int8)
    cp_cred = np.full(n, np.nan)
    log_bf = np.full(n, np.nan)
    model = np.full(n, "", dtype=object)

    for i in range(w, n, stride):
        yw = log_price[i - w : i]
        if np.any(~np.isfinite(yw)):
            continue
        yw = yw - yw[0]
        inf = _window_inference(yw, cfg)
        j1, j2 = i, min(i + stride, n)
        slope[j1:j2] = inf["slope"]
        regime[j1:j2] = inf["regime"]
        cp_cred[j1:j2] = inf["cp_credibility"]
        log_bf[j1:j2] = inf["log_bf_linear"]
        model[j1:j2] = inf["model"]

    out = pd.DataFrame(
        {
            "slope": slope,
            "regime": regime,
            "cp_credibility": cp_cred,
            "log_bf_linear": log_bf,
            "model": model,
        }
    )
    out["slope"] = out["slope"].ffill().bfill()
    out["cp_credibility"] = out["cp_credibility"].ffill().bfill()
    out["log_bf_linear"] = out["log_bf_linear"].ffill().bfill()
    out["regime"] = out["regime"].ffill().bfill().astype(int)
    return out


def compute_trend_on_bbo(bbo: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample BBO to freq ('5s' or '60s') and run sliding Bayes trend."""
    cfg = HORIZON_CONFIG[freq]
    b = bbo.sort_values("ts_us").copy()
    b.index = pd.to_datetime(b["ts_us"], unit="us", utc=True)
    rule = freq if freq.endswith("s") else f"{freq}s"
    if freq == "60s":
        rule = "60s"
    s = b.resample(rule).agg(
        {
            "ts_us": "last",
            "microprice": "last",
            "mid": "last",
            "spread_bps": "mean",
            "book_imbalance": "mean",
        }
    ).dropna(subset=["microprice"])
    lp = np.log(s["microprice"].astype(float).values)
    bt = sliding_bayes_trend(lp, cfg)
    out = pd.concat([s.reset_index(drop=True), bt], axis=1)
    out = out.rename(
        columns={
            "slope": f"trend_slope_{freq}",
            "regime": f"trend_regime_{freq}",
            "cp_credibility": f"cp_cred_{freq}",
            "log_bf_linear": f"log_bf_linear_{freq}",
            "model": f"trend_model_{freq}",
        }
    )
    return out
