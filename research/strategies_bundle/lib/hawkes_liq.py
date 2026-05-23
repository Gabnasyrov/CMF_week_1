"""Univariate exponential Hawkes MLE for liquidation timestamps."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def hawkes_loglik(events: np.ndarray, mu: float, alpha: float, kappa: float) -> float:
    if len(events) < 3:
        return -1e12
    if mu <= 0 or alpha < 0 or kappa <= 1e-6:
        return -1e12
    if alpha / kappa >= 0.999:
        return -1e12
    t = events.astype(np.float64)
    t0, t1 = t[0], t[-1]
    span = t1 - t0
    if span <= 0:
        return -1e12
    s = 0.0
    ll = 0.0
    comp = 0.0
    for i, ti in enumerate(t):
        if i > 0:
            dt = ti - t[i - 1]
            comp += mu * dt + alpha * s * (1.0 - np.exp(-kappa * dt)) / kappa
            s *= np.exp(-kappa * dt)
        lam = mu + alpha * s
        if lam <= 1e-12:
            return -1e12
        ll += np.log(lam)
        s += 1.0
    comp += mu * max(1e-6, 1e-3)
    return float(ll - comp)


def fit_hawkes_mle(events_us: np.ndarray, max_events: int = 50_000) -> dict:
    if len(events_us) < 50:
        return {"ok": False, "reason": "too_few_events"}
    ev = np.sort(events_us.astype(np.float64)) / 1_000_000.0
    if len(ev) > max_events:
        ev = ev[-max_events:]
    n = len(ev)
    span = ev[-1] - ev[0]
    rate = n / max(span, 1.0)
    x0 = [rate * 0.3, rate * 0.5, 0.05]

    def nll(x):
        return -hawkes_loglik(ev, x[0], x[1], x[2])

    bounds = [(1e-8, rate * 10), (1e-8, rate * 10), (1e-4, 5.0)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        return {"ok": False, "reason": res.message}
    mu, alpha, kappa = res.x
    return {
        "ok": True,
        "mu": float(mu),
        "alpha": float(alpha),
        "kappa": float(kappa),
        "branching_ratio": float(alpha / kappa),
        "n_events_fit": n,
        "span_sec": float(span),
        "loglik": float(-res.fun),
    }
