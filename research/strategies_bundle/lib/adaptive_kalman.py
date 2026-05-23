"""2D adaptive Kalman (level + velocity) on log-mid."""

from __future__ import annotations

import numpy as np


def filter_log_price(
    log_mid: np.ndarray,
    q_level: float = 1e-6,
    q_vel: float = 1e-8,
    r_obs: float = 1e-4,
    adapt_alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(log_mid)
    mu = np.full(n, np.nan)
    vel = np.full(n, np.nan)
    innov = np.full(n, np.nan)
    var = np.full(n, np.nan)
    if n == 0:
        return mu, vel, innov, var

    x = np.array([log_mid[0], 0.0])
    P = np.diag([1e-4, 1e-6])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([q_level, q_vel])
    R = r_obs

    for i in range(n):
        z = log_mid[i]
        if not np.isfinite(z):
            if i > 0:
                mu[i], vel[i], innov[i], var[i] = mu[i - 1], vel[i - 1], 0.0, var[i - 1]
            continue
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y = z - (H @ x_pred).item()
        S = (H @ P_pred @ H.T).item() + R
        K = (P_pred @ H.T) / S
        x = x_pred + (K * y).ravel()
        P = (np.eye(2) - K.reshape(2, 1) @ H) @ P_pred
        mu[i] = x[0]
        vel[i] = x[1]
        innov[i] = y
        var[i] = P[0, 0]
        R = (1 - adapt_alpha) * R + adapt_alpha * max(y * y, 1e-12)

    return mu, vel, innov, var
