"""Evaluation: sign(mu[t+h]-mu[t]) agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mu_direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if m.sum() < 50:
        return {"hit_rate": np.nan, "rank_ic": np.nan, "n": int(m.sum())}
    yt, yp = y_true[m], y_pred[m]
    hit = float((np.sign(yt) == np.sign(yp)).mean())
    ic = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
    return {"hit_rate": hit, "rank_ic": ic, "n": int(m.sum())}
