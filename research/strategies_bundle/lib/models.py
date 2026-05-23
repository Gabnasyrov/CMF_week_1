"""LGBM direction models with save/load."""

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluate import mu_direction_metrics


def _binary_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(y) & (y != 0)
    yb = (y[m] > 0).astype(int)
    return m, yb


class LGBMDirectionModel:
    def __init__(self, **kwargs):
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": kwargs.get("num_leaves", 31),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "min_child_samples": kwargs.get("min_child_samples", 50),
            "feature_fraction": kwargs.get("feature_fraction", 0.8),
            "n_estimators": kwargs.get("n_estimators", 300),
        }
        self.model = None
        self.feat_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray):
        self.feat_cols = list(X.columns)
        m_tr, yb_tr = _binary_labels(y)
        m_va, yb_va = _binary_labels(y_val)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X.loc[m_tr, self.feat_cols],
            yb_tr,
            eval_set=[(X_val.loc[m_va, self.feat_cols], yb_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return self

    def predict_sign(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X[self.feat_cols])[:, 1]
        return np.where(proba >= 0.5, 1.0, -1.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"params": self.params, "feat_cols": self.feat_cols, "model": self.model}, path)

    @classmethod
    def load(cls, path: Path) -> "LGBMDirectionModel":
        d = joblib.load(path)
        obj = cls(**{k: d["params"][k] for k in ("num_leaves", "learning_rate") if k in d["params"]})
        obj.params = d["params"]
        obj.feat_cols = d["feat_cols"]
        obj.model = d["model"]
        return obj


def score_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return mu_direction_metrics(y_true, y_pred)
