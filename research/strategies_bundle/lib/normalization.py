"""Z-score normalization fit on train only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ZScoreNormalizer:
    def __init__(self, cols: list[str], clip: float = 8.0):
        self.cols = cols
        self.clip = clip
        self.mean_: dict[str, float] = {}
        self.std_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "ZScoreNormalizer":
        for c in self.cols:
            v = df[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) < 10:
                self.mean_[c], self.std_[c] = 0.0, 1.0
            else:
                self.mean_[c] = float(v.mean())
                self.std_[c] = float(max(v.std(), 1e-9))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.cols:
            if c not in out.columns:
                continue
            z = (out[c] - self.mean_[c]) / self.std_[c]
            out[c + "_z"] = z.clip(-self.clip, self.clip)
        return out

    def z_columns(self) -> list[str]:
        return [c + "_z" for c in self.cols if c in self.mean_]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"mean": self.mean_, "std": self.std_, "cols": self.cols, "clip": self.clip},
            indent=2,
        )
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ZScoreNormalizer":
        d = json.loads(path.read_text())
        obj = cls(d["cols"], clip=d.get("clip", 8.0))
        obj.mean_ = d["mean"]
        obj.std_ = d["std"]
        return obj
