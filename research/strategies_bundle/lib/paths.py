"""Data root resolution for standalone runs."""

from __future__ import annotations

import os
from pathlib import Path


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """liquidation_task repo root (parent of research/)."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    env = os.environ.get("LIQUIDATION_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "data"


def enriched_paths(sym: str) -> dict[str, Path]:
    base = data_root() / "enriched"
    return {
        "bbo": base / "binance_bbo" / f"perp_{sym}.parquet",
        "trades": base / "binance_trades" / f"perp_{sym}.parquet",
        "liq_binance": base / "binance_liquidations" / f"perp_{sym}.parquet",
        "liq_bybit": base / "bybit_liquidations" / f"{sym}.parquet",
    }
