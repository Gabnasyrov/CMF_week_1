"""Paths and constants for TZ assignment (no data shipped in repo)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
# One data root only. Priority: env > parent ../data (when inside liquidation_task) > tz_assignment/data/
_parent_data = (PKG_ROOT.parent / "data").resolve()
_local_data = (PKG_ROOT / "data").resolve()
if os.environ.get("LIQUIDATION_DATA_ROOT"):
    DATA = Path(os.environ["LIQUIDATION_DATA_ROOT"]).resolve()
elif (_local_data / "binance_trades").is_dir():
    DATA = _local_data
elif (_parent_data / "binance_trades").is_dir():
    DATA = _parent_data
else:
    DATA = _local_data
RESULTS = PKG_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
NOTEBOOKS = PKG_ROOT / "notebooks"

SYMBOLS = ("btcusdt", "ethusdt")
BYBIT_LATENCY_US = 200_000
HOUR_US = 3_600_000_000
DAY_US = 86_400_000_000

TRAIN_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
VAL_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
VAL_END = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

SAMPLE_DAYS = (
    datetime(2025, 12, 15, tzinfo=timezone.utc),
    datetime(2026, 1, 10, tzinfo=timezone.utc),
    datetime(2026, 2, 10, tzinfo=timezone.utc),
)

LEAD_LAG_SEC = (-5, -2, -1, 0, 1, 2, 5, 10, 30, 60)
BBO_SAMPLE_ROWS = 200_000
TRADE_SAMPLE_ROWS = 200_000


def paths_for_symbol(sym: str) -> dict[str, Path]:
    return {
        "trades": DATA / "binance_trades" / f"perp_{sym}.parquet",
        "bbo": DATA / "binance_booktickers" / f"perp_{sym}.parquet",
        "liq_binance": DATA / "binance_liquidations" / f"perp_{sym}.parquet",
        "liq_bybit": DATA / "bybit_liquidations" / f"{sym}.parquet",
    }


def enriched_paths(sym: str) -> dict[str, Path]:
    base = DATA / "enriched"
    return {
        "bbo": base / "binance_bbo" / f"perp_{sym}.parquet",
        "trades": base / "binance_trades" / f"perp_{sym}.parquet",
        "liq_binance": base / "binance_liquidations" / f"perp_{sym}.parquet",
        "liq_bybit": base / "bybit_liquidations" / f"{sym}.parquet",
    }


def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def ensure_dirs() -> None:
    for d in (FIGURES, TABLES, NOTEBOOKS, DATA):
        d.mkdir(parents=True, exist_ok=True)


def has_raw_data(symbols: tuple[str, ...] = SYMBOLS) -> bool:
    """All 8 raw parquet files present under DATA."""
    for sym in symbols:
        for path in paths_for_symbol(sym).values():
            if not path.is_file():
                return False
    return True


def has_enriched(symbols: tuple[str, ...] = SYMBOLS) -> bool:
    """Enriched BBO exists for each symbol (proxy for full feature build)."""
    for sym in symbols:
        if not enriched_paths(sym)["bbo"].is_file():
            return False
        try:
            if enriched_paths(sym)["bbo"].stat().st_size < 10_000:
                return False
        except OSError:
            return False
    return True
