import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("LIQUIDATION_DATA_ROOT"):
    DATA = Path(os.environ["LIQUIDATION_DATA_ROOT"]).resolve()
else:
    DATA = ROOT / "data"
FIGURES = Path(__file__).resolve().parent / "figures"
TABLES = Path(__file__).resolve().parent / "tables"

SYMBOLS = ("btcusdt", "ethusdt")
BINANCE_TICKER = {s: f"perp:{s}" for s in SYMBOLS}
BYBIT_LATENCY_US = 200_000
TAUS_SEC = (30, 120, 300)
MAKER_REBATE_BPS = 0.5
WEIGHT_CAP_USD = 100_000
MIN_TURNOVER_PER_DAY = 500_000

TRAIN_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
VAL_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
VAL_END = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

# Sample days for heavy trade/BBO joins (full period in liq-only scripts)
SAMPLE_DAYS = (
    datetime(2025, 12, 15, tzinfo=timezone.utc),
    datetime(2026, 1, 10, tzinfo=timezone.utc),
    datetime(2026, 2, 10, tzinfo=timezone.utc),
)

RESAMPLE_MS = (100, 500, 1000, 5000)
LEAD_LAG_SEC = (-5, -2, -1, 0, 1, 2, 5, 10, 30, 60)


def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def us_to_dt(us: int) -> datetime:
    return datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc)


def paths_for_symbol(sym: str) -> dict[str, Path]:
    return {
        "trades": DATA / "binance_trades" / f"perp_{sym}.parquet",
        "bbo": DATA / "binance_booktickers" / f"perp_{sym}.parquet",
        "liq_binance": DATA / "binance_liquidations" / f"perp_{sym}.parquet",
        "liq_bybit": DATA / "bybit_liquidations" / f"{sym}.parquet",
    }


def bybit_market_paths(sym: str) -> dict[str, Path]:
    """Native Bybit perp market data (separate from Binance paths)."""
    return {
        "trades": DATA / "bybit_trades" / f"{sym}.parquet",
        "bbo": DATA / "bybit_booktickers" / f"{sym}.parquet",
        "liq_bybit": DATA / "bybit_liquidations" / f"{sym}.parquet",
    }


ENRICHED = DATA / "enriched"


def enriched_paths(sym: str) -> dict[str, Path]:
    return {
        "bbo": ENRICHED / "binance_bbo" / f"perp_{sym}.parquet",
        "trades": ENRICHED / "binance_trades" / f"perp_{sym}.parquet",
        "liq_binance": ENRICHED / "binance_liquidations" / f"perp_{sym}.parquet",
        "liq_bybit": ENRICHED / "bybit_liquidations" / f"{sym}.parquet",
    }
