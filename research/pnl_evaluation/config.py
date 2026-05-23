"""PnL / markout evaluation config (task spec + research strategies)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
TAB = OUT / "tables"

SYMBOLS = ("btcusdt", "ethusdt")
VENUES = ("binance", "bybit")
TAUS_SEC = (30, 120, 300)

# Chronological split: first TRAIN_FRAC train (LGBM), rest test (PnL)
TRAIN_FRAC = 0.5

MAKER_REBATE_BPS = 0.5
WEIGHT_CAP_USD = 100_000
BYBIT_LATENCY_US = 200_000
MIN_TURNOVER_PER_DAY = 500_000

# Avellaneda–Stoikov skew: fair = mid * (1 + signal * skew_bps / 1e4)
AS_SKEW_BPS = 2.0
AS_HALF_SPREAD_BPS = 1.0

# Strategy definitions: (id, kind, forecast_horizon_sec)
STRATEGIES = (
    ("baseline", "baseline", None),
    ("kalman_30s", "kalman", 30),
    ("kalman_300s", "kalman", 300),
    ("lgbm_nk_30s", "lgbm_nk", 30),
    ("lgbm_nk_300s", "lgbm_nk", 300),
    ("lgbm_full_30s", "lgbm_full", 30),
    ("ensemble_30s", "ensemble", 30),
)

BUNDLE_ROOT = ROOT / "research" / "strategies_bundle"
