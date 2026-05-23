"""Strategy bundle configuration."""

from __future__ import annotations

from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BUNDLE_ROOT / "config"
ARTIFACTS = BUNDLE_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"

HORIZONS_SEC = (1, 3, 5, 30, 300)
PRIMARY_HORIZONS = (30, 300)
TRAIN_FRAC = 0.4
MAX_PANEL_ROWS = 80_000
MAX_DAYS_DEFAULT = 90

FEATURE_GROUPS = {
    "kalman": ["kalman_mu", "kalman_vel", "kalman_innov", "kalman_var"],
    "hawkes": ["hawkes_lambda", "hawkes_branching"],
    "bayes": ["trend_regime_5s", "trend_slope_5s", "cp_cred_5s"],
    "micro": ["microprice_g", "book_imbalance", "ofi_increment", "spread_bps"],
    "flow": ["signed_vol_1s", "signed_vol_5s", "trades_per_sec"],
    "vol": ["vol_30s", "vov_30s"],
    "misprice": ["misprice_bid_bps", "misprice_ask_bps"],
    "liq": ["liq_bid_5s", "liq_ask_5s"],
    "depth": ["bid_depth_usd", "ask_depth_usd", "signed_depth_l1"],
}

ALL_FEATURES = sorted({f for fs in FEATURE_GROUPS.values() for f in fs})
FEATURES_NO_KALMAN = [f for f in ALL_FEATURES if not f.startswith("kalman_")]

SYMBOLS = ("btcusdt", "ethusdt")
