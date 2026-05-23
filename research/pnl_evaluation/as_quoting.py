"""
Avellaneda–Stoikov (2018) reservation price with strategy signal instead of microprice.

Stoikov microprice: psi = (P_b*V_a + P_a*V_b)/(V_b+V_a) = mid + I*(spread/2).
Here: fair = mid + signal * AS_SKEW_BPS/1e4 * mid  (directional skew from Binance forecast).
Quotes: bid = fair - half_spread, ask = fair + half_spread.
"""

from __future__ import annotations

import numpy as np

from .config import AS_HALF_SPREAD_BPS, AS_SKEW_BPS


def reservation_price(mid: np.ndarray, signal: np.ndarray, skew_bps: float = AS_SKEW_BPS) -> np.ndarray:
    """Fair price from mid and directional signal in {-1, 0, 1, nan}."""
    s = np.nan_to_num(signal, nan=0.0)
    return mid * (1.0 + s * skew_bps / 10_000.0)


def as_quotes(mid: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric spread around reservation (inventory q=0)."""
    fair = reservation_price(mid, signal)
    hs = mid * AS_HALF_SPREAD_BPS / 10_000.0
    return fair - hs, fair + hs


def filter_adverse_trades(
    signal: np.ndarray,
    taker_side: np.ndarray,
    trade_price: np.ndarray | None = None,
    mid_at_trade: np.ndarray | None = None,
    use_as_price: bool = False,
) -> np.ndarray:
    """
    Binary filter f: 1 = drop trade, 0 = keep (per description.md).

    Primary rule: drop when signal aligns with taker flow (adverse for passive maker).
      f = 1 if signal * s > 0,  s = +1 taker buy (maker sell).

    Optional AS rule (use_as_price): also drop if fill is worse than reservation.
    """
    if taker_side.dtype.kind in ("i", "u", "f"):
        s = taker_side.astype(np.float64)
    else:
        s = np.where(taker_side == "buy", 1.0, -1.0)
    sig = np.nan_to_num(signal, nan=0.0)
    f = (sig * s > 0).astype(np.int8)

    if use_as_price and trade_price is not None and mid_at_trade is not None:
        fair = reservation_price(mid_at_trade, signal)
        # maker sell (s=+1): drop if sold below fair
        adverse_price = ((s > 0) & (trade_price < fair)) | ((s < 0) & (trade_price > fair))
        f = np.maximum(f, adverse_price.astype(np.int8))

    return f
