"""Order book / order flow microstructure (Stoikov 2018 microprice, OFI, book imbalance)."""

from __future__ import annotations

import numpy as np


def book_imbalance(bid_vol: np.ndarray, ask_vol: np.ndarray) -> np.ndarray:
    """I = (V_b - V_a) / (V_b + V_a), in [-1, 1]."""
    denom = bid_vol + ask_vol
    out = np.zeros_like(bid_vol, dtype=np.float64)
    m = denom > 0
    out[m] = (bid_vol[m] - ask_vol[m]) / denom[m]
    return out


def microprice_stoikov(
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    bid_vol: np.ndarray,
    ask_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
  Stoikov (2018) microprice at touch:
    psi = (P_b * V_a + P_a * V_b) / (V_b + V_a)
    mid = (P_b + P_a) / 2
    g   = psi - mid = I * (P_a - P_b) / 2   where I is book imbalance

  Returns (microprice, mid, g_bps) with g in price units.
    """
    denom = bid_vol + ask_vol
    mid = (bid_price + ask_price) * 0.5
    psi = mid.copy()
    m = denom > 0
    psi[m] = (bid_price[m] * ask_vol[m] + ask_price[m] * bid_vol[m]) / denom[m]
    g = psi - mid
    return psi, mid, g


def ofi_increment_cont(
    bid_p: np.ndarray,
    ask_p: np.ndarray,
    bid_v: np.ndarray,
    ask_v: np.ndarray,
    bid_p_prev: float,
    ask_p_prev: float,
    bid_v_prev: float,
    ask_v_prev: float,
) -> tuple[np.ndarray, float, float, float, float]:
    """
  Cont–Kukanov–Stoikov style OFI increment per row (vectorized except first row).

  First row uses previous state from prior chunk.
    """
    n = len(bid_p)
    e = np.zeros(n, dtype=np.float64)

    # row 0 vs previous chunk state
    e[0] = _ofi_event(bid_p[0], ask_p[0], bid_v[0], ask_v[0], bid_p_prev, ask_p_prev, bid_v_prev, ask_v_prev)

    if n > 1:
        # bid side
        bp0, bp1 = bid_p[:-1], bid_p[1:]
        bv0, bv1 = bid_v[:-1], bid_v[1:]
        up = bp1 > bp0
        same = bp1 == bp0
        down = bp1 < bp0
        e[1:][up] += bv1[up]
        e[1:][same] += (bv1[same] - bv0[same])
        e[1:][down] -= bv0[down]

        ap0, ap1 = ask_p[:-1], ask_p[1:]
        av0, av1 = ask_v[:-1], ask_v[1:]
        down_a = ap1 < ap0
        same_a = ap1 == ap0
        up_a = ap1 > ap0
        e[1:][down_a] += av1[down_a]
        e[1:][same_a] -= av1[same_a] - av0[same_a]
        e[1:][up_a] -= av0[up_a]

    return (
        e,
        float(bid_p[-1]),
        float(ask_p[-1]),
        float(bid_v[-1]),
        float(ask_v[-1]),
    )


def _ofi_event(bp, ap, bv, av, bp0, ap0, bv0, av0) -> float:
    e = 0.0
    if bp > bp0:
        e += bv
    elif bp == bp0:
        e += bv - bv0
    else:
        e -= bv0
    if ap < ap0:
        e += av
    elif ap == ap0:
        e -= av - av0
    else:
        e -= av0
    return e


def trade_flow_imbalance(signed_notional: np.ndarray) -> np.ndarray:
    """Cumulative signed taker flow (buy positive)."""
    return np.cumsum(signed_notional)


def asof_lookup(event_ts: np.ndarray, feat_ts: np.ndarray, feat_val: np.ndarray) -> np.ndarray:
    """Last feature value at or before each event timestamp."""
    idx = np.searchsorted(feat_ts, event_ts, side="right") - 1
    out = np.full(len(event_ts), np.nan, dtype=np.float64)
    ok = idx >= 0
    out[ok] = feat_val[idx[ok]]
    return out
