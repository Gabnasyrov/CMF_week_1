"""Build 1s Binance panel with ensemble microstructure features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .adaptive_kalman import filter_log_price
from .config import ALL_FEATURES
from .hawkes_liq import fit_hawkes_mle
from .paths import enriched_paths


def _agg_trades_1s(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.sort_values("timestamp")
    t["sec"] = (t["timestamp"] // 1_000_000).astype(np.int64)
    t["signed"] = np.where(
        t["side"] == "buy",
        t["price"] * t["amount"],
        -(t["price"] * t["amount"]),
    )
    g = t.groupby("sec", sort=True)
    return pd.DataFrame(
        {
            "ts_us": g["timestamp"].max(),
            "signed_vol_1s": g["signed"].sum(),
            "trades_per_sec": g.size(),
        }
    ).reset_index(drop=True)


def _agg_liq_sides_1s(liq: pd.DataFrame) -> pd.DataFrame:
    t = liq.sort_values("timestamp")
    t["notional"] = t["price"] * t["amount"]
    t["sec"] = (t["timestamp"] // 1_000_000).astype(np.int64)
    rows = []
    for sec, g in t.groupby("sec", sort=True):
        buy = g.loc[g["side"] == "buy", "notional"].sum()
        sell = g.loc[g["side"] == "sell", "notional"].sum()
        rows.append({"ts_us": int(g["timestamp"].max()), "liq_bid_5s": buy, "liq_ask_5s": sell})
    return pd.DataFrame(rows)


def _merge_asof(panel_ts: np.ndarray, aux: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    if aux.empty:
        return {c: np.zeros(len(panel_ts)) for c in cols}
    bts = aux["ts_us"].values.astype(np.int64)
    out = {}
    idx = np.searchsorted(bts, panel_ts, side="right") - 1
    for c in cols:
        v = aux[c].values.astype(np.float64)
        arr = np.zeros(len(panel_ts))
        ok = idx >= 0
        arr[ok] = v[idx[ok]]
        out[c] = np.nan_to_num(arr, nan=0.0)
    return out


def _sliding_bayes_light(log_p: np.ndarray, window: int = 48, stride: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal trend: only closed windows [i-window, i) — label at index i-1 when window completes.
    No forward fill into future stride slots; ffill from past only (no bfill).
    """
    n = len(log_p)
    regime = np.zeros(n)
    slope = np.zeros(n)
    cred = np.full(n, 0.5)
    for i in range(window, n, stride):
        yw = log_p[i - window : i]
        if not np.isfinite(yw).all():
            continue
        yw = yw - yw[0]
        t = np.linspace(0, 1, len(yw))
        b1 = np.polyfit(t, yw, 1)[0]
        idx = i - 1
        slope[idx] = b1
        regime[idx] = np.sign(b1)
        cred[idx] = min(1.0, abs(b1) * 1e4)
    slope = pd.Series(slope).replace(0, np.nan).ffill().fillna(0).values
    regime = pd.Series(regime).replace(0, np.nan).ffill().fillna(0).values
    cred = pd.Series(cred).ffill().fillna(0.5).values
    return regime, slope, cred


def add_hawkes_features(panel: pd.DataFrame, sym: str, fit_end_us: int) -> pd.DataFrame:
    """Fit Hawkes MLE on liquidations with timestamp <= fit_end_us; causal λ recurrence on panel."""
    out = panel.copy()
    liq = pd.read_parquet(
        enriched_paths(sym)["liq_binance"],
        columns=["timestamp"],
    )
    liq_tr = liq[liq["timestamp"] <= fit_end_us]
    hawkes_fit = fit_hawkes_mle(liq_tr["timestamp"].values)

    liq_cnt = liq.copy()
    liq_cnt["sec"] = (liq_cnt["timestamp"] // 1_000_000).astype(np.int64)
    cnt_1s = liq_cnt.groupby("sec").size().reset_index(name="n_liq")
    cnt_1s["ts_us"] = cnt_1s["sec"] * 1_000_000
    mcnt = _merge_asof(out["ts_us"].values.astype(np.int64), cnt_1s, ["n_liq"])
    nliq = mcnt["n_liq"]

    if hawkes_fit.get("ok"):
        kappa = hawkes_fit["kappa"]
        mu_h, alpha = hawkes_fit["mu"], hawkes_fit["alpha"]
        decay = np.exp(-kappa)
        r = 0.0
        lam = np.zeros(len(nliq))
        for i in range(len(nliq)):
            r = r * decay + nliq[i]
            lam[i] = mu_h + alpha * r
        out["hawkes_lambda"] = lam
        out["hawkes_branching"] = hawkes_fit["branching_ratio"]
    else:
        out["hawkes_lambda"] = 0.0
        out["hawkes_branching"] = 0.0
    return out


def build_panel(
    sym: str,
    max_days: int | None = 90,
    kalman_q_level: float = 1e-6,
    kalman_q_vel: float = 1e-8,
    kalman_r_obs: float = 1e-4,
    kalman_adapt: float = 0.05,
    hawkes_fit_end_us: int | None = None,
    skip_hawkes: bool = False,
) -> pd.DataFrame:
    paths = enriched_paths(sym)
    bbo_path = paths["bbo"]
    if not bbo_path.exists():
        raise FileNotFoundError(f"Missing BBO: {bbo_path}. Set LIQUIDATION_DATA_ROOT.")

    pf = pq.ParquetFile(bbo_path)
    liq = pd.read_parquet(paths["liq_binance"], columns=["timestamp", "side", "price", "amount"])
    liq_1s = _agg_liq_sides_1s(liq)

    chunks = []
    seen = set()
    for rg in range(pf.metadata.num_row_groups):
        day = int(pc.min(pf.read_row_group(rg, columns=["timestamp"]).column("timestamp")).as_py() // 86_400_000_000)
        if day in seen:
            continue
        seen.add(day)
        if max_days is not None and len(seen) > max_days:
            break

        df = pf.read_row_group(
            rg,
            columns=[
                "timestamp",
                "bid_price",
                "ask_price",
                "bid_amount",
                "ask_amount",
                "mid",
                "microprice",
                "book_imbalance",
                "ofi_increment",
                "spread_bps",
                "microprice_g",
            ],
        ).to_pandas()
        df.index = pd.to_datetime(df["timestamp"], unit="us", utc=True)
        s = df.resample("1s", closed="right", label="right").last().dropna(subset=["mid"])
        if len(s) > 1:
            s = s.iloc[:-1]
        s["ts_us"] = s["timestamp"].astype(np.int64)
        s["ret_1s_bps"] = (s["mid"] / s["mid"].shift(1) - 1) * 10_000
        s["vol_30s"] = s["ret_1s_bps"].shift(1).rolling(30, min_periods=5).std()
        s["vov_30s"] = s["vol_30s"].shift(1).rolling(30, min_periods=5).std()

        log_mid = np.log(s["mid"].astype(float).values)
        mu, vel, innov, kv = filter_log_price(
            log_mid, kalman_q_level, kalman_q_vel, kalman_r_obs, kalman_adapt
        )
        s["kalman_mu"] = mu
        s["kalman_vel"] = vel
        s["kalman_innov"] = innov
        s["kalman_var"] = kv

        reg, sl, cr = _sliding_bayes_light(log_mid)
        s["trend_regime_5s"] = reg
        s["trend_slope_5s"] = sl
        s["cp_cred_5s"] = cr

        s["bid_depth_usd"] = s["bid_price"] * s["bid_amount"]
        s["ask_depth_usd"] = s["ask_price"] * s["ask_amount"]
        s["signed_depth_l1"] = s["bid_depth_usd"] - s["ask_depth_usd"]
        s["misprice_bid_bps"] = (s["mid"] - s["bid_price"]) / s["mid"] * 10_000
        s["misprice_ask_bps"] = (s["ask_price"] - s["mid"]) / s["mid"] * 10_000

        ts = s["ts_us"].values.astype(np.int64)
        liq_m = _merge_asof(ts, liq_1s, ["liq_bid_5s", "liq_ask_5s"])
        for k, v in liq_m.items():
            s[k] = v
        chunks.append(s.reset_index(drop=True))

    if not chunks:
        return pd.DataFrame()

    panel = pd.concat(chunks, ignore_index=True).sort_values("ts_us")
    if skip_hawkes:
        panel["hawkes_lambda"] = 0.0
        panel["hawkes_branching"] = 0.0
    elif hawkes_fit_end_us is not None:
        panel = add_hawkes_features(panel, sym, hawkes_fit_end_us)
    else:
        panel = add_hawkes_features(panel, sym, int(liq["timestamp"].max()))

    panel["signed_vol_1s"] = 0.0
    panel["signed_vol_5s"] = 0.0
    panel["trades_per_sec"] = 0.0
    try:
        tp = pq.ParquetFile(paths["trades"])
        t_chunks = []
        seen_t = set()
        for rg in range(min(tp.metadata.num_row_groups, len(seen) + 5)):
            tday = int(
                pc.min(tp.read_row_group(rg, columns=["timestamp"]).column("timestamp")).as_py()
                // 86_400_000_000
            )
            if tday in seen_t:
                continue
            seen_t.add(tday)
            td = tp.read_row_group(
                rg,
                columns=["timestamp", "price", "amount", "side", "order_flow_imbalance_1s"],
            ).to_pandas()
            if "order_flow_imbalance_1s" in td.columns:
                td["signed_vol_1s"] = td["order_flow_imbalance_1s"]
            else:
                td = _agg_trades_1s(td)
            t_chunks.append(td)
        if t_chunks:
            tr = pd.concat(t_chunks, ignore_index=True)
            m = _merge_asof(panel["ts_us"].values, tr, ["signed_vol_1s", "trades_per_sec"])
            panel["signed_vol_1s"] = m["signed_vol_1s"]
            panel["trades_per_sec"] = m["trades_per_sec"]
            panel["signed_vol_5s"] = panel["signed_vol_1s"].shift(1).rolling(5, min_periods=1).sum()
    except Exception:
        panel["signed_vol_5s"] = panel["signed_vol_1s"].shift(1).rolling(5, min_periods=1).sum()

    panel["symbol"] = sym
    for c in ALL_FEATURES:
        if c not in panel.columns:
            panel[c] = 0.0
    return panel.dropna(subset=["kalman_mu", "mid"]).reset_index(drop=True)
