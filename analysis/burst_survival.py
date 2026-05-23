"""Survival / duration models for liquidation bursts."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from burst_features_extra import EXTRA_FEATURES, enrich_burst_row, prepare_liq_arrays  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES_BASE = [
    "spread_bps",
    "book_imbalance",
    "microprice_g",
    "ofi_5s",
    "log_bid_depth",
    "log_ask_depth",
    "signed_depth_l1",
    "ofi_30s_pre",
    "bybit_L_60s_pre",
    "ret_micro_5s_pre",
]
FEATURES = FEATURES_BASE + EXTRA_FEATURES

TRAIN_FRAC = 0.6
DURATION_CAP_SEC = 600.0


def _book_at(bbo: pd.DataFrame, ts_us: int) -> dict | None:
    bts = bbo["ts_us"].values
    idx = int(np.searchsorted(bts, ts_us, side="right") - 1)
    if idx < 0:
        return None
    row = bbo.iloc[idx]
    return {
        "spread_bps": float(row["spread_bps"]),
        "book_imbalance": float(row["book_imbalance"]),
        "microprice_g": float(row["microprice_g"]),
        "ofi_5s": float(row["ofi_5s"]),
        "bid_depth_usd": float(row["bid_depth_usd"]),
        "ask_depth_usd": float(row["ask_depth_usd"]),
        "signed_depth_l1": float(row["signed_depth_l1"]),
        "ret_micro_bps": float(row["ret_micro_bps"]),
    }


def _window_sum(bbo: pd.DataFrame, col: str, t_end_us: int, window_sec: int) -> float:
    bts = bbo["ts_us"].values
    t0 = t_end_us - window_sec * 1_000_000
    lo = int(np.searchsorted(bts, t0, side="left"))
    hi = int(np.searchsorted(bts, t_end_us, side="right"))
    if hi <= lo:
        return 0.0
    return float(bbo[col].iloc[lo:hi].sum())


def _bybit_intensity_before(bybit: pd.DataFrame, ts_us: int, window_sec: int = 60) -> float:
    bts = bybit["ts_us"].values
    t0 = ts_us - window_sec * 1_000_000
    lo = int(np.searchsorted(bts, t0, side="left"))
    hi = int(np.searchsorted(bts, ts_us, side="left"))
    if hi <= lo:
        return 0.0
    return float(bybit["bybit_L_abs"].iloc[lo:hi].sum())


def build_burst_table(
    bursts: pd.DataFrame,
    bbo: pd.DataFrame,
    bybit: pd.DataFrame,
    venue: str,
    sym: str,
    bursts_other: pd.DataFrame | None = None,
    liq_binance: pd.DataFrame | None = None,
    liq_bybit: pd.DataFrame | None = None,
    trend_5s: pd.DataFrame | None = None,
    trend_60s: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if bursts_other is None:
        bursts_other = pd.DataFrame()
    lb_ts = lb_sn = lb_ab = np.array([], dtype=np.int64)
    ly_ts = ly_sn = ly_ab = np.array([], dtype=np.int64)
    if liq_binance is not None and len(liq_binance):
        lb_ts, lb_sn, lb_ab = prepare_liq_arrays(liq_binance)
    if liq_bybit is not None and len(liq_bybit):
        ly_ts, ly_sn, ly_ab = prepare_liq_arrays(liq_bybit)

    rows = []
    last_end = None
    for _, b in bursts.iterrows():
        t0 = int(b["t_start"])
        t1 = int(b["t_end"])
        book = _book_at(bbo, t0)
        if book is None:
            continue
        dur = float(b["duration_sec"])
        if dur <= 0 or dur > DURATION_CAP_SEC:
            continue
        sn = float(b["signed_notional"])
        tn = float(b["total_notional"])
        row = {
            "venue": venue,
            "symbol": sym,
            "t_start": t0,
            "duration_sec": dur,
            "n_events": int(b["n_events"]),
            "total_notional": tn,
            "signed_notional": sn,
            **book,
            "log_bid_depth": np.log1p(book["bid_depth_usd"]),
            "log_ask_depth": np.log1p(book["ask_depth_usd"]),
            "ofi_30s_pre": _window_sum(bbo, "ofi_increment", t0, 30),
            "ret_micro_5s_pre": _window_sum(bbo, "ret_micro_bps", t0, 5),
            "bybit_L_60s_pre": _bybit_intensity_before(bybit, t0, 60),
        }
        row.update(
            enrich_burst_row(
                t0,
                t1,
                sn,
                tn,
                int(b["n_events"]),
                venue,
                bursts_other,
                lb_ts,
                lb_sn,
                lb_ab,
                ly_ts,
                ly_sn,
                ly_ab,
                last_end,
                trend_5s,
                trend_60s,
            )
        )
        rows.append(row)
        last_end = t1
    return pd.DataFrame(rows)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("t_start").reset_index(drop=True)
    cut = int(len(df) * TRAIN_FRAC)
    return df.iloc[:cut], df.iloc[cut:]


def _select_feats(df: pd.DataFrame, feat_list: list[str], min_std: float = 1e-8) -> list[str]:
    feats = [c for c in feat_list if c in df.columns]
    keep = []
    for c in feats:
        if float(df[c].std()) > min_std:
            keep.append(c)
    return keep


def fit_cox(train: pd.DataFrame, test: pd.DataFrame, feat_list: list[str] | None = None) -> dict:
    feats = _select_feats(train, feat_list or FEATURES)
    tr = train[feats + ["duration_sec"]].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[feats + ["duration_sec"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 80 or len(te) < 30 or len(feats) < 2:
        return {"ok": False, "reason": "too_few_rows"}
    try:
        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(tr, duration_col="duration_sec")
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    pred_tr = cph.predict_partial_hazard(tr[feats]).values.ravel()
    pred_te = cph.predict_partial_hazard(te[feats]).values.ravel()
    ci_tr = float(concordance_index(tr["duration_sec"], -pred_tr))
    ci_te = float(concordance_index(te["duration_sec"], -pred_te))
    coef = {f: float(cph.params_[f]) for f in feats if f in cph.params_.index}
    return {
        "ok": True,
        "c_index_train": ci_tr,
        "c_index_test": ci_te,
        "coef": coef,
        "summary": str(cph.summary[["coef", "p"]].head(8)),
    }


def fit_lgbm_duration(train: pd.DataFrame, test: pd.DataFrame, feat_list: list[str] | None = None) -> dict:
    feats = _select_feats(train, feat_list or FEATURES)
    tr = train[feats + ["duration_sec"]].replace([np.inf, -np.inf], np.nan).dropna()
    te = test[feats + ["duration_sec"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tr) < 80 or len(te) < 30:
        return {"ok": False}
    y_tr = np.log1p(tr["duration_sec"].values)
    y_te = np.log1p(te["duration_sec"].values)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        verbose=-1,
    )
    model.fit(tr[feats], y_tr)
    pred_te = np.expm1(model.predict(te[feats]))
    pred_te = np.clip(pred_te, 0.1, DURATION_CAP_SEC)
    mae = float(mean_absolute_error(te["duration_sec"], pred_te))
    r2 = float(r2_score(te["duration_sec"], pred_te))
    ci = float(concordance_index(te["duration_sec"], pred_te))
    imp = dict(zip(feats, model.feature_importances_.astype(float).tolist()))
    return {
        "ok": True,
        "mae_sec_test": mae,
        "r2_test": r2,
        "c_index_test": ci,
        "median_baseline_mae": float(
            mean_absolute_error(te["duration_sec"], np.full(len(te), tr["duration_sec"].median()))
        ),
        "feature_importance": imp,
        "pred_test": pred_te,
        "actual_test": te["duration_sec"].values,
    }


def plot_km(df: pd.DataFrame, out_path: Path, title: str):
    if len(df) < 40:
        return
    terc = pd.qcut(df["spread_bps"], 3, labels=["low_spread", "mid", "high_spread"], duplicates="drop")
    fig, ax = plt.subplots(figsize=(8, 4))
    kmf = KaplanMeierFitter()
    for label in terc.unique():
        m = terc == label
        if m.sum() < 10:
            continue
        kmf.fit(df.loc[m, "duration_sec"], label=str(label))
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("seconds since burst start")
    ax.set_ylabel("P(burst still active)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pred_scatter(actual, pred, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(actual, pred, alpha=0.25, s=8)
    mx = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, mx], [0, mx], "r--", lw=1)
    ax.set_xlabel("actual duration (s)")
    ax.set_ylabel("predicted (s)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_burst_survival(
    sym: str,
    venue: str,
    bursts: pd.DataFrame,
    bbo: pd.DataFrame,
    bybit: pd.DataFrame,
    out_dir: Path,
    tab_dir: Path,
    bursts_other: pd.DataFrame | None = None,
    liq_binance: pd.DataFrame | None = None,
    liq_bybit: pd.DataFrame | None = None,
    trend_5s: pd.DataFrame | None = None,
    trend_60s: pd.DataFrame | None = None,
) -> dict:
    df = build_burst_table(
        bursts,
        bbo,
        bybit,
        venue,
        sym,
        bursts_other=bursts_other,
        liq_binance=liq_binance,
        liq_bybit=liq_bybit,
        trend_5s=trend_5s,
        trend_60s=trend_60s,
    )
    if len(df) < 100:
        return {"ok": False, "n": len(df)}
    df.to_csv(tab_dir / f"05_burst_features_{venue}_{sym}.csv", index=False)
    tr, te = _temporal_split(df)
    cox = fit_cox(tr, te, FEATURES)
    cox_base = fit_cox(tr, te, FEATURES_BASE)
    lgbm = fit_lgbm_duration(tr, te, FEATURES)
    lgbm_base = fit_lgbm_duration(tr, te, FEATURES_BASE)
    plot_km(
        df,
        out_dir / f"05_km_spread_{venue}_{sym}.png",
        f"KM by spread tercile — {venue} bursts {sym}",
    )
    if lgbm.get("ok"):
        plot_pred_scatter(
            lgbm["actual_test"],
            lgbm["pred_test"],
            out_dir / f"05_lgbm_duration_{venue}_{sym}.png",
            f"LGBM duration test — {venue} {sym}",
        )
    row = {
        "symbol": sym,
        "venue": venue,
        "n_bursts": len(df),
        "median_duration_sec": float(df["duration_sec"].median()),
        "cox": cox,
        "cox_base": cox_base,
        "lgbm": {k: v for k, v in lgbm.items() if k not in ("pred_test", "actual_test")},
        "lgbm_base": {k: v for k, v in lgbm_base.items() if k not in ("pred_test", "actual_test")},
    }
    pd.DataFrame([row]).to_json(tab_dir / f"05_survival_metrics_{venue}_{sym}.json", indent=2)
    if lgbm.get("ok") and lgbm.get("feature_importance"):
        pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in lgbm["feature_importance"].items()]
        ).sort_values("importance", ascending=False).to_csv(
            tab_dir / f"05_lgbm_importance_{venue}_{sym}.csv", index=False
        )
    return row
