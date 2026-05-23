# Фичи панели (1s)

Все колонки строятся в `lib/feature_builder.py` из `data/enriched/`.

## Kalman (`kalman_*`)

| Колонка | Описание |
|---------|----------|
| `kalman_mu` | Уровень log(mid) после 2D Kalman (level + velocity) |
| `kalman_vel` | Скорость уровня |
| `kalman_innov` | Инновация наблюдения |
| `kalman_var` | Постериорная дисперсия уровня |

Фильтр: `lib/adaptive_kalman.py` — `filter_log_price(log_mid, q_level, q_vel, r_obs, adapt_alpha)`.

**Таргет:** `y_mu_sign_{h}s = sign(μ[t+h] − μ[t])` (`lib/targets.py`).

## Micro / book

| Колонка | Источник |
|---------|----------|
| `microprice_g` | enriched BBO |
| `book_imbalance` | enriched BBO |
| `ofi_increment` | enriched BBO |
| `spread_bps` | enriched BBO |
| `misprice_bid_bps`, `misprice_ask_bps` | mid vs touch |
| `bid_depth_usd`, `ask_depth_usd` | L1 notional |
| `signed_depth_l1` | bid − ask depth USD |

## Flow / vol

| Колонка | Описание |
|---------|----------|
| `signed_vol_1s`, `signed_vol_5s` | Signed trade notional (или OFI из enriched) |
| `trades_per_sec` | Число сделок / сек |
| `vol_30s`, `vov_30s` | Rolling std return bps |

## Liquidations / Hawkes

| Колонка | Описание |
|---------|----------|
| `liq_bid_5s`, `liq_ask_5s` | Notional buy/sell liq за сек (asof merge) |
| `hawkes_lambda` | λ(t) exp-Hawkes по всем liq (MLE на train liq) |
| `hawkes_branching` | α/κ (константа на панели) |

Hawkes: `fit_hawkes_mle` on train liquidations only (`timestamp <= train_end_us`); λ recurrence is causal on panel.
Trend: slope only at **closed** window end (`i-1`), `ffill` from past (no `bfill`, no forward stride fill).
1s bars: `resample(..., closed='right', label='right')`, drop last incomplete bar; `vol_*` / `signed_vol_5s` use `.shift(1).rolling(...)`.
LGBM train: `lgbm_train_mask` excludes last `h` train rows per horizon (no μ[t+h] from test).

## Bayes trend (light)

| Колонка | Описание |
|---------|----------|
| `trend_regime_5s` | sign(slope) линейной регрессии log-price в окне 48s |
| `trend_slope_5s` | Наклон |
| `cp_cred_5s` | min(1, \|slope\|×1e4) |

## Нормализация

Train-only z-score, clip ±8: `lib/normalization.py` → колонки `{feature}_z`.

**LGBM no-K:** все `ALL_FEATURES` кроме `kalman_*` (`FEATURES_NO_KALMAN` в `lib/config.py`).
