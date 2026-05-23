# Стратегии (liquidation signal → maker filter)

Пакет TZ запускает те же модели, что `research/strategies_bundle` и PnL из `research/pnl_evaluation`.

## Binance vs Bybit в отчётах (прочитать первым)

| | **Binance** (`venue=binance`) | **Bybit** (`venue=bybit`) |
|---|------------------------------|---------------------------|
| **Официальная задача** | ✅ Фильтр **реальных trades** | ❌ Не площадка исполнения в ТЗ |
| **События в PnL** | Binance trades | **Ликвидации** как proxy (+200 ms) |
| **Сигнал** | Binance 1s (LGBM / Kalman) | **Тот же** Binance-сигнал (asof) |
| **Markout** | Binance BBO mid | **Тот же** Binance BBO mid |
| **Метрика качества фильтра** | `score_bps` | **`score_bps`** (не сырой pnl_kept) |
| **WR / hit_rate** | `direction_metrics_*.csv` | **Нет** отдельного WR на Bybit |

Готовая таблица Bybit: **`results/strategies/bybit_filter_summary.csv`** (τ=30s).

## Запуск из `tz_assignment/`

```bash
cd tz_assignment
pip install -r requirements.txt -r requirements-strategies.txt

# 1) Сырые parquet в data/ (см. data/README.md)
python build_features.py

# 2) Обучение + WR + PnL (долго на полных 90 днях)
python run_strategies.py

# Быстрый smoke (~14 дней, subsample trades)
python run_strategies.py --quick

# Только направление, без PnL
python run_strategies.py --skip-pnl

# Полный пайплайн: фичи + стратегии + EDA
python run_all.py
```

**Выход стратегий:** `results/strategies/`

| Файл | Содержание |
|------|------------|
| `direction_metrics_{sym}.csv` | WR / hit_rate по горизонтам (Kalman, LGBM no-K, ensemble) |
| `direction_eval_{sym}.json` | Полные метрики evaluate |
| `pnl_report.csv`, `PNL_REPORT.md` | Maker markout τ∈{30,120,300}s, Binance trades + **Bybit liq** |
| `bybit_filter_summary.csv` | **Bybit-only** filter scores @ 30s |
| `run_meta.json` | Параметры прогона |

Веса моделей: `../research/strategies_bundle/artifacts/models/{sym}/`.

**Фичи:** см. [FEATURES.md](FEATURES.md) (в ноутбуке — отдельная секция).

---

# Удачные стратегии (WR / hit_rate > 0.53 на test)

**Задача:** предсказать `sign(μ_Kalman[t+h] − μ_Kalman[t])` на 1s панели Binance.  
**Оценка:** хронологический split **40% train / 60% test** (bundle) или **50/50** (PnL в `run_strategies.py`), WR = доля совпадения знака.

## Рекомендуемые (честные)

### S1 — LGBM без Kalman-фичей @ 30s (основная)

| | BTC | ETH |
|---|-----|-----|
| **WR test** | **0.560** | **0.544** |
| rank IC | ~0.33 | ~0.21 |

- Microstructure + flow + liq + Hawkes + trend **без** `kalman_*`.
- PnL id: `lgbm_nk_30s`, `lgbm_nk_300s`.

### S2 — Kalman-only (без ML)

| Symbol | Лучший h | WR test |
|--------|----------|---------|
| BTC | 1s | **0.537** |
| ETH | 30s | **0.537** |

- Знак `μ[t+h]−μ[t]` из 2D adaptive Kalman.
- PnL id: `kalman_30s`, `kalman_300s`.

### S3 — Ensemble (0.4 Kalman + 0.6 LGBM no-K)

- PnL id: `ensemble_30s`.

### S4 — Baseline (без фильтра)

- Все сделки kept; `score_bps` = 0 по определению.

## Не использовать для честной оценки edge

### S5 — LGBM **с** Kalman-фичами (`lgbm_full_*`)

Таргет из `kalman_mu`; `kalman_*` в X завышают WR на 1–5s. В PnL включён только для сравнения.

## Maker PnL (описание)

1. **Сигнал** на Binance 1s (asof backward на сделку).
2. **Фильтр:** убрать сделку, если `signal × s_taker > 0` (adverse для мейкера).
3. **Binance:** реальные trades. **Bybit:** ликвидации как proxy, **+200 ms**.
4. **Markout:** Binance BBO mid на `t+τ`, τ ∈ {30, 120, 300} s, rebate +0.5 bps.

См. `docs/description.md` в корне репозитория.

## Эталон

`../research/strategies_bundle/artifacts/metrics_reference.json`
