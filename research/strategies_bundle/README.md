# Strategies bundle (WR > 0.53)

Самодостаточный пакет для передачи и отдельного запуска удачных стратегий прогноза направления **μ Kalman** на Binance (1s panel, 90 дней, split train 40% / test 60%).

## Требования

- Python 3.10+
- Данные: `data/enriched/` из основного репозитория (BBO, trades, liquidations)

```bash
cd research/strategies_bundle
pip install -r requirements.txt
```

## Пути к данным

По умолчанию ищется `../../data` относительно этой папки. Или явно:

```bash
export LIQUIDATION_DATA_ROOT=/absolute/path/to/liquidation_task/data
```

## Быстрый старт

```bash
# Обучить и сохранить модели (BTC, горизонты 30s и 300s — основные стратегии)
python scripts/train_models.py --symbol btcusdt --max-days 90

# Оценка на test (должна быть близка к reference metrics)
python scripts/evaluate_models.py --symbol btcusdt

# Сигнал на последней строке панели
python scripts/predict_signal.py --symbol btcusdt --horizon 30
```

ETH:

```bash
python scripts/train_models.py --symbol ethusdt --max-days 90
python scripts/evaluate_models.py --symbol ethusdt
```

Все горизонты (1, 3, 5, 30, 300):

```bash
python scripts/train_models.py --symbol btcusdt --all-horizons
```

## Структура

| Путь | Назначение |
|------|------------|
| `STRATEGIES.md` | Описание стратегий с WR > 0.53, оговорки |
| `FEATURES.md` | Словарь фичей и формул |
| `config/*.json` | Лучшие гиперпараметры Kalman + LGBM |
| `artifacts/metrics_reference.json` | Эталонные метрики с полного research |
| `lib/` | Панель, Kalman, нормализация, модели |
| `scripts/` | train / evaluate / predict |
| `artifacts/models/{symbol}/` | Сохранённые веса после `train_models.py` |

## Оценка PnL (markout)

См. **`../pnl_evaluation/`** — AS-котирование по сигналу, τ ∈ {30,120,300}s, Binance trades + Bybit liq (+200ms), отчёт `results/PNL_REPORT.md`.

## Стратегии (кратко)

См. **`STRATEGIES.md`**. Рекомендуемые для продакшена:

1. **LGBM без Kalman-фичей @ 30s** — WR ~0.54–0.56 (честный ML).
2. **Kalman-only** — WR ~0.53–0.54 на коротких горизонтах (без обучения).
3. **Ensemble** (40% Kalman + 60% LGBM no-K) — как в research pipeline.

Не использовать для оценки edge: **LGBM с `kalman_*` в фичах** (завышенный WR на 1–5s из-за корреляции с таргетом).
