# Описание репозитория (для GitHub About)

**Короткое описание (скопировать в Settings → About):**

> EDA ликвидаций BTC/ETH (Binance+Bybit), 1s-панель признаков, LGBM/Kalman, maker-фильтр на Binance; Bybit liq как прокси. Без сырых данных — готовые `tz_assignment/results/`.

**Topics (теги):** `python` `jupyter` `cryptocurrency` `market-microstructure` `eda` `machine-learning`

---

## Что внутри

| Раздел | Содержание |
|--------|------------|
| **tz_assignment/** | Основной deliverable: ТЗ 1.1–1.3, пайплайн, ноутбук, CSV/PNG |
| **docs/description.md** | Формальное ТЗ: сигнал, markout, score, turnover |
| **analysis/** | Построение enriched parquet (`build_features.py`) |
| **research/** | Обучение моделей и PnL (Binance trades + Bybit liq filter) |

## Проверка без данных

```bash
cd tz_assignment
jupyter notebook notebooks/Liquidation_EDA_TZ.ipynb
```

Или: [tz_assignment/SUBMISSION.md](tz_assignment/SUBMISSION.md), [results/EXECUTIVE_SUMMARY.md](tz_assignment/results/EXECUTIVE_SUMMARY.md).

## Полное воспроизведение

1. Положить 8 parquet в `tz_assignment/data/` (см. `tz_assignment/data/README.md`).
2. `pip install -r tz_assignment/requirements.txt -r tz_assignment/requirements-strategies.txt`
3. `cd tz_assignment && python run_all.py`
