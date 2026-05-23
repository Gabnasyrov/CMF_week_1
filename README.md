# CMF Week 1 — Liquidation research (Binance / Bybit)

**Кратко:** исследование ликвидаций BTCUSDT и ETHUSDT (Binance + Bybit), EDA по ТЗ (1.1–1.3), построение 1s-панели признаков, обучение направленческих моделей (LGBM, Kalman, ансамбль) и оценка **maker-фильтра** на сделках Binance; ликвидации Bybit — как прокси-события с тем же сигналом. Сырые parquet в git **не входят**; для проверки без данных — `tz_assignment/results/` и ноутбук.

Research repo: liquidation-driven **maker filter** on **Binance** trades (see [docs/description.md](docs/description.md)).

## Repository map

| Path | Purpose |
|------|---------|
| **[tz_assignment/](tz_assignment/)** | **Deliverable package** — EDA, features, strategies, notebook, precomputed `results/` |
| [docs/description.md](docs/description.md) | Official task: signal, markout, score, turnover |
| [analysis/](analysis/) | Feature build (`build_features.py`), cross-lead, Bybit market download |
| [research/strategies_bundle/](research/strategies_bundle/) | LGBM / Kalman training, model weights |
| [research/pnl_evaluation/](research/pnl_evaluation/) | Maker PnL on **Binance trades** + **Bybit liq proxy** |
| [data/](data/) | Parquet (not in git) — or reviewer copies into `tz_assignment/data/` |

## Quick paths

**Reviewer without data** — open bundled outputs:

```bash
cd tz_assignment
jupyter notebook notebooks/Liquidation_EDA_TZ.ipynb
# or read results/EXECUTIVE_SUMMARY.md, results/strategies/PNL_REPORT.md
```

**Reviewer with data** — full reproduce:

```bash
cd tz_assignment
pip install -r requirements.txt -r requirements-strategies.txt
# place parquet per tz_assignment/data/README.md
python run_all.py
```

Start with **[tz_assignment/README.md](tz_assignment/README.md)** and **[tz_assignment/SUBMISSION.md](tz_assignment/SUBMISSION.md)**.
