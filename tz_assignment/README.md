# TZ Assignment — Liquidation research package

Self-contained deliverable: **EDA** (1.1–1.3), **features**, **strategies + PnL (Binance + Bybit filter)**, notebook, precomputed `results/`.

**No raw data in git** — see [SUBMISSION.md](SUBMISSION.md) for reviewers.

## Two ways to use this folder

### A) Review only (no parquet)

```bash
cd tz_assignment
pip install -r requirements.txt   # notebook deps only
jupyter notebook notebooks/Liquidation_EDA_TZ.ipynb
```

Open also:

- [results/EXECUTIVE_SUMMARY.md](results/EXECUTIVE_SUMMARY.md)
- [results/strategies/PNL_REPORT.md](results/strategies/PNL_REPORT.md) — **includes `venue=bybit`**
- [results/strategies/bybit_filter_summary.csv](results/strategies/bybit_filter_summary.csv)

### B) Full reproduce (with data)

```bash
pip install -r requirements.txt -r requirements-strategies.txt
# 8 parquet files → data/  (see data/README.md)
python run_all.py
```

`run_all.py` automatically:

1. `build_features.py` → `data/enriched/` (skip if already built)
2. `run_strategies.py` → train LGBM, WR, **PnL on Binance trades + Bybit liq**
3. `run_analysis.py` → EDA CSV/PNG

## Official task spec

[../docs/description.md](../docs/description.md) — **filter Binance trades**, markout on **Binance BBO mid**, Bybit liq in features (+200 ms).

## Documentation index

| Doc | Content |
|-----|---------|
| [SUBMISSION.md](SUBMISSION.md) | What to commit, reviewer checklist, **Bybit filter** |
| [data/README.md](data/README.md) | Parquet layout, `LIQUIDATION_DATA_ROOT` |
| [STRATEGIES.md](STRATEGIES.md) | S1–S6, WR vs PnL, commands |
| [FEATURES.md](FEATURES.md) | 1s panel feature dictionary |
| [results/README.md](results/README.md) | Bundled artifacts |
| [results/strategies/README.md](results/strategies/README.md) | PnL + **Bybit filter** details |

## Commands

```bash
python run_all.py              # features → strategies → EDA
python run_all.py --view-only  # no data: print paths to bundled results
python run_all.py --eda-only   # only EDA (raw data required)
python run_all.py --quick      # 14-day strategy smoke

python build_features.py
python run_strategies.py       # train + WR + PnL (both venues)
python run_strategies.py --skip-pnl   # direction metrics only

python plot_price_liq.py --symbol ethusdt --day 2025-12-15
python build_notebook.py
```

## Metrics glossary (avoid confusion)

| Metric | Where | Meaning |
|--------|-------|---------|
| **hit_rate / WR** | `direction_metrics_*.csv` | Binance 1s panel: sign(μ[t+h]−μ[t]) — **not Bybit** |
| **score_bps** | `pnl_report.csv` | PnL_kept − PnL_all — **use this** for filter quality |
| **winrate_kept** | PnL | Share of kept events with positive markout |
| **venue=bybit** | PnL | Liquidations as proxy events, **same Binance signal** |

## Project layout (code)

```
tz_assignment/
  config.py           # DATA root resolution
  run_all.py          # main entry
  build_features.py   # → analysis/build_features.py
  run_strategies.py   # → strategies_bundle + pnl_evaluation
  run_analysis.py     # EDA tiers
  plot_price_liq.py   # optional price + liq chart
  results/            # SHIP THIS (tables, figures, strategies)
  notebooks/
```

Implementation lives in parent repo: `analysis/`, `research/strategies_bundle/`, `research/pnl_evaluation/`.

## Runtime & troubleshooting

| Issue | Action |
|-------|--------|
| Wrong DATA root | `export LIQUIDATION_DATA_ROOT=/path` or put files in `tz_assignment/data/` |
| `build_features` hours / OOM | Normal on 90d; needs ~100GB disk for enriched |
| Corrupt enriched parquet | Delete partial file, re-run `build_features.py` |
| EDA fails on partial enriched | Re-run after build completes; tier 1.2 skips bad files |
| PnL very slow | Use `--quick` or `--skip-pnl` first |

## Strategies (summary)

| ID | Role |
|----|------|
| **lgbm_nk_30s** | Main ML filter (recommended) |
| **kalman_30s** | No-training baseline |
| **ensemble_30s** | 0.4 Kalman + 0.6 LGBM no-K |
| **baseline** | No filter (score=0) |

Weights: `../research/strategies_bundle/artifacts/models/{sym}/`
