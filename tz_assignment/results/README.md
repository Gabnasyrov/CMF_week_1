# Bundled results (ship with submission)

Precomputed outputs for reviewers **without** raw parquet.

## EDA

| Path | Description |
|------|-------------|
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | Key findings |
| [tables/TIER_SUMMARY.csv](tables/TIER_SUMMARY.csv) | Task checklist |
| [tables/tier11_*.csv](tables/) | Data quality, frequency, liq stats |
| [tables/tier12_*.csv](tables/) | Spread, regimes, ACF, signal-to-noise |
| [tables/tier13_*.csv](tables/) | Heatmaps, lead-lag, depth@liq |
| [figures/*.png](figures/) | All tier plots + `ethusdt_price_liq_2025-12-15.png` |

## Strategies & PnL

| Path | Description |
|------|-------------|
| [strategies/PNL_REPORT.md](strategies/PNL_REPORT.md) | Full markdown table |
| [strategies/pnl_report.csv](strategies/pnl_report.csv) | All symbol × **venue** × strategy × τ |
| [strategies/bybit_filter_summary.csv](strategies/bybit_filter_summary.csv) | **Bybit liq filter** @ τ=30s |
| [strategies/direction_metrics_*.csv](strategies/) | Binance direction WR |
| [strategies/run_meta.json](strategies/run_meta.json) | Last run parameters |

Regenerate: `python run_strategies.py` from `tz_assignment/` with data present.
