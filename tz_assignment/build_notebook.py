#!/usr/bin/env python3
"""Generate unified Jupyter notebook from template cells."""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "notebooks" / "Liquidation_EDA_TZ.ipynb"
PKG = Path(__file__).resolve().parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# Liquidation research — TZ package

**Reviewer:** run this notebook **without** raw data — it displays bundled `results/`.

**Reproduce:** add parquet per `data/README.md`, then `python run_all.py` (features → strategies incl. **Bybit filter** → EDA).

See **[SUBMISSION.md](SUBMISSION.md)** · Spec: [docs/description.md](../docs/description.md)

Symbols: `btcusdt`, `ethusdt` · Dec 2025 – Feb 2026"""
    ),
    md(
        """## Review checklist (no data)

1. [EXECUTIVE_SUMMARY.md](../results/EXECUTIVE_SUMMARY.md)
2. **Bybit liquidation filter** → `results/strategies/bybit_filter_summary.csv` (below)
3. Full PnL → `results/strategies/PNL_REPORT.md`
4. EDA tiers 1.1–1.3 (sections below)
5. Optional chart: `ethusdt_price_liq_2025-12-15.png`"""
    ),
    code(
        """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display, Markdown, HTML

ROOT = Path('.').resolve()
if not (ROOT / 'config.py').exists():
    ROOT = Path('..').resolve()
import sys
sys.path.insert(0, str(ROOT))
from config import FIGURES, TABLES, DATA, SYMBOLS

REPO = ROOT.parent if (ROOT.parent / 'research').is_dir() else ROOT
STRAT = ROOT / 'results' / 'strategies'
PNL_TZ = STRAT / 'pnl_report.csv'
PNL_REPO = REPO / 'research' / 'pnl_evaluation' / 'results' / 'tables' / 'pnl_report.csv'
BUNDLE = REPO / 'research' / 'strategies_bundle'

print('ROOT:', ROOT)
print('DATA:', DATA, 'exists:', DATA.is_dir())
print('EDA tables:', len(list(TABLES.glob('*.csv'))), 'figures:', len(list(FIGURES.glob('*.png'))))
print('Strategy dir:', STRAT, 'exists:', STRAT.is_dir())"""
    ),
    md("## Executive summary (EDA)"),
    code(
        """p = ROOT / 'results' / 'EXECUTIVE_SUMMARY.md'
if p.exists():
    display(Markdown(p.read_text()))
else:
    print('Run: python run_all.py --eda-only')"""
    ),
    md("## Tier summary"),
    code(
        """t = TABLES / 'TIER_SUMMARY.csv'
if t.exists():
    display(pd.read_csv(t))
else:
    print('Missing TIER_SUMMARY.csv')"""
    ),
    md("---\n# Feature documentation"),
    md(
        """Микроструктурные фичи для 1s панели стратегий. Источник: `FEATURES.md` (копия bundle) + enriched parquet.

Сборка: `python build_features.py` → `data/enriched/`."""
    ),
    code(
        """feat_paths = [
    ROOT / 'FEATURES.md',
    BUNDLE / 'FEATURES.md',
]
feat_doc = next((p for p in feat_paths if p.is_file()), None)
if feat_doc:
    display(Markdown(feat_doc.read_text()))
else:
    print('FEATURES.md not found — run from tz_assignment/')"""
    ),
    md("### Enriched parquet (проверка наличия)"),
    code(
        """from config import enriched_paths
rows = []
for sym in SYMBOLS:
    ep = enriched_paths(sym)
    for k, path in ep.items():
        rows.append({'symbol': sym, 'stream': k, 'path': str(path), 'exists': path.is_file()})
display(pd.DataFrame(rows))"""
    ),
    md("---\n# Task 1.1 — Foundational EDA"),
    md("### 1. Data quality & stream frequency"),
    code(
        """display(pd.read_csv(TABLES / 'tier11_01_data_quality.csv').head(20))
display(pd.read_csv(TABLES / 'tier11_01b_stream_frequency.csv'))
display(Image(filename=str(FIGURES / 'tier11_01b_stream_frequency.png')))"""
    ),
    md("### 2. Univariate & time-of-day"),
    code(
        """display(pd.read_csv(TABLES / 'tier11_02_univariate_summary.csv'))
display(Image(filename=str(FIGURES / 'tier11_02_hour_of_day_liq.png')))
display(Image(filename=str(FIGURES / 'tier11_02_liq_notional_hist.png')))"""
    ),
    md("### 3. Cross-exchange liquidations"),
    code(
        """display(pd.read_csv(TABLES / 'tier11_03_cross_liq_alignment.csv'))
display(Image(filename=str(FIGURES / 'tier11_03_cross_liq_alignment.png')))"""
    ),
    md("### 4. Price + liquidations (ETH sample day)"),
    code(
        """p = FIGURES / 'ethusdt_price_liq_2025-12-15.png'
if p.exists():
    display(Image(filename=str(p)))
else:
    print('Missing — run: python plot_price_liq.py --symbol ethusdt --day 2025-12-15')"""
    ),
    md("---\n# Task 1.2 — Intermediate Analysis"),
    md("### Broad market (spread, trades sample)"),
    code(
        """display(pd.read_csv(TABLES / 'tier12_01_spread_quantiles.csv'))
display(pd.read_csv(TABLES / 'tier12_01_trade_stats_sample.csv'))
display(Image(filename=str(FIGURES / 'tier12_01_spread_distribution.png')))"""
    ),
    md("### ML EDA: stationarity, regimes, outliers, ACF, signal-to-noise"),
    code(
        """display(pd.read_csv(TABLES / 'tier12_02_stationarity_adf.csv'))
display(pd.read_csv(TABLES / 'tier12_02_vol_regimes.csv'))
display(pd.read_csv(TABLES / 'tier12_02_outlier_structure.csv'))
display(Image(filename=str(FIGURES / 'tier12_02_vol_regimes.png')))
display(Image(filename=str(FIGURES / 'tier12_02_autocorrelation.png')))
sn = TABLES / 'tier12_02_signal_to_noise.csv'
if sn.exists():
    display(pd.read_csv(sn))"""
    ),
    md("---\n# Task 1.3 — Advanced Exploration"),
    md("### Orderbook at liquidation & side imbalance"),
    code(
        """display(pd.read_csv(TABLES / 'tier13_01_depth_at_liq_sample.csv').describe())
display(pd.read_csv(TABLES / 'tier13_02_side_imbalance_liq.csv'))
display(Image(filename=str(FIGURES / 'tier13_02_side_imbalance.png')))"""
    ),
    md("### Liquidation heatmaps (hour × day-of-week)"),
    code(
        """for sym in SYMBOLS:
    p = FIGURES / f'tier13_03_liq_heatmap_{sym}.png'
    if p.exists():
        print(sym)
        display(Image(filename=str(p)))"""
    ),
    md("### Cross-exchange lead-lag & div proxy"),
    code(
        """display(pd.read_csv(TABLES / 'tier13_04_leadlag_all.csv'))
display(pd.read_csv(TABLES / 'tier13_04_cross_venue_div_proxy.csv'))
for sym in SYMBOLS:
    p = FIGURES / f'tier13_04_leadlag_heatmap_{sym}.png'
    if p.exists():
        display(Image(filename=str(p)))"""
    ),
    md("---\n# Strategies & PnL"),
    md(
        """## Bybit liquidation filter (bundled)

**Not** the official submission metric (that is **Binance trades** in [description.md](../docs/description.md)).

Stress test: **Bybit liquidations** as events, **Binance** signal, filter adverse flow, markout on **Binance mid**.

Use **`score_bps`** = improvement vs keeping all liq events."""
    ),
    code(
        """bybit_csv = STRAT / 'bybit_filter_summary.csv'
if bybit_csv.is_file():
    display(pd.read_csv(bybit_csv))
else:
    print('Run: python run_strategies.py')"""
    ),
    md("### Strategy docs & Binance direction WR"),
    md("Краткое описание: `STRATEGIES.md`. Прогон: `python run_strategies.py` / `python run_all.py`."),
    code(
        """strat_doc = ROOT / 'STRATEGIES.md'
if strat_doc.is_file():
    display(Markdown(strat_doc.read_text()))
else:
    print('Missing STRATEGIES.md')"""
    ),
    md("### Run metadata (`results/strategies/run_meta.json`)"),
    code(
        """meta_p = STRAT / 'run_meta.json'
if meta_p.is_file():
    meta = json.loads(meta_p.read_text())
    display(pd.DataFrame([meta]).T.rename(columns={0: 'value'}))
    print('Source:', meta_p)
else:
    print('No TZ strategy run yet. Use: python run_strategies.py')"""
    ),
    md("### Direction accuracy (WR / hit_rate on test)"),
    code(
        """ref_p = BUNDLE / 'artifacts' / 'metrics_reference.json'
if ref_p.is_file():
    print('Reference WR (full research):')
    display(pd.DataFrame(json.loads(ref_p.read_text())).T.head(10))

dir_csvs = sorted(STRAT.glob('direction_metrics_*.csv')) if STRAT.is_dir() else []
if not dir_csvs:
    print('No direction_metrics_*.csv in', STRAT)
else:
    for p in dir_csvs:
        sym = p.stem.replace('direction_metrics_', '')
        df = pd.read_csv(p)
        mtime = pd.Timestamp(p.stat().st_mtime, unit='s')
        print(f'\\n=== {sym} (from {p.name}, mtime={mtime}) ===')
        pivot = df.pivot_table(index='horizon_sec', columns='model', values='hit_rate', aggfunc='first')
        display(pivot.style.format('{:.3f}').background_gradient(cmap='RdYlGn', vmin=0.48, vmax=0.58))
        display(df)"""
    ),
    md("### Full maker PnL (Binance trades + Bybit liq, τ = 30 / 120 / 300 s)"),
    code(
        """pnl_csv = PNL_TZ if PNL_TZ.is_file() else (PNL_REPO if PNL_REPO.is_file() else None)
pnl_md_paths = [STRAT / 'PNL_REPORT.md', REPO / 'research' / 'pnl_evaluation' / 'results' / 'PNL_REPORT.md']

if pnl_csv:
    src_label = 'TZ results/strategies' if pnl_csv == PNL_TZ else 'research/pnl_evaluation (latest full run)'
    print('PnL table:', pnl_csv, '|', src_label)
    pnl = pd.read_csv(pnl_csv)
    display(Markdown(f'**Rows:** {len(pnl)} · **Symbols:** ' + str(pnl['symbol'].unique().tolist())))

    # Top score_bps per symbol × venue (τ=30s primary)
    sub = pnl[pnl['tau_sec'] == 30].copy()
    sub = sub[sub['strategy'] != 'baseline'].sort_values('score_bps', ascending=False)
    print('\\nTop score_bps @ τ=30s (excl. baseline):')
    display(sub.groupby(['symbol', 'venue'], as_index=False).head(3))

    for sym in sorted(pnl['symbol'].unique()):
        for venue in sorted(pnl['venue'].unique()):
            s = pnl[(pnl.symbol == sym) & (pnl.venue == venue) & (pnl.tau_sec == 30)]
            if s.empty:
                continue
            print(f'\\n{sym} / {venue} @ 30s')
            cols = ['strategy', 'score_bps', 'pnl_kept_bps', 'winrate_kept', 'turnover_kept_usd_day', 'meets_turnover_constraint']
            display(s[cols].sort_values('score_bps', ascending=False))
else:
    print('No pnl_report.csv — run: python run_strategies.py (or full run_report.py in main repo)')"""
    ),
    code(
        """for p in pnl_md_paths:
    if p.is_file():
        print('Markdown report:', p)
        display(Markdown(p.read_text()))
        break
else:
    print('No PNL_REPORT.md found')"""
    ),
    md(
        """---
## Reproducibility

```bash
cd tz_assignment
pip install -r requirements.txt -r requirements-strategies.txt
# optional: export LIQUIDATION_DATA_ROOT=/path/to/data
python run_all.py --view-only   # no data: list bundled paths
python run_all.py               # full reproduce when data present
python build_notebook.py
jupyter notebook notebooks/Liquidation_EDA_TZ.ipynb
```"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", NB)
