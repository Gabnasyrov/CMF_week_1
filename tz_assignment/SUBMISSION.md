# Submission guide (no raw data in git)

This folder is self-contained for review. **Precomputed outputs are included** so you can inspect results **without** parquet files.

## What to open first (no data required)

1. **[notebooks/Liquidation_EDA_TZ.ipynb](notebooks/Liquidation_EDA_TZ.ipynb)** — Run All cells (displays CSV/PNG from `results/`).
2. **[results/EXECUTIVE_SUMMARY.md](results/EXECUTIVE_SUMMARY.md)** — EDA conclusions.
3. **[results/strategies/PNL_REPORT.md](results/strategies/PNL_REPORT.md)** — Maker PnL including **Bybit liquidation filter** (`venue=bybit`).
4. **[results/strategies/bybit_filter_summary.csv](results/strategies/bybit_filter_summary.csv)** — Bybit-only score table (τ=30s).
5. **[STRATEGIES.md](STRATEGIES.md)** — Strategy definitions and metric glossary (WR vs PnL vs Bybit).

## What is committed vs not

| Include in git | Do **not** commit |
|----------------|-------------------|
| `results/tables/*.csv` | `data/*.parquet` (multi‑GB) |
| `results/figures/*.png` | `data/enriched/` (~100GB) |
| `results/strategies/*` | `__pycache__/`, `.ipynb_checkpoints/` |
| `notebooks/*.ipynb` | |
| `*.md`, `*.py`, `requirements*.txt` | |
| `data/README.md`, `data/.gitkeep` only | |

## Bybit filter (important)

Per [docs/description.md](../docs/description.md), **execution** is on **Binance**. The **Bybit row** in PnL is a **stress test**:

- Events = **Bybit liquidations** (+200 ms latency shift).
- Signal = same **Binance** 1s model (LGBM / Kalman / ensemble).
- Filter = drop event when `signal × s_taker > 0` (adverse for maker).
- Markout = still **Binance BBO mid** at `t+τ`.

Use **`score_bps`** (kept vs all), not raw `pnl_kept_bps` on Bybit (proxy events are not real fills).

## Reproduce with your data

1. Copy parquet into `tz_assignment/data/` (layout in [data/README.md](data/README.md)) **or** symlink parent `../data`.
2. Install deps: `pip install -r requirements.txt -r requirements-strategies.txt`
3. Run full pipeline (auto-skips steps already done):

```bash
cd tz_assignment
python run_all.py
```

Pipeline order: **enriched features** → **train + PnL** → **EDA**. Runtime: hours on full 90 days / ~50M+ trade rows.

Smoke test: `python run_all.py --quick`

View only (no data): `python run_all.py --view-only`

## Optional figures

```bash
python plot_price_liq.py --symbol ethusdt --day 2025-12-15
```
