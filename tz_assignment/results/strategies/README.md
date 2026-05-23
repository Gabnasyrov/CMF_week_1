# Strategy & PnL results

## Primary task (docs/description.md)

**Filter Binance trades** using a signal built on Binance microstructure + **both venues' liquidations** (Bybit liq shifted **+200 ms** in features).

Markout and score use **Binance BBO mid** at τ ∈ {30, 120, 300} seconds.

## Two evaluation modes in this folder

### 1) Direction WR (`direction_metrics_{sym}.csv`)

- **Venue:** Binance only (1s panel).
- **Metric:** `hit_rate` = agreement with `sign(μ_Kalman[t+h] − μ[t])`.
- **Not** the same as PnL winrate on Bybit.

### 2) Maker PnL (`pnl_report.csv`, `PNL_REPORT.md`)

| `venue` | Events | Signal | Markout mid |
|---------|--------|--------|-------------|
| **binance** | Real trades | Binance 1s | Binance BBO |
| **bybit** | **Liquidations as proxy** (+200 ms) | Same Binance signal | Binance BBO |

## Bybit liquidation filter (stress test)

For each Bybit liq event treated as a pseudo-trade:

1. **Signal** at event time: latest Binance model output (asof backward).
2. **Filter** `f=1`: remove if `signal × s_taker > 0` (toxic for passive maker).
3. **score_bps** = weighted PnL on **kept** events minus PnL on **all** events.

**Do not** compare absolute `pnl_kept_bps` on Bybit to Binance — liq notionals and economics differ. Compare **score_bps** and **kept fraction**.

See [bybit_filter_summary.csv](bybit_filter_summary.csv) for τ=30s highlights.

## Files

| File | Notes |
|------|-------|
| `pnl_report.csv` | Full grid; filter `venue == 'bybit'` |
| `PNL_REPORT.md` | Human-readable table |
| `direction_eval_*.json` | Raw evaluate output |
| `run_meta.json` | `data_root`, `pnl_ran`, symbols |

## Re-run

```bash
cd tz_assignment
python run_strategies.py          # full train + PnL both venues
python run_strategies.py --quick  # smoke
```

Requires `data/enriched/` (from `python build_features.py`).
