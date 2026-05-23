# Executive Summary — Liquidation EDA (TZ Assignment)

## Key findings

1. **Data coverage**: ~90 days (Dec 2025 – Feb 2026), BTC/ETH perps. Trades are dense (~4–7M rows/day); BBO ~1.1M/day; liquidations sparse (~1–2k/day/venue).
2. **Bybit vs Binance liquidations**: Bybit shows **higher total liquidation notional** over the window; short-horizon **Binance→Bybit** follow-through exceeds the reverse (consistent with +200ms Bybit latency convention).
3. **Microstructure**: Spreads are usually sub–few bps; distributions are heavy-tailed for trade/liq notionals.
4. **Lead-lag (sample day)**: Bybit 1s liq flow vs Binance 1s returns shows **weak but positive** correlation at **+2…+5s** lags (see `tier13_04_leadlag_heatmap_*.png`).
5. **Signal-to-noise**: Book imbalance / OFI vs 5s forward return — **low rank IC** on BBO sample (enriched); better suited to **filtering toxic flow** than price forecasting.

## Actionable insights

- Use **Binance** for execution markout and maker PnL; treat **Bybit liquidations (+200ms)** as **early warning**, not as fill prices.
- Focus strategies on **burst regimes** and **cross-venue pressure** rather than point forecasts of mid.
- Validate any filter on **second-half** chronological test with full trade stream (not liq-only proxy).

## Limitations & data caveats

- **No Bybit BBO/trades** in base TZ dataset (only liquidations); lead-lag uses liq flow proxies. Native Bybit trades/BBO path documented for optional extension.
- Trade/BBO EDA uses **random samples** / **sample days** — not full 1B-row scans.
- Timestamp duplicates at parquet row-group boundaries possible; monotonicity checked per row-group sample.
- **ADF/stationarity** on intraday returns: often rejects unit root on short windows — use returns, not prices, for ML.
- Enriched features path optional: `data/enriched/` (if missing, signal-to-noise table may be partial).

## Strategies & Bybit filter (bundled)

- **Binance trades PnL + Bybit liq filter:** `results/strategies/PNL_REPORT.md`, **`bybit_filter_summary.csv`** (τ=30s).
- Reproduce: `python run_all.py` with data in `data/` — see [SUBMISSION.md](../SUBMISSION.md).

## Beyond TZ (also in main repo)

- Further research: cross-lead, QuantFormer, native Bybit trades/BBO.


**BTC 30s alignment**: B→Y rate=0.03%, Y→B rate=0.01%.

### Signal-to-noise (sample)

```
 symbol        feature horizon  rank_ic     n
btcusdt book_imbalance      5s 0.331012 79995
btcusdt  ofi_increment      5s 0.054514 79995
```
