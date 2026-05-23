# ТЗ: Cross-venue анализ (стр. 273+, Queue-reactive / Jane Street)

Источник: `docs/Queue-reactive model.pdf`, стр. 273–279.

**Контекст проекта:**
- **B (execution / maker):** Binance perp — `trades`, `bbo`, `liquidations`
- **A (ранний сигнал):** Bybit — только `liquidations`, `timestamp += 200ms` перед join
- Время: `int64` μs UTC; ресемпл 100ms / 500ms / 1s; asof-alignment

---

## 1. Синхронизация и качество времени

| Поле | Binance | Bybit |
|------|---------|-------|
| timestamp | μs UTC | μs UTC + 200_000 |
| mid, bid/ask, spread | BBO enriched | — (proxy: liq price) |
| depth L1 | bid/ask amount | — |
| trades / TFI | trades enriched | liq signed flow |

**Проверки:**
- clock skew / monotonicity timestamps
- sweep latency Bybit shift: {0, 100, 200, 500} ms
- asof: последнее состояние A на или до t (B)

**Deliverable:** `01_sync_quality.csv`, график latency sensitivity

---

## 2. Cross-venue lead-lag карта

Для сигналов A и target на B:

```
corr(signal_A[t], return_B[t+h]),  h ∈ {50ms, 100ms, 250ms, 500ms, 1s, 2s, 5s}
corr(signal_B[t], return_A_proxy[t+h])  # обратное направление
```

Сигналы A: `TFI_bybit`, `L_net_bybit`, `L_abs_bybit`  
Target B: `ret_micro_1s`, `ret_mid_1s` (Binance BBO)

**Критерий лидера:** A→B сильнее и стабильнее, чем B→A на test (60/40 split).

**Deliverable:** heatmap `02_lead_lag_heatmap.png`, `02_lead_lag.csv`

---

## 3. Главные фичи

| Фича | Формула / источник |
|------|-------------------|
| `div_adj` | proxy: `binance_mid / rolling_mean(binance_mid) - 1` (нет Bybit mid) |
| `div_liq` | `binance_liq_vwap / bybit_liq_vwap - 1` − rolling mean (1s bars) |
| `TFI_binance` | buy_usd − sell_usd (1s trades) |
| `TFI_bybit` | signed liq USD (1s) |
| `OFI_binance` | Δbid − Δask (BBO enriched) |
| `book_imbalance` | (Vb−Va)/(Vb+Va) |
| `microprice`, `microprice_g` | Stoikov |
| `spread_bps` | (ask−bid)/mid × 1e4 |
| `depth_cost_N` | L1 proxy: spread_bps × N/2 (N=10k USD) |

**Deliverable:** `data/enriched` + panel `03_features_1s.parquet` (sample period)

---

## 4. Price discovery (регрессия)

```
future_ret_B[t+h] = β0 + β1·ret_B + β2·OFI_B + β3·TFI_B + β4·div + β5·cross_A_lags + ε
```

Сравнить `local_only` vs `local_plus_cross` на train; метрики на test: R², rank IC.

**Deliverable:** `04_regression_coefs.csv`, bar chart

---

## 5. Edge после costs (упрощённо)

```
net_edge_bps ≈ gross_move_bps − spread_bps/2 − fee_proxy
```

fee_proxy: maker rebate 0.5 bps (из description.md).

**Deliverable:** `05_net_edge_by_signal.csv`

---

## 6. Maker vs taker framing

- **Taker:** Bybit pressure → ожидаемый move Binance → hedge
- **Maker:** conditional markout Binance trades при bullish/bearish cross signal

**Deliverable:** `06_maker_markout_by_signal.csv`

---

## 7. Сильные тесты

| Test | Описание |
|------|----------|
| 7.1 Decile | signal → future_ret, монотонность top vs bottom |
| 7.2 Conditional markout | bid vs ask markout при bullish signal |
| 7.3 Latency decay | edge vs assumed delay 0–500ms |
| 7.4 Regime split | tight/wide spread, high/low vol, liq burst |

**Deliverable:** figures `07_*.png`, tables

---

## 8. Пять типов alpha (чеклист)

1. Lead-lag  
2. Dislocation mean reversion  
3. Cross-venue liquidity pressure  
4. Cross-venue toxic flow  
5. Fee/liquidity asymmetry  

**Deliverable:** `08_alpha_checklist.md` с pass/fail по test

---

## 9. Минимальная модель

LGBM: `local_only` vs `local_plus_cross` → `future_ret_500ms`, `future_ret_1s`  
Target для MM: markout proxy на subsample trades

**Deliverable:** `09_model_compare.json`

---

## 10. Вывод

Сводка: есть ли cross-venue alpha, кто лидер, практичный горизонт, применимость к фильтру ликвидаций.

**Split:** train 60% / test 40% chronological.
