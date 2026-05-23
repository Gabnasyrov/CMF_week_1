# Удачные стратегии (WR / hit_rate > 0.53 на test)

**Задача:** предсказать `sign(μ_Kalman[t+h] − μ_Kalman[t])` на 1s панели Binance.  
**Оценка:** хронологический split **40% train / 60% test**, ~90 дней, WR = доля совпадения знака предсказания и таргета.

---

## Рекомендуемые (честные)

### S1 — LGBM без Kalman-фичей @ 30s (основная)

| | BTC | ETH |
|---|-----|-----|
| **WR test** | **0.560** | **0.544** |
| rank IC | ~0.33 | ~0.21 |
| n test | ~48k | ~48k |

- **Идея:** microstructure + flow + liq + Hawkes + trend **без** `kalman_*` предсказывают будущее смещение уровня Kalman.
- **Скрипт:** `scripts/train_models.py` → `artifacts/models/{sym}/lgbm_nk_30s.joblib`
- **Инференс:** `scripts/predict_signal.py --horizon 30`
- **Параметры LGBM:** `num_leaves=15`, `lr=0.02` (BTC), `lr=0.05` (ETH) — см. `config/`.

### S2 — Kalman-only (без ML)

| Symbol | Лучший h | WR test |
|--------|----------|---------|
| BTC | 1s | **0.537** |
| ETH | 5s | **0.535** |
| ETH | 30s | **0.537** |
| BTC/ETH | 300s | ~0.533–0.535 |

- **Идея:** знак `μ[t+h]−μ[t]` из уже отфильтрованного ряда (2D Kalman level+velocity, adaptive R).
- **Скрипт:** `lib/signals.py` → `kalman_direction_sign(panel, horizon_sec)`
- **Параметры:** `config/btcusdt.json`, `config/ethusdt.json` → блок `kalman`.

### S3 — Composite score `score_long_micro`

| Symbol | WR (avg @ 30s + 300s, LGBM no-K) |
|--------|----------------------------------|
| BTC | **0.536** |
| ETH | **0.530** |

Использовался как objective при hyperparameter search. Практически = среднее S1 @ 30s и @ 300s.

### S4 — Ensemble (0.4 Kalman + 0.6 LGBM no-K)

На 30s совпадает с S1 в отчёте (Kalman слабее, доминирует LGBM). Реализация: `lib/signals.py` → `ensemble_sign`.

---

## Дополнительные (WR > 0.53, осторожно с горизонтом)

### S5 — LGBM no-K @ коротких горизонтах

| h | BTC WR | ETH WR |
|---|--------|--------|
| 1s | 0.704 | 0.622 |
| 3s | 0.665 | 0.603 |
| 5s | 0.638 | 0.587 |

Часть edge может быть из **микроструктурного арбитража** и шума 1s; для maker/markout лучше **30–300s**.

### S6 — LGBM **с** Kalman-фичами (не для честной оценки)

| h | BTC WR |
|---|--------|
| 1s | 0.846 |
| 30s | 0.567 |

Таргет построен из `kalman_mu`; включение `kalman_*` в X **завышает** WR на коротких h. В пакете сохраняется опционально (`lgbm_full_*`) только для сравнения.

---

## Не прошли порог WR > 0.53

| Модель | WR @ 30s |
|--------|----------|
| HMM K=5 | ~0.50 |
| QuantFormer | ~0.50 |
| LGBM no-K @ 300s | ~0.51–0.52 |

---

## Файлы запуска

```bash
# Обучение + сохранение весов
python scripts/train_models.py --symbol btcusdt

# Проверка WR на test
python scripts/evaluate_models.py --symbol btcusdt

# Один сигнал (-1 / +1)
python scripts/predict_signal.py --symbol btcusdt --horizon 30 --strategy lgbm_nk
```

Эталонные метрики без переобучения: `artifacts/metrics_reference.json`.
