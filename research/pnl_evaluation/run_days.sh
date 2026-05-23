#!/bin/bash
# Run one test day per subprocess (OOM-safe). Usage:
#   ./run_days.sh btcusdt binance 35 45
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SYM="${1:?symbol}"
VENUE="${2:?venue}"
D0="${3:?start_day}"
D1="${4:?end_day}"
FRAC="${5:-50}"
SIG="research/pnl_evaluation/results/tables/signals_${SYM}_train${FRAC}_causal.parquet"
META="research/pnl_evaluation/results/tables/split_meta_${SYM}.json"
TAB="research/pnl_evaluation/results/tables"
CKPT_IN=""
for ((d=D0; d<=D1; d++)); do
  CKPT_OUT="${TAB}/ckpt_${SYM}_${VENUE}_train${FRAC}_d${d}.pkl"
  CMD=(python3 research/pnl_evaluation/run_venue_pnl.py
    --symbol "$SYM" --venue "$VENUE" --subsample 1
    --start-day "$d" --end-day "$d"
    --signals "$SIG" --meta "$META" --ckpt-out "$CKPT_OUT")
  if [[ -n "$CKPT_IN" ]]; then CMD+=(--ckpt-in "$CKPT_IN"); fi
  echo "=== ${SYM} ${VENUE} day ${d} ==="
  PYTHONUNBUFFERED=1 "${CMD[@]}"
  CKPT_IN="$CKPT_OUT"
done
