#!/usr/bin/env python3
"""Process one symbol/venue over all test days; save aggregation checkpoint (OOM isolation)."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.pnl_evaluation.aggregate import empty_state, flush_day, update_state  # noqa: E402
from research.pnl_evaluation.as_quoting import filter_adverse_trades  # noqa: E402
from research.pnl_evaluation.config import STRATEGIES, TAUS_SEC, TAB  # noqa: E402
from research.pnl_evaluation.data_loaders import (  # noqa: E402
    day_ranges_us,
    load_bbo_mid,
    load_binance_trades,
    load_bybit_liq_as_trades,
)
from research.pnl_evaluation.markout_metrics import maker_pnl_bps, mid_at_horizon  # noqa: E402
from research.pnl_evaluation.run_report import SIG_COLS, TRADE_BATCH, _signal_arrays  # noqa: E402
from research.pnl_evaluation.strategy_signals import asof_signal_to_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--venue", required=True, choices=["binance", "bybit"])
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--start-day", type=int, default=1)
    ap.add_argument("--end-day", type=int, default=0, help="0 = through last test day")
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--ckpt-out", type=Path, required=True)
    ap.add_argument("--ckpt-in", type=Path, default=None)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    t0, t1 = meta["test_t0_us"], meta["test_t1_us"]
    days = day_ranges_us(t0, t1)

    table = pq.read_table(args.signals, columns=SIG_COLS, memory_map=True)
    sig_df = table.to_pandas(self_destruct=True)
    del table
    sig_bts, sig_cols = _signal_arrays(sig_df)
    del sig_df
    gc.collect()

    if args.ckpt_in and args.ckpt_in.exists():
        states = pickle.loads(args.ckpt_in.read_bytes())
    else:
        states = {sid: {tau: empty_state() for tau in TAUS_SEC} for sid, _, _ in STRATEGIES}

    end_day = args.end_day if args.end_day > 0 else len(days)
    for di, (d0, d1) in enumerate(days):
        if di + 1 < args.start_day or di + 1 > end_day:
            continue
        print(f"    day {di+1}/{len(days)} {d0}..{d1}", flush=True)
        bbo = load_bbo_mid(args.symbol, d0, d1, tau_pad_sec=300)
        if bbo.empty:
            continue
        bbo_ts = bbo["timestamp"].values.astype(np.int64)
        bbo_mid = bbo["mid"].values.astype(np.float64)
        del bbo

        if args.venue == "binance":
            events = load_binance_trades(args.symbol, d0, d1, subsample=args.subsample)
        else:
            events = load_bybit_liq_as_trades(args.symbol, d0, d1, subsample=args.subsample)
        if events.empty:
            continue

        ts = events["timestamp"].values.astype(np.int64)
        price = events["price"].values.astype(np.float64)
        side = events["side_code"].values
        w = events["w"].values.astype(np.float64)
        del events
        n_ev = len(ts)

        for start in range(0, n_ev, TRADE_BATCH):
            sl = slice(start, min(start + TRADE_BATCH, n_ev))
            ts_b = ts[sl]
            price_b = price[sl]
            side_b = side[sl]
            w_b = w[sl]
            mid0_b = mid_at_horizon(ts_b, bbo_ts, bbo_mid, 0)
            pnl_by_tau = {
                tau: maker_pnl_bps(
                    price_b,
                    mid_at_horizon(ts_b, bbo_ts, bbo_mid, tau * 1_000_000),
                    side_b,
                ).astype(np.float32)
                for tau in TAUS_SEC
            }
            for sid, kind, _fh in STRATEGIES:
                signal = asof_signal_to_events(sig_bts, sig_cols[sid], ts_b)
                f = filter_adverse_trades(signal, side_b, price_b, mid0_b, use_as_price=False)
                if kind == "baseline":
                    f = np.zeros(len(ts_b), dtype=np.int8)
                for tau in TAUS_SEC:
                    states[sid][tau] = update_state(states[sid][tau], pnl_by_tau[tau], w_b, f, ts_b)

        del ts, price, side, w, bbo_ts, bbo_mid
        for sid, _, _ in STRATEGIES:
            for tau in TAUS_SEC:
                states[sid][tau] = flush_day(states[sid][tau])
        gc.collect()

    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    args.ckpt_out.write_bytes(pickle.dumps(states))
    print(f"  checkpoint -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
