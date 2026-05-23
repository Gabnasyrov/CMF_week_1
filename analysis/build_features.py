#!/usr/bin/env python3
"""
Build microstructure features and write enriched parquet (strict timestamp join).

Binance (perp:btcusdt, perp:ethusdt):
  - BBO: book_imbalance, microprice, mid, spread, g, ofi_increment, ofi_cumulative
  - Trades: asof BBO state + trade_flow_imbalance (1s / 5s rolling USD)

Bybit (btcusdt, ethusdt) — no BBO in dataset:
  - Liquidations: order_flow_imbalance (event + 1s bars), book fields NaN

Output:
  data/enriched/binance_bbo/perp_<sym>.parquet
  data/enriched/binance_trades/perp_<sym>.parquet
  data/enriched/bybit_liquidations/<sym>.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BYBIT_LATENCY_US, DATA, SYMBOLS, paths_for_symbol
from microstructure import (
    asof_lookup,
    book_imbalance,
    microprice_stoikov,
    ofi_increment_cont,
)

ENRICHED = DATA / "enriched"
BBO_OUT = ENRICHED / "binance_bbo"
TRADES_OUT = ENRICHED / "binance_trades"
BYBIT_OUT = ENRICHED / "bybit_liquidations"

BBO_SCHEMA_EXTRA = [
    ("mid", pa.float64()),
    ("spread", pa.float64()),
    ("spread_bps", pa.float64()),
    ("book_imbalance", pa.float64()),
    ("microprice", pa.float64()),
    ("microprice_g", pa.float64()),
    ("ofi_increment", pa.float64()),
    ("ofi_cumulative", pa.float64()),
]

TRADES_SCHEMA_EXTRA = [
    ("mid", pa.float64()),
    ("book_imbalance", pa.float64()),
    ("microprice", pa.float64()),
    ("microprice_g", pa.float64()),
    ("ofi_cumulative", pa.float64()),
    ("trade_flow_1s", pa.float64()),
    ("trade_flow_5s", pa.float64()),
    ("order_flow_imbalance_1s", pa.float64()),
    ("order_flow_imbalance_5s", pa.float64()),
]

BYBIT_SCHEMA_EXTRA = [
    ("order_flow_imbalance", pa.float64()),
    ("order_flow_imbalance_1s", pa.float64()),
    ("order_flow_cumulative", pa.float64()),
]


def _day_us_bounds(us: int) -> tuple[int, int]:
    day = us // 86_400_000_000
    t0 = day * 86_400_000_000
    return t0, t0 + 86_400_000_000 - 1


def enrich_bbo_symbol(sym: str, row_group_batch: int = 4) -> Path:
    src = paths_for_symbol(sym)["bbo"]
    out_dir = BBO_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perp_{sym}.parquet"

    pf = pq.ParquetFile(src)
    writer = None
    ofi_cum = 0.0
    bp_prev = ap_prev = bv_prev = av_prev = np.nan

    n_groups = pf.metadata.num_row_groups
    for start in range(0, n_groups, row_group_batch):
        tables = [pf.read_row_group(i) for i in range(start, min(start + row_group_batch, n_groups))]
        table = pa.concat_tables(tables)
        df = table.to_pandas()

        bp = df["bid_price"].to_numpy(np.float64)
        ap = df["ask_price"].to_numpy(np.float64)
        bv = df["bid_amount"].to_numpy(np.float64)
        av = df["ask_amount"].to_numpy(np.float64)

        imb = book_imbalance(bv, av)
        psi, mid, g = microprice_stoikov(bp, ap, bv, av)
        spread = ap - bp
        spread_bps = np.where(mid > 0, spread / mid * 10_000, 0.0)

        if np.isnan(bp_prev):
            bp_prev, ap_prev, bv_prev, av_prev = bp[0], ap[0], bv[0], av[0]
            e = np.zeros(len(df), dtype=np.float64)
        else:
            e, bp_prev, ap_prev, bv_prev, av_prev = ofi_increment_cont(
                bp, ap, bv, av, bp_prev, ap_prev, bv_prev, av_prev
            )

        ofi_cum_arr = np.empty(len(df), dtype=np.float64)
        for i in range(len(df)):
            ofi_cum += e[i]
            ofi_cum_arr[i] = ofi_cum

        df["mid"] = mid
        df["spread"] = spread
        df["spread_bps"] = spread_bps
        df["book_imbalance"] = imb
        df["microprice"] = psi
        df["microprice_g"] = g
        df["ofi_increment"] = e
        df["ofi_cumulative"] = ofi_cum_arr
        out_table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path, out_table.schema, compression="zstd")
        writer.write_table(out_table)
        print(f"  bbo {sym}: row groups {start}-{min(start + row_group_batch, n_groups)}/{n_groups}")

    if writer:
        writer.close()
    print(f"Wrote {out_path}")
    return out_path


def _rolling_flow_1s(ts: np.ndarray, signed: np.ndarray, window_us: int) -> np.ndarray:
    """Rolling sum of signed notional over trailing window_us (sorted ts)."""
    n = len(ts)
    out = np.zeros(n, dtype=np.float64)
    j = 0
    acc = 0.0
    for i in range(n):
        acc += signed[i]
        while ts[i] - ts[j] > window_us:
            acc -= signed[j]
            j += 1
        out[i] = acc
    return out


def _load_bbo_window(bbo_path: Path, t0: int, t1: int) -> tuple[np.ndarray, ...]:
    from utils import load_parquet_window

    cols = ["timestamp", "mid", "book_imbalance", "microprice", "microprice_g", "ofi_cumulative"]
    bbo = load_parquet_window(bbo_path, t0, t1, columns=cols)
    if bbo.empty:
        n = 0
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty
    return (
        bbo["timestamp"].values.astype(np.int64),
        bbo["mid"].values.astype(np.float64),
        bbo["book_imbalance"].values.astype(np.float64),
        bbo["microprice"].values.astype(np.float64),
        bbo["microprice_g"].values.astype(np.float64),
        bbo["ofi_cumulative"].values.astype(np.float64),
    )


def enrich_trades_symbol(sym: str, bbo_enriched: Path, row_group_batch: int = 2) -> Path:
    src = paths_for_symbol(sym)["trades"]
    out_dir = TRADES_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perp_{sym}.parquet"

    pf = pq.ParquetFile(src)
    writer = None
    n_groups = pf.metadata.num_row_groups
    pad_us = 5_000_000  # 5s lookback for asof + rolling

    for start in range(0, n_groups, row_group_batch):
        tables = [pf.read_row_group(i) for i in range(start, min(start + row_group_batch, n_groups))]
        df = pa.concat_tables(tables).to_pandas()
        ts = df["timestamp"].values.astype(np.int64)
        t0, t1 = int(ts.min()) - pad_us, int(ts.max())
        bbo_ts, bbo_mid, bbo_imb, bbo_psi, bbo_g, bbo_ofi = _load_bbo_window(bbo_enriched, t0, t1)

        notional = (df["price"] * df["amount"]).values.astype(np.float64)
        signed = np.where(df["side"] == "buy", notional, -notional)
        flow_1s = _rolling_flow_1s(ts, signed, 1_000_000)
        flow_5s = _rolling_flow_1s(ts, signed, 5_000_000)

        df["mid"] = asof_lookup(ts, bbo_ts, bbo_mid)
        df["book_imbalance"] = asof_lookup(ts, bbo_ts, bbo_imb)
        df["microprice"] = asof_lookup(ts, bbo_ts, bbo_psi)
        df["microprice_g"] = asof_lookup(ts, bbo_ts, bbo_g)
        df["ofi_cumulative"] = asof_lookup(ts, bbo_ts, bbo_ofi)
        df["trade_flow_1s"] = flow_1s
        df["trade_flow_5s"] = flow_5s
        df["order_flow_imbalance_1s"] = flow_1s
        df["order_flow_imbalance_5s"] = flow_5s

        out_table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, out_table.schema, compression="zstd")
        writer.write_table(out_table)
        print(f"  trades {sym}: row groups {start}-{min(start + row_group_batch, n_groups)}/{n_groups}")

    if writer:
        writer.close()
    print(f"Wrote {out_path}")
    return out_path


def enrich_bybit_liq(sym: str) -> Path:
    src = paths_for_symbol(sym)["liq_bybit"]
    out_dir = BYBIT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sym}.parquet"

    df = pq.read_table(src).to_pandas()
    df = df.sort_values("timestamp").reset_index(drop=True)
    # availability delay per task spec
    df["timestamp"] = df["timestamp"] + BYBIT_LATENCY_US

    notional = df["price"].values * df["amount"].values
    signed = np.where(df["side"] == "buy", notional, -notional)
    ts = df["timestamp"].values.astype(np.int64)

    flow_cum = np.cumsum(signed)
    flow_1s = _rolling_flow_1s(ts, signed, 1_000_000)

    n = len(df)
    nan = np.full(n, np.nan)

    out = df.copy()
    out["order_flow_imbalance"] = signed
    out["order_flow_imbalance_1s"] = flow_1s
    out["order_flow_cumulative"] = flow_cum
    out["book_imbalance"] = nan
    out["microprice"] = nan
    out["microprice_g"] = nan

    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")
    print(f"Wrote {out_path} ({len(out):,} rows)")
    return out_path


def enrich_binance_liq(sym: str, bbo_enriched: Path) -> Path:
    """Binance liquidations + asof book state + event order flow."""
    src = paths_for_symbol(sym)["liq_binance"]
    out_dir = ENRICHED / "binance_liquidations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"perp_{sym}.parquet"

    df = pq.read_table(src).to_pandas().sort_values("timestamp").reset_index(drop=True)
    t0 = int(df["timestamp"].min()) - 5_000_000
    t1 = int(df["timestamp"].max())
    bbo_ts, bbo_mid, bbo_imb, bbo_psi, bbo_g, bbo_ofi = _load_bbo_window(bbo_enriched, t0, t1)
    ts = df["timestamp"].values.astype(np.int64)
    notional = df["price"].values * df["amount"].values
    signed = np.where(df["side"] == "buy", notional, -notional)

    out = df.copy()
    out["order_flow_imbalance"] = signed
    out["order_flow_imbalance_1s"] = _rolling_flow_1s(ts, signed, 1_000_000)
    out["order_flow_cumulative"] = np.cumsum(signed)
    out["mid"] = asof_lookup(ts, bbo_ts, bbo_mid)
    out["book_imbalance"] = asof_lookup(ts, bbo_ts, bbo_imb)
    out["microprice"] = asof_lookup(ts, bbo_ts, bbo_psi)
    out["microprice_g"] = asof_lookup(ts, bbo_ts, bbo_g)
    out["ofi_cumulative"] = asof_lookup(ts, bbo_ts, bbo_ofi)

    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd")
    print(f"Wrote {out_path} ({len(out):,} rows)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    parser.add_argument("--skip-bbo", action="store_true")
    parser.add_argument("--skip-trades", action="store_true")
    parser.add_argument("--bbo-batch", type=int, default=4)
    parser.add_argument("--trades-batch", type=int, default=2)
    args = parser.parse_args()

    bbo_paths = {}
    for sym in args.symbols:
        if not args.skip_bbo:
            print(f"=== BBO {sym} ===")
            bbo_paths[sym] = enrich_bbo_symbol(sym, row_group_batch=args.bbo_batch)
        else:
            bbo_paths[sym] = BBO_OUT / f"perp_{sym}.parquet"

    for sym in args.symbols:
        if not args.skip_trades:
            print(f"=== Trades {sym} ===")
            enrich_trades_symbol(sym, bbo_paths[sym], row_group_batch=args.trades_batch)
        print(f"=== Binance liq {sym} ===")
        enrich_binance_liq(sym, bbo_paths[sym])
        print(f"=== Bybit liq {sym} ===")
        enrich_bybit_liq(sym)

    print("Done.")


if __name__ == "__main__":
    main()
