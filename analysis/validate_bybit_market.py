#!/usr/bin/env python3
"""Sanity checks for native Bybit parquet (does not touch Binance files)."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SYMBOLS, bybit_market_paths, paths_for_symbol  # noqa: E402


def _check_monotonic(path: Path) -> None:
    pf = pq.ParquetFile(path)
    prev = -1
    n = 0
    for rg in range(pf.metadata.num_row_groups):
        ts = pf.read_row_group(rg, columns=["timestamp"]).column("timestamp")
        arr = ts.to_numpy()
        if len(arr) and arr[0] < prev:
            raise AssertionError(f"{path}: non-monotonic at rg{rg}: {arr[0]} < {prev}")
        if len(arr) and arr.min() < prev:
            raise AssertionError(f"{path}: decrease inside rg{rg}")
        for v in arr:
            if v < prev:
                raise AssertionError(f"{path}: ts decrease {v} < {prev}")
            prev = int(v)
        n += len(arr)
    mn = pc.min(pf.read(columns=["timestamp"]).column("timestamp")).as_py()
    mx = pc.max(pf.read(columns=["timestamp"]).column("timestamp")).as_py()
    print(f"  {path.name}: rows={n:,} ts=[{mn},{mx}] monotonic=ok")


def main() -> None:
    for sym in SYMBOLS:
        bp = bybit_market_paths(sym)
        liq_p = paths_for_symbol(sym)["liq_bybit"]
        print(f"\n=== {sym} ===")
        for key in ("trades", "bbo"):
            p = bp[key]
            if not p.exists():
                print(f"  MISSING {key}: {p}")
                continue
            _check_monotonic(p)
        if liq_p.exists():
            liq = pq.read_table(liq_p, columns=["timestamp"])
            print(
                f"  liq ref: ts=[{pc.min(liq['timestamp']).as_py()},"
                f"{pc.max(liq['timestamp']).as_py()}] (unchanged)"
            )


if __name__ == "__main__":
    main()
