#!/usr/bin/env python3
"""
Download Bybit USDT linear perp trades + order book (ob200) and write parquet
matching Binance raw schema (timestamp in microseconds UTC).

Writes ONLY to:
  data/bybit_trades/<sym>.parquet
  data/bybit_booktickers/<sym>.parquet

Staging (safe resume): data/staging/bybit_{trades,booktickers}/<sym>/YYYY-MM-DD.parquet

Does not read or modify binance_* / bybit_liquidations source trees.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA, SYMBOLS, TRAIN_START, VAL_END, bybit_market_paths  # noqa: E402

US_PER_DAY = 86_400_000_000

TRADE_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("ticker", pa.string()),
        ("side", pa.string()),
        ("price", pa.float64()),
        ("amount", pa.float64()),
    ]
)

BBO_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("ticker", pa.string()),
        ("bid_price", pa.float64()),
        ("bid_amount", pa.float64()),
        ("ask_price", pa.float64()),
        ("ask_amount", pa.float64()),
    ]
)

BYBIT_SYMBOL = {"btcusdt": "BTCUSDT", "ethusdt": "ETHUSDT"}

TRADE_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz"  # day = YYYY-MM-DD
OB_URL = "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{day}_{symbol}_ob200.data.zip"


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "liquidation-task/1.0"})
    with urllib.request.urlopen(req, context=_ssl_ctx(), timeout=timeout) as resp:
        return resp.read()


def sym_to_exchange(sym: str) -> str:
    if sym not in BYBIT_SYMBOL:
        raise ValueError(sym)
    return BYBIT_SYMBOL[sym]


def daterange(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def day_us_bounds(d: date) -> tuple[int, int]:
    dt0 = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    t0 = int(dt0.timestamp() * 1_000_000)
    return t0, t0 + US_PER_DAY - 1


def sec_to_us(ts_sec: float) -> int:
    return int(round(ts_sec * 1_000_000))


def ms_to_us(ts_ms: int) -> int:
    return int(ts_ms) * 1_000


class _L2Book:
    """Top-of-book from Bybit ob200 snapshot/delta lines."""

    __slots__ = ("bids", "asks")

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    @staticmethod
    def _apply(side: dict[float, float], levels) -> None:
        for px, sz in levels:
            p = float(px)
            q = float(sz)
            if q <= 0:
                side.pop(p, None)
            else:
                side[p] = q

    def load_snapshot(self, bids, asks) -> None:
        self.clear()
        self._apply(self.bids, bids)
        self._apply(self.asks, asks)

    def apply_delta(self, bids, asks) -> None:
        if bids:
            self._apply(self.bids, bids)
        if asks:
            self._apply(self.asks, asks)

    def top(self) -> tuple[float, float, float, float]:
        if not self.bids or not self.asks:
            return (np.nan, np.nan, np.nan, np.nan)
        bp = max(self.bids)
        ap = min(self.asks)
        return bp, self.bids[bp], ap, self.asks[ap]


def _write_day_table(path: Path, table: pa.Table, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.cast(schema), path, compression="zstd", compression_level=3)


def _manifest_path(staging_dir: Path) -> Path:
    return staging_dir.parent / f".{staging_dir.name}_manifest.txt"


def _load_manifest(staging_dir: Path) -> set[str]:
    mp = _manifest_path(staging_dir)
    if not mp.exists():
        return set()
    return {ln.strip() for ln in mp.read_text().splitlines() if ln.strip()}


def _mark_day(staging_dir: Path, day_str: str) -> None:
    mp = _manifest_path(staging_dir)
    done = _load_manifest(staging_dir)
    done.add(day_str)
    mp.write_text("\n".join(sorted(done)) + "\n")


def build_trades_day(sym: str, d: date, out_path: Path) -> tuple[int, int, int]:
    """Returns (n_rows, ts_min, ts_max)."""
    exch = sym_to_exchange(sym)
    day_str = d.isoformat()
    url = TRADE_URL.format(symbol=exch, day=day_str)

    raw = _http_get(url)
    ticker = sym
    t0_us, t1_us = day_us_bounds(d)

    ts_list: list[int] = []
    side_list: list[str] = []
    price_list: list[float] = []
    amt_list: list[float] = []

    with gzip.open(BytesIO(raw), mode="rt", newline="") as f:
        reader = csv.DictReader(f)
        prev_ts = -1
        for row in reader:
            ts_us = sec_to_us(float(row["timestamp"]))
            if ts_us < t0_us or ts_us > t1_us:
                continue
            if ts_us < prev_ts:
                raise ValueError(f"{sym} {day_str}: non-monotonic trade ts {ts_us} < {prev_ts}")
            prev_ts = ts_us
            side = row["side"].strip().lower()
            price = float(row["price"])
            amount = float(row["size"])
            ts_list.append(ts_us)
            side_list.append(side)
            price_list.append(price)
            amt_list.append(amount)

    if not ts_list:
        raise ValueError(f"{sym} {day_str}: no trades in UTC day window")

    table = pa.table(
        {
            "timestamp": pa.array(ts_list, type=pa.int64()),
            "ticker": pa.array([ticker] * len(ts_list), type=pa.string()),
            "side": pa.array(side_list, type=pa.string()),
            "price": pa.array(price_list, type=pa.float64()),
            "amount": pa.array(amt_list, type=pa.float64()),
        }
    )
    _write_day_table(out_path, table, TRADE_SCHEMA)
    return len(ts_list), ts_list[0], ts_list[-1]


def build_bbo_day(sym: str, d: date, out_path: Path) -> tuple[int, int, int]:
    exch = sym_to_exchange(sym)
    day_str = d.isoformat()
    url = OB_URL.format(symbol=exch, day=day_str)
    raw = _http_get(url, timeout=600)
    ticker = sym
    t0_us, t1_us = day_us_bounds(d)

    book = _L2Book()
    ts_list: list[int] = []
    bp_list: list[float] = []
    bv_list: list[float] = []
    ap_list: list[float] = []
    av_list: list[float] = []

    prev_top: tuple[float, float, float, float] | None = None
    prev_ts = -1

    with zipfile.ZipFile(BytesIO(raw)) as zf:
        data_name = zf.namelist()[0]
        with zf.open(data_name) as f:
            for line in f:
                obj = json.loads(line)
                ts_us = ms_to_us(int(obj["ts"]))
                if ts_us < t0_us or ts_us > t1_us:
                    continue
                typ = obj.get("type")
                data = obj.get("data") or {}
                bids = data.get("b") or []
                asks = data.get("a") or []
                if typ == "snapshot":
                    book.load_snapshot(bids, asks)
                elif typ == "delta":
                    book.apply_delta(bids, asks)
                else:
                    continue

                top = book.top()
                if any(np.isnan(x) for x in top):
                    continue
                if top == prev_top:
                    continue
                if ts_us < prev_ts:
                    raise ValueError(f"{sym} {day_str}: non-monotonic bbo ts {ts_us} < {prev_ts}")
                prev_ts = ts_us
                prev_top = top
                ts_list.append(ts_us)
                bp, bv, ap, av = top
                bp_list.append(bp)
                bv_list.append(bv)
                ap_list.append(ap)
                av_list.append(av)

    if not ts_list:
        raise ValueError(f"{sym} {day_str}: no BBO rows in UTC day window")

    table = pa.table(
        {
            "timestamp": pa.array(ts_list, type=pa.int64()),
            "ticker": pa.array([ticker] * len(ts_list), type=pa.string()),
            "bid_price": pa.array(bp_list, type=pa.float64()),
            "bid_amount": pa.array(bv_list, type=pa.float64()),
            "ask_price": pa.array(ap_list, type=pa.float64()),
            "ask_amount": pa.array(av_list, type=pa.float64()),
        }
    )
    _write_day_table(out_path, table, BBO_SCHEMA)
    return len(ts_list), ts_list[0], ts_list[-1]


def merge_staging(staging_dir: Path, out_path: Path, schema: pa.Schema) -> int:
    day_files = sorted(staging_dir.glob("*.parquet"))
    if not day_files:
        raise FileNotFoundError(f"No staging parquet in {staging_dir}")
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}; remove manually if intended")

    writer = None
    total = 0
    prev_last_ts = -1
    for fp in day_files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            ts = table.column("timestamp").to_numpy()
            if len(ts) and ts[0] < prev_last_ts:
                raise ValueError(f"merge order break at {fp}: {ts[0]} < {prev_last_ts}")
            if len(ts):
                prev_last_ts = int(ts[-1])
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(out_path, schema, compression="zstd", compression_level=3)
            writer.write_table(table.cast(schema))
            total += len(ts)
    if writer is not None:
        writer.close()
    return total


def run_build(
    sym: str,
    d0: date,
    d1: date,
    *,
    trades: bool,
    bbo: bool,
    merge: bool,
    clean_staging: bool,
) -> None:
    paths = bybit_market_paths(sym)
    trade_stage = DATA / "staging" / "bybit_trades" / sym
    bbo_stage = DATA / "staging" / "bybit_booktickers" / sym

    for d in daterange(d0, d1):
        day_str = d.isoformat()
        if trades:
            done = _load_manifest(trade_stage)
            out_day = trade_stage / f"{day_str}.parquet"
            if day_str in done and out_day.exists():
                print(f"[trades] {sym} {day_str} skip (manifest)", flush=True)
            else:
                print(f"[trades] {sym} {day_str} download...", flush=True)
                n, t0, t1 = build_trades_day(sym, d, out_day)
                _mark_day(trade_stage, day_str)
                print(f"[trades] {sym} {day_str} rows={n:,} ts=[{t0},{t1}]", flush=True)
        if bbo:
            done = _load_manifest(bbo_stage)
            out_day = bbo_stage / f"{day_str}.parquet"
            if day_str in done and out_day.exists():
                print(f"[bbo] {sym} {day_str} skip (manifest)", flush=True)
            else:
                print(f"[bbo] {sym} {day_str} download...", flush=True)
                n, t0, t1 = build_bbo_day(sym, d, out_day)
                _mark_day(bbo_stage, day_str)
                print(f"[bbo] {sym} {day_str} rows={n:,} ts=[{t0},{t1}]", flush=True)

    if merge:
        if trades:
            out = paths["trades"]
            if not out.exists():
                n = merge_staging(trade_stage, out, TRADE_SCHEMA)
                print(f"[merge trades] {out} rows={n:,}")
            else:
                print(f"[merge trades] exists {out}")
        if bbo:
            out = paths["bbo"]
            if not out.exists():
                n = merge_staging(bbo_stage, out, BBO_SCHEMA)
                print(f"[merge bbo] {out} rows={n:,}")
            else:
                print(f"[merge bbo] exists {out}")

    if clean_staging:
        import shutil

        for sd in (trade_stage, bbo_stage):
            if sd.exists():
                shutil.rmtree(sd)
                mp = _manifest_path(sd)
                if mp.exists():
                    mp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Bybit trades + bookticker parquet")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--start-date", default=TRAIN_START.date().isoformat())
    ap.add_argument("--end-date", default=VAL_END.date().isoformat())
    ap.add_argument("--only", choices=("trades", "bbo", "both"), default="both")
    ap.add_argument("--no-merge", action="store_true", help="Keep daily staging only")
    ap.add_argument("--clean-staging", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    d0 = date.fromisoformat(args.start_date)
    d1 = date.fromisoformat(args.end_date)
    syms = [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
    do_trades = args.only in ("trades", "both")
    do_bbo = args.only in ("bbo", "both")

    if args.merge_only:
        for sym in syms:
            paths = bybit_market_paths(sym)
            if do_trades and not paths["trades"].exists():
                merge_staging(DATA / "staging" / "bybit_trades" / sym, paths["trades"], TRADE_SCHEMA)
            if do_bbo and not paths["bbo"].exists():
                merge_staging(DATA / "staging" / "bybit_booktickers" / sym, paths["bbo"], BBO_SCHEMA)
        return

    for sym in syms:
        run_build(
            sym,
            d0,
            d1,
            trades=do_trades,
            bbo=do_bbo,
            merge=not args.no_merge,
            clean_staging=args.clean_staging,
        )


if __name__ == "__main__":
    main()
