"""Append-only CRSP daily pull from the CIZ v2 tables (2025+).

The legacy crsp.dsf / crsp.dsi are the annual-update product and stop at
2024-12-31. Post-2024 daily data lives in the CIZ v2 query views:
  - crsp.wrds_dsfv2_query        (daily security data; dly* columns)
  - crsp.wrds_dailyindexret_query (daily market index)

This script pulls ONLY the new window (default 2025-01-01..2026-06-30) and
writes legacy-schema CSVs (permno, date, ret, retx, prc, vol, shrout [+open,
high, low]) so they concatenate cleanly onto the frozen legacy
pre_prediction_cache/event_study/ files for the new Part 2 multi-target run.

It does NOT touch the legacy event_study/ dir. One psycopg2 connection,
one Duo push.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
DEFAULT_USER = os.environ.get("WRDS_USER", "upticks100")
DEFAULT_META = ROOT / "prediction_new" / "tensor_cache" / "meta.pkl"
DEFAULT_OUT = ROOT / "pre_prediction_cache" / "event_study_ext_2025_2026"
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2026-06-30"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def connect() -> "psycopg2.extensions.connection":
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        os.environ.setdefault("PGPASSFILE", str(pgpass))
    print(f"  connecting host={WRDS_HOST} user={DEFAULT_USER!r} (Duo push if not cached) ...",
          flush=True)
    conn = psycopg2.connect(host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
                            user=DEFAULT_USER, connect_timeout=120)
    conn.set_session(readonly=True, autocommit=True)
    print("  connected", flush=True)
    return conn


def main() -> int:
    args = parse_args()
    print("== fetch_crsp_v2_append.py ==", flush=True)
    print(f"  window {args.start_date}..{args.end_date}", flush=True)
    print(f"  out_dir {args.out_dir}", flush=True)

    meta = joblib.load(args.meta)
    gvkeys = sorted({str(g) for g in meta["firms"]})
    print(f"  {len(gvkeys)} gvkeys from meta", flush=True)

    conn = connect()
    try:
        link = pd.read_sql_query(
            """
            SELECT gvkey, lpermno AS permno, lpermco AS permco,
                   linktype, linkprim, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE gvkey = ANY(%(g)s) AND linktype IN ('LC','LU')
              AND linkprim IN ('P','C')
            ORDER BY gvkey, linkdt
            """,
            conn, params={"g": gvkeys})
        permnos = sorted(link["permno"].dropna().astype(int).unique().tolist())
        print(f"  resolved {len(permnos)} permnos", flush=True)

        t0 = time.perf_counter()
        ret = pd.read_sql_query(
            """
            SELECT permno,
                   dlycaldt AS date,
                   dlyret   AS ret,
                   dlyretx  AS retx,
                   dlyprc   AS prc,
                   dlyvol   AS vol,
                   shrout,
                   dlyopen  AS open,
                   dlyhigh  AS high,
                   dlylow   AS low
            FROM crsp.wrds_dsfv2_query
            WHERE permno = ANY(%(p)s)
              AND dlycaldt >= %(s)s AND dlycaldt <= %(e)s
            ORDER BY permno, dlycaldt
            """,
            conn, params={"p": [int(x) for x in permnos],
                          "s": args.start_date, "e": args.end_date})
        print(f"  daily_returns: {len(ret):,} rows in {time.perf_counter()-t0:.1f}s "
              f"({ret['date'].min()}..{ret['date'].max() if len(ret) else 'NA'})",
              flush=True)

        mkt = pd.read_sql_query(
            """
            SELECT dlycaldt AS date, vwretd, vwretx, ewretd, ewretx, sprtrn
            FROM crsp.wrds_dailyindexret_query
            WHERE dlycaldt >= %(s)s AND dlycaldt <= %(e)s
            ORDER BY dlycaldt
            """,
            conn, params={"s": args.start_date, "e": args.end_date})
        print(f"  daily_market: {len(mkt):,} rows "
              f"({mkt['date'].min()}..{mkt['date'].max() if len(mkt) else 'NA'})",
              flush=True)
    finally:
        conn.close()

    if ret.empty or mkt.empty:
        raise SystemExit("v2 pull returned no rows; aborting (check window).")
    rd = float(ret["ret"].notna().mean())
    print(f"  ret density {rd*100:.2f}%", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("link_table.csv", link),
                     ("daily_returns.csv", ret),
                     ("daily_market.csv", mkt)):
        out = args.out_dir / name
        df.to_csv(out, index=False)
        print(f"  wrote {name} ({df.shape[0]:,}x{df.shape[1]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
