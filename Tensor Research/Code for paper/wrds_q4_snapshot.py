"""One-quarter sanity snapshot of comp.fundq for our 499 gvkeys.

Pulls the rows for 2024-Q4 (datadate 2024-10-01..2024-12-31), reports the
density of every column, and highlights the 15 YTD cash-flow columns the
local CSV is currently missing. Does NOT write any file. One Duo push.

Usage:
    python wrds_q4_snapshot.py
    python wrds_q4_snapshot.py --start 2024-09-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pre_prediction_config import LOCAL_GICS_FILE  # noqa: E402


WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
DEFAULT_USER = os.environ.get("WRDS_USER", "upticks100")

REQUIRED_YTD_COLUMNS = (
    "capxy", "dvy", "txbcofy", "fincfy", "fopoy",
    "ivncfy", "ivacoy", "dltisy", "oancfy", "sstky",
    "sivy", "sppivy", "aolochy", "aqcy", "ibcy",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-10-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        os.environ.setdefault("PGPASSFILE", str(pgpass))

    gvkeys = sorted(
        pd.read_csv(LOCAL_GICS_FILE, dtype={"gvkey": str}, usecols=["gvkey"])
        ["gvkey"].dropna().unique().tolist()
    )
    print(f"gvkeys: {len(gvkeys)}", flush=True)
    print(f"window: {args.start}..{args.end}", flush=True)
    print(
        f"connecting host={WRDS_HOST} db={WRDS_DB} user={DEFAULT_USER!r} (Duo push) ...",
        flush=True,
    )

    conn = psycopg2.connect(
        host=WRDS_HOST,
        port=WRDS_PORT,
        dbname=WRDS_DB,
        user=DEFAULT_USER,
        connect_timeout=120,
    )
    conn.set_session(readonly=True, autocommit=True)
    try:
        sql = """
            SELECT *
            FROM comp.fundq
            WHERE gvkey = ANY(%(gvkeys)s)
              AND datadate >= %(start)s
              AND datadate <= %(end)s
              AND consol = 'C'
              AND popsrc = 'D'
              AND datafmt = 'STD'
              AND indfmt = 'INDL'
            ORDER BY gvkey, datadate
        """
        df = pd.read_sql_query(
            sql,
            conn,
            params={"gvkeys": gvkeys, "start": args.start, "end": args.end},
        )
    finally:
        conn.close()

    print(f"\nfetched {len(df):,} rows x {df.shape[1]} columns")
    if df.empty:
        return 1

    print(f"distinct gvkeys in result: {df['gvkey'].nunique()}")
    print(f"distinct datadates:       {df['datadate'].nunique()}")

    print("\n=== density of the 15 previously-missing YTD columns ===")
    for col in REQUIRED_YTD_COLUMNS:
        if col not in df.columns:
            print(f"  {col:10s}  MISSING from result columns!")
            continue
        n = df[col].notna().sum()
        print(f"  {col:10s}  non-null={n:4d}/{len(df)}  density={n/len(df)*100:6.2f}%")

    print("\n=== top 25 columns by density ===")
    densities = df.notna().mean().sort_values(ascending=False)
    for col, d in densities.head(25).items():
        print(f"  {col:25s}  density={d * 100:6.2f}%")

    print("\n=== bottom 25 columns by density (out of {}) ===".format(len(densities)))
    for col, d in densities.tail(25).items():
        print(f"  {col:25s}  density={d * 100:6.2f}%")

    print("\n=== density bucket histogram ===")
    bins = [0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0001]
    labels = ["0%", "0-1%", "1-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]
    counts = [int((densities == 0).sum())]
    for lo, hi in zip(bins[:-1], bins[1:]):
        counts.append(int(((densities > lo) & (densities <= hi)).sum()))
    counts = [int((densities == 0).sum())] + [
        int(((densities > lo) & (densities <= hi)).sum())
        for lo, hi in zip(bins[:-1], bins[1:])
    ]
    for label, count in zip(labels, counts):
        print(f"  {label:8s}: {count:4d} columns")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
