"""Pull a complete refresh of comp.fundq from WRDS for our gvkey universe.

Connects via psycopg2 (bypasses the wrds Python wrapper, which prompts
interactively even when given wrds_username). Pulls every column of
comp.fundq for the gvkeys listed in gvkeys_to_gics.csv, restricted to
1990-01-01 .. 2024-12-31 and the standard Compustat filter
(consol='C', popsrc='D', datafmt='STD', indfmt='INDL').

Saves to 90-25_Q_Fundamentals_v2.csv (does NOT overwrite the existing
90-25_Q_Fundamentals.csv) so the original can be diffed before swapping.

Triggers exactly one WRDS Duo push when started.

Usage
-----
    python fetch_fundamentals_wrds.py
    python fetch_fundamentals_wrds.py --output /tmp/fundq.csv  # custom path
    python fetch_fundamentals_wrds.py --no-confirm             # skip prompt

Sanity assertions before the file is written:
  * non-empty result
  * all 15 currently-empty YTD cash-flow columns must be present and have
    at least 1 % observed density across the pulled rows
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import psycopg2

import config

# The MFI-era fundamentals pull spans the full 1990-2024 window regardless of
# the prediction panel's (later) START_DATE; both are CLI-overridable.
LOCAL_GICS_FILE = config.GICS_FILE
START_DATE = config.MFI_START_DATE
END_DATE = config.MFI_END_DATE


WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
DEFAULT_USER = os.environ.get("WRDS_USER", "upticks100")

# These were 0.00% dense in the previous local extract; if the new pull
# doesn't have them populated, something went wrong.
REQUIRED_YTD_COLUMNS = (
    "capxy", "dvy", "txbcofy", "fincfy", "fopoy",
    "ivncfy", "ivacoy", "dltisy", "oancfy", "sstky",
    "sivy", "sppivy", "aolochy", "aqcy", "ibcy",
)
DENSITY_FLOOR = 0.01  # 1% of pulled rows must have a non-null value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.MFI_FUNDAMENTALS_FILE,
        help="Destination CSV (default: the configured MFI fundamentals file).",
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help=f"Earliest datadate (default: {START_DATE!r}).",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help=f"Latest datadate (default: {END_DATE!r}).",
    )
    parser.add_argument(
        "--gvkey-source",
        type=Path,
        default=LOCAL_GICS_FILE,
        help="CSV containing a 'gvkey' column to use as the firm universe.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the are-you-sure prompt before connecting to WRDS.",
    )
    return parser.parse_args()


def load_gvkey_universe(path: Path) -> list[str]:
    df = pd.read_csv(path, dtype={"gvkey": str}, usecols=["gvkey"])
    gvkeys = sorted(df["gvkey"].dropna().unique().tolist())
    if not gvkeys:
        raise SystemExit(f"no gvkeys found in {path}")
    print(f"  loaded {len(gvkeys)} unique gvkeys from {path}", flush=True)
    return gvkeys


def connect() -> psycopg2.extensions.connection:
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        os.environ.setdefault("PGPASSFILE", str(pgpass))
    print(
        f"  connecting host={WRDS_HOST} db={WRDS_DB} user={DEFAULT_USER!r} "
        f"(Duo push expected) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    conn = psycopg2.connect(
        host=WRDS_HOST,
        port=WRDS_PORT,
        dbname=WRDS_DB,
        user=DEFAULT_USER,
        connect_timeout=120,
    )
    conn.set_session(readonly=True, autocommit=True)
    print(f"  connected in {time.perf_counter() - t0:.1f}s", flush=True)
    return conn


def pull_fundq(
    conn: psycopg2.extensions.connection,
    gvkeys: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
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
    print(
        f"  querying comp.fundq for {len(gvkeys)} gvkeys, "
        f"{start_date}..{end_date} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    df = pd.read_sql_query(
        sql,
        conn,
        params={"gvkeys": gvkeys, "start": start_date, "end": end_date},
    )
    elapsed = time.perf_counter() - t0
    print(
        f"  fetched {len(df):,} rows x {df.shape[1]} columns in {elapsed:.1f}s "
        f"(~{len(df) / max(elapsed, 1e-3):.0f} rows/s)",
        flush=True,
    )
    return df


def assert_sane(df: pd.DataFrame) -> None:
    if df.empty:
        raise SystemExit("query returned 0 rows; aborting before any file write")

    missing = [c for c in REQUIRED_YTD_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            "expected YTD columns absent from result: "
            + ", ".join(missing)
        )

    too_sparse = []
    for col in REQUIRED_YTD_COLUMNS:
        density = df[col].notna().mean()
        flag = "OK" if density >= DENSITY_FLOOR else "LOW"
        print(f"    [{flag}] {col:10s} density={density * 100:6.2f}%")
        if density < DENSITY_FLOOR:
            too_sparse.append((col, density))
    if too_sparse:
        raise SystemExit(
            f"these columns are below {DENSITY_FLOOR * 100:.0f}% density: "
            + ", ".join(f"{c}={d * 100:.2f}%" for c, d in too_sparse)
        )


def main() -> int:
    args = parse_args()
    print("== fetch_fundamentals_wrds.py ==", flush=True)
    print(f"  start_date  = {args.start_date}", flush=True)
    print(f"  end_date    = {args.end_date}", flush=True)
    print(f"  gvkey_src   = {args.gvkey_source}", flush=True)
    print(f"  output_path = {args.output}", flush=True)

    if args.output.exists():
        print(
            f"  NOTE: output already exists ({args.output.stat().st_size:,} bytes); "
            "will be overwritten on success.",
            flush=True,
        )

    if not args.no_confirm:
        try:
            answer = input(
                "  press Enter to proceed (Duo push will fire), Ctrl-C to abort: "
            )
            del answer
        except KeyboardInterrupt:
            print("\naborted")
            return 1

    gvkeys = load_gvkey_universe(args.gvkey_source)

    conn = connect()
    try:
        df = pull_fundq(conn, gvkeys, args.start_date, args.end_date)
    finally:
        conn.close()

    print("  sanity-checking required columns ...", flush=True)
    assert_sane(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"  writing CSV to {args.output} ...", flush=True)
    t0 = time.perf_counter()
    df.to_csv(args.output, index=False)
    elapsed = time.perf_counter() - t0
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(
        f"  wrote {size_mb:.1f} MB in {elapsed:.1f}s. "
        f"shape = {df.shape[0]:,} rows x {df.shape[1]} cols",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
