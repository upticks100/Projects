"""Pull CRSP daily returns + market index for the prediction universe.

Mirrors fetch_fundamentals_wrds.py: one psycopg2 connection, one Duo push,
all queries run inside it.

Pulls three tables:
  1. crsp.ccmxpf_linktable  -- gvkey <-> permno mapping (LC/LU links only)
  2. crsp.dsf               -- daily stock file (permno, date, ret, prc, vol)
  3. crsp.dsi               -- daily index file (date, vwretd, ewretd, sprtrn)

Restricted to the prediction universe (50 gvkeys from meta.pkl by default,
overridable) and 2005-01-01..2024-12-31.

Outputs three CSVs in pre_prediction_cache/event_study/:
  - link_table.csv
  - daily_returns.csv  (permno-date rows with ret, prc, vol)
  - daily_market.csv   (date rows with vwretd, ewretd, sprtrn)

Usage
-----
    python fetch_crsp_returns_wrds.py            # default universe, 2005-2024
    python fetch_crsp_returns_wrds.py --no-confirm
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import psycopg2

import config

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
DEFAULT_USER = os.environ.get("WRDS_USER", "upticks100")

OUT_DIR = config.PRE_PRED_CACHE / "event_study"
DEFAULT_META = config.meta_path()
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2024-12-31"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", type=Path, default=DEFAULT_META,
                   help="tensor cache meta.pkl (source of the universe gvkeys).")
    p.add_argument("--gvkeys", default=None,
                   help="Optional comma-separated gvkeys to override the meta universe.")
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip the are-you-sure prompt before connecting.")
    return p.parse_args()


def resolve_gvkeys(args: argparse.Namespace) -> list[str]:
    if args.gvkeys:
        gvkeys = sorted({g.strip() for g in args.gvkeys.split(",") if g.strip()})
        print(f"  using {len(gvkeys)} gvkeys from --gvkeys", flush=True)
        return gvkeys
    meta = joblib.load(args.meta)
    gvkeys = sorted({str(g) for g in meta["firms"]})
    print(f"  using {len(gvkeys)} gvkeys from {args.meta}", flush=True)
    return gvkeys


def connect() -> psycopg2.extensions.connection:
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        os.environ.setdefault("PGPASSFILE", str(pgpass))
    print(f"  connecting host={WRDS_HOST} db={WRDS_DB} user={DEFAULT_USER!r} "
          f"(Duo push expected) ...", flush=True)
    t0 = time.perf_counter()
    conn = psycopg2.connect(
        host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
        user=DEFAULT_USER, connect_timeout=120,
    )
    conn.set_session(readonly=True, autocommit=True)
    print(f"  connected in {time.perf_counter() - t0:.1f}s", flush=True)
    return conn


def pull_link_table(conn, gvkeys: list[str]) -> pd.DataFrame:
    sql = """
        SELECT gvkey, lpermno AS permno, lpermco AS permco,
               linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE gvkey = ANY(%(gvkeys)s)
          AND linktype IN ('LC', 'LU')
          AND linkprim IN ('P', 'C')
        ORDER BY gvkey, linkdt
    """
    print(f"  querying crsp.ccmxpf_linktable for {len(gvkeys)} gvkeys ...", flush=True)
    t0 = time.perf_counter()
    df = pd.read_sql_query(sql, conn, params={"gvkeys": gvkeys})
    print(f"    {len(df):,} link rows in {time.perf_counter() - t0:.1f}s", flush=True)
    return df


def pull_daily_returns(conn, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    sql = """
        SELECT permno, date, ret, retx, prc, vol, shrout
        FROM crsp.dsf
        WHERE permno = ANY(%(permnos)s)
          AND date >= %(start)s
          AND date <= %(end)s
        ORDER BY permno, date
    """
    print(f"  querying crsp.dsf for {len(permnos)} permnos, {start}..{end} ...",
          flush=True)
    t0 = time.perf_counter()
    df = pd.read_sql_query(
        sql, conn,
        params={"permnos": [int(p) for p in permnos], "start": start, "end": end},
    )
    elapsed = time.perf_counter() - t0
    print(f"    {len(df):,} daily-return rows in {elapsed:.1f}s "
          f"(~{len(df)/max(elapsed,1e-3):.0f}/s)", flush=True)
    return df


def pull_market_index(conn, start: str, end: str) -> pd.DataFrame:
    sql = """
        SELECT date, vwretd, vwretx, ewretd, ewretx, sprtrn
        FROM crsp.dsi
        WHERE date >= %(start)s AND date <= %(end)s
        ORDER BY date
    """
    print(f"  querying crsp.dsi for market index {start}..{end} ...", flush=True)
    t0 = time.perf_counter()
    df = pd.read_sql_query(sql, conn, params={"start": start, "end": end})
    print(f"    {len(df):,} market rows in {time.perf_counter() - t0:.1f}s", flush=True)
    return df


def assert_sane(link: pd.DataFrame, ret: pd.DataFrame, mkt: pd.DataFrame,
                gvkeys: list[str]) -> None:
    if link.empty:
        raise SystemExit("link_table query returned 0 rows; aborting")
    linked_gvkeys = set(link["gvkey"].astype(str))
    missing_gvkeys = set(gvkeys) - linked_gvkeys
    if missing_gvkeys:
        print(f"    WARN: {len(missing_gvkeys)} gvkeys with no link: "
              f"{sorted(missing_gvkeys)[:10]}...", flush=True)
    if ret.empty:
        raise SystemExit("daily_returns query returned 0 rows; aborting")
    if mkt.empty:
        raise SystemExit("market_index query returned 0 rows; aborting")
    ret_density = float(ret["ret"].notna().mean())
    if ret_density < 0.95:
        raise SystemExit(f"daily ret column too sparse: {ret_density*100:.1f}%")
    print(f"    ret density: {ret_density*100:.2f}%", flush=True)


def main() -> int:
    args = parse_args()
    print("== fetch_crsp_returns_wrds.py ==", flush=True)
    print(f"  start_date  = {args.start_date}", flush=True)
    print(f"  end_date    = {args.end_date}", flush=True)
    print(f"  out_dir     = {args.out_dir}", flush=True)

    if not args.no_confirm:
        try:
            input("  press Enter to proceed (Duo push will fire), Ctrl-C to abort: ")
        except KeyboardInterrupt:
            print("\naborted")
            return 1

    gvkeys = resolve_gvkeys(args)

    conn = connect()
    try:
        link = pull_link_table(conn, gvkeys)
        permnos = sorted(link["permno"].dropna().astype(int).unique().tolist())
        print(f"    resolved {len(permnos)} unique permnos", flush=True)
        ret = pull_daily_returns(conn, permnos, args.start_date, args.end_date)
        mkt = pull_market_index(conn, args.start_date, args.end_date)
    finally:
        conn.close()

    print("  sanity-checking ...", flush=True)
    assert_sane(link, ret, mkt, gvkeys)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "link_table.csv": link,
        "daily_returns.csv": ret,
        "daily_market.csv": mkt,
    }
    for name, df in paths.items():
        out = args.out_dir / name
        t0 = time.perf_counter()
        df.to_csv(out, index=False)
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"  wrote {name} ({df.shape[0]:,}x{df.shape[1]}, "
              f"{size_mb:.1f} MB, {time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
