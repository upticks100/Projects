"""Append-only OptionMetrics (IvyDB US) implied-vol pull for the event study.

Motivation: the in-hand vol-risk result shows |CP increment| forecasts realized
vol INCREMENTAL to the firm's own lagged realized vol. The *correct* expected-vol
benchmark for a tradeable (straddle) claim is option-IMPLIED vol, not lagged
realized vol. This script pulls the constant-maturity ATM implied vol from the
OptionMetrics volatility surface for our universe so we can test:
    straddle PnL proxy  ~  realized_vol[+2,+30]  -  implied_vol[pre-event]

Design mirrors fetch_crsp_v2_append.py: one psycopg2 connection / one Duo push,
versioned output CSV, never overwrites existing caches.

Run `--probe` FIRST (one push) to confirm the optionm schema / link table /
column names on this WRDS instance, then run the full pull.

Usage
-----
    python fetch_optionmetrics_iv.py --probe
    python fetch_optionmetrics_iv.py --start-date 2005-01-01 --end-date 2025-12-31 \
        --out pre_prediction_cache/event_study_extended/optionmetrics_iv.csv
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
DEFAULT_META = config.meta_path()
DEFAULT_OUT = config.EVENT_DIR_EXTENDED / "optionmetrics_iv.csv"
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2025-12-31"
# Constant-maturity horizons to keep (days). 30d matches our [+2,+30] window.
TARGET_DAYS = (30, 60)
# ATM definition: |delta| == 50 (OptionMetrics surface delta is in pct points).
ATM_DELTAS = (50, -50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--probe", action="store_true",
                   help="discover schema/link/columns and a tiny sample, then exit")
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


def resolve_permnos(conn, gvkeys) -> pd.DataFrame:
    link = pd.read_sql_query(
        """
        SELECT gvkey, lpermno AS permno
        FROM crsp.ccmxpf_linktable
        WHERE gvkey = ANY(%(g)s) AND linktype IN ('LC','LU')
          AND linkprim IN ('P','C')
        """,
        conn, params={"g": list(gvkeys)})
    link = link.dropna(subset=["permno"])
    link["permno"] = link["permno"].astype(int)
    return link.drop_duplicates()


def probe(conn) -> None:
    print("\n== PROBE: optionm / wrdsapps schema ==", flush=True)
    tabs = pd.read_sql_query(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('optionm','wrdsapps')
          AND (table_name ILIKE '%vsurf%' OR table_name ILIKE '%stdopd%'
               OR table_name ILIKE '%secur%' OR table_name ILIKE '%opcrsp%'
               OR table_name ILIKE '%secnm%' OR table_name ILIKE '%opvol%'
               OR table_name ILIKE '%link%optionm%' OR table_name ILIKE '%optionm%link%')
        ORDER BY table_schema, table_name
        """, conn)
    print(tabs.to_string(index=False), flush=True)

    # newest volatility-surface table name
    vsurf = tabs[(tabs.table_schema == "optionm") &
                 (tabs.table_name.str.contains("vsurf"))]
    if not vsurf.empty:
        latest = sorted(vsurf["table_name"].tolist())[-1]
        print(f"\n-- columns of optionm.{latest} --", flush=True)
        cols = pd.read_sql_query(
            """
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema='optionm' AND table_name=%(t)s
            ORDER BY ordinal_position
            """, conn, params={"t": latest})
        print(cols.to_string(index=False), flush=True)
        print(f"\n-- sample rows from optionm.{latest} (days IN {TARGET_DAYS}, "
              f"|delta|=50) --", flush=True)
        try:
            samp = pd.read_sql_query(
                f"SELECT * FROM optionm.{latest} "
                f"WHERE days = ANY(%(d)s) AND delta = ANY(%(x)s) LIMIT 8",
                conn, params={"d": list(TARGET_DAYS), "x": list(ATM_DELTAS)})
            print(samp.to_string(index=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  sample query failed: {e}", flush=True)

    # link table columns
    opl = tabs[tabs.table_name.str.contains("opcrsp")]
    if not opl.empty:
        sch, tname = opl.iloc[0]["table_schema"], opl.iloc[0]["table_name"]
        print(f"\n-- columns of {sch}.{tname} --", flush=True)
        cols = pd.read_sql_query(
            """
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema=%(s)s AND table_name=%(t)s
            ORDER BY ordinal_position
            """, conn, params={"s": sch, "t": tname})
        print(cols.to_string(index=False), flush=True)


def fetch_link_secid(conn, permnos) -> pd.DataFrame:
    """permno -> secid via wrdsapps.opcrsphist (best score per permno)."""
    link = pd.read_sql_query(
        """
        SELECT secid, permno, score, sdate, edate
        FROM wrdsapps.opcrsphist
        WHERE permno = ANY(%(p)s)
        """, conn, params={"p": [int(x) for x in permnos]})
    if link.empty:
        return link
    link = link.sort_values(["permno", "score"]).drop_duplicates("permno", keep="first")
    return link


def fetch_iv(conn, secids, start, end) -> pd.DataFrame:
    """Constant-maturity ATM implied vol from optionm.vsurfd<yyyy>, per year."""
    y0, y1 = int(start[:4]), int(end[:4])
    frames = []
    for yr in range(y0, y1 + 1):
        tbl = f"optionm.vsurfd{yr}"
        try:
            df = pd.read_sql_query(
                f"""
                SELECT secid, date, days, cp_flag, delta, impl_volatility
                FROM {tbl}
                WHERE secid = ANY(%(s)s)
                  AND days = ANY(%(d)s) AND delta = ANY(%(x)s)
                  AND date >= %(a)s AND date <= %(b)s
                """,
                conn, params={"s": [int(x) for x in secids],
                              "d": list(TARGET_DAYS), "x": list(ATM_DELTAS),
                              "a": start, "b": end})
        except Exception as e:  # noqa: BLE001
            print(f"  {tbl}: skipped ({e})", flush=True)
            continue
        if not df.empty:
            frames.append(df)
        print(f"  {tbl}: {len(df):,} rows", flush=True)
    if not frames:
        return pd.DataFrame()
    iv = pd.concat(frames, ignore_index=True)
    # ATM IV = mean over call+put at |delta|=50 per (secid,date,days)
    atm = (iv.groupby(["secid", "date", "days"])["impl_volatility"]
             .mean().reset_index())
    wide = atm.pivot_table(index=["secid", "date"], columns="days",
                           values="impl_volatility").reset_index()
    wide.columns = (["secid", "date"] +
                    [f"iv_{int(c)}d" for c in wide.columns[2:]])
    return wide


def main() -> int:
    args = parse_args()
    print("== fetch_optionmetrics_iv.py ==", flush=True)
    meta = joblib.load(args.meta)
    gvkeys = sorted({str(g) for g in meta["firms"]})
    print(f"  {len(gvkeys)} gvkeys from meta", flush=True)

    conn = connect()
    try:
        if args.probe:
            probe(conn)
            return 0

        link_pm = resolve_permnos(conn, gvkeys)
        permnos = sorted(link_pm["permno"].unique().tolist())
        print(f"  resolved {len(permnos)} permnos", flush=True)

        sec = fetch_link_secid(conn, permnos)
        if sec.empty:
            raise SystemExit("no secids resolved from wrdsapps.opcrsphist")
        secids = sorted(sec["secid"].unique().tolist())
        print(f"  resolved {len(secids)} secids", flush=True)

        t0 = time.perf_counter()
        iv = fetch_iv(conn, secids, args.start_date, args.end_date)
        print(f"  IV: {len(iv):,} secid-dates in {time.perf_counter()-t0:.1f}s",
              flush=True)
    finally:
        conn.close()

    if iv.empty:
        raise SystemExit("IV pull returned no rows; aborting (check schema/window).")

    # attach permno + gvkey
    iv = iv.merge(sec[["secid", "permno"]], on="secid", how="left")
    iv = iv.merge(link_pm.rename(columns={"permno": "permno"}), on="permno", how="left")
    iv["date"] = pd.to_datetime(iv["date"])
    iv = iv.sort_values(["gvkey", "date"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iv.to_csv(args.out, index=False)
    print(f"  wrote {args.out} ({iv.shape[0]:,}x{iv.shape[1]}) "
          f"{iv['date'].min().date()}..{iv['date'].max().date()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
