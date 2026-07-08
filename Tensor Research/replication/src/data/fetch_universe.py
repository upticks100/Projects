"""One-connection WRDS pull for the FULL gvkey universe (Master List lever L1).

Scales the event-study/veer data layer from 50 to ~499 firms so the
pre-registered veer hypotheses (drift_cashflow -> dDD, veers -> dIV) can be
falsified at real cross-sectional power. Pulls, in ONE psycopg2 connection
(= one Duo push):

  1. crsp.ccmxpf_linktable        gvkey <-> permno for the whole universe
  2. crsp.dsf                     legacy daily, 2005-01-01..2024-12-31
  3. crsp.wrds_dsfv2_query        CIZ v2 daily, 2025-01-01..end (legacy schema)
  4. crsp.dsi + wrds_dailyindexret_query   market index across both eras
  5. wrdsapps.opcrsphist + optionm.vsurfd<yyyy>  ATM 30/60d implied vol

Universe = all distinct gvkeys in the extended fundamentals CSV (~499).
Outputs to a NEW dir (append-only rule; never touches the 50-firm caches):
  pre_prediction_cache/event_study_499/
    link_table.csv
    daily_returns.csv.gz     (permno,date,ret,retx,prc,vol,shrout,open,high,low)
    daily_market.csv
    optionmetrics_iv.csv.gz  (secid,date,iv_30d,iv_60d,permno,gvkey)

Usage
-----
    python fetch_universe_499.py            # full pull, Duo push on connect
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent
WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
DEFAULT_USER = os.environ.get("WRDS_USER", "upticks100")

DEFAULT_FUNDAMENTALS = ROOT / "90-26_Q_Fundamentals_v2_extended.csv"
DEFAULT_OUT = ROOT / "pre_prediction_cache" / "event_study_499"
DEFAULT_START = "2005-01-01"
LEGACY_END = "2024-12-31"      # crsp.dsf / crsp.dsi stop here
V2_START = "2025-01-01"        # CIZ v2 tables begin here
DEFAULT_END = "2026-07-06"

TARGET_DAYS = (30, 60)         # constant-maturity horizons (days)
ATM_DELTAS = (50, -50)         # |delta| = 50 => ATM on the OptionMetrics surface


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamentals", type=Path, default=DEFAULT_FUNDAMENTALS)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def connect() -> "psycopg2.extensions.connection":
    pgpass = Path.home() / ".pgpass"
    if pgpass.exists():
        os.environ.setdefault("PGPASSFILE", str(pgpass))
    print(f"  connecting host={WRDS_HOST} user={DEFAULT_USER!r} "
          f"(DUO PUSH EXPECTED NOW) ...", flush=True)
    conn = psycopg2.connect(host=WRDS_HOST, port=WRDS_PORT, dbname=WRDS_DB,
                            user=DEFAULT_USER, connect_timeout=120)
    conn.set_session(readonly=True, autocommit=True)
    print("  connected", flush=True)
    return conn


def q(conn, sql, params=None, label="") -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_sql_query(sql, conn, params=params)
    print(f"  {label}: {len(df):,} rows in {time.perf_counter()-t0:.1f}s",
          flush=True)
    return df


def main() -> int:
    args = parse_args()
    print("== fetch_universe_499.py ==", flush=True)
    print(f"  window {args.start_date}..{args.end_date}", flush=True)
    print(f"  out    {args.out_dir}", flush=True)

    fund = pd.read_csv(args.fundamentals, usecols=["gvkey"], dtype={"gvkey": str})
    gvkeys = sorted(fund["gvkey"].dropna().unique().tolist())
    print(f"  universe: {len(gvkeys)} gvkeys from {args.fundamentals.name}",
          flush=True)

    conn = connect()
    try:
        link = q(conn, """
            SELECT gvkey, lpermno AS permno, lpermco AS permco,
                   linktype, linkprim, linkdt, linkenddt
            FROM crsp.ccmxpf_linktable
            WHERE gvkey = ANY(%(g)s) AND linktype IN ('LC','LU')
              AND linkprim IN ('P','C')
            ORDER BY gvkey, linkdt
            """, {"g": gvkeys}, "link_table")
        permnos = sorted(link["permno"].dropna().astype(int).unique().tolist())
        no_link = len(set(gvkeys) - set(link["gvkey"].astype(str)))
        print(f"  resolved {len(permnos)} permnos "
              f"({no_link} gvkeys without link)", flush=True)

        ret_legacy = q(conn, """
            SELECT permno, date, ret, retx, prc, vol, shrout
            FROM crsp.dsf
            WHERE permno = ANY(%(p)s)
              AND date >= %(s)s AND date <= %(e)s
            ORDER BY permno, date
            """, {"p": permnos, "s": args.start_date, "e": LEGACY_END},
            "crsp.dsf (legacy)")

        ret_v2 = q(conn, """
            SELECT permno, dlycaldt AS date, dlyret AS ret, dlyretx AS retx,
                   dlyprc AS prc, dlyvol AS vol, shrout,
                   dlyopen AS open, dlyhigh AS high, dlylow AS low
            FROM crsp.wrds_dsfv2_query
            WHERE permno = ANY(%(p)s)
              AND dlycaldt >= %(s)s AND dlycaldt <= %(e)s
            ORDER BY permno, dlycaldt
            """, {"p": permnos, "s": V2_START, "e": args.end_date},
            "wrds_dsfv2 (2025+)")

        mkt_legacy = q(conn, """
            SELECT date, vwretd, vwretx, ewretd, ewretx, sprtrn
            FROM crsp.dsi
            WHERE date >= %(s)s AND date <= %(e)s ORDER BY date
            """, {"s": args.start_date, "e": LEGACY_END}, "crsp.dsi (legacy)")

        mkt_v2 = q(conn, """
            SELECT dlycaldt AS date, vwretd, vwretx, ewretd, ewretx, sprtrn
            FROM crsp.wrds_dailyindexret_query
            WHERE dlycaldt >= %(s)s AND dlycaldt <= %(e)s ORDER BY dlycaldt
            """, {"s": V2_START, "e": args.end_date}, "index v2 (2025+)")

        sec = q(conn, """
            SELECT secid, permno, score, sdate, edate
            FROM wrdsapps.opcrsphist
            WHERE permno = ANY(%(p)s)
            """, {"p": permnos}, "opcrsphist")
        sec = (sec.sort_values(["permno", "score"])
                  .drop_duplicates("permno", keep="first"))
        secids = sorted(sec["secid"].unique().tolist())
        print(f"  resolved {len(secids)} secids", flush=True)

        frames = []
        y0, y1 = int(args.start_date[:4]), int(args.end_date[:4])
        for yr in range(y0, y1 + 1):
            tbl = f"optionm.vsurfd{yr}"
            try:
                df = q(conn, f"""
                    SELECT secid, date, days, impl_volatility
                    FROM {tbl}
                    WHERE secid = ANY(%(s)s)
                      AND days = ANY(%(d)s) AND delta = ANY(%(x)s)
                      AND date >= %(a)s AND date <= %(b)s
                    """, {"s": [int(x) for x in secids],
                          "d": list(TARGET_DAYS), "x": list(ATM_DELTAS),
                          "a": args.start_date, "b": args.end_date}, tbl)
            except Exception as e:  # noqa: BLE001
                print(f"  {tbl}: skipped ({e})", flush=True)
                continue
            if not df.empty:
                frames.append(df)
    finally:
        conn.close()

    # ---- assemble daily returns across the source break ----
    for c in ("open", "high", "low"):
        ret_legacy[c] = np.nan
    ret = pd.concat([ret_legacy, ret_v2], ignore_index=True)
    ret = ret.sort_values(["permno", "date"])
    if ret.empty:
        raise SystemExit("daily pull returned 0 rows; aborting")
    dens = float(ret["ret"].notna().mean())
    print(f"  daily_returns: {len(ret):,} rows, {ret['permno'].nunique()} "
          f"permnos, ret density {dens*100:.2f}%", flush=True)
    if dens < 0.90:
        raise SystemExit("ret density < 90%; aborting")

    mkt = pd.concat([mkt_legacy, mkt_v2], ignore_index=True).sort_values("date")

    # ---- ATM IV wide table ----
    if not frames:
        raise SystemExit("IV pull returned 0 rows; aborting")
    iv = pd.concat(frames, ignore_index=True)
    atm = (iv.groupby(["secid", "date", "days"])["impl_volatility"]
             .mean().reset_index())
    wide = atm.pivot_table(index=["secid", "date"], columns="days",
                           values="impl_volatility").reset_index()
    wide.columns = ["secid", "date"] + [f"iv_{int(c)}d" for c in wide.columns[2:]]
    wide = wide.merge(sec[["secid", "permno"]], on="secid", how="left")
    pm2gv = (link.dropna(subset=["permno"])
                 .assign(permno=lambda d: d["permno"].astype(int))
                 .sort_values(["permno", "linkprim"])   # 'C' < 'P': keep P last
                 .drop_duplicates("permno", keep="last")[["permno", "gvkey"]])
    wide = wide.merge(pm2gv, on="permno", how="left")
    print(f"  optionmetrics_iv: {len(wide):,} secid-dates, "
          f"{wide['permno'].nunique()} permnos", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = (("link_table.csv", link, None),
               ("daily_returns.csv.gz", ret, "gzip"),
               ("daily_market.csv", mkt, None),
               ("optionmetrics_iv.csv.gz", wide, "gzip"))
    for name, df, comp in outputs:
        out = args.out_dir / name
        t0 = time.perf_counter()
        df.to_csv(out, index=False, compression=comp)
        print(f"  wrote {name} ({df.shape[0]:,}x{df.shape[1]}, "
              f"{out.stat().st_size/1e6:.1f} MB, "
              f"{time.perf_counter()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
