"""Markit single-name CDS pull for the 499 universe (H1 economic translation).

Pre-registered extension (RESEARCH_LOG 2026-07-07): translate the confirmed
drift_cashflow -> d_dd effect into market credit-spread units. Spec declared
before pulling:

  - markit_cds.cds<yyyy>, 2021-07-01..2026-07-06 (covers the veer panels'
    2022Q1-2026Q2 announcements plus the -65td control window).
  - tenor 5Y, currency USD, tier SNRFOR, docclause in (XR, XR14) with XR14
    preferred when both quote on a date, country = United States.
  - Link: CRSP stocknames ticker <-> Markit ticker, name rows valid inside
    the sample window; a gvkey matching several redcodes resolves to the
    redcode with the most spread observations.

One psycopg2 connection (= one Duo push). Outputs (append-only, new files):
  pre_prediction_cache/event_study_499/cds_markit.csv.gz
      gvkey, permno, ticker, redcode, date, parspread, docclause, depth5y
  pre_prediction_cache/event_study_499/cds_link_audit.csv
      per-gvkey match audit (CRSP name vs Markit shortname, obs counts)
"""
from __future__ import annotations

import sys
import time

import pandas as pd

import config
from src.data.fetch_universe import connect, q

OUT_DIR = config.EVENT_DIR_499
START, END = "2021-07-01", "2026-07-06"
YEARS = range(2021, 2027)


def main() -> int:
    print("== fetch_cds_markit.py ==", flush=True)
    link = pd.read_csv(OUT_DIR / "link_table.csv", dtype={"gvkey": str})
    permnos = sorted(link["permno"].dropna().astype(int).unique().tolist())
    print(f"  universe: {link['gvkey'].nunique()} gvkeys, "
          f"{len(permnos)} permnos", flush=True)

    conn = connect()
    try:
        names = q(conn, """
            SELECT permno, ticker, comnam, namedt, nameenddt
            FROM crsp.stocknames
            WHERE permno = ANY(%(p)s) AND ticker IS NOT NULL
              AND nameenddt >= %(s)s AND namedt <= %(e)s
            """, {"p": permnos, "s": START, "e": END}, "crsp.stocknames")
        tickers = sorted(names["ticker"].unique().tolist())
        print(f"  {len(tickers)} distinct in-window tickers", flush=True)

        chunks = []
        for yr in YEARS:
            t0 = time.perf_counter()
            try:
                df = pd.read_sql_query(f"""
                    SELECT date, ticker, redcode, shortname, docclause,
                           parspread, compositedepth5y AS depth5y
                    FROM markit_cds.cds{yr}
                    WHERE country = 'United States' AND tenor = '5Y'
                      AND currency = 'USD' AND tier = 'SNRFOR'
                      AND docclause IN ('XR', 'XR14')
                      AND parspread IS NOT NULL
                      AND ticker = ANY(%(t)s)
                      AND date BETWEEN %(s)s AND %(e)s
                    """, conn, params={"t": tickers, "s": START, "e": END})
            except Exception as exc:  # missing year table etc.
                print(f"  cds{yr}: skipped ({exc})", flush=True)
                conn.rollback()
                continue
            print(f"  cds{yr}: {len(df):,} rows in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
            chunks.append(df)
    finally:
        conn.close()

    cds = pd.concat(chunks, ignore_index=True)
    cds["date"] = pd.to_datetime(cds["date"])

    # XR14 preferred over XR when both quote on the same redcode-date
    cds["dc_rank"] = (cds["docclause"] == "XR14").astype(int)
    cds = (cds.sort_values(["redcode", "date", "dc_rank"])
              .drop_duplicates(["redcode", "date"], keep="last")
              .drop(columns="dc_rank"))
    print(f"  after docclause dedup: {len(cds):,} rows, "
          f"{cds['redcode'].nunique()} redcodes", flush=True)

    # ticker -> (gvkey, permno); one Markit ticker can hit several name rows.
    nm = names[["permno", "ticker", "comnam"]].drop_duplicates()
    lk = (link[["gvkey", "permno"]].drop_duplicates()
          .merge(nm, on="permno"))
    obs = (cds.groupby(["ticker", "redcode"])
              .agg(n_obs=("parspread", "size"),
                   shortname=("shortname", "first"))
              .reset_index())
    cand = lk.merge(obs, on="ticker")
    # gvkey resolves to its single best redcode (most observations)
    best = (cand.sort_values(["gvkey", "n_obs"], ascending=[True, False])
                .drop_duplicates("gvkey"))
    # a redcode must belong to only one gvkey too (drop collisions to the
    # gvkey with more CDS observations is meaningless here — same n_obs — so
    # drop ALL colliding gvkeys and audit them)
    dup_rc = best["redcode"].duplicated(keep=False)
    if dup_rc.any():
        print(f"  WARNING: {dup_rc.sum()} gvkeys collide on a redcode "
              f"-> dropped (see audit file)", flush=True)
    best["collision"] = dup_rc
    best.to_csv(OUT_DIR / "cds_link_audit.csv", index=False)
    best = best[~dup_rc]
    print(f"  matched {len(best)} gvkeys to unique redcodes", flush=True)

    out = (cds.merge(best[["gvkey", "permno", "ticker", "redcode"]],
                     on=["ticker", "redcode"])
              [["gvkey", "permno", "ticker", "redcode", "date",
                "parspread", "docclause", "depth5y"]]
              .sort_values(["gvkey", "date"]))
    dest = OUT_DIR / "cds_markit.csv.gz"
    out.to_csv(dest, index=False, compression="gzip")
    print(f"  wrote {len(out):,} rows / {out['gvkey'].nunique()} gvkeys "
          f"-> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
