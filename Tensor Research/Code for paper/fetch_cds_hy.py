"""Markit 5Y CDS pull for the curated HY universe (H-HY pre-registration).

Unlike fetch_cds_markit.py (which ticker-matched at pull time), this pulls
by REDCODE from the hand-curated hy_universe_link.csv, so the link decisions
are frozen in build_hy_universe.py and audited there.

Window starts 2021-07-01 (Markit pull convention from the 499 test) — the
veer panels' test quarters begin 2022Q1, and the -65td control window never
reaches before 2021-10.

Output: pre_prediction_cache/event_study_hy/cds_markit_hy.csv.gz
        (gvkey, permno, redcode, date, parspread, docclause, depth5y)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fetch_universe_499 import connect  # noqa: E402

CACHE = ROOT / "pre_prediction_cache"
OUT_DIR = CACHE / "event_study_hy"
START, END = "2021-07-01", "2026-07-06"


def main() -> int:
    print("== fetch_cds_hy.py ==", flush=True)
    link = pd.read_csv(CACHE / "hy_universe_link.csv", dtype={"gvkey": str})
    redcodes = sorted(link["redcode"].unique().tolist())
    print(f"  {len(redcodes)} curated redcodes", flush=True)

    conn = connect()
    try:
        chunks = []
        for yr in range(2021, 2027):
            t0 = time.perf_counter()
            df = pd.read_sql_query(f"""
                SELECT date, redcode, docclause, parspread,
                       compositedepth5y AS depth5y
                FROM markit_cds.cds{yr}
                WHERE tenor = '5Y' AND currency = 'USD' AND tier = 'SNRFOR'
                  AND docclause IN ('XR', 'XR14')
                  AND parspread IS NOT NULL
                  AND redcode = ANY(%(r)s)
                  AND date BETWEEN %(s)s AND %(e)s
                """, conn, params={"r": redcodes, "s": START, "e": END})
            print(f"  cds{yr}: {len(df):,} rows in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
            chunks.append(df)
    finally:
        conn.close()

    cds = pd.concat(chunks, ignore_index=True)
    cds["date"] = pd.to_datetime(cds["date"])
    cds["dc_rank"] = (cds["docclause"] == "XR14").astype(int)
    cds = (cds.sort_values(["redcode", "date", "dc_rank"])
              .drop_duplicates(["redcode", "date"], keep="last")
              .drop(columns="dc_rank"))

    out = (cds.merge(link[["gvkey", "permno", "redcode"]], on="redcode")
              [["gvkey", "permno", "redcode", "date", "parspread",
                "docclause", "depth5y"]]
              .sort_values(["gvkey", "date"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "cds_markit_hy.csv.gz"
    out.to_csv(dest, index=False, compression="gzip")
    print(f"  wrote {len(out):,} rows / {out['gvkey'].nunique()} gvkeys "
          f"(median spread "
          f"{out['parspread'].median() * 1e4:.0f} bp) -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
