"""WRDS schema discovery for Compustat fundq via psycopg2 (no interactive prompts).

Read-only: connects directly using psycopg2 + PGPASSFILE, queries
information_schema for the columns of comp.fundq (and a couple of adjacent
tables), buckets columns by suffix (q / y / a / 12 / other), and reports
which of the columns referenced by FEATURE_SPECS exist in WRDS, plus their
quarterly siblings.

The goal is to definitively answer "does WRDS have a quarterly version of
<this YTD field>?" before we write a fetch script.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pre_prediction_config import FEATURE_SPECS  # noqa: E402


WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DB = "wrds"
DEFAULT_USER = os.environ.get("WRDS_USER", "upticks100")
DEFAULT_PGPASSFILE = str(Path.home() / ".pgpass")

WANTED_TABLES = [
    ("comp", "fundq"),
    ("comp", "funda"),
    ("comp_na_daily_all", "fundq"),
]


def stem_of(col: str) -> str:
    m = re.match(r"^(.*?)(q|y|a|12|p)$", col)
    return m.group(1) if m else col


def bucket_by_suffix(cols: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for c in cols:
        if c.endswith("12"):
            buckets["12"].append(c)
        elif c.endswith("q"):
            buckets["q"].append(c)
        elif c.endswith("y"):
            buckets["y"].append(c)
        elif c.endswith("a"):
            buckets["a"].append(c)
        else:
            buckets["other"].append(c)
    return buckets


def list_columns(cur, schema: str, table: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return [row[0] for row in cur.fetchall()]


def list_schemas(cur) -> list[str]:
    cur.execute(
        """
        SELECT schema_name FROM information_schema.schemata
        WHERE schema_name NOT LIKE 'pg\\_%' ESCAPE '\\'
          AND schema_name <> 'information_schema'
        ORDER BY schema_name
        """,
    )
    return [row[0] for row in cur.fetchall()]


def main() -> int:
    os.environ.setdefault("PGPASSFILE", DEFAULT_PGPASSFILE)
    print(f"connecting host={WRDS_HOST} db={WRDS_DB} user={DEFAULT_USER!r} ...", flush=True)
    conn = psycopg2.connect(
        host=WRDS_HOST,
        port=WRDS_PORT,
        dbname=WRDS_DB,
        user=DEFAULT_USER,
        connect_timeout=20,
    )
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    try:
        schemas = list_schemas(cur)
        print(f"available schemas (count={len(schemas)}, first 30 shown):")
        print("  " + ", ".join(schemas[:30]))
        comp_like = [s for s in schemas if "comp" in s or "ciq" in s][:30]
        print(f"  comp/ciq-like schemas: {comp_like}")

        for library, table in WANTED_TABLES:
            print(f"\n{'=' * 70}\nschema={library!r} table={table!r}\n{'=' * 70}", flush=True)
            try:
                cols = list_columns(cur, library, table)
            except Exception as exc:
                print(f"  query failed: {exc}")
                continue
            if not cols:
                print(f"  table not found or no permissions on {library}.{table}")
                continue

            print(f"total columns: {len(cols)}")
            buckets = bucket_by_suffix(cols)
            for k in ("q", "y", "a", "12", "other"):
                items = buckets.get(k, [])
                print(f"  suffix {k!r}: {len(items)} columns")

            if (library, table) == ("comp", "fundq"):
                fundq_set = set(cols)
                print("\n--- audit: columns referenced by FEATURE_SPECS ---")
                missing_in_wrds: list[str] = []
                for spec in FEATURE_SPECS:
                    for src in spec.source_columns:
                        in_wrds = src in fundq_set
                        stem = stem_of(src)
                        siblings = sorted(
                            c
                            for c in cols
                            if c != src and (c.startswith(stem) or stem_of(c) == stem)
                        )
                        flag = "OK" if in_wrds else "MISSING"
                        print(
                            f"  [{flag:7s}] {spec.label[:48]:48s} "
                            f"src={src:10s} siblings={siblings[:6]}"
                        )
                        if not in_wrds:
                            missing_in_wrds.append(src)
                print(f"\n  missing-in-WRDS: {missing_in_wrds}")

                print("\n--- explicit YTD cash-flow check ---")
                ytd_check = [
                    "capxy", "dvy", "txbcofy", "fincfy", "fopoy", "ivncfy",
                    "ivacoy", "dltisy", "oancfy", "sstky", "sivy", "sppivy",
                    "aolochy", "aqcy", "ibcy", "iby", "epsfxy", "epspxy",
                ]
                for col in ytd_check:
                    in_wrds = col in fundq_set
                    stem = col[:-1]
                    q_sibling = stem + "q"
                    q_sibling_present = q_sibling in fundq_set
                    print(
                        f"  {col:10s}  in_fundq={in_wrds!s:5s}  "
                        f"q_sibling={q_sibling:11s} present={q_sibling_present!s}"
                    )
    finally:
        cur.close()
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
