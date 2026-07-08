#!/usr/bin/env python
"""Stage 1 — WRDS data pulls (requires WRDS credentials; one Duo push each).

Usage:
    python scripts/01_fetch_data.py <what> [args passed to the fetcher]

where <what> is one of:
    fundamentals    comp.fundq quarterly fundamentals (MFI 1990-2024 window)
    universe        gvkey<->permno links, CRSP daily, market index,
                    OptionMetrics IV for the full ~499-firm universe
    crsp            CRSP daily + links for the 50-firm cache universe
    optionmetrics   ATM 30/60d implied vol for the 50-firm universe
    cds             Markit 5Y single-name CDS for the 499 universe
    hy              curated high-yield/crossover universe from Markit names

Each fetcher has its own --help. All are append-only: they write new files and
never overwrite the original pulls.
"""
import sys

import _bootstrap  # noqa: F401

FETCHERS = {
    "fundamentals": "src.data.fetch_fundamentals",
    "universe": "src.data.fetch_universe",
    "crsp": "src.data.fetch_crsp",
    "optionmetrics": "src.data.fetch_optionmetrics",
    "cds": "src.data.fetch_cds",
    "hy": "src.data.hy_universe",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") \
            or sys.argv[1] not in FETCHERS:
        print(__doc__)
        return 0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1
    what = sys.argv[1]
    sys.argv = [f"01_fetch_data.py {what}"] + sys.argv[2:]
    import importlib
    mod = importlib.import_module(FETCHERS[what])
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
