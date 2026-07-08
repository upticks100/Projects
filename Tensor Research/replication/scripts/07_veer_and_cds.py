#!/usr/bin/env python
"""Stage 7 — veer anomaly tests + CDS economic translation.

Usage:
    python scripts/07_veer_and_cds.py veer [args]   # studentized forecast-
                                                    # surprise (veer) battery:
                                                    # H1 drift_cashflow->dDD,
                                                    # H2 veers->dIV, clustering
    python scripts/07_veer_and_cds.py cds  [args]   # translate H1 into credit-
                                                    # spread units on Markit CDS
"""
import sys

import _bootstrap  # noqa: F401

STAGES = {
    "veer": "src.analysis.veer",
    "cds": "src.analysis.cds_translation",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        return 1
    what = sys.argv[1]
    sys.argv = [f"07_veer_and_cds.py {what}"] + sys.argv[2:]
    import importlib
    mod = importlib.import_module(STAGES[what])
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
