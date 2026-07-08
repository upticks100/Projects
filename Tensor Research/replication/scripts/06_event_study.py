#!/usr/bin/env python
"""Stage 6 — event-study dataset build + multi-target analysis.

Usage:
    python scripts/06_event_study.py dataset [builder args]
    python scripts/06_event_study.py analyze [analyzer args]

`dataset` joins stage-3 prediction dumps with announcement dates, CRSP daily
returns, and OptionMetrics IV into per-event rows. `analyze` runs the
multi-target battery (returns, realized/idio vol, IV subsumption, straddle
long-short) with two-way clustered SEs and BH/BY multiple-testing discipline.
"""
import sys

import _bootstrap  # noqa: F401

STAGES = {
    "dataset": "src.analysis.event_study_dataset",
    "analyze": "src.analysis.event_study_multi",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        return 1
    what = sys.argv[1]
    sys.argv = [f"06_event_study.py {what}"] + sys.argv[2:]
    import importlib
    mod = importlib.import_module(STAGES[what])
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
