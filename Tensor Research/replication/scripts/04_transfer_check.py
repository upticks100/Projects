#!/usr/bin/env python
"""Stage 4 — universe-transfer gate.

Recomputes pooled mask-aware R2 deltas from the stage-3 prediction dumps and
compares them to the locked 50-firm reference. Exit 0 = PASS (>=1 cell with a
positive CP delta), 2 = FAIL (all four collapse; downstream veer tests must
not run), 1 = missing dumps.
"""
import sys

import _bootstrap  # noqa: F401

from src.model.transfer_check import main

if __name__ == "__main__":
    sys.exit(main() or 0)
