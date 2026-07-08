#!/usr/bin/env python
"""Stage 3 — refit the four locked cells and dump test predictions.

Hyperparameters come from locked_cells.csv (no Optuna search needed). For each
cell (objective x L) this refits baseline + gamma*CP on the dev block and
writes per-cell prediction pickles + a summary CSV to REPL_RESULTS_DIR.

Set REPL_CP_LOWMEM=1 to use the exact-math low-memory CP fitter (required at
~499 firms; validated equivalent in tests/test_cp_lowmem_equiv.py).
"""
import sys

import _bootstrap  # noqa: F401

from src.model.refit_and_dump import main

if __name__ == "__main__":
    sys.exit(main() or 0)
