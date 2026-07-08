#!/usr/bin/env python
"""Stage 2 — build rolling tensor caches (mask-aware Tucker imputation).

Reads the fundamentals CSV configured in config.py (env: REPL_FUNDAMENTALS,
REPL_TOP_N, REPL_END_DATE), writes per-(mode, L) pickles to REPL_CACHE_DIR.
"""
import sys

import _bootstrap  # noqa: F401

from src.tensors.build_caches import main

if __name__ == "__main__":
    sys.exit(main() or 0)
