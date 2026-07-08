#!/usr/bin/env python
"""Stage 5 — rebuild the MFI (clean v2/40 tensor) and its exhibits.

Runs the full-universe 1990Q1-2024Q4 Tucker decomposition, writes the MFI v2
series, MFI<->FCIX correlations, and the L_n/I_n independence tests to
REPL_RESULTS_DIR/mfi_v2/. Pass --figures to also regenerate the paper's MFI
and cross-correlation figures from the rebuilt series.
"""
import sys

import _bootstrap  # noqa: F401


def main() -> int:
    make_figures = "--figures" in sys.argv
    if make_figures:
        sys.argv.remove("--figures")

    from src.mfi.build_mfi import main as build_main
    build_main()

    if make_figures:
        from src.mfi.figures import main as figures_main
        figures_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
