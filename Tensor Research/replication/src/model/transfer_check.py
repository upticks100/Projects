"""Transfer-check gate for wider-universe scale-ups (locked hyperparameters).

Loads each prediction dump from a scale-up run (e.g. 499 firms or the HY
universe), recomputes pooled mask-aware test R2 (base / ensemble / delta) with
the exact evaluate_model convention, and compares against the 50-firm values
in the reference locked-cells CSV.

Gate semantics (declared ex ante, research log 2026-07-06): if the CP ensemble
delta collapses (<= 0) in ALL FOUR cells on the wider universe, the declared
veer test must NOT run — that outcome is itself a finding ("CP structure is a
mega-cap phenomenon"). Exit code 0 = PASS (>=1 cell with positive delta),
2 = FAIL, 1 = missing dumps.

Run from the replication root:
    python -m src.model.transfer_check <dump_dir> --reference-summary locked_cells.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np  # noqa: F401  (kept: parity with original module)
import pandas as pd

from src.model.baselines import evaluate_model

CELLS = (("ridge_delta_v3", 2), ("ridge_delta_v3", 4),
         ("residual_delta_v3", 2), ("residual_delta_v3", 4))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("holdout_499_dir", type=Path,
                   help="directory holding predictions_*_rank1.pkl dumps")
    p.add_argument("--reference-summary", type=Path,
                   default=None,
                   help="locked_cells.csv or a 50-firm aggregate_summary.csv "
                        "(default: the package locked_cells.csv)")
    args = p.parse_args()

    import config
    ref_csv = args.reference_summary or config.LOCKED_CELLS_FILE
    ref = pd.read_csv(ref_csv)
    if "rank_order" not in ref.columns:  # locked_cells.csv keeps only rank-1 rows
        ref = ref.assign(rank_order=1)
    rows = []
    missing = []
    for obj, L in CELLS:
        pkl = args.holdout_499_dir / f"predictions_{obj}_L{L}_rank1.pkl"
        if not pkl.exists():
            missing.append(pkl.name)
            continue
        d = joblib.load(pkl)
        Y, M = d["realized"], d["mask"]
        base = float(evaluate_model(Y, d["predicted_base"], M))
        ens = float(evaluate_model(Y, d["predicted_ensemble"], M))
        r = ref[(ref["objective"] == obj) & (ref["L"] == L) &
                (ref["rank_order"] == 1)].iloc[0]
        rows.append({
            "objective": obj, "L": L,
            "n_firms": len(d["firm_gvkeys"]),
            "test_windows": int(Y.shape[0]),
            "base_r2_499": base, "ensemble_r2_499": ens,
            "delta_499": ens - base,
            "base_r2_50": float(r["base_test_r2"]),
            "ensemble_r2_50": float(r["ensemble_test_r2"]),
            "delta_50": float(r["test_delta"]),
        })

    if missing:
        print(f"MISSING dumps: {missing}")
        return 1

    df = pd.DataFrame(rows)
    out = args.holdout_499_dir / "transfer_check_499.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.5f}"
          if isinstance(v, float) else str(v)))

    n_pos = int((df["delta_499"] > 0).sum())
    verdict = "PASS" if n_pos >= 1 else "FAIL"
    (args.holdout_499_dir / "transfer_check_verdict.txt").write_text(
        f"{verdict}: {n_pos}/4 cells with positive ensemble delta at 499 "
        f"firms (50-firm deltas: "
        f"{', '.join(f'{d:+.4f}' for d in df['delta_50'])})\n")
    print(f"\nTRANSFER CHECK: {verdict} ({n_pos}/4 cells delta>0) -> {out}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
