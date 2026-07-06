"""Refit one (objective, mode, L, rank) model and dump test predictions.

Same refit logic as evaluate_per_feature.py / evaluate_top_trials_test.py,
but saves the full (W_test, F, K) baseline + ensemble prediction tensors
to a joblib pickle so downstream event-study analyses can reuse them
without re-running the slow CP fit.

Outputs `<holdout_dir>/predictions_<objective>_L<L>_rank<r>.pkl` with
keys:

  predicted_ensemble : (W_test, F, K) ndarray
  predicted_base     : (W_test, F, K) ndarray  (Ridge or FE depending on cell)
  realized           : (W_test, F, K) ndarray
  mask               : (W_test, F, K) ndarray (1 = observed)
  firm_gvkeys        : list[str]
  feature_names      : list[str]
  quarters_test      : list[str]   target calendar quarter PER TEST WINDOW
                                   (i.e. quarters[split_idx + L : ...]).
                                   Use this to join with rdq.
  input_quarters     : list[list[str]]  one list per test window of the L
                                        calendar quarters that fed into
                                        X[w]. For audit / look-ahead checks.
  trial_meta         : dict (the row from aggregate_summary.csv)

Window indexing convention (matches build_prediction_caches.py:104):
  cache window w  →  X[w] = tensor[:, :, w : w+L]
                     Y[w] = tensor[:, :, w + L]
So for test window w_test (= split_idx + w within the cache),
  target quarter      = quarters[split_idx + L + w_test]
  input quarters used = quarters[split_idx + w_test : split_idx + w_test + L]

The earlier version of this script wrote quarters[split_idx + w_test]
which is L quarters too early — that is the audit "off-by-L" bug.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorly as tl  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PARENT_DIR))

from prediction_config import SEED, meta_path  # noqa: E402
from evaluate_per_feature import load_split, refit_and_predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("holdout_dir", type=Path)
    p.add_argument("--objective", required=True)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--rank-order", type=int, default=1)
    p.add_argument("--out", type=Path, default=None,
                   help="Override output path (default: <holdout_dir>/predictions_<obj>_L<L>_rank<r>.pkl).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    holdout_dir = args.holdout_dir.resolve()
    summary_csv = holdout_dir / "aggregate_summary.csv"
    if not summary_csv.exists():
        sys.exit(f"missing {summary_csv}")
    summary = pd.read_csv(summary_csv)
    row = summary[
        (summary["objective"] == args.objective) &
        (summary["L"] == args.L) &
        (summary["rank_order"] == args.rank_order)
    ]
    if row.empty:
        sys.exit(f"no trial for objective={args.objective} L={args.L} rank={args.rank_order}")
    trial = row.iloc[0].to_dict()
    print(f"trial: {trial}")

    meta = joblib.load(meta_path())
    feature_names = list(meta["feature_names"])
    firm_gvkeys = list(meta["firms"])
    quarters = list(meta["quarters"])

    tl.set_backend("numpy")
    np.random.seed(SEED)

    L = int(trial["L"])
    X_cv, Y_cv, M_cv, X_test, Y_test, M_test, split_idx = load_split(
        str(trial["mode"]), L,
    )
    base_test, ensemble_test = refit_and_predict(
        trial, X_cv, Y_cv, M_cv, X_test, Y_test,
    )

    # Quarter labels for the PREDICTION TARGET of each test window.
    # See module docstring: cache window w → Y[w] = quarters[w + L].
    W_test = int(Y_test.shape[0])
    target_start = split_idx + L
    quarters_test = list(map(str, quarters[target_start: target_start + W_test]))
    if len(quarters_test) != W_test:
        raise SystemExit(
            f"quarter label window misalignment: have {len(quarters_test)} "
            f"labels for {W_test} test windows (split_idx={split_idx}, L={L}, "
            f"len(quarters)={len(quarters)})"
        )

    # For each test window, list the L calendar quarters that fed into X[w].
    input_quarters = [
        list(map(str, quarters[split_idx + w: split_idx + w + L]))
        for w in range(W_test)
    ]

    out = {
        "predicted_ensemble": np.asarray(ensemble_test),
        "predicted_base":     np.asarray(base_test),
        "realized":           np.asarray(Y_test),
        "mask":               np.asarray(M_test),
        "firm_gvkeys":        firm_gvkeys,
        "feature_names":      feature_names,
        "quarters_test":      quarters_test,
        "input_quarters":     input_quarters,
        "split_index":        int(split_idx),
        "L":                  L,
        "trial_meta":         trial,
    }

    default_name = (
        f"predictions_{args.objective}_L{args.L}_rank{args.rank_order}.pkl"
    )
    out_path = args.out if args.out is not None else (holdout_dir / default_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path, compress=3)
    print(f"\nWrote {out_path}  shape={out['realized'].shape}  "
          f"size={out_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
