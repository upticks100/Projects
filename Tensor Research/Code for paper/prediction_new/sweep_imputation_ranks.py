"""prediction_new/sweep_imputation_ranks.py

Re-derive Tucker imputation ranks for the prediction tensor caches on the v2
fundamentals + top-50-by-mkvaltq universe + 40 features.

For every (L, r_firm, r_feat) combination we run masked Tucker per rolling
window with a 10% held-out entry split, then evaluate relative error on the
held-out entries only. We sweep over the dev portion (first 80% of windows).

Time-mode rank is pinned at L (full rank) since L ∈ {2, 4} is tiny.

Outputs:
    sweep_results/imputation_rank_sweep.csv  (full grid)
    prints recommended ranks per L

Usage:
    python sweep_imputation_ranks.py
    python sweep_imputation_ranks.py --r1 5,10,15,20,25,30,40,50 --r2 3,5,8,12,16,20,30,40
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import tensorly as tl  # noqa: E402
from tensorly.decomposition import tucker  # noqa: E402
from tensorly.tucker_tensor import tucker_to_tensor  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR.parent))

from prediction_config import (  # noqa: E402
    LOOKBACKS, SEED, START_DATE, END_DATE,
)
from build_prediction_caches import load_filtered_panel, build_raw_tensor  # noqa: E402

SWEEP_DIR = ROOT_DIR / "sweep_results"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

DEV_FRACTION = 0.80
HOLDOUT_FRAC = 0.10
N_ITER_MAX = 30
TOL = 1e-3


def evaluate_window(t: int, tensor: np.ndarray, L: int,
                    r1: int, r2: int, holdout_seed: int):
    """Return (windowed_train_err, windowed_holdout_err, n_holdout, ok)."""
    tl.set_backend("numpy")
    X_raw = tensor[:, :, t : t + L]  # shape (firms, features, L)
    obs = ~np.isnan(X_raw)
    if obs.sum() < 50:  # too sparse to evaluate
        return None

    # rms-scale to keep magnitudes comparable (mirrors build script)
    rms = float(np.sqrt(np.mean(X_raw[obs] ** 2))) + 1e-8
    X_scaled = X_raw / rms
    X_filled = np.nan_to_num(X_scaled, nan=0.0)

    # build held-out mask: 10% of observed entries
    rng = np.random.default_rng(holdout_seed + t)
    obs_idx = np.flatnonzero(obs.ravel())
    n_hold = max(5, int(round(HOLDOUT_FRAC * obs_idx.size)))
    hold_pick = rng.choice(obs_idx, size=n_hold, replace=False)
    hold_mask = np.zeros(X_raw.size, dtype=bool)
    hold_mask[hold_pick] = True
    hold_mask = hold_mask.reshape(X_raw.shape)
    train_mask_bool = obs & ~hold_mask
    train_mask = train_mask_bool.astype(int)

    if train_mask.sum() < 30:
        return None

    rank = [min(r1, X_raw.shape[0]),
            min(r2, X_raw.shape[1]),
            min(L, X_raw.shape[2])]

    try:
        core, factors = tucker(
            X_filled, rank=rank, mask=train_mask,
            n_iter_max=N_ITER_MAX, tol=TOL,
            init="random", random_state=int(SEED + t),
            verbose=0,
        )
        X_hat = tucker_to_tensor((core, factors))
    except Exception:
        return None

    # in-sample (train mask) error
    tr_num = float(np.linalg.norm((X_hat - X_filled) * train_mask_bool))
    tr_den = float(np.linalg.norm(X_filled * train_mask_bool)) + 1e-12
    tr_err = tr_num / tr_den

    # held-out error: only on the masked entries
    ho_num = float(np.linalg.norm((X_hat - X_filled) * hold_mask))
    ho_den = float(np.linalg.norm(X_filled * hold_mask)) + 1e-12
    ho_err = ho_num / ho_den

    return tr_err, ho_err, int(hold_mask.sum())


def sweep_for_L(tensor: np.ndarray, L: int, r1_grid: list[int],
                r2_grid: list[int], n_jobs: int) -> pd.DataFrame:
    n_time = tensor.shape[2]
    n_wins = n_time - L
    n_dev = int(DEV_FRACTION * n_wins)
    print(f"\n=== L={L}: {n_dev} dev windows (of {n_wins} total) ===")

    rows = []
    for r1 in r1_grid:
        for r2 in r2_grid:
            t0 = time.perf_counter()
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(evaluate_window)(t, tensor, L, r1, r2, SEED)
                for t in range(n_dev)
            )
            valid = [r for r in results if r is not None]
            if not valid:
                continue
            tr = np.array([r[0] for r in valid])
            ho = np.array([r[1] for r in valid])
            n_h = np.array([r[2] for r in valid])
            elapsed = time.perf_counter() - t0
            rows.append({
                "L": L, "r_firm": r1, "r_feat": r2,
                "n_windows": len(valid),
                "train_err_mean": float(tr.mean()),
                "train_err_med": float(np.median(tr)),
                "holdout_err_mean": float(ho.mean()),
                "holdout_err_med": float(np.median(ho)),
                "holdout_err_std": float(ho.std()),
                "n_holdout_avg": float(n_h.mean()),
                "elapsed_s": elapsed,
            })
            print(f"  r1={r1:>2} r2={r2:>2}  train={tr.mean():.4f}  "
                  f"holdout={ho.mean():.4f} (std={ho.std():.4f})  "
                  f"n_wins={len(valid)}/{n_dev}  {elapsed:.1f}s")
    return pd.DataFrame(rows)


def recommend(df_L: pd.DataFrame, L: int) -> tuple[int, int]:
    sub = df_L.copy()
    best = sub.sort_values("holdout_err_mean").iloc[0]
    print(f"\n  optimum holdout err for L={L}: r_firm={int(best['r_firm'])}, "
          f"r_feat={int(best['r_feat'])}, mean={best['holdout_err_mean']:.4f}")

    # Apply 1-sigma elbow rule: smallest model whose mean is within
    # one std of the absolute optimum (parsimony preference).
    best_err = best["holdout_err_mean"]
    best_std = best["holdout_err_std"]
    threshold = best_err + best_std
    candidates = sub[sub["holdout_err_mean"] <= threshold].copy()
    candidates["complexity"] = candidates["r_firm"] * candidates["r_feat"]
    parsimony = candidates.sort_values(["complexity", "holdout_err_mean"]).iloc[0]
    print(f"  parsimonious pick (within 1σ of best, smallest r_firm×r_feat):")
    print(f"    r_firm={int(parsimony['r_firm'])}, r_feat={int(parsimony['r_feat'])}, "
          f"holdout={parsimony['holdout_err_mean']:.4f}")
    return int(parsimony["r_firm"]), int(parsimony["r_feat"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r1", type=str, default="5,10,15,20,25,30,40,50")
    p.add_argument("--r2", type=str, default="3,5,8,12,16,20,30,40")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    r1_grid = [int(x) for x in args.r1.split(",")]
    r2_grid = [int(x) for x in args.r2.split(",")]
    print(f"sweep grid: r_firm ∈ {r1_grid}, r_feat ∈ {r2_grid}")
    print(f"dev fraction: {DEV_FRACTION}, hold-out frac per window: {HOLDOUT_FRAC}")
    print(f"START..END: {START_DATE} .. {END_DATE}")

    df, feature_names, firms, quarters = load_filtered_panel()
    raw_tensor = build_raw_tensor(df, feature_names, firms, quarters)
    print(f"\nraw tensor: {raw_tensor.shape}, observed density: "
          f"{(~np.isnan(raw_tensor)).mean()*100:.2f}%")

    all_rows = []
    rec = {}
    for L in LOOKBACKS:
        df_L = sweep_for_L(raw_tensor, L, r1_grid, r2_grid, n_jobs=args.n_jobs)
        all_rows.append(df_L)
        rec[L] = recommend(df_L, L)

    full = pd.concat(all_rows, ignore_index=True)
    out_csv = SWEEP_DIR / "imputation_rank_sweep.csv"
    full.to_csv(out_csv, index=False)
    print(f"\nfull sweep saved: {out_csv}")

    print("\n=== RECOMMENDED IMPUTATION_RANKS ===")
    print("update prediction_config.py:")
    print("IMPUTATION_RANKS: dict[int, list[int]] = {")
    for L, (r1, r2) in rec.items():
        print(f"    {L}: [{r1}, {r2}, {L}],")
    print("}")


if __name__ == "__main__":
    main()
