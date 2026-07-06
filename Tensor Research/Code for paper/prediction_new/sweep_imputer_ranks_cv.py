"""Cross-validate Tucker ranks for prediction-window imputation.

The imputer's job is to fill missing X entries before CP regression. Since true
missing cells have no labels, we hide observed entries, fit mask-aware Tucker on
the remaining entries, and score only the hidden cells.

This version stratifies the hidden entries by feature within each rolling
window, so sparse features are represented in the validation score. It uses
SVD-initialized Tucker because the full MFI audit showed SVD init is stable and
random init can be badly under-optimized for Tucker.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR.parent))

from build_prediction_caches import build_raw_tensor, load_filtered_panel  # noqa: E402
from prediction_config import LOOKBACKS, SEED  # noqa: E402

SWEEP_DIR = ROOT_DIR / "sweep_results"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1", default="1,2,3,4,5,8,10")
    parser.add_argument("--r2", default="1,2,3,4,5,8,10")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--holdout-frac", type=float, default=0.10)
    parser.add_argument("--dev-frac", type=float, default=0.80)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=SWEEP_DIR / "imputer_rank_cv_stratified.csv",
    )
    return parser.parse_args()


def stratified_holdout_mask(
    observed: np.ndarray,
    seed: int,
    holdout_frac: float,
) -> np.ndarray:
    """Sample held-out observed entries separately for each feature slice."""
    rng = np.random.default_rng(seed)
    holdout = np.zeros(observed.shape, dtype=bool)
    n_features = observed.shape[1]
    for j in range(n_features):
        idx = np.flatnonzero(observed[:, j, :].ravel())
        if idx.size < 5:
            continue
        n_hold = max(1, int(round(holdout_frac * idx.size)))
        chosen = rng.choice(idx, size=n_hold, replace=False)
        feature_mask = holdout[:, j, :].copy().ravel()
        feature_mask[chosen] = True
        holdout[:, j, :] = feature_mask.reshape(holdout[:, j, :].shape)
    return holdout


def evaluate_window(
    tensor: np.ndarray,
    t: int,
    L: int,
    r_firm: int,
    r_feat: int,
    seed: int,
    holdout_frac: float,
    max_iter: int,
    tol: float,
) -> dict[str, float | int | str] | None:
    tl.set_backend("numpy")
    X_raw = tensor[:, :, t : t + L]
    observed = np.isfinite(X_raw)
    if observed.sum() < 50:
        return None

    holdout = stratified_holdout_mask(
        observed=observed,
        seed=SEED + 10_000 * seed + t,
        holdout_frac=holdout_frac,
    )
    if holdout.sum() < 10:
        return None

    train_mask = observed & ~holdout
    train_values = X_raw[train_mask]
    rms = float(np.sqrt(np.mean(train_values**2))) + 1e-8 if train_values.size else 1.0
    X_scaled = X_raw / rms
    X_filled = np.nan_to_num(X_scaled, nan=0.0)

    rank = [
        min(r_firm, X_raw.shape[0]),
        min(r_feat, X_raw.shape[1]),
        min(L, X_raw.shape[2]),
    ]

    try:
        core, factors = tucker(
            X_filled,
            rank=rank,
            mask=train_mask.astype(np.float64),
            n_iter_max=max_iter,
            tol=tol,
            init="svd",
            random_state=SEED,
            verbose=False,
        )
        X_hat = tucker_to_tensor((core, factors))
    except Exception as exc:  # noqa: BLE001
        return {
            "window": t,
            "L": L,
            "r_firm": r_firm,
            "r_feat": r_feat,
            "seed": seed,
            "holdout_n": int(holdout.sum()),
            "holdout_err": np.nan,
            "train_err": np.nan,
            "status": f"{type(exc).__name__}: {exc}",
        }

    holdout_err = float(
        np.linalg.norm((X_hat - X_filled) * holdout)
        / (np.linalg.norm(X_filled * holdout) + 1e-12)
    )
    train_err = float(
        np.linalg.norm((X_hat - X_filled) * train_mask)
        / (np.linalg.norm(X_filled * train_mask) + 1e-12)
    )
    return {
        "window": t,
        "L": L,
        "r_firm": r_firm,
        "r_feat": r_feat,
        "seed": seed,
        "holdout_n": int(holdout.sum()),
        "holdout_err": holdout_err,
        "train_err": train_err,
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    r1_grid = parse_csv_ints(args.r1)
    r2_grid = parse_csv_ints(args.r2)
    seeds = parse_csv_ints(args.seeds)

    print(
        f"Imputer rank CV: r1={r1_grid}, r2={r2_grid}, seeds={seeds}, "
        f"holdout_frac={args.holdout_frac}, max_iter={args.max_iter}, "
        f"tol={args.tol}, n_jobs={args.n_jobs}",
        flush=True,
    )

    df, feature_names, firms, quarters = load_filtered_panel()
    raw_tensor = build_raw_tensor(df, feature_names, firms, quarters)

    rows: list[dict[str, float | int | str]] = []
    for L in LOOKBACKS:
        n_wins = raw_tensor.shape[2] - L
        n_dev = int(args.dev_frac * n_wins)
        jobs = [
            (t, r1, r2, seed)
            for r1 in r1_grid
            for r2 in r2_grid
            for seed in seeds
            for t in range(n_dev)
        ]
        print(f"\n=== L={L}: {len(jobs)} window/rank/seed jobs ===", flush=True)
        start = time.perf_counter()
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(evaluate_window)(
                raw_tensor,
                t,
                L,
                r1,
                r2,
                seed,
                args.holdout_frac,
                args.max_iter,
                args.tol,
            )
            for t, r1, r2, seed in jobs
        )
        rows.extend([row for row in results if row is not None])
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"L={L} done in {time.perf_counter() - start:.1f}s", flush=True)

    full = pd.DataFrame(rows)
    full.to_csv(args.output, index=False)
    print(f"\nSaved CV rows: {args.output}", flush=True)

    ok = full[full["status"] == "ok"].copy()
    summary = (
        ok.groupby(["L", "r_firm", "r_feat"])
        .agg(
            holdout_mean=("holdout_err", "mean"),
            holdout_median=("holdout_err", "median"),
            holdout_std=("holdout_err", "std"),
            train_mean=("train_err", "mean"),
            n_windows=("window", "count"),
        )
        .reset_index()
        .sort_values(["L", "holdout_mean", "r_firm", "r_feat"])
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}", flush=True)

    print("\nBest by L:", flush=True)
    for L, group in summary.groupby("L"):
        best = group.nsmallest(10, "holdout_mean")
        print(f"\nL={L}", flush=True)
        print(best.to_string(index=False), flush=True)

    print("\nParsimony picks within one standard error of best:", flush=True)
    for L, group in summary.groupby("L"):
        best = group.iloc[0]
        best_se = best["holdout_std"] / np.sqrt(best["n_windows"])
        threshold = best["holdout_mean"] + best_se
        candidates = group[group["holdout_mean"] <= threshold].copy()
        candidates["complexity"] = candidates["r_firm"] * candidates["r_feat"]
        pick = candidates.sort_values(["complexity", "holdout_mean"]).iloc[0]
        print(
            f"L={L}: ranks=[{int(pick['r_firm'])}, {int(pick['r_feat'])}, {L}] "
            f"holdout_mean={pick['holdout_mean']:.6f} "
            f"(best={best['holdout_mean']:.6f}, one_se={best_se:.6f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
