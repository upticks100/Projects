"""Per-feature R² breakdown for the rank-1 trial in each cell.

For each (objective, mode, L) found in the holdout `aggregate_summary.csv`,
picks the rank-1 trial and:

1. Refits the model on the full dev set using the trial's hyperparameters
   (mirroring evaluate_top_trials_test.py exactly).
2. Predicts on the held-out test set.
3. Computes per-feature R² on test for the baseline (Ridge or FE) and the
   ensemble (baseline + GAMMA · CP_residual).
4. Emits `<holdout_dir>/per_feature_r2.csv` with one row per
   (objective, mode, L, feature_index, feature_name).

Per-feature R² uses the same numpy `evaluate_model` helper as the
pooled metric, restricted to mask cells of that single feature. Features
with too few observed test cells are reported as NaN.

This is a diagnostic, not a model — it tells us *where* CP's gain
concentrates, not how to forecast.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorly as tl  # noqa: E402
from tensorly.regression.cp_regression import CPRegressor  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PARENT_DIR))

from prediction_config import N_ITER_MAX, SEED, cache_path, meta_path  # noqa: E402
from CP_struct_test_new import evaluate_model, firm_feature_means  # noqa: E402
from evaluate_top_trials_test import (  # noqa: E402
    BOOSTER_OBJECTIVES,
    _compute_ridge_oof_drop_fallback,
    _per_feature_x_scale,
    get_min_valid_entries,
    load_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "holdout_dir",
        type=Path,
        help="Path to results/v3_holdout_<ts>/ directory with aggregate_summary.csv.",
    )
    parser.add_argument(
        "--ranks",
        default="1",
        help="Comma-separated rank_order values to evaluate (default: 1).",
    )
    parser.add_argument(
        "--objective",
        default=None,
        help="Filter to a single objective name (default: all in summary).",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Filter to a single lookback (default: all in summary).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Override output CSV path (default: <holdout_dir>/per_feature_r2.csv).",
    )
    return parser.parse_args()


def refit_and_predict(trial: dict, X_cv, Y_cv, M_cv, X_test, Y_test):
    """Refit one trial on full dev set and return (base_test, ensemble_test).

    Mirrors evaluate_top_trials_test.evaluate_trial exactly, but returns
    raw prediction tensors so per-feature R² can be computed downstream.
    """
    objective_name = trial["objective"]

    if objective_name in BOOSTER_OBJECTIVES:
        ridge_oof, ridge_test, ridge_valid = _compute_ridge_oof_drop_fallback(
            X_cv, Y_cv, M_cv, X_test,
        )
        if ridge_valid.sum() == 0:
            raise RuntimeError(f"No honest Ridge OOF rows for {trial}")
        X_fit_raw = X_cv[ridge_valid]
        Y_fit = Y_cv[ridge_valid]
        M_fit = M_cv[ridge_valid]
        base_fit = ridge_oof[ridge_valid]
        base_test = ridge_test
    else:
        mu_ff = firm_feature_means(Y_cv, M_cv)
        X_fit_raw = X_cv
        Y_fit = Y_cv
        M_fit = M_cv
        base_fit = np.broadcast_to(mu_ff[None, :, :], Y_cv.shape)
        base_test = np.broadcast_to(mu_ff[None, :, :], Y_test.shape)

    if trial["FEATURE_X_SCALE"]:
        feat_x_scale = _per_feature_x_scale(X_fit_raw)
        X_fit = X_fit_raw / feat_x_scale[None, None, :, None]
        X_test_in = X_test / feat_x_scale[None, None, :, None]
    else:
        X_fit = X_fit_raw
        X_test_in = X_test

    Y_cent = (Y_fit - base_fit) * M_fit

    if trial["FEATURE_TARGET_SCALE"]:
        feat_sse = np.sum((Y_cent ** 2) * M_fit, axis=(0, 1))
        feat_n = np.sum(M_fit, axis=(0, 1))
        feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
        feat_scale = np.where(
            np.isfinite(feat_scale) & (feat_scale > 1e-8),
            feat_scale,
            1.0,
        ).astype(Y_fit.dtype)
    else:
        feat_scale = np.ones(Y_fit.shape[2], dtype=Y_fit.dtype)

    Y_scaled = Y_cent / feat_scale[None, None, :]

    if trial["USE_RMS_SCALING"]:
        y_obs = Y_scaled[M_fit > 0]
        if y_obs.size <= get_min_valid_entries(M_fit):
            raise RuntimeError("too few observed cp target cells")
        y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
        Y_target = Y_scaled / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target = Y_scaled

    # PRED_CP_LOWMEM=1: sample-blocked normal equations (identical math,
    # verified bit-equal by test_cp_lowmem_equiv.py). Needed at 498 firms
    # where stock CPRegressor's design matrix (~65 GB at rank 13) OOMs the
    # 62 GB lab hosts.
    if os.environ.get("PRED_CP_LOWMEM") == "1":
        from cp_regressor_lowmem import LowMemCPRegressor
        cp_cls, cp_extra = LowMemCPRegressor, {
            "block_size": int(os.environ.get("PRED_CP_LOWMEM_BLOCK", "4")),
        }
    else:
        cp_cls, cp_extra = CPRegressor, {}
    cp = cp_cls(
        weight_rank=int(trial["RANK_REGRESS"]),
        reg_W=float(trial["REG_W"]),
        n_iter_max=N_ITER_MAX,
        random_state=SEED,
        **cp_extra,
    )
    cp.fit(X_fit, Y_target)
    cp_residual_test = cp.predict(X_test_in) * y_rms * feat_scale[None, None, :]
    ensemble_test = base_test + float(trial["GAMMA"]) * cp_residual_test

    return np.asarray(base_test), np.asarray(ensemble_test)


def per_feature_r2(Y_true: np.ndarray, Y_pred: np.ndarray, M: np.ndarray,
                   min_cells: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature R² on observed cells, plus per-feature observed counts."""
    n_feat = Y_true.shape[2]
    r2 = np.full(n_feat, np.nan)
    counts = np.zeros(n_feat, dtype=int)
    for f in range(n_feat):
        m = (M[:, :, f] > 0)
        counts[f] = int(m.sum())
        if counts[f] < min_cells:
            continue
        Y_slice = Y_true[:, :, f][m]
        P_slice = Y_pred[:, :, f][m]
        ss_res = float(np.sum((Y_slice - P_slice) ** 2))
        ss_tot = float(np.sum((Y_slice - np.mean(Y_slice)) ** 2))
        r2[f] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return r2, counts


def main() -> None:
    args = parse_args()
    holdout_dir = args.holdout_dir.resolve()
    summary_csv = holdout_dir / "aggregate_summary.csv"
    if not summary_csv.exists():
        sys.exit(f"missing aggregate_summary.csv in {holdout_dir}")

    ranks_to_run = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    summary = pd.read_csv(summary_csv)
    summary = summary[summary["rank_order"].isin(ranks_to_run)].copy()
    if args.objective is not None:
        summary = summary[summary["objective"] == args.objective].copy()
    if args.L is not None:
        summary = summary[summary["L"] == args.L].copy()
    if summary.empty:
        sys.exit(
            f"no trials matching ranks={ranks_to_run} objective={args.objective} L={args.L}"
        )

    meta = joblib.load(meta_path())
    feature_names = list(meta["feature_names"])
    tl.set_backend("numpy")
    np.random.seed(SEED)

    print(f"Evaluating {len(summary)} trials from {summary_csv.name}")

    rows = []
    for _, trial in summary.iterrows():
        cell = f"{trial['objective']} {trial['mode']} L={int(trial['L'])} rank{int(trial['rank_order'])} trial{int(trial['trial_number'])}"
        print(f"\n--- {cell} ---")
        t0 = time.perf_counter()

        X_cv, Y_cv, M_cv, X_test, Y_test, M_test, split_idx = load_split(
            str(trial["mode"]), int(trial["L"])
        )
        base_test, ensemble_test = refit_and_predict(
            trial.to_dict(), X_cv, Y_cv, M_cv, X_test, Y_test,
        )

        base_r2 = per_feature_r2(Y_test, base_test, M_test)
        ens_r2 = per_feature_r2(Y_test, ensemble_test, M_test)
        base_r2_arr, base_counts = base_r2
        ens_r2_arr, _ = ens_r2

        for f in range(Y_test.shape[2]):
            rows.append(
                {
                    "objective": trial["objective"],
                    "mode": trial["mode"],
                    "L": int(trial["L"]),
                    "rank_order": int(trial["rank_order"]),
                    "trial_number": int(trial["trial_number"]),
                    "feature_index": f,
                    "feature_name": feature_names[f] if f < len(feature_names) else f"feat_{f}",
                    "test_cells": int(base_counts[f]),
                    "base_r2": float(base_r2_arr[f]),
                    "ensemble_r2": float(ens_r2_arr[f]),
                    "delta": float(ens_r2_arr[f] - base_r2_arr[f]),
                }
            )

        elapsed = time.perf_counter() - t0
        baseline_pooled = float(evaluate_model(Y_test, base_test, M_test) or np.nan)
        ensemble_pooled = float(evaluate_model(Y_test, ensemble_test, M_test) or np.nan)
        print(
            f"  pooled base_R2={baseline_pooled:.5f} ensemble_R2={ensemble_pooled:.5f} "
            f"pooled_delta={ensemble_pooled - baseline_pooled:+.5f}  "
            f"per_feat n_eval={np.isfinite(base_r2_arr).sum()}/{Y_test.shape[2]}  "
            f"elapsed={elapsed:.1f}s"
        )

    out = pd.DataFrame(rows)
    out_csv = args.out_csv if args.out_csv is not None else (holdout_dir / "per_feature_r2.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Wrote {out_csv}")

    print("\n=== Top 10 features by booster delta (ridge_delta_v3) ===")
    booster = out[out["objective"] == "ridge_delta_v3"].copy()
    if not booster.empty:
        booster_top = booster.sort_values("delta", ascending=False).head(15)
        print(booster_top[
            ["mode", "L", "feature_name", "base_r2", "ensemble_r2", "delta", "test_cells"]
        ].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

        print("\n=== Bottom 5 features by booster delta (where CP HURTS) ===")
        booster_bot = booster.sort_values("delta").head(5)
        print(booster_bot[
            ["mode", "L", "feature_name", "base_r2", "ensemble_r2", "delta", "test_cells"]
        ].to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

        print("\n=== Win/loss count (booster) ===")
        for (mode, L), grp in booster.groupby(["mode", "L"]):
            wins = int((grp["delta"] > 0).sum())
            losses = int((grp["delta"] < 0).sum())
            ties = int((grp["delta"] == 0).sum())
            evaluable = int(grp["delta"].notna().sum())
            print(f"  {mode} L={L}: {wins} wins / {ties} ties / {losses} losses ({evaluable} evaluable)")


if __name__ == "__main__":
    main()
