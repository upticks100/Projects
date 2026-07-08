"""Refit one locked cell and dump holdout prediction tensors.

Reads the locked hyperparameters from locked_cells.csv (or any CSV with the
same columns, e.g. an original aggregate_summary.csv via --cells-csv), refits
the cell's baseline + CP booster on the full training block, predicts the
calendar-fixed holdout, and writes a joblib pickle consumed by every
downstream analysis (event study, veer, CDS):

  predicted_ensemble : (W_test, F, K) ndarray
  predicted_base     : (W_test, F, K) ndarray  (ridge or FE, per cell)
  realized           : (W_test, F, K) ndarray
  mask               : (W_test, F, K) ndarray  (1 = observed)
  firm_gvkeys, feature_names, quarters_test, input_quarters, split_index,
  L, trial_meta

Window indexing convention (matches build_caches.py): cache window w has
X[w] = tensor[:, :, w : w+L] and target Y[w] = tensor[:, :, w+L], so the
target quarter of test window w is quarters[split_idx + L + w].

Refit logic is ported verbatim from evaluate_per_feature.refit_and_predict /
evaluate_top_trials_test.evaluate_trial. Set REPL_CP_LOWMEM=1 to use the
Gram-identity low-memory CP fitter (required at ~500 firms; bit-equivalent,
see tests/test_cp_lowmem_equiv.py).

Run from the replication root:
    python -m src.model.refit_and_dump --objective residual_delta_v3 --L 2 \
        --out-dir results/run1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.regression.cp_regression import CPRegressor

import config
from src.model.baselines import (
    BOOSTER_OBJECTIVES,
    compute_ridge_oof,
    evaluate_model,
    firm_feature_means,
    get_min_valid_entries,
    load_split,
    per_feature_x_scale,
)
from src.tensors.cp_lowmem import LowMemCPRegressor

N_ITER_MAX = 150  # CP ALS iteration cap (matches the locked search protocol)


def refit_and_predict(trial: dict, X_cv, Y_cv, M_cv, X_test, Y_test):
    """Refit one locked cell on the full training block; return
    (base_test, ensemble_test) prediction tensors."""
    objective_name = trial["objective"]

    if objective_name in BOOSTER_OBJECTIVES:
        ridge_oof, ridge_test, ridge_valid = compute_ridge_oof(
            X_cv, Y_cv, M_cv, X_test,
        )
        if ridge_valid.sum() == 0:
            raise RuntimeError(f"No valid ridge OOF rows for {trial}")
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
        feat_x_scale = per_feature_x_scale(X_fit_raw)
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

    if os.environ.get("REPL_CP_LOWMEM") == "1":
        cp_cls = LowMemCPRegressor
    else:
        cp_cls = CPRegressor
    cp = cp_cls(
        weight_rank=int(trial["RANK_REGRESS"]),
        reg_W=float(trial["REG_W"]),
        n_iter_max=N_ITER_MAX,
        random_state=config.SEED,
    )
    cp.fit(X_fit, Y_target)
    cp_residual_test = cp.predict(X_test_in) * y_rms * feat_scale[None, None, :]
    ensemble_test = base_test + float(trial["GAMMA"]) * cp_residual_test

    return np.asarray(base_test), np.asarray(ensemble_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True,
                   choices=["residual_delta_v3", "ridge_delta_v3"])
    p.add_argument("--L", type=int, required=True, choices=[2, 4])
    p.add_argument("--cells-csv", type=Path, default=config.LOCKED_CELLS_FILE,
                   help="CSV with locked hyperparameters (default: locked_cells.csv)")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cells = pd.read_csv(args.cells_csv)
    if "rank_order" in cells.columns:  # accept an original aggregate_summary.csv
        cells = cells[cells["rank_order"] == 1]
    row = cells[(cells["objective"] == args.objective) & (cells["L"] == args.L)]
    if row.empty:
        raise SystemExit(f"no locked cell for objective={args.objective} L={args.L}")
    trial = row.iloc[0].to_dict()
    print(f"locked cell: {trial}")

    meta = joblib.load(config.meta_path())
    feature_names = list(meta["feature_names"])
    firm_gvkeys = list(meta["firms"])
    quarters = list(meta["quarters"])

    tl.set_backend("numpy")
    np.random.seed(config.SEED)

    L = int(trial["L"])
    X_cv, Y_cv, M_cv, X_test, Y_test, M_test, split_idx = load_split(
        str(trial["mode"]), L,
    )
    base_test, ensemble_test = refit_and_predict(
        trial, X_cv, Y_cv, M_cv, X_test, Y_test,
    )

    base_r2 = evaluate_model(Y_test, base_test, M_test)
    ens_r2 = evaluate_model(Y_test, ensemble_test, M_test)
    print(f"holdout: base_R2={base_r2:.5f} ensemble_R2={ens_r2:.5f} "
          f"delta={ens_r2 - base_r2:+.5f} "
          f"(locked 50-firm delta: {float(trial.get('test_delta', np.nan)):+.5f})")

    # Quarter labels for the prediction target of each test window.
    W_test = int(Y_test.shape[0])
    target_start = split_idx + L
    quarters_test = list(map(str, quarters[target_start: target_start + W_test]))
    if len(quarters_test) != W_test:
        raise SystemExit(
            f"quarter label window misalignment: have {len(quarters_test)} "
            f"labels for {W_test} test windows (split_idx={split_idx}, L={L})"
        )
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"predictions_{args.objective}_L{args.L}_rank1.pkl"
    joblib.dump(out, out_path, compress=3)
    print(f"\nWrote {out_path}  shape={out['realized'].shape}  "
          f"size={out_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
