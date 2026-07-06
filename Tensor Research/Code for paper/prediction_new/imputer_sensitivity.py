"""Downstream sensitivity of prediction results to Tucker imputer ranks.

This is intentionally not a full hyperparameter search. Imputer ranks were
selected by held-out-cell CV; here we check whether downstream CP/Ridge
forecasting is robust to reasonable neighboring rank choices.
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
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
from tensorly.regression.cp_regression import CPRegressor

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PARENT_DIR))

from build_prediction_caches import (  # noqa: E402
    build_raw_tensor,
    load_filtered_panel,
    process_window,
)
from prediction_config import (  # noqa: E402
    LEGACY_WARM_START,
    LOOKBACKS,
    MODES,
    N_ITER_MAX,
    RESULTS_DIR,
    SEED,
)

from CP_struct_test_new import (  # noqa: E402
    evaluate_model,
    firm_feature_means,
    ridge_structured_cp_matched_zero_filled_ts_cv,
)


SENSITIVITY_CONFIGS: dict[str, dict[int, list[int]]] = {
    "validated_cv": {2: [2, 2, 2], 4: [4, 4, 4]},
    "small_22": {2: [2, 2, 2], 4: [2, 2, 4]},
    "hidden_best": {2: [3, 3, 2], 4: [4, 4, 4]},
    "medium_55": {2: [5, 5, 2], 4: [5, 5, 4]},
    "legacy_4020": {2: [40, 20, 2], 4: [40, 20, 4]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default=",".join(SENSITIVITY_CONFIGS.keys()),
        help="Comma-separated config names.",
    )
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--lookbacks", default=",".join(str(x) for x in LOOKBACKS))
    parser.add_argument("--n-jobs-cache", type=int, default=4)
    parser.add_argument("--n-jobs-eval", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "imputer_sensitivity_results.csv",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_csv_str(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_csv_int(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_cache_arrays(
    raw_tensor: np.ndarray,
    mode: str,
    L: int,
    ranks: list[int],
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_time = raw_tensor.shape[2]
    n_wins = n_time - L
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_window)(t, raw_tensor, L, ranks, mode=mode)
        for t in range(n_wins)
    )
    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError(f"No valid windows for mode={mode} L={L} ranks={ranks}")
    X_all = np.stack([r[0] for r in valid])
    Y_all = np.stack([r[1] for r in valid])
    M_all = np.stack([r[2] for r in valid])
    recon_errors = np.array([r[3] for r in valid], dtype=float)
    return X_all, Y_all, M_all, recon_errors


def eval_cp(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    M_tr: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    M_val: np.ndarray,
    mode: str,
    L: int,
) -> float:
    params = LEGACY_WARM_START[(mode, L)]
    rank_reg = int(params["RANK_REGRESS"])
    reg_w = float(params["REG_W"])
    use_rms = bool(params["USE_RMS_SCALING"])

    mu_ff = firm_feature_means(Y_tr, M_tr)
    Y_tr_cent = (Y_tr - mu_ff[None, :, :]) * M_tr

    if use_rms:
        y_obs = Y_tr_cent[M_tr > 0]
        y_rms = float(np.sqrt(np.mean(y_obs**2))) if y_obs.size else 1.0
        Y_target = Y_tr_cent / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target = Y_tr_cent

    cp = CPRegressor(
        weight_rank=rank_reg,
        reg_W=reg_w,
        n_iter_max=N_ITER_MAX,
        random_state=SEED,
    )
    cp.fit(X_tr, Y_target)
    pred = cp.predict(X_val) * y_rms + mu_ff[None, :, :]
    score = evaluate_model(Y_val, pred, M_val)
    return float(score) if score is not None and np.isfinite(score) else np.nan


def eval_one_combo(
    config_name: str,
    mode: str,
    L: int,
    ranks: list[int],
    raw_tensor_path: Path,
    n_jobs_cache: int,
) -> dict[str, float | int | str]:
    start = time.perf_counter()
    raw_tensor = joblib.load(raw_tensor_path, mmap_mode="r")
    X_all, Y_all, M_all, recon_errors = build_cache_arrays(
        raw_tensor=raw_tensor,
        mode=mode,
        L=L,
        ranks=ranks,
        n_jobs=n_jobs_cache,
    )

    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]
    X_test, Y_test, M_test = X_all[split_idx:], Y_all[split_idx:], M_all[split_idx:]

    tscv = TimeSeriesSplit(n_splits=3)
    cp_scores: list[float] = []
    ridge_scores: list[float] = []
    for tr_idx, va_idx in tscv.split(X_cv):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_va, Y_va, M_va = X_cv[va_idx], Y_cv[va_idx], M_cv[va_idx]

        ridge_pred = ridge_structured_cp_matched_zero_filled_ts_cv(X_tr, Y_tr, M_tr, X_va)
        ridge = evaluate_model(Y_va, ridge_pred, M_va)
        ridge_scores.append(float(ridge) if ridge is not None else np.nan)

        cp_scores.append(eval_cp(X_tr, Y_tr, M_tr, X_va, Y_va, M_va, mode, L))

    ridge_test_pred = ridge_structured_cp_matched_zero_filled_ts_cv(X_cv, Y_cv, M_cv, X_test)
    ridge_test = evaluate_model(Y_test, ridge_test_pred, M_test)
    cp_test = eval_cp(X_cv, Y_cv, M_cv, X_test, Y_test, M_test, mode, L)

    return {
        "config": config_name,
        "mode": mode,
        "L": L,
        "ranks": "-".join(str(x) for x in ranks),
        "n_windows": len(X_all),
        "imputer_recon_mean": float(np.mean(recon_errors)),
        "imputer_recon_median": float(np.median(recon_errors)),
        "ridge_cv_mean": float(np.nanmean(ridge_scores)),
        "cp_cv_mean": float(np.nanmean(cp_scores)),
        "ridge_test_r2": float(ridge_test) if ridge_test is not None else np.nan,
        "cp_test_r2": cp_test,
        "delta_cv_cp_minus_ridge": float(np.nanmean(cp_scores) - np.nanmean(ridge_scores)),
        "delta_test_cp_minus_ridge": (
            float(cp_test - ridge_test)
            if ridge_test is not None and np.isfinite(cp_test)
            else np.nan
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tl.set_backend("numpy")

    config_names = parse_csv_str(args.configs)
    modes = parse_csv_str(args.modes)
    lookbacks = parse_csv_int(args.lookbacks)

    invalid = [name for name in config_names if name not in SENSITIVITY_CONFIGS]
    if invalid:
        raise ValueError(f"Unknown config(s): {invalid}")

    existing_rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    if args.resume and args.output.exists():
        existing = pd.read_csv(args.output)
        existing_rows = existing.to_dict("records")
        done = set(zip(existing["config"], existing["mode"], existing["L"].astype(int)))

    print("Building raw prediction tensor once...", flush=True)
    df, feature_names, firms, quarters = load_filtered_panel()
    raw_tensor = build_raw_tensor(df, feature_names, firms, quarters)
    raw_tensor_path = RESULTS_DIR / "imputer_sensitivity_raw_tensor.joblib"
    joblib.dump(raw_tensor, raw_tensor_path)

    jobs = []
    for name in config_names:
        for mode in modes:
            for L in lookbacks:
                if (name, mode, L) in done:
                    continue
                jobs.append((name, mode, L, SENSITIVITY_CONFIGS[name][L]))

    print(
        f"Running {len(jobs)} sensitivity jobs "
        f"(configs={config_names}, modes={modes}, L={lookbacks}); "
        f"n_jobs_eval={args.n_jobs_eval}",
        flush=True,
    )

    rows = list(existing_rows)
    parallel_results = Parallel(n_jobs=args.n_jobs_eval, verbose=10, return_as="generator_unordered")(
        delayed(eval_one_combo)(
            name,
            mode,
            L,
            ranks,
            raw_tensor_path,
            args.n_jobs_cache,
        )
        for name, mode, L, ranks in jobs
    )
    for row in parallel_results:
        rows.append(row)
        pd.DataFrame(rows).sort_values(["config", "mode", "L"]).to_csv(args.output, index=False)
        print(
            f"finished {row['config']} {row['mode']} L={row['L']} "
            f"cp_cv={row['cp_cv_mean']:.5f} ridge_cv={row['ridge_cv_mean']:.5f} "
            f"delta={row['delta_cv_cp_minus_ridge']:.5f}",
            flush=True,
        )

    out = pd.DataFrame(rows).sort_values(["config", "mode", "L"])
    out.to_csv(args.output, index=False)
    print(f"\nSaved sensitivity results: {args.output}", flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
