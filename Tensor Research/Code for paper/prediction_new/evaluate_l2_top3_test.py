"""Evaluate top completed L=2 Optuna CP trials against Ridge on the test set."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "3")

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorly as tl
from joblib import Parallel, delayed
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from tensorly.regression.cp_regression import CPRegressor

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PARENT_DIR))

from prediction_config import (  # noqa: E402
    MODES,
    N_ITER_MAX,
    RESULTS_DIR,
    SEED,
    cache_path,
    journal_path,
    study_name,
)
from CP_struct_test_new import (  # noqa: E402
    _within_firm_means_y,
    evaluate_model,
    firm_feature_means,
    get_min_obs_per_feat,
    ridge_structured_cp_matched_zero_filled_ts_cv,
    ridge_structured_fixed_effects_ts_cv,
)

RIDGE_ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-jobs-cp", type=int, default=2)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "l2_top3_test_vs_ridge.csv",
    )
    return parser.parse_args()


def load_top_trials(mode: str, top_k: int) -> list[dict]:
    storage = JournalStorage(
        JournalFileBackend(
            str(journal_path(mode, 2)),
            lock_obj=JournalFileSymlinkLock(str(journal_path(mode, 2))),
        )
    )
    study = optuna.load_study(study_name=study_name(mode, 2), storage=storage)
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: float(t.value), reverse=True)
    if len(completed) < top_k:
        raise RuntimeError(f"{mode} L=2 has only {len(completed)} completed trials; need {top_k}")
    return [
        {
            "mode": mode,
            "L": 2,
            "rank_order": i + 1,
            "trial_number": t.number,
            "cv_r2": float(t.value),
            "RANK_REGRESS": int(t.params["RANK_REGRESS"]),
            "REG_W": float(t.params["REG_W"]),
            "USE_RMS_SCALING": bool(t.params["USE_RMS_SCALING"]),
        }
        for i, t in enumerate(completed[:top_k])
    ]


def load_split(mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = joblib.load(cache_path(mode, 2))
    X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
    split_idx = int(0.8 * len(X_all))
    return (
        X_all[:split_idx],
        Y_all[:split_idx],
        M_all[:split_idx],
        X_all[split_idx:],
        Y_all[split_idx:],
        M_all[split_idx:],
    )


def ridge_predict_fixed_global_alpha(
    X_tr_4d: np.ndarray,
    Y_tr_3d: np.ndarray,
    M_tr_3d: np.ndarray,
    X_val_4d: np.ndarray,
    alpha: float,
) -> np.ndarray:
    n_tr_t, n_f, n_feat, n_l = X_tr_4d.shape
    n_val_t = X_val_4d.shape[0]
    p = n_feat * n_l
    min_obs_per_feat = get_min_obs_per_feat(n_f, n_tr_t, M_tr_3d)

    X_tr_2d = X_tr_4d.transpose(0, 1, 3, 2).reshape(n_tr_t * n_f, p)
    X_val_2d = X_val_4d.transpose(0, 1, 3, 2).reshape(n_val_t * n_f, p)
    Y_tr_f = Y_tr_3d.reshape(n_tr_t * n_f, n_feat)
    M_tr_f = M_tr_3d.reshape(n_tr_t * n_f, n_feat)

    firm_ids_tr = np.tile(np.arange(n_f), n_tr_t)
    firm_ids_val = np.tile(np.arange(n_f), n_val_t)
    denom = M_tr_3d.sum(axis=(0, 1)) + 1e-8
    y_global_mean = (Y_tr_3d * M_tr_3d).sum(axis=(0, 1)) / denom

    y_pred_val_f = np.zeros((n_val_t * n_f, n_feat), dtype=float)
    for j in range(n_feat):
        yj = Y_tr_f[:, j]
        mj = M_tr_f[:, j]
        y_firm_mean = _within_firm_means_y(
            y_tr=yj,
            m_tr=mj,
            firm_ids=firm_ids_tr,
            n_firms=n_f,
            fallback_global=float(y_global_mean[j]),
        )
        obs = mj > 0
        if obs.sum() < min_obs_per_feat:
            y_pred_val_f[:, j] = y_firm_mean[firm_ids_val]
            continue

        rg = Ridge(alpha=alpha, fit_intercept=False, solver="auto", random_state=SEED)
        rg.fit(X_tr_2d[obs], (yj - y_firm_mean[firm_ids_tr])[obs])
        y_pred_val_f[:, j] = rg.predict(X_val_2d) + y_firm_mean[firm_ids_val]

    return y_pred_val_f.reshape(n_val_t, n_f, n_feat)


def evaluate_global_alpha_ridge(mode: str) -> tuple[float, float]:
    start = time.perf_counter()
    X_cv, Y_cv, M_cv, X_test, Y_test, M_test = load_split(mode)
    tscv = TimeSeriesSplit(n_splits=3)
    alpha_scores: list[tuple[float, float]] = []

    for alpha in RIDGE_ALPHAS:
        scores = []
        for tr_idx, va_idx in tscv.split(X_cv):
            pred = ridge_predict_fixed_global_alpha(
                X_cv[tr_idx],
                Y_cv[tr_idx],
                M_cv[tr_idx],
                X_cv[va_idx],
                alpha=float(alpha),
            )
            score = evaluate_model(Y_cv[va_idx], pred, M_cv[va_idx])
            if score is not None and np.isfinite(score):
                scores.append(float(score))
        alpha_scores.append((float(alpha), float(np.mean(scores)) if scores else -np.inf))

    best_alpha, best_cv = max(alpha_scores, key=lambda item: item[1])
    pred_test = ridge_predict_fixed_global_alpha(X_cv, Y_cv, M_cv, X_test, alpha=best_alpha)
    test_score = evaluate_model(Y_test, pred_test, M_test)
    elapsed = time.perf_counter() - start
    print(
        f"[{mode} L=2] Matched global-alpha Ridge test R2={test_score:.6f} "
        f"alpha={best_alpha:g} inner_cv={best_cv:.6f} ({elapsed:.1f}s)",
        flush=True,
    )
    return float(test_score) if test_score is not None and np.isfinite(test_score) else np.nan, best_alpha


def evaluate_cp_matched_ridge(mode: str) -> float:
    start = time.perf_counter()
    X_cv, Y_cv, M_cv, X_test, Y_test, M_test = load_split(mode)
    pred = ridge_structured_cp_matched_zero_filled_ts_cv(X_cv, Y_cv, M_cv, X_test)
    score = evaluate_model(Y_test, pred, M_test)
    elapsed = time.perf_counter() - start
    cp_matched_score = float(score) if score is not None and np.isfinite(score) else np.nan
    print(
        f"[{mode} L=2] CP-matched zero-filled Ridge test R2={cp_matched_score:.6f} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return cp_matched_score


def evaluate_ridges(mode: str) -> tuple[float, float, float, float]:
    start = time.perf_counter()
    X_cv, Y_cv, M_cv, X_test, Y_test, M_test = load_split(mode)
    pred = ridge_structured_fixed_effects_ts_cv(X_cv, Y_cv, M_cv, X_test)
    score = evaluate_model(Y_test, pred, M_test)
    elapsed = time.perf_counter() - start
    per_feature_score = float(score) if score is not None and np.isfinite(score) else np.nan
    print(
        f"[{mode} L=2] Per-feature-alpha Ridge test R2={per_feature_score:.6f} "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    global_score, global_alpha = evaluate_global_alpha_ridge(mode)
    cp_matched_score = evaluate_cp_matched_ridge(mode)
    return per_feature_score, global_score, global_alpha, cp_matched_score


def evaluate_cp_trial(
    trial: dict,
    ridge_per_feature_test_r2: float,
    ridge_global_alpha_test_r2: float,
    ridge_global_alpha: float,
    ridge_cp_matched_test_r2: float,
) -> dict:
    start = time.perf_counter()
    mode = trial["mode"]
    X_cv, Y_cv, M_cv, X_test, Y_test, M_test = load_split(mode)

    mu_ff = firm_feature_means(Y_cv, M_cv)
    Y_cv_cent = (Y_cv - mu_ff[None, :, :]) * M_cv

    if trial["USE_RMS_SCALING"]:
        y_obs = Y_cv_cent[M_cv > 0]
        y_rms = float(np.sqrt(np.mean(y_obs ** 2))) if y_obs.size else 1.0
        Y_target = Y_cv_cent / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target = Y_cv_cent

    cp = CPRegressor(
        weight_rank=trial["RANK_REGRESS"],
        reg_W=trial["REG_W"],
        n_iter_max=N_ITER_MAX,
        random_state=SEED,
    )
    cp.fit(X_cv, Y_target)
    pred = cp.predict(X_test) * y_rms + mu_ff[None, :, :]
    score = evaluate_model(Y_test, pred, M_test)
    cp_test_r2 = float(score) if score is not None and np.isfinite(score) else np.nan
    elapsed = time.perf_counter() - start

    row = dict(trial)
    row.update(
        {
            "ridge_per_feature_alpha_test_r2": ridge_per_feature_test_r2,
            "ridge_global_alpha_test_r2": ridge_global_alpha_test_r2,
            "ridge_global_alpha": ridge_global_alpha,
            "ridge_cp_matched_zero_filled_test_r2": ridge_cp_matched_test_r2,
            "cp_test_r2": cp_test_r2,
            "delta_cp_minus_per_feature_ridge": cp_test_r2 - ridge_per_feature_test_r2,
            "delta_cp_minus_global_alpha_ridge": cp_test_r2 - ridge_global_alpha_test_r2,
            "delta_cp_minus_cp_matched_ridge": cp_test_r2 - ridge_cp_matched_test_r2,
            "elapsed_seconds": elapsed,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    print(
        f"[{mode} L=2 rank#{trial['rank_order']}] "
        f"trial={trial['trial_number']} cv={trial['cv_r2']:.6f} "
        f"test_cp={cp_test_r2:.6f} "
        f"ridge_pf={ridge_per_feature_test_r2:.6f} "
        f"ridge_global={ridge_global_alpha_test_r2:.6f} "
        f"ridge_cp_matched={ridge_cp_matched_test_r2:.6f} "
        f"delta_cp_matched={row['delta_cp_minus_cp_matched_ridge']:.6f} ({elapsed:.1f}s)",
        flush=True,
    )
    return row


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tl.set_backend("numpy")
    np.random.seed(SEED)

    modes = [m.strip().upper() for m in args.modes.split(",") if m.strip()]
    top_trials: list[dict] = []
    ridge_scores: dict[str, tuple[float, float, float, float]] = {}
    for mode in modes:
        top = load_top_trials(mode, args.top_k)
        top_trials.extend(top)
        print(f"\nTop {args.top_k} completed trials for {mode} L=2:")
        for t in top:
            print(
                f"  rank#{t['rank_order']} trial={t['trial_number']} "
                f"cv={t['cv_r2']:.6f} rank={t['RANK_REGRESS']} "
                f"reg_w={t['REG_W']:.6g} rms={t['USE_RMS_SCALING']}",
                flush=True,
            )
        ridge_scores[mode] = evaluate_ridges(mode)

    rows = Parallel(n_jobs=args.n_jobs_cp, verbose=10)(
        delayed(evaluate_cp_trial)(trial, *ridge_scores[trial["mode"]])
        for trial in top_trials
    )

    df = pd.DataFrame(rows).sort_values(["mode", "rank_order"])
    df.to_csv(args.output, index=False)
    print(f"\n[DONE] Wrote {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
