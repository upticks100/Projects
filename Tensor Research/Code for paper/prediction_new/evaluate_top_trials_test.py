"""Evaluate top completed Optuna CP trials on the held-out test set.

Supports:
- pooled_r2
- residual_delta
- residual_delta_v2
- residual_delta_v3
- ridge_delta_v3

For `ridge_delta_v3`, replays the post-2026-06-19 worker convention:
  - Computes Ridge OOF rows on dev with the inner-TimeSeriesSplit skip
    (`inner_tr_idx.size < 5`) — this MUST stay in sync with
    `worker.py::_compute_ridge_predictions_for_fold`.
  - Drops un-initialized rows from CP training.
  - Test prediction is `Ridge_test + GAMMA * CP_test`, where Ridge_test
    is fit on the FULL dev set with `ridge_structured_cp_matched_zero_filled_ts_cv`.

Emits two CSVs:
- summary: one row per (objective, mode, L, trial) with pooled test R²,
  ensemble R², test delta.
- per-window: one row per (objective, mode, L, trial, test window)
  with per-window base R², ensemble R², delta. Used for regime
  / per-window analysis.

Both CSVs are written incrementally to `.partial` files; the final
versions are written at completion and the partial files removed.
"""

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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import tensorly as tl  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from optuna.storages import JournalStorage  # noqa: E402
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from tensorly.regression.cp_regression import CPRegressor  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PARENT_DIR))

from prediction_config import (  # noqa: E402
    LOOKBACKS,
    MODES,
    N_ITER_MAX,
    RESULTS_DIR,
    SEED,
    TEST_START_TARGET_QUARTER,
    cache_path,
    journal_path,
    meta_path,
    study_name,
)
from CP_struct_test_new import (  # noqa: E402
    evaluate_model,
    firm_feature_means,
    ridge_structured_cp_matched_zero_filled_ts_cv,
)

OBJECTIVES = (
    "pooled_r2",
    "residual_delta",
    "residual_delta_v2",
    "residual_delta_v3",
    "ridge_delta_v3",
)
RESIDUAL_OBJECTIVES = (
    "residual_delta",
    "residual_delta_v2",
    "residual_delta_v3",
    "ridge_delta_v3",
)
V3_OBJECTIVES = ("residual_delta_v3", "ridge_delta_v3")
BOOSTER_OBJECTIVES = ("ridge_delta_v3",)

# MUST match worker.py::_compute_ridge_predictions_for_fold. Changing one
# without the other silently breaks the booster train/test replay.
MIN_INNER_TR_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="residual_delta", choices=list(OBJECTIVES))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-completed", type=int, default=10)
    parser.add_argument("--n-jobs-cp", type=int, default=2)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--lookbacks", default=",".join(str(x) for x in LOOKBACKS))
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "top_trials_test_summary.csv",
    )
    parser.add_argument(
        "--per-window-output",
        type=Path,
        default=None,
        help="CSV for per-window rows. Default: <output stem>_per_window.csv",
    )
    return parser.parse_args()


def objective_journal_path(mode: str, L: int, objective_name: str) -> Path:
    base = journal_path(mode, L)
    if objective_name == "pooled_r2":
        return base
    return base.with_name(f"{base.stem}_{objective_name}{base.suffix}")


def objective_study_name(mode: str, L: int, objective_name: str) -> str:
    base = study_name(mode, L)
    if objective_name == "pooled_r2":
        return base
    return f"{base}_{objective_name}"


def parse_csv_str(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def parse_csv_int(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def get_min_valid_entries(mask: np.ndarray, min_frac: float = 0.05,
                          floor: int = 100, cap: int = 5000) -> int:
    return max(floor, min(cap, int(min_frac * mask.size)))


def _per_feature_x_scale(X: np.ndarray) -> np.ndarray:
    n_w, n_f, n_feat, n_l = X.shape
    feat_sse = np.sum(X ** 2, axis=(0, 1, 3))
    feat_n = float(n_w * n_f * n_l)
    feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
    return np.where(
        np.isfinite(feat_scale) & (feat_scale > 1e-8), feat_scale, 1.0,
    ).astype(X.dtype)


def load_top_trials(mode: str, L: int, objective_name: str,
                    top_k: int, min_completed: int) -> list[dict]:
    jp = objective_journal_path(mode, L, objective_name)
    if not jp.exists():
        raise RuntimeError(
            f"Missing journal for {mode} L={L} objective={objective_name}: {jp}"
        )

    storage = JournalStorage(
        JournalFileBackend(str(jp), lock_obj=JournalFileSymlinkLock(str(jp)))
    )
    study = optuna.load_study(
        study_name=objective_study_name(mode, L, objective_name),
        storage=storage,
    )
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if len(completed) < min_completed:
        raise RuntimeError(
            f"{objective_study_name(mode, L, objective_name)} has "
            f"{len(completed)} completed trials; minimum required is {min_completed}"
        )

    completed.sort(key=lambda t: float(t.value), reverse=True)
    rows = []
    for i, t in enumerate(completed[:top_k]):
        rows.append(
            {
                "mode": mode,
                "L": int(L),
                "objective": objective_name,
                "rank_order": i + 1,
                "trial_number": int(t.number),
                "cv_objective_value": float(t.value),
                "RANK_REGRESS": int(t.params["RANK_REGRESS"]),
                "REG_W": float(t.params["REG_W"]),
                "USE_RMS_SCALING": bool(t.params["USE_RMS_SCALING"]),
                "GAMMA": float(t.params.get("GAMMA", 1.0)),
                "FEATURE_TARGET_SCALE": bool(t.params.get("FEATURE_TARGET_SCALE", False)),
                "FEATURE_X_SCALE": bool(t.params.get("FEATURE_X_SCALE", False)),
            }
        )
    return rows


def _calendar_split_idx(L: int, n_windows: int) -> int:
    """Calendar-fixed train/test boundary (replaces length-based 0.8 split).

    Cache window w has prediction target quarters[w + L]
    (build_prediction_caches: Y[w] = tensor[:, :, w + L]). We freeze the first
    TEST target quarter to TEST_START_TARGET_QUARTER, so split_idx =
    index(TEST_START_TARGET_QUARTER) - L. Because earlier quarters never shift
    when the panel is extended, this keeps the train block and the original
    Part 1 test quarters byte-identical and only appends new quarters to the
    test side. For the original 80-quarter panel it reproduces the old
    int(0.8 * n_windows) split exactly (L=2 -> 62, L=4 -> 60).
    """
    meta = joblib.load(meta_path())
    quarters = [str(q) for q in meta["quarters"]]
    try:
        qi = quarters.index(str(TEST_START_TARGET_QUARTER))
    except ValueError:
        raise SystemExit(
            f"TEST_START_TARGET_QUARTER {TEST_START_TARGET_QUARTER!r} not in "
            f"meta quarters ({quarters[0]}..{quarters[-1]})"
        )
    split_idx = qi - L
    if not (0 < split_idx < n_windows):
        raise SystemExit(
            f"calendar split_idx={split_idx} out of range (L={L}, "
            f"n_windows={n_windows}, qi={qi}, q={TEST_START_TARGET_QUARTER})"
        )
    return split_idx


def load_split(mode: str, L: int):
    cache = joblib.load(cache_path(mode, L))
    X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
    split_idx = _calendar_split_idx(L, len(X_all))
    return (
        X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx],
        X_all[split_idx:], Y_all[split_idx:], M_all[split_idx:],
        split_idx,
    )


def _compute_ridge_oof_drop_fallback(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    M_tr: np.ndarray,
    X_test: np.ndarray,
    n_inner_splits: int = 3,
    inner_alpha_search_splits: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the post-2026-06-19 worker Ridge-OOF convention.

    Must match worker.py::_compute_ridge_predictions_for_fold exactly
    (skip rule MIN_INNER_TR_SIZE, inner_alpha_search_splits, ridge baseline).
    Returns (ridge_oof_tr, ridge_test, initialized).
    """
    ridge_oof_tr = np.zeros_like(Y_tr)
    initialized = np.zeros(X_tr.shape[0], dtype=bool)

    inner_tscv = TimeSeriesSplit(n_splits=n_inner_splits)
    for inner_tr_idx, inner_va_idx in inner_tscv.split(X_tr):
        if inner_tr_idx.size < MIN_INNER_TR_SIZE or inner_va_idx.size == 0:
            continue
        preds = ridge_structured_cp_matched_zero_filled_ts_cv(
            X_tr[inner_tr_idx],
            Y_tr[inner_tr_idx],
            M_tr[inner_tr_idx],
            X_tr[inner_va_idx],
            inner_splits=inner_alpha_search_splits,
        )
        ridge_oof_tr[inner_va_idx] = preds.astype(Y_tr.dtype, copy=False)
        initialized[inner_va_idx] = True

    ridge_test = ridge_structured_cp_matched_zero_filled_ts_cv(
        X_tr, Y_tr, M_tr, X_test,
        inner_splits=inner_alpha_search_splits,
    ).astype(Y_tr.dtype, copy=False)

    return ridge_oof_tr, ridge_test, initialized


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    score = evaluate_model(y_true, y_pred, mask)
    return float(score) if score is not None and np.isfinite(score) else np.nan


def _per_window_rows(
    trial: dict,
    split_idx: int,
    Y_test: np.ndarray,
    M_test: np.ndarray,
    base_test: np.ndarray,
    ensemble_test: np.ndarray,
) -> list[dict]:
    rows = []
    for w in range(Y_test.shape[0]):
        base_r2 = _safe_r2(Y_test[w:w + 1], base_test[w:w + 1], M_test[w:w + 1])
        ensemble_r2 = _safe_r2(Y_test[w:w + 1], ensemble_test[w:w + 1], M_test[w:w + 1])
        rows.append(
            {
                "objective": trial["objective"],
                "mode": trial["mode"],
                "L": trial["L"],
                "rank_order": trial["rank_order"],
                "trial_number": trial["trial_number"],
                "cv_objective_value": trial["cv_objective_value"],
                "window_index": w,
                "global_window_index": split_idx + w,
                "base_r2": base_r2,
                "ensemble_r2": ensemble_r2,
                "delta": ensemble_r2 - base_r2,
            }
        )
    return rows


def evaluate_trial(trial: dict) -> tuple[dict, list[dict]]:
    start = time.perf_counter()
    mode = trial["mode"]
    L = int(trial["L"])
    objective_name = trial["objective"]

    X_cv, Y_cv, M_cv, X_test, Y_test, M_test, split_idx = load_split(mode, L)

    if objective_name in BOOSTER_OBJECTIVES:
        ridge_oof, ridge_test, ridge_valid = _compute_ridge_oof_drop_fallback(
            X_cv, Y_cv, M_cv, X_test,
        )
        if ridge_valid.sum() == 0:
            raise RuntimeError(
                f"No honest Ridge OOF rows for {mode} L={L} trial={trial['trial_number']}"
            )
        X_fit_raw = X_cv[ridge_valid]
        Y_fit = Y_cv[ridge_valid]
        M_fit = M_cv[ridge_valid]
        base_fit = ridge_oof[ridge_valid]
        base_test = ridge_test
        baseline_name = "ridge_cp_matched_zero_filled"
        dropped_oof_rows = int((~ridge_valid).sum())
    else:
        mu_ff = firm_feature_means(Y_cv, M_cv)
        X_fit_raw = X_cv
        Y_fit = Y_cv
        M_fit = M_cv
        base_fit = np.broadcast_to(mu_ff[None, :, :], Y_cv.shape)
        base_test = np.broadcast_to(mu_ff[None, :, :], Y_test.shape)
        baseline_name = "firm_feature_fe"
        dropped_oof_rows = 0

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
            raise RuntimeError(
                f"Too few observed CP target cells for {mode} L={L} "
                f"trial={trial['trial_number']}"
            )
        y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
        Y_target = Y_scaled / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target = Y_scaled

    cp = CPRegressor(
        weight_rank=trial["RANK_REGRESS"],
        reg_W=trial["REG_W"],
        n_iter_max=N_ITER_MAX,
        random_state=SEED,
    )
    cp.fit(X_fit, Y_target)
    cp_residual_test = cp.predict(X_test_in) * y_rms * feat_scale[None, None, :]
    ensemble_test = base_test + trial["GAMMA"] * cp_residual_test

    base_test_r2 = _safe_r2(Y_test, base_test, M_test)
    ensemble_test_r2 = _safe_r2(Y_test, ensemble_test, M_test)
    elapsed = time.perf_counter() - start

    summary = dict(trial)
    summary.update(
        {
            "baseline_name": baseline_name,
            "base_test_r2": base_test_r2,
            "ensemble_test_r2": ensemble_test_r2,
            "test_delta": ensemble_test_r2 - base_test_r2,
            "dropped_oof_rows": dropped_oof_rows,
            "cp_train_windows": int(X_fit.shape[0]),
            "test_windows": int(X_test.shape[0]),
            "elapsed_seconds": elapsed,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    return summary, _per_window_rows(trial, split_idx, Y_test, M_test, base_test, ensemble_test)


def main() -> None:
    args = parse_args()
    if args.per_window_output is None:
        args.per_window_output = args.output.with_name(
            f"{args.output.stem}_per_window{args.output.suffix}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.per_window_output.parent.mkdir(parents=True, exist_ok=True)

    tl.set_backend("numpy")
    np.random.seed(SEED)

    modes = parse_csv_str(args.modes)
    lookbacks = parse_csv_int(args.lookbacks)

    top_trials: list[dict] = []
    print(f"Preparing evaluation: objective={args.objective} modes={modes} lookbacks={lookbacks}")
    for mode in modes:
        for L in lookbacks:
            trials = load_top_trials(
                mode, L, args.objective, args.top_k, args.min_completed,
            )
            top_trials.extend(trials)
            print(f"[{mode} L={L}] loaded top {len(trials)} trials")

    summary_partial = args.output.with_suffix(args.output.suffix + ".partial")
    window_partial = args.per_window_output.with_suffix(args.per_window_output.suffix + ".partial")
    for path in (summary_partial, window_partial):
        if path.exists():
            path.unlink()

    completed_summary: list[dict] = []
    completed_windows: list[dict] = []

    total = len(top_trials)
    t_run = time.time()
    parallel = Parallel(
        n_jobs=args.n_jobs_cp,
        verbose=10,
        return_as="generator_unordered",
    )
    row_iter = parallel(delayed(evaluate_trial)(trial) for trial in top_trials)

    for i, (summary, window_rows) in enumerate(row_iter, start=1):
        completed_summary.append(summary)
        completed_windows.extend(window_rows)

        pd.DataFrame(completed_summary).to_csv(summary_partial, index=False)
        pd.DataFrame(completed_windows).to_csv(window_partial, index=False)

        elapsed = time.time() - t_run
        print(
            f"[partial {i}/{total}] objective={summary['objective']} mode={summary['mode']} "
            f"L={summary['L']} trial={summary['trial_number']} "
            f"base={summary['base_test_r2']:.5f} ensemble={summary['ensemble_test_r2']:.5f} "
            f"delta={summary['test_delta']:.5f} elapsed={elapsed:.0f}s",
            flush=True,
        )

    summary_df = pd.DataFrame(completed_summary).sort_values(
        ["objective", "mode", "L", "rank_order"]
    )
    window_df = pd.DataFrame(completed_windows).sort_values(
        ["objective", "mode", "L", "rank_order", "window_index"]
    )

    summary_df.to_csv(args.output, index=False)
    window_df.to_csv(args.per_window_output, index=False)

    for path in (summary_partial, window_partial):
        if path.exists():
            path.unlink()

    print(f"\n[DONE] Wrote {args.output}")
    print(f"[DONE] Wrote {args.per_window_output}")
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
