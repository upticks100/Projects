"""Baselines, evaluation, and the calendar-fixed split.

Numerical core shared by the refit and transfer-check stages. Function bodies
are ported verbatim from the original pipeline (CP_struct_test_new.py and
evaluate_top_trials_test.py) so results reproduce exactly:

  - evaluate_model: pooled, mask-aware R^2 with an adaptive validity threshold.
  - firm_feature_means: the fixed-effects (FE) baseline.
  - ridge_structured_cp_matched_zero_filled_ts_cv: the per-feature ridge
    baseline, with per-feature alpha chosen by an inner TimeSeriesSplit CV on
    firm-demeaned, zero-filled residual targets (matching how CPRegressor
    handles missing targets).
  - compute_ridge_oof: out-of-fold ridge predictions on the training block
    (the "OOF discipline" for the ridge-booster cells).
  - load_split: calendar-fixed train/test split anchored at
    config.TEST_START_TARGET_QUARTER.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

import config

SEED = config.SEED

# Objectives whose baseline is the per-feature ridge (trained on OOF residuals).
BOOSTER_OBJECTIVES = ("ridge_delta_v3",)

# Inner-fold skip rule for the OOF pass. Must stay in sync between training
# and replay; changing it silently breaks the booster train/test replay.
MIN_INNER_TR_SIZE = 5

RIDGE_ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4], dtype=float)


def get_min_valid_entries(mask: np.ndarray, min_frac: float = 0.05,
                          floor: int = 100, cap: int = 5000) -> int:
    """Adaptive threshold for R^2 evaluation: 5% of entries, bounded [100, 5000]."""
    return max(floor, min(cap, int(min_frac * mask.size)))


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   mask: np.ndarray) -> float | None:
    """Pooled, mask-aware R^2 with adaptive threshold."""
    y_t, y_p, m = y_true.flatten(), y_pred.flatten(), mask.flatten()
    valid = m > 0
    min_valid = get_min_valid_entries(mask)
    if np.sum(valid) < min_valid:
        return None
    clean_t, clean_p = y_t[valid], y_p[valid]
    sst = np.sum((clean_t - np.mean(clean_t)) ** 2)
    if sst <= 1e-8:
        return None
    return 1.0 - np.sum((clean_t - clean_p) ** 2) / sst


def _within_firm_means_y(y_tr: np.ndarray, m_tr: np.ndarray,
                         firm_ids: np.ndarray, n_firms: int,
                         fallback_global: float) -> np.ndarray:
    obs = (m_tr > 0)
    sums = np.bincount(firm_ids[obs], weights=y_tr[obs], minlength=n_firms).astype(float)
    cnts = np.bincount(firm_ids[obs], minlength=n_firms).astype(float)
    means = np.empty(n_firms, dtype=float)
    means[:] = fallback_global
    nz = cnts > 0
    means[nz] = sums[nz] / (cnts[nz] + 1e-12)
    return means


def firm_feature_means(Y_tr: np.ndarray, M_tr: np.ndarray) -> np.ndarray:
    """Mask-aware mean over time per (firm, feature), falling back to the
    global feature mean where a pair is never observed.
    Y_tr, M_tr: (T, Firms, Features) -> mu_ff: (Firms, Features)."""
    denom = M_tr.sum(axis=0)
    mu_ff = (Y_tr * M_tr).sum(axis=0) / (denom + 1e-8)

    denom_feat = M_tr.sum(axis=(0, 1))
    mu_feat = (Y_tr * M_tr).sum(axis=(0, 1)) / (denom_feat + 1e-8)
    missing = denom <= 0
    if np.any(missing):
        idx_firm, idx_feat = np.where(missing)
        mu_ff[idx_firm, idx_feat] = mu_feat[idx_feat]
    return mu_ff


def ridge_structured_cp_matched_zero_filled_ts_cv(
    X_tr_4d: np.ndarray, Y_tr_3d: np.ndarray, M_tr_3d: np.ndarray,
    X_val_4d: np.ndarray,
    inner_splits: int = 3,
) -> np.ndarray:
    """Ridge baseline matched to TensorLy CPRegressor target handling.

    CPRegressor has no target mask, so CP is trained on firm-feature demeaned
    residuals with missing target cells set to zero. This ridge variant uses
    the same zero-filled residual target and fits on all rows, then adds the
    firm-feature fixed effect back at prediction time. Validation/test scoring
    remains mask-aware through evaluate_model().
    """
    n_tr_t, n_f, n_feat, n_l = X_tr_4d.shape
    n_val_t = X_val_4d.shape[0]
    p = n_feat * n_l

    X_tr_2d = X_tr_4d.transpose(0, 1, 3, 2).reshape(n_tr_t * n_f, p)
    X_val_2d = X_val_4d.transpose(0, 1, 3, 2).reshape(n_val_t * n_f, p)

    Y_tr_f = Y_tr_3d.reshape(n_tr_t * n_f, n_feat)
    M_tr_f = M_tr_3d.reshape(n_tr_t * n_f, n_feat)

    time_ids_tr = np.repeat(np.arange(n_tr_t), n_f)
    firm_ids_tr = np.tile(np.arange(n_f), n_tr_t)
    firm_ids_val = np.tile(np.arange(n_f), n_val_t)

    denom = M_tr_3d.sum(axis=(0, 1)) + 1e-8
    y_global_mean = (Y_tr_3d * M_tr_3d).sum(axis=(0, 1)) / denom

    inner_tscv = TimeSeriesSplit(n_splits=inner_splits)
    y_pred_val_f = np.zeros((n_val_t * n_f, n_feat), dtype=float)

    for j in range(n_feat):
        yj = Y_tr_f[:, j]
        mj = M_tr_f[:, j]
        y_firm_mean_full = _within_firm_means_y(
            y_tr=yj,
            m_tr=mj,
            firm_ids=firm_ids_tr,
            n_firms=n_f,
            fallback_global=float(y_global_mean[j]),
        )
        yj_zero_filled_full = (yj - y_firm_mean_full[firm_ids_tr]) * mj

        best_alpha = 100.0
        best_score = -np.inf
        for alpha in RIDGE_ALPHAS:
            fold_scores = []
            for tr_time_idx, va_time_idx in inner_tscv.split(np.arange(n_tr_t)):
                tr_rows = np.isin(time_ids_tr, tr_time_idx)
                va_rows = np.isin(time_ids_tr, va_time_idx)

                # Inner-fold leakage fix: firm means from inner-training rows only.
                obs_inner_tr = mj[tr_rows] > 0
                if obs_inner_tr.sum() < 10:
                    continue
                inner_global = float(np.mean(yj[tr_rows][obs_inner_tr]))
                y_firm_mean_inner = _within_firm_means_y(
                    y_tr=yj[tr_rows],
                    m_tr=mj[tr_rows],
                    firm_ids=firm_ids_tr[tr_rows],
                    n_firms=n_f,
                    fallback_global=inner_global,
                )
                yj_zero_filled_inner = (
                    yj[tr_rows] - y_firm_mean_inner[firm_ids_tr[tr_rows]]
                ) * mj[tr_rows]

                rg = Ridge(alpha=float(alpha), fit_intercept=False,
                           solver="auto", random_state=SEED)
                rg.fit(X_tr_2d[tr_rows], yj_zero_filled_inner)
                pred_va = rg.predict(X_tr_2d[va_rows]) + y_firm_mean_inner[firm_ids_tr[va_rows]]

                obs_va = mj[va_rows] > 0
                if obs_va.sum() < 10:
                    continue
                y_true_va = yj[va_rows][obs_va]
                y_pred_va = pred_va[obs_va]
                sst = np.sum((y_true_va - np.mean(y_true_va)) ** 2)
                if sst > 1e-8:
                    fold_scores.append(1.0 - np.sum((y_true_va - y_pred_va) ** 2) / sst)

            if fold_scores:
                score = float(np.mean(fold_scores))
                if score > best_score:
                    best_score = score
                    best_alpha = float(alpha)

        rg_final = Ridge(alpha=best_alpha, fit_intercept=False,
                         solver="auto", random_state=SEED)
        rg_final.fit(X_tr_2d, yj_zero_filled_full)
        y_pred_val_f[:, j] = rg_final.predict(X_val_2d) + y_firm_mean_full[firm_ids_val]

    return y_pred_val_f.reshape(n_val_t, n_f, n_feat)


def compute_ridge_oof(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    M_tr: np.ndarray,
    X_test: np.ndarray,
    n_inner_splits: int = 3,
    inner_alpha_search_splits: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold ridge predictions on the training block, plus the full-fit
    test prediction. Windows never covered by an OOF fold are flagged so they
    can be dropped from CP training. Returns (ridge_oof_tr, ridge_test,
    initialized)."""
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


def per_feature_x_scale(X: np.ndarray) -> np.ndarray:
    """Per-feature RMS scale of the input tensor (FEATURE_X_SCALE switch)."""
    n_w, n_f, n_feat, n_l = X.shape
    feat_sse = np.sum(X ** 2, axis=(0, 1, 3))
    feat_n = float(n_w * n_f * n_l)
    feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
    return np.where(
        np.isfinite(feat_scale) & (feat_scale > 1e-8), feat_scale, 1.0,
    ).astype(X.dtype)


def calendar_split_idx(L: int, n_windows: int) -> int:
    """Calendar-fixed train/test boundary.

    Cache window w has prediction target quarters[w + L]. The first TEST
    target quarter is frozen (config.TEST_START_TARGET_QUARTER), so
    split_idx = index(test_start_quarter) - L. Extending the panel only
    appends test windows; the training block never shifts.
    """
    meta = joblib.load(config.meta_path())
    quarters = [str(q) for q in meta["quarters"]]
    try:
        qi = quarters.index(str(config.TEST_START_TARGET_QUARTER))
    except ValueError:
        raise SystemExit(
            f"TEST_START_TARGET_QUARTER {config.TEST_START_TARGET_QUARTER!r} "
            f"not in meta quarters ({quarters[0]}..{quarters[-1]})"
        )
    split_idx = qi - L
    if not (0 < split_idx < n_windows):
        raise SystemExit(
            f"calendar split_idx={split_idx} out of range (L={L}, "
            f"n_windows={n_windows}, qi={qi})"
        )
    return split_idx


def load_split(mode: str, L: int):
    """Load a tensor cache and split it at the calendar boundary.

    Returns (X_cv, Y_cv, M_cv, X_test, Y_test, M_test, split_idx)."""
    cache = joblib.load(config.cache_path(mode, L))
    X_all, Y_all, M_all = cache["X"], cache["Y"], cache["Mask"]
    split_idx = calendar_split_idx(L, len(X_all))
    return (
        X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx],
        X_all[split_idx:], Y_all[split_idx:], M_all[split_idx:],
        split_idx,
    )
