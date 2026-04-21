#!/usr/bin/env python3
"""
CP_struct_test_top3.py

Top-3 CP ensemble vs Ridge baseline, for BOTH lookbacks L=2 and L=4
(and for both MODE=LEVELS/SURPRISE).

Based on CP_struct_test_new.py but uses the top 3 Optuna trials per
(mode, L) instead of just the single best.  Reports individual per-trial
results AND an ensemble (averaged predictions) result.

Outputs (new filenames to avoid overwriting originals):
  - cp_vs_ridge_top3_results.csv       (aggregate: 4 rows per mode/L)
  - cp_vs_ridge_top3_per_feature.csv   (per-feature R^2)
  - cp_vs_ridge_top3_persistence.csv   (AR1 rho vs delta)
  - cp_vs_ridge_top3_stats.csv         (stats with BH-adjusted p-values)
  - persistence_scatter_top3_*.png     (scatter plots)

Run:
  python CP_struct_test_top3.py
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import joblib
import numpy as np
import optuna
import pandas as pd
import tensorly as tl

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from tensorly.regression.cp_regression import CPRegressor
from scipy.stats import spearmanr, wilcoxon, linregress
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

SEED = 42
TOP_K = 3

# ---------------------------------------------------------------------------
# Adaptive threshold helpers (unchanged from original)
# ---------------------------------------------------------------------------

def get_min_valid_entries(mask: np.ndarray, min_frac: float = 0.05,
                          floor: int = 100, cap: int = 5000) -> int:
    total_entries = int(mask.size)
    return max(floor, min(cap, int(min_frac * total_entries)))


def get_min_obs_per_feat(n_firms: int, n_tr_t: int, mask: np.ndarray,
                          frac: float = 0.05, floor: int = 10, cap: int = 200) -> int:
    mean_density = float(mask.mean()) if mask.size > 0 else 0.5
    adaptive = int(frac * n_firms * n_tr_t * mean_density)
    return max(floor, min(cap, adaptive))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES = ["LEVELS", "SURPRISE"]
LOOKBACKS = [2, 4]

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_v2_L2.pkl", 4: "tensor_levels_v2_L4.pkl"},
    "SURPRISE": {2: "tensor_surprise_v2_L2.pkl", 4: "tensor_surprise_v2_L4.pkl"},
}

RIDGE_ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4], dtype=float)

FEATURES = [
    "aoq", "aqaq", "atq", "ltq", "ceqq", "cheq", "ciq", "cogsq", "dlcq", "dlttq",
    "epsfxq", "epspxq", "ibq", "invtq", "intanq", "mibq", "nopiq", "oiadpq",
    "piq", "ppentq", "pstkq", "rectq", "txtq", "xidoq"
]


def get_feature_names(n_feat: int) -> List[str]:
    if len(FEATURES) == n_feat:
        return FEATURES
    return [f"feat_{i}" for i in range(n_feat)]


# ---------------------------------------------------------------------------
# Evaluation helpers (unchanged)
# ---------------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   mask: np.ndarray) -> Optional[float]:
    """Pooled, mask-aware R² with adaptive threshold."""
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


def evaluate_model_per_feature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    min_obs: Optional[int] = None
) -> np.ndarray:
    """Feature-wise, mask-aware R². Returns array of shape (Features,)."""
    n_t, n_f, n_feat = y_true.shape
    if min_obs is None:
        min_obs = get_min_obs_per_feat(n_f, n_t, mask)
    scores = np.full(n_feat, np.nan, dtype=float)
    for j in range(n_feat):
        y_t = y_true[:, :, j].ravel()
        y_p = y_pred[:, :, j].ravel()
        m = mask[:, :, j].ravel()
        valid = m > 0
        if valid.sum() < min_obs:
            continue
        clean_t, clean_p = y_t[valid], y_p[valid]
        sst = np.sum((clean_t - np.mean(clean_t)) ** 2)
        if sst <= 1e-8:
            continue
        scores[j] = 1.0 - np.sum((clean_t - clean_p) ** 2) / sst
    return scores


# ---------------------------------------------------------------------------
# Top-K trial selection (replaces get_best_trial_for)
# ---------------------------------------------------------------------------

def get_top_k_trials_for(
    study: optuna.Study,
    mode: str,
    L: int,
    k: int = TOP_K,
) -> List[optuna.trial.FrozenTrial]:
    """Return the top *k* trials for a given L, sorted by best CV value.

    Handles missing ``L`` params gracefully and respects the study's
    optimization direction (maximize or minimize).
    """
    candidates = []
    for t in study.get_trials(deepcopy=False):
        if t.value is None:
            continue
        L_param = t.params.get("L")
        if L_param is None:
            continue
        try:
            if int(L_param) != int(L):
                continue
        except (ValueError, TypeError):
            continue
        candidates.append(t)

    descending = study.direction == optuna.study.StudyDirection.MAXIMIZE
    candidates.sort(key=lambda t: t.value, reverse=descending)

    if len(candidates) < k:
        raise RuntimeError(
            f"Only {len(candidates)} complete trials for MODE={mode}, L={L}; need {k}"
        )
    return candidates[:k]


# ---------------------------------------------------------------------------
# Fixed-effects helpers (unchanged)
# ---------------------------------------------------------------------------

def _within_firm_means_y(
    y_tr: np.ndarray,
    m_tr: np.ndarray,
    firm_ids: np.ndarray,
    n_firms: int,
    fallback_global: float
) -> np.ndarray:
    obs = (m_tr > 0)
    sums = np.bincount(firm_ids[obs], weights=y_tr[obs], minlength=n_firms).astype(float)
    cnts = np.bincount(firm_ids[obs], minlength=n_firms).astype(float)
    means = np.empty(n_firms, dtype=float)
    means[:] = fallback_global
    nz = cnts > 0
    means[nz] = sums[nz] / (cnts[nz] + 1e-12)
    return means


def firm_feature_means(Y_tr: np.ndarray, M_tr: np.ndarray) -> np.ndarray:
    """Mask-aware mean over time for each (firm, feature) with fallback to global."""
    denom = M_tr.sum(axis=0)
    mu_ff = (Y_tr * M_tr).sum(axis=0) / (denom + 1e-8)
    denom_feat = M_tr.sum(axis=(0, 1))
    mu_feat = (Y_tr * M_tr).sum(axis=(0, 1)) / (denom_feat + 1e-8)
    missing = denom <= 0
    if np.any(missing):
        idx_firm, idx_feat = np.where(missing)
        mu_ff[idx_firm, idx_feat] = mu_feat[idx_feat]
    return mu_ff


def ar1_persistence_per_feature(
    Y_all: np.ndarray,
    M_all: np.ndarray,
    min_obs: int = 8
) -> np.ndarray:
    """AR(1) persistence (rho) per feature: fit per firm, then average."""
    n_firms = Y_all.shape[1]
    n_feat = Y_all.shape[2]
    rhos = np.full(n_feat, np.nan, dtype=float)
    for j in range(n_feat):
        firm_rhos = []
        for i in range(n_firms):
            mask = M_all[:, i, j] > 0
            series = Y_all[:, i, j][mask]
            if series.size < min_obs:
                continue
            y_t = series[1:]
            y_lag = series[:-1]
            if y_lag.size < 2:
                continue
            var_lag = np.var(y_lag, ddof=1)
            if var_lag < 1e-12:
                continue
            rho = np.cov(y_t, y_lag)[0, 1] / var_lag
            if np.isfinite(rho):
                firm_rhos.append(float(np.clip(rho, -1.0, 1.0)))
        if firm_rhos:
            rhos[j] = float(np.mean(firm_rhos))
    return rhos


# ---------------------------------------------------------------------------
# Ridge baseline (unchanged)
# ---------------------------------------------------------------------------

def ridge_structured_fixed_effects_ts_cv(
    X_tr_4d: np.ndarray, Y_tr_3d: np.ndarray, M_tr_3d: np.ndarray,
    X_val_4d: np.ndarray,
    min_obs_per_feat: Optional[int] = None,
    inner_splits: int = 3
) -> np.ndarray:
    n_tr_t, n_f, n_feat, n_l = X_tr_4d.shape
    n_val_t = X_val_4d.shape[0]
    p = n_feat * n_l

    if min_obs_per_feat is None:
        min_obs_per_feat = get_min_obs_per_feat(n_f, n_tr_t, M_tr_3d)

    X_tr_2d = X_tr_4d.transpose(0, 1, 3, 2).reshape(n_tr_t * n_f, p)
    X_val_2d = X_val_4d.transpose(0, 1, 3, 2).reshape(n_val_t * n_f, p)

    Y_tr_f = Y_tr_3d.reshape(n_tr_t * n_f, n_feat)
    M_tr_f = M_tr_3d.reshape(n_tr_t * n_f, n_feat)

    time_ids_tr = np.repeat(np.arange(n_tr_t), n_f)
    firm_ids_tr = np.tile(np.arange(n_f), n_tr_t)
    firm_ids_val = np.tile(np.arange(n_f), n_val_t)

    denom = M_tr_3d.sum(axis=(0, 1)) + 1e-8
    y_global_mean = (Y_tr_3d * M_tr_3d).sum(axis=(0, 1)) / denom

    X_tr_sc = X_tr_2d
    X_val_sc = X_val_2d

    inner_tscv = TimeSeriesSplit(n_splits=inner_splits)
    y_pred_val_f = np.zeros((n_val_t * n_f, n_feat), dtype=float)
    n_too_sparse = 0
    n_default_alpha = 0

    for j in range(n_feat):
        yj = Y_tr_f[:, j]
        mj = M_tr_f[:, j]

        y_firm_mean_full = _within_firm_means_y(
            y_tr=yj, m_tr=mj, firm_ids=firm_ids_tr, n_firms=n_f,
            fallback_global=float(y_global_mean[j])
        )

        obs_all = (mj > 0)
        if obs_all.sum() < min_obs_per_feat:
            y_pred_val_f[:, j] = y_firm_mean_full[firm_ids_val]
            n_too_sparse += 1
            continue

        best_alpha = None
        best_score = np.inf

        for alpha in RIDGE_ALPHAS:
            fold_mses = []
            for tr_times, va_times in inner_tscv.split(np.arange(n_tr_t)):
                tr_mask = (mj > 0) & np.isin(time_ids_tr, tr_times)
                va_mask = (mj > 0) & np.isin(time_ids_tr, va_times)
                if tr_mask.sum() < min_obs_per_feat or va_mask.sum() < min_obs_per_feat:
                    continue
                inner_obs = mj[tr_mask]
                inner_y = yj[tr_mask]
                inner_global_denom = inner_obs.sum()
                inner_global_mean = (
                    float((inner_y * inner_obs).sum() / (inner_global_denom + 1e-8))
                    if inner_global_denom > 0 else 0.0
                )
                y_firm_mean_inner = _within_firm_means_y(
                    y_tr=inner_y, m_tr=inner_obs,
                    firm_ids=firm_ids_tr[tr_mask], n_firms=n_f,
                    fallback_global=inner_global_mean
                )
                yj_dm_inner_tr = yj[tr_mask] - y_firm_mean_inner[firm_ids_tr[tr_mask]]
                yj_dm_inner_va = yj[va_mask] - y_firm_mean_inner[firm_ids_tr[va_mask]]
                rg = Ridge(alpha=float(alpha), fit_intercept=False, solver="auto",
                           random_state=SEED)
                rg.fit(X_tr_sc[tr_mask], yj_dm_inner_tr)
                pred = rg.predict(X_tr_sc[va_mask])
                mse = float(np.mean((yj_dm_inner_va - pred) ** 2))
                fold_mses.append(mse)

            if len(fold_mses) == 0:
                continue
            avg_mse = float(np.mean(fold_mses))
            if avg_mse < best_score:
                best_score = avg_mse
                best_alpha = float(alpha)

        if best_alpha is None:
            best_alpha = 100.0
            n_default_alpha += 1

        yj_dm_full = yj - y_firm_mean_full[firm_ids_tr]
        rg_final = Ridge(alpha=best_alpha, fit_intercept=False, solver="auto",
                         random_state=SEED)
        rg_final.fit(X_tr_sc[obs_all], yj_dm_full[obs_all])
        y_pred_val_f[:, j] = rg_final.predict(X_val_sc) + y_firm_mean_full[firm_ids_val]

    if n_too_sparse > 0:
        print(f"  [WARN] Ridge: {n_too_sparse}/{n_feat} features too sparse")
    if n_default_alpha > 0:
        print(f"  [WARN] Ridge: {n_default_alpha}/{n_feat} features used default alpha=100")

    return y_pred_val_f.reshape(n_val_t, n_f, n_feat)


# ---------------------------------------------------------------------------
# Single CP model helper (factored out for reuse across trials)
# ---------------------------------------------------------------------------

def _fit_predict_cp(
    X_tr: np.ndarray,
    Y_tr_cent: np.ndarray,
    M_tr: np.ndarray,
    X_val: np.ndarray,
    mu_ff: np.ndarray,
    rank_reg: int,
    reg_w: float,
    use_rms: bool,
) -> Optional[np.ndarray]:
    """Fit a single CP model and return predictions in original scale.

    Returns None when RMS scaling has insufficient support.
    """
    if use_rms:
        y_obs = Y_tr_cent[M_tr > 0]
        min_rms_entries = get_min_valid_entries(M_tr)
        if y_obs.size <= min_rms_entries:
            return None
        y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
        Y_target = Y_tr_cent / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target = Y_tr_cent

    cp_model = CPRegressor(weight_rank=rank_reg, reg_W=reg_w,
                           n_iter_max=150, random_state=SEED)
    cp_model.fit(X_tr, Y_target)
    preds = cp_model.predict(X_val)
    return (preds * y_rms) + mu_ff[None, :, :]


# ---------------------------------------------------------------------------
# Main evaluation loop -- top-K version
# ---------------------------------------------------------------------------

def run_one_combo(
    study: optuna.Study,
    cache_path: Path,
    mode: str,
    L: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Run evaluation for one (mode, L) using the top-K CP trials.

    Returns:
        results_list  -- list of aggregate dicts (K individual + 1 ensemble)
        per_feat_list -- list of per-feature dicts (K individual + 1 ensemble)
    """
    cache = joblib.load(cache_path)
    X_all = cache["X"]
    Y_all = cache["Y"]
    M_all = cache["Mask"]
    feature_names = get_feature_names(Y_all.shape[2])
    n_feat = len(feature_names)

    top_trials = get_top_k_trials_for(study, mode, L)

    trial_params = []
    for idx, bt in enumerate(top_trials):
        rank_reg = int(bt.params["RANK_REGRESS"])
        reg_w = float(bt.params["REG_W"])
        raw_rms = bt.params.get("USE_RMS_SCALING", bt.params.get("RMS_Scaling", False))
        if isinstance(raw_rms, str):
            use_rms = raw_rms.strip().lower() not in ("false", "0", "no", "")
        else:
            use_rms = bool(raw_rms)
        trial_params.append({
            "rank": rank_reg, "reg_w": reg_w, "rms": use_rms,
            "trial_number": bt.number, "trial_cv_value": bt.value,
        })

    print(f"\n=== TOP-{TOP_K} CP ENSEMBLE vs RIDGE | MODE={mode} L={L} ===")
    for idx, tp in enumerate(trial_params):
        print(f"  Trial {idx+1} (#{tp['trial_number']}, cv={tp['trial_cv_value']:.6g}): "
              f"rank={tp['rank']} reg_w={tp['reg_w']:.6g} rms={tp['rms']}")

    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]
    X_test, Y_test, M_test = X_all[split_idx:], Y_all[split_idx:], M_all[split_idx:]

    ar1_rhos = ar1_persistence_per_feature(Y_cv, M_cv)

    # ------------------------------------------------------------------
    # 1.  Cross-Validation on Dev set (80%)
    # ------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=3)

    ridge_scores: List[float] = []
    ridge_feat_scores: List[np.ndarray] = []
    # Per-trial CP scores
    cp_scores_by_trial: List[List[float]] = [[] for _ in range(TOP_K)]
    cp_feat_by_trial: List[List[np.ndarray]] = [[] for _ in range(TOP_K)]
    # Ensemble CP scores
    ens_scores: List[float] = []
    ens_feat_scores: List[np.ndarray] = []

    print("--- Running Cross-Validation on Dev Set (80%) ---")
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]

        mu_ff = firm_feature_means(Y_tr, M_tr)
        Y_tr_cent = (Y_tr - mu_ff[None, :, :]) * M_tr

        # Ridge always runs (independent of CP failures)
        y_ridge_val = ridge_structured_fixed_effects_ts_cv(X_tr, Y_tr, M_tr, X_val)
        r2_ridge = evaluate_model(Y_val, y_ridge_val, M_val)
        r2_ridge_feat = evaluate_model_per_feature(Y_val, y_ridge_val, M_val)
        ridge_scores.append(max(r2_ridge, -1.0) if r2_ridge is not None else np.nan)
        ridge_feat_scores.append(r2_ridge_feat)

        # Fit each CP model independently; failures don't block others
        cp_preds_this_fold: List[Optional[np.ndarray]] = []
        for tp in trial_params:
            preds = _fit_predict_cp(
                X_tr, Y_tr_cent, M_tr, X_val, mu_ff,
                tp["rank"], tp["reg_w"], tp["rms"],
            )
            cp_preds_this_fold.append(preds)

        # Individual CP trial scores (NaN for failed trials)
        for k, preds in enumerate(cp_preds_this_fold):
            if preds is None:
                cp_scores_by_trial[k].append(np.nan)
                cp_feat_by_trial[k].append(np.full(n_feat, np.nan, dtype=float))
                print(f"  [WARN] Fold {fold_idx}: CP trial {k+1} failed (RMS support)")
            else:
                r2 = evaluate_model(Y_val, preds, M_val)
                r2_feat = evaluate_model_per_feature(Y_val, preds, M_val)
                cp_scores_by_trial[k].append(max(r2, -1.0) if r2 is not None else np.nan)
                cp_feat_by_trial[k].append(r2_feat)

        # Ensemble from valid predictions only (consistent with test-time)
        valid_fold_preds = [p for p in cp_preds_this_fold if p is not None]
        if valid_fold_preds:
            ens_pred = np.mean(np.stack(valid_fold_preds, axis=0), axis=0)
            r2_ens = evaluate_model(Y_val, ens_pred, M_val)
            r2_ens_feat = evaluate_model_per_feature(Y_val, ens_pred, M_val)
            ens_scores.append(max(r2_ens, -1.0) if r2_ens is not None else np.nan)
            ens_feat_scores.append(r2_ens_feat)
            if len(valid_fold_preds) < TOP_K:
                print(f"  [WARN] Fold {fold_idx}: ensemble from {len(valid_fold_preds)}/{TOP_K} valid CP models")
        else:
            ens_scores.append(np.nan)
            ens_feat_scores.append(np.full(n_feat, np.nan, dtype=float))

    ridge_mean_cv = float(np.nanmean(ridge_scores))
    ridge_feat_cv = (np.nanmean(np.stack(ridge_feat_scores), axis=0)
                     if ridge_feat_scores else np.full(n_feat, np.nan))

    cp_mean_cv_by_trial = [float(np.nanmean(s)) for s in cp_scores_by_trial]
    cp_feat_cv_by_trial = [
        np.nanmean(np.stack(f), axis=0) if f else np.full(n_feat, np.nan)
        for f in cp_feat_by_trial
    ]
    ens_mean_cv = float(np.nanmean(ens_scores))
    ens_feat_cv = (np.nanmean(np.stack(ens_feat_scores), axis=0)
                   if ens_feat_scores else np.full(n_feat, np.nan))

    # ------------------------------------------------------------------
    # 2.  Hold-out test (train on full dev, predict test)
    # ------------------------------------------------------------------
    print("--- Running Final Evaluation on Hold-Out Test Set (20%) ---")

    y_ridge_test = ridge_structured_fixed_effects_ts_cv(X_cv, Y_cv, M_cv, X_test)
    r2_ridge_test = evaluate_model(Y_test, y_ridge_test, M_test)
    if r2_ridge_test is None:
        r2_ridge_test = np.nan
    ridge_feat_test = evaluate_model_per_feature(Y_test, y_ridge_test, M_test)

    mu_ff_cv = firm_feature_means(Y_cv, M_cv)
    Y_cv_cent = (Y_cv - mu_ff_cv[None, :, :]) * M_cv

    cp_test_preds: List[Optional[np.ndarray]] = []
    for tp in trial_params:
        preds = _fit_predict_cp(
            X_cv, Y_cv_cent, M_cv, X_test, mu_ff_cv,
            tp["rank"], tp["reg_w"], tp["rms"],
        )
        cp_test_preds.append(preds)

    # Individual test results per trial
    r2_cp_test_by_trial = []
    cp_feat_test_by_trial = []
    for k, preds in enumerate(cp_test_preds):
        if preds is None:
            print(f"  [WARN] CP trial {k+1}: insufficient RMS support; skipping test eval")
            r2_cp_test_by_trial.append(np.nan)
            cp_feat_test_by_trial.append(np.full(n_feat, np.nan, dtype=float))
        else:
            r2 = evaluate_model(Y_test, preds, M_test)
            r2_cp_test_by_trial.append(r2 if r2 is not None else np.nan)
            cp_feat_test_by_trial.append(evaluate_model_per_feature(Y_test, preds, M_test))

    # Ensemble test
    valid_test_preds = [p for p in cp_test_preds if p is not None]
    if valid_test_preds:
        ens_test_pred = np.mean(np.stack(valid_test_preds, axis=0), axis=0)
        r2_ens_test = evaluate_model(Y_test, ens_test_pred, M_test)
        if r2_ens_test is None:
            r2_ens_test = np.nan
        ens_feat_test = evaluate_model_per_feature(Y_test, ens_test_pred, M_test)
    else:
        r2_ens_test = np.nan
        ens_feat_test = np.full(n_feat, np.nan, dtype=float)

    print(f"Test Ridge R2:    {r2_ridge_test:.5f}")
    for k in range(TOP_K):
        print(f"Test CP#{k+1} R2:    {r2_cp_test_by_trial[k]:.5f}")
    print(f"Test Ensemble R2: {r2_ens_test:.5f}")

    # ------------------------------------------------------------------
    # 3.  Assemble output rows
    # ------------------------------------------------------------------
    def _safe(v):
        return float(v) if np.isfinite(v) else np.nan

    agg_rows: List[Dict] = []
    feat_rows: List[Dict] = []

    # Per-trial rows
    for k in range(TOP_K):
        tp = trial_params[k]
        cp_test_r2 = r2_cp_test_by_trial[k]
        delta = cp_test_r2 - r2_ridge_test if np.isfinite(cp_test_r2) and np.isfinite(r2_ridge_test) else np.nan

        agg_rows.append({
            "Mode": mode, "L": L, "Trial_Rank": str(k + 1),
            "Trial_Number": tp["trial_number"],
            "Trial_CV_Value": tp["trial_cv_value"],
            "CP_Rank": tp["rank"], "CP_Reg_W": tp["reg_w"], "CP_RMS": tp["rms"],
            "Ridge_CV_Mean": ridge_mean_cv,
            "CP_CV_Mean": cp_mean_cv_by_trial[k],
            "Ridge_Test_R2": r2_ridge_test,
            "CP_Test_R2": cp_test_r2,
            "Delta_Test": delta,
        })

        cp_feat_cv_k = cp_feat_cv_by_trial[k]
        cp_feat_test_k = cp_feat_test_by_trial[k]
        for j, feat in enumerate(feature_names):
            d = (
                float(cp_feat_test_k[j] - ridge_feat_test[j])
                if np.isfinite(cp_feat_test_k[j]) and np.isfinite(ridge_feat_test[j])
                else np.nan
            )
            feat_rows.append({
                "Mode": mode, "L": L, "Trial_Rank": str(k + 1),
                "Feature": feat,
                "Ridge_CV_R2": _safe(ridge_feat_cv[j]),
                "CP_CV_R2": _safe(cp_feat_cv_k[j]),
                "Ridge_Test_R2": _safe(ridge_feat_test[j]),
                "CP_Test_R2": _safe(cp_feat_test_k[j]),
                "AR1_rho": _safe(ar1_rhos[j]),
                "Delta_Test": d,
            })

    # Ensemble row
    ens_delta = r2_ens_test - r2_ridge_test if np.isfinite(r2_ens_test) and np.isfinite(r2_ridge_test) else np.nan
    agg_rows.append({
        "Mode": mode, "L": L, "Trial_Rank": "Ensemble",
        "Trial_Number": "", "Trial_CV_Value": "",
        "CP_Rank": "", "CP_Reg_W": "", "CP_RMS": "",
        "Ridge_CV_Mean": ridge_mean_cv,
        "CP_CV_Mean": ens_mean_cv,
        "Ridge_Test_R2": r2_ridge_test,
        "CP_Test_R2": r2_ens_test,
        "Delta_Test": ens_delta,
    })

    for j, feat in enumerate(feature_names):
        d = (
            float(ens_feat_test[j] - ridge_feat_test[j])
            if np.isfinite(ens_feat_test[j]) and np.isfinite(ridge_feat_test[j])
            else np.nan
        )
        feat_rows.append({
            "Mode": mode, "L": L, "Trial_Rank": "Ensemble",
            "Feature": feat,
            "Ridge_CV_R2": _safe(ridge_feat_cv[j]),
            "CP_CV_R2": _safe(ens_feat_cv[j]),
            "Ridge_Test_R2": _safe(ridge_feat_test[j]),
            "CP_Test_R2": _safe(ens_feat_test[j]),
            "AR1_rho": _safe(ar1_rhos[j]),
            "Delta_Test": d,
        })

    return agg_rows, feat_rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    tl.set_backend("numpy")
    np.random.seed(SEED)

    here = Path(__file__).resolve().parent
    cache_dir = here / "tensor_cache"

    results_list: List[Dict] = []
    per_feature_rows_all: List[Dict] = []

    for mode in MODES:
        db_path = here / f"research_cp_fe_v2_{mode.lower()}.db"
        study_name = f"research_cp_fe_v2_{mode.lower()}"

        if not db_path.exists():
            print(f"[SKIP] Missing study database for {mode}: {db_path}")
            continue

        storage = f"sqlite:///{db_path.as_posix()}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception as e:
            print(f"[SKIP] Could not load study for {mode}: {e}")
            continue

        for L in LOOKBACKS:
            cache_file = cache_dir / FILE_MAP[mode][L]
            if not cache_file.exists():
                print(f"[SKIP] Missing cache: {cache_file}")
                continue
            try:
                agg_rows, feat_rows = run_one_combo(study, cache_file, mode, L)
                results_list.extend(agg_rows)
                per_feature_rows_all.extend(feat_rows)
            except Exception as e:
                print(f"[FAIL] MODE={mode} L={L}: {e}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if not results_list:
        print("\n[WARN] No results collected.")
        return

    df = pd.DataFrame(results_list)
    cols = [
        "Mode", "L", "Trial_Rank",
        "Trial_Number", "Trial_CV_Value",
        "CP_Rank", "CP_Reg_W", "CP_RMS",
        "Ridge_CV_Mean", "CP_CV_Mean",
        "Ridge_Test_R2", "CP_Test_R2", "Delta_Test",
    ]
    df = df[[c for c in cols if c in df.columns]]

    out_file = here / "cp_vs_ridge_top3_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\n[DONE] Results saved to {out_file}")
    print(df.to_string(index=False))

    if not per_feature_rows_all:
        return

    df_feat = pd.DataFrame(per_feature_rows_all)
    feat_out_file = here / "cp_vs_ridge_top3_per_feature.csv"
    df_feat.to_csv(feat_out_file, index=False)
    print(f"\n[DONE] Per-feature results saved to {feat_out_file}")

    # Persistence CSV (ensemble rows only)
    ens_feat = df_feat[df_feat["Trial_Rank"] == "Ensemble"]
    persist_out = ens_feat[["Mode", "L", "Feature", "AR1_rho", "Delta_Test"]]
    persist_out_file = here / "cp_vs_ridge_top3_persistence.csv"
    persist_out.to_csv(persist_out_file, index=False)
    print(f"\n[DONE] Persistence metrics saved to {persist_out_file}")

    # ------------------------------------------------------------------
    # Stats & scatter plots (computed on Ensemble rows)
    # ------------------------------------------------------------------
    stats_rows = []
    for (m, L_val), grp in ens_feat.groupby(["Mode", "L"]):
        plot_df = grp.dropna(subset=["AR1_rho", "Delta_Test"])
        test_df = grp.dropna(subset=["CP_Test_R2", "Ridge_Test_R2"])

        if len(plot_df) >= 5:
            spear_r, spear_p = spearmanr(plot_df["AR1_rho"], plot_df["Delta_Test"])
        else:
            spear_r, spear_p = np.nan, np.nan

        if len(test_df) >= 5:
            try:
                _, wilcox_p = wilcoxon(test_df["CP_Test_R2"], test_df["Ridge_Test_R2"])
            except Exception:
                wilcox_p = np.nan
        else:
            wilcox_p = np.nan

        mean_delta = test_df["Delta_Test"].mean() if len(test_df) > 0 else np.nan

        stats_rows.append({
            "Mode": m, "L": L_val,
            "N_features": len(plot_df),
            "Mean_Delta_Test": mean_delta,
            "Spearman_r": spear_r,
            "Spearman_p": spear_p,
            "Wilcoxon_p": wilcox_p,
        })

        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(plot_df["AR1_rho"], plot_df["Delta_Test"], alpha=0.7, s=60)

        x = plot_df["AR1_rho"].values
        y = plot_df["Delta_Test"].values
        if len(x) >= 3:
            slope, intercept, *_ = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', linewidth=2, label=f"OLS: slope={slope:.4f}")

            n = len(x)
            y_pred_line = slope * x + intercept
            resid_std = np.sqrt(np.sum((y - y_pred_line)**2) / (n - 2)) if n > 2 else 0
            x_mean = np.mean(x)
            x_var = np.sum((x - x_mean)**2)
            if x_var > 1e-12:
                se_line = resid_std * np.sqrt(1/n + (x_line - x_mean)**2 / x_var)
                ax.fill_between(x_line, y_line - 1.96*se_line, y_line + 1.96*se_line,
                                alpha=0.2, color='red', label="95% CI")

        stats_text = (f"Spearman r = {spear_r:.3f}, p = {spear_p:.4f} (raw)\n"
                      f"Wilcoxon p = {wilcox_p:.4f} (raw)")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("AR(1) rho (higher = more persistent)", fontsize=11)
        ax.set_ylabel("Delta_Test (CP Ensemble - Ridge)", fontsize=11)
        ax.set_title(f"Persistence vs Ensemble Delta | MODE={m}, L={L_val}", fontsize=12)
        ax.legend(loc='lower right')
        plt.tight_layout()
        plot_file = here / f"persistence_scatter_top3_{m}_L{L_val}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"[DONE] Saved scatter plot: {plot_file}")

    stats_df = pd.DataFrame(stats_rows)

    spear_pvals = stats_df["Spearman_p"].values
    wilcox_pvals = stats_df["Wilcoxon_p"].values

    valid_spear = ~np.isnan(spear_pvals)
    if valid_spear.sum() > 0:
        _, spear_adj, _, _ = multipletests(spear_pvals[valid_spear], method='fdr_bh')
        stats_df.loc[valid_spear, "Spearman_p_adj"] = spear_adj
    else:
        stats_df["Spearman_p_adj"] = np.nan

    valid_wilcox = ~np.isnan(wilcox_pvals)
    if valid_wilcox.sum() > 0:
        _, wilcox_adj, _, _ = multipletests(wilcox_pvals[valid_wilcox], method='fdr_bh')
        stats_df.loc[valid_wilcox, "Wilcoxon_p_adj"] = wilcox_adj
    else:
        stats_df["Wilcoxon_p_adj"] = np.nan

    col_order = [
        "Mode", "L", "N_features", "Mean_Delta_Test",
        "Spearman_r", "Spearman_p", "Spearman_p_adj",
        "Wilcoxon_p", "Wilcoxon_p_adj",
    ]
    stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]

    stats_file = here / "cp_vs_ridge_top3_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n[DONE] Statistical summary saved to {stats_file}")
    print("\n--- Statistical Significance Summary (BH-adjusted p-values) ---")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
