#!/usr/bin/env python3
"""
CP_struct_test_new.py

Pure CP (Structured) vs Ridge baseline, for BOTH lookbacks L=2 and L=4
(and for both MODE=LEVELS/SURPRISE).

Features:
  - Calculates Mean CV R^2 (on 80% dev set)
  - Calculates Final Hold-Out Test R^2 (fitting on 80% dev, testing on 20% hold-out)
  - Per-feature R^2 and AR(1) persistence analysis
  - Benjamini-Hochberg FDR correction for multiple testing

Outputs:
  - cp_vs_ridge_fe_v2_results.csv      (aggregate results)
  - cp_vs_ridge_fe_v2_per_feature.csv  (per-feature R^2)
  - cp_vs_ridge_fe_v2_persistence.csv  (AR1 rho vs delta)
  - cp_vs_ridge_fe_v2_stats.csv        (stats with BH-adjusted p-values)
  - persistence_scatter_v2_*.png       (scatter plots)

Run:
  python CP_struct_test_new.py
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
import tensorly as tl

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from tensorly.regression.cp_regression import CPRegressor
from scipy.stats import spearmanr, wilcoxon, linregress
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# --- thread safety ---
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

SEED = 42

# --- ADAPTIVE THRESHOLD HELPERS ---
def get_min_valid_entries(mask: np.ndarray, min_frac: float = 0.05, 
                          floor: int = 100, cap: int = 5000) -> int:
    """
    Adaptive threshold for R² evaluation.
    Require 5% of total possible entries, bounded [100, 5000].
    """
    total_entries = int(mask.size)
    return max(floor, min(cap, int(min_frac * total_entries)))

def get_min_obs_per_feat(n_firms: int, n_tr_t: int, mask: np.ndarray,
                          frac: float = 0.05, floor: int = 10, cap: int = 200) -> int:
    """
    Adaptive threshold for per-feature Ridge CV.
    Scale with firms × time × density, clipped to [10, 200].
    """
    mean_density = float(mask.mean()) if mask.size > 0 else 0.5
    adaptive = int(frac * n_firms * n_tr_t * mean_density)
    return max(floor, min(cap, adaptive))

MODES = ["LEVELS", "SURPRISE"]
LOOKBACKS = [2, 4]

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_v2_L2.pkl", 4: "tensor_levels_v2_L4.pkl"},
    "SURPRISE": {2: "tensor_surprise_v2_L2.pkl", 4: "tensor_surprise_v2_L4.pkl"},
}

RIDGE_ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4], dtype=float)

CP_PARAMS: Dict[Tuple[str, int], Dict] = {
    ("LEVELS",   2): {"rank": 8,  "reg_w": 98.1655,  "rms": True},
    ("LEVELS",   4): {"rank": 5,  "reg_w": 67.3035,  "rms": True},
    ("SURPRISE", 2): {"rank": 43, "reg_w": 52.4373,  "rms": True},
    ("SURPRISE", 4): {"rank": 49, "reg_w": 85.3185,  "rms": True},
}

FEATURES = [
    "aoq", "aqaq", "atq", "ltq", "ceqq", "cheq", "ciq", "cogsq", "dlcq", "dlttq",
    "epsfxq", "epspxq", "ibq", "invtq", "intanq", "mibq", "nopiq", "oiadpq",
    "piq", "ppentq", "pstkq", "rectq", "txtq", "xidoq"
]

def get_feature_names(n_feat: int) -> List[str]:
    if len(FEATURES) == n_feat:
        return FEATURES
    return [f"feat_{i}" for i in range(n_feat)]

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
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
    """Feature-wise, mask-aware R². Returns array of shape (Features,).
    If min_obs is None, uses adaptive threshold based on data dimensions.
    """
    n_t, n_f, n_feat = y_true.shape
    if min_obs is None:
        # Adaptive: scale with firms × time × density, clipped to [10, 200]
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
    """
    Mask-aware mean over time for each (firm, feature) with fallback to global feature mean.
    Y_tr, M_tr: (T, Firms, Features)
    Returns mu_ff: (Firms, Features)
    """
    denom = M_tr.sum(axis=0)                 # (Firms, Features)
    mu_ff = (Y_tr * M_tr).sum(axis=0) / (denom + 1e-8)

    denom_feat = M_tr.sum(axis=(0, 1))       # (Features,)
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
    """AR(1) persistence (rho) per feature: fit per firm, then average.
    
    Args:
        min_obs: Minimum observations per firm series (default 8 = 2 years quarterly).
                 Uses a fixed threshold since AR(1) is computed per-firm, not panel-wide.
    
    NOTE: Missing quarters are dropped, treating non-contiguous observations
    as contiguous (lag-1). This is a simplification; true AR(1) with gaps
    would require explicit time indexing. Acceptable as a rough persistence proxy.
    """
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
            # AR(1): y_t = rho * y_{t-1} + e
            y_t = series[1:]
            y_lag = series[:-1]
            if y_lag.size < 2:
                continue
            # OLS: rho = cov(y_t, y_lag) / var(y_lag)
            # Use ddof=1 for consistency (np.cov uses ddof=1 by default)
            var_lag = np.var(y_lag, ddof=1)
            if var_lag < 1e-12:
                continue
            rho = np.cov(y_t, y_lag)[0, 1] / var_lag
            if np.isfinite(rho):
                firm_rhos.append(float(np.clip(rho, -1.0, 1.0)))
        if firm_rhos:
            rhos[j] = float(np.mean(firm_rhos))
    return rhos

def ridge_structured_fixed_effects_ts_cv(
    X_tr_4d: np.ndarray, Y_tr_3d: np.ndarray, M_tr_3d: np.ndarray,
    X_val_4d: np.ndarray,
    min_obs_per_feat: Optional[int] = None,
    inner_splits: int = 3
) -> np.ndarray:
    n_tr_t, n_f, n_feat, n_l = X_tr_4d.shape
    n_val_t = X_val_4d.shape[0]
    p = n_feat * n_l

    # Compute adaptive threshold if not provided
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

    # Use raw X (no firm demeaning or scaling) to match CP preprocessing
    X_tr_sc = X_tr_2d
    X_val_sc = X_val_2d

    inner_tscv = TimeSeriesSplit(n_splits=inner_splits)
    y_pred_val_f = np.zeros((n_val_t * n_f, n_feat), dtype=float)
    n_too_sparse = 0      # Features with too few observations (firm-mean only)
    n_default_alpha = 0   # Features with data but no CV evidence (default alpha=100)

    for j in range(n_feat):
        yj = Y_tr_f[:, j]
        mj = M_tr_f[:, j]

        # Compute firm means on FULL outer training data (for final model + prediction)
        y_firm_mean_full = _within_firm_means_y(
            y_tr=yj, m_tr=mj, firm_ids=firm_ids_tr, n_firms=n_f,
            fallback_global=float(y_global_mean[j])
        )

        obs_all = (mj > 0)
        if obs_all.sum() < min_obs_per_feat:
            # Too sparse: use firm-mean baseline only (no Ridge model)
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

                # Compute firm means on INNER training data only (no leakage)
                # Global fallback also computed from inner training only
                inner_obs = mj[tr_mask]
                inner_y = yj[tr_mask]
                inner_global_denom = inner_obs.sum()
                inner_global_mean = float((inner_y * inner_obs).sum() / (inner_global_denom + 1e-8)) if inner_global_denom > 0 else 0.0
                
                y_firm_mean_inner = _within_firm_means_y(
                    y_tr=inner_y, m_tr=inner_obs, 
                    firm_ids=firm_ids_tr[tr_mask], n_firms=n_f,
                    fallback_global=inner_global_mean
                )
                
                # Demean using inner-fold means
                yj_dm_inner_tr = yj[tr_mask] - y_firm_mean_inner[firm_ids_tr[tr_mask]]
                yj_dm_inner_va = yj[va_mask] - y_firm_mean_inner[firm_ids_tr[va_mask]]

                rg = Ridge(alpha=float(alpha), fit_intercept=False, solver="auto", random_state=SEED)
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
            # Has data but all inner folds were skipped: use default alpha
            best_alpha = 100.0
            n_default_alpha += 1

        # Final model uses full outer training data
        yj_dm_full = yj - y_firm_mean_full[firm_ids_tr]
        rg_final = Ridge(alpha=best_alpha, fit_intercept=False, solver="auto", random_state=SEED)
        rg_final.fit(X_tr_sc[obs_all], yj_dm_full[obs_all])
        y_pred_val_f[:, j] = rg_final.predict(X_val_sc) + y_firm_mean_full[firm_ids_val]

    if n_too_sparse > 0:
        print(f"  [WARN] Ridge: {n_too_sparse}/{n_feat} features too sparse (firm-mean baseline only)")
    if n_default_alpha > 0:
        print(f"  [WARN] Ridge: {n_default_alpha}/{n_feat} features used default alpha=100 (no inner CV evidence)")

    return y_pred_val_f.reshape(n_val_t, n_f, n_feat)

def run_one_combo(
    cache_path: Path,
    mode: str,
    L: int
) -> Tuple[Dict, List[Dict]]:
    cache = joblib.load(cache_path)
    X_all = cache["X"]     # (T, Firms, Features, L)
    Y_all = cache["Y"]     # (T, Firms, Features)
    M_all = cache["Mask"]  # (T, Firms, Features)
    feature_names = get_feature_names(Y_all.shape[2])

    cp = CP_PARAMS[(mode, L)]
    rank_reg = cp["rank"]
    reg_w = cp["reg_w"]
    use_rms = cp["rms"]

    print(f"\n=== PURE CP STRUCT vs RIDGE (FE+TS-CV Ridge) | MODE={mode} L={L} ===")
    print(f"CP Params: rank={rank_reg} reg_w={reg_w:.6g} rms={use_rms}")

    # --- Split Data: 80% Dev (CV), 20% Hold-out Test ---
    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]
    X_test, Y_test, M_test = X_all[split_idx:], Y_all[split_idx:], M_all[split_idx:]

    # Compute AR1 persistence on training data only (avoid test leakage)
    ar1_rhos = ar1_persistence_per_feature(Y_cv, M_cv)

    # 1. Cross-Validation Loop (on Dev set)
    tscv = TimeSeriesSplit(n_splits=3)
    ridge_scores: List[float] = []
    cp_scores: List[float] = []
    ridge_feat_scores: List[np.ndarray] = []
    cp_feat_scores: List[np.ndarray] = []

    print("--- Running Cross-Validation on Dev Set (80%) ---")
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]

        # CP prep
        mu_ff = firm_feature_means(Y_tr, M_tr)
        # NOTE: Missing entries become 0 after masking. CPRegressor fits on ALL entries
        # (no mask support), so it implicitly assumes missing residuals = 0. This is a
        # known limitation; evaluation is mask-aware so scoring is unaffected.
        Y_tr_cent = (Y_tr - mu_ff[None, :, :]) * M_tr

        # If CP RMS support is too low, skip the fold for BOTH models
        if use_rms:
            y_obs = Y_tr_cent[M_tr > 0]
            min_rms_entries = get_min_valid_entries(M_tr)
            if y_obs.size <= min_rms_entries:
                ridge_scores.append(np.nan)
                cp_scores.append(np.nan)
                ridge_feat_scores.append(np.full(len(feature_names), np.nan, dtype=float))
                cp_feat_scores.append(np.full(len(feature_names), np.nan, dtype=float))
                continue

        # Ridge (only if CP fold is valid)
        y_ridge_val = ridge_structured_fixed_effects_ts_cv(X_tr, Y_tr, M_tr, X_val)
        r2_ridge = evaluate_model(Y_val, y_ridge_val, M_val)
        r2_ridge_feat = evaluate_model_per_feature(Y_val, y_ridge_val, M_val)
        
        # CP
        if use_rms:
            y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
            Y_target_scaled = Y_tr_cent / (y_rms + 1e-8)
        else:
            y_rms = 1.0
            Y_target_scaled = Y_tr_cent

        cp_model = CPRegressor(weight_rank=rank_reg, reg_W=reg_w, n_iter_max=150, random_state=SEED)
        cp_model.fit(X_tr, Y_target_scaled)
        preds = cp_model.predict(X_val)
        y_cp_final = (preds * y_rms) + mu_ff[None, :, :]
        r2_cp = evaluate_model(Y_val, y_cp_final, M_val)
        r2_cp_feat = evaluate_model_per_feature(Y_val, y_cp_final, M_val)

        r_sc = max(r2_ridge, -1.0) if r2_ridge is not None else np.nan
        c_sc = max(r2_cp, -1.0) if r2_cp is not None else np.nan
        ridge_scores.append(r_sc)
        cp_scores.append(c_sc)
        ridge_feat_scores.append(r2_ridge_feat)
        cp_feat_scores.append(r2_cp_feat)

    ridge_mean_cv = float(np.nanmean(ridge_scores))
    cp_mean_cv = float(np.nanmean(cp_scores))
    ridge_feat_cv = np.nanmean(np.stack(ridge_feat_scores), axis=0) if ridge_feat_scores else np.full(len(feature_names), np.nan)
    cp_feat_cv = np.nanmean(np.stack(cp_feat_scores), axis=0) if cp_feat_scores else np.full(len(feature_names), np.nan)

    # 2. Final Hold-Out Test (Train on full Dev, Predict Test)
    print("--- Running Final Evaluation on Hold-Out Test Set (20%) ---")
    
    # Ridge Final
    y_ridge_test = ridge_structured_fixed_effects_ts_cv(X_cv, Y_cv, M_cv, X_test)
    r2_ridge_test = evaluate_model(Y_test, y_ridge_test, M_test)
    if r2_ridge_test is None: r2_ridge_test = np.nan
    ridge_feat_test = evaluate_model_per_feature(Y_test, y_ridge_test, M_test)

    # CP Final
    mu_ff_cv = firm_feature_means(Y_cv, M_cv)
    # NOTE: Missing entries become 0 (see CV loop comment for rationale)
    Y_cv_cent = (Y_cv - mu_ff_cv[None, :, :]) * M_cv

    if use_rms:
        y_obs = Y_cv_cent[M_cv > 0]
        min_rms_entries = get_min_valid_entries(M_cv)
        if y_obs.size <= min_rms_entries:
            print("  [WARN] CP: insufficient RMS support on full dev set; skipping CP test eval")
            r2_cp_test = np.nan
            cp_feat_test = np.full(len(feature_names), np.nan, dtype=float)
        else:
            y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
            Y_target_final = Y_cv_cent / (y_rms + 1e-8)

            cp_final = CPRegressor(weight_rank=rank_reg, reg_W=reg_w, n_iter_max=150, random_state=SEED)
            cp_final.fit(X_cv, Y_target_final)
            preds_test = cp_final.predict(X_test)
            y_cp_test = (preds_test * y_rms) + mu_ff_cv[None, :, :]
            r2_cp_test = evaluate_model(Y_test, y_cp_test, M_test)
            if r2_cp_test is None: r2_cp_test = np.nan
            cp_feat_test = evaluate_model_per_feature(Y_test, y_cp_test, M_test)
    else:
        y_rms = 1.0
        Y_target_final = Y_cv_cent

        cp_final = CPRegressor(weight_rank=rank_reg, reg_W=reg_w, n_iter_max=150, random_state=SEED)
        cp_final.fit(X_cv, Y_target_final)
        preds_test = cp_final.predict(X_test)
        y_cp_test = (preds_test * y_rms) + mu_ff_cv[None, :, :]
        r2_cp_test = evaluate_model(Y_test, y_cp_test, M_test)
        if r2_cp_test is None: r2_cp_test = np.nan
        cp_feat_test = evaluate_model_per_feature(Y_test, y_cp_test, M_test)

    print(f"Test Ridge R2: {r2_ridge_test:.5f}")
    print(f"Test CP R2:    {r2_cp_test:.5f}")

    per_feature_rows: List[Dict] = []
    for j, feat in enumerate(feature_names):
        per_feature_rows.append({
            "Mode": mode,
            "L": L,
            "Feature": feat,
            "Ridge_CV_R2": float(ridge_feat_cv[j]) if np.isfinite(ridge_feat_cv[j]) else np.nan,
            "CP_CV_R2": float(cp_feat_cv[j]) if np.isfinite(cp_feat_cv[j]) else np.nan,
            "Ridge_Test_R2": float(ridge_feat_test[j]) if np.isfinite(ridge_feat_test[j]) else np.nan,
            "CP_Test_R2": float(cp_feat_test[j]) if np.isfinite(cp_feat_test[j]) else np.nan,
            "AR1_rho": float(ar1_rhos[j]) if np.isfinite(ar1_rhos[j]) else np.nan,
            "Delta_Test": float(cp_feat_test[j] - ridge_feat_test[j]) if np.isfinite(cp_feat_test[j]) and np.isfinite(ridge_feat_test[j]) else np.nan
        })

    return {
        "Mode": mode,
        "L": L,
        "Ridge_CV_Mean": ridge_mean_cv,
        "CP_CV_Mean": cp_mean_cv,
        "Ridge_Test_R2": r2_ridge_test,
        "CP_Test_R2": r2_cp_test,
        "Delta_Test": r2_cp_test - r2_ridge_test
    }, per_feature_rows

def main():
    tl.set_backend("numpy")
    np.random.seed(SEED)

    here = Path(__file__).resolve().parent
    cache_dir = here / "tensor_cache"

    results_list = []
    per_feature_rows_all: List[Dict] = []

    for mode in MODES:
        for L in LOOKBACKS:
            cache_file = cache_dir / FILE_MAP[mode][L]
            if not cache_file.exists():
                print(f"[SKIP] Missing cache: {cache_file}")
                continue
            try:
                res, per_rows = run_one_combo(cache_file, mode, L)
                results_list.append(res)
                per_feature_rows_all.extend(per_rows)
            except Exception as e:
                print(f"[FAIL] MODE={mode} L={L}: {e}")

    # Create DataFrame and save
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Reorder columns for readability
        cols = ["Mode", "L", "Ridge_CV_Mean", "CP_CV_Mean", "Ridge_Test_R2", "CP_Test_R2", "Delta_Test"]
        df = df[cols]
        
        out_file = here / "cp_vs_ridge_fe_v2_results.csv"
        df.to_csv(out_file, index=False)
        
        print(f"\n[DONE] Results saved to {out_file}")
        print(df.to_string(index=False))

        if per_feature_rows_all:
            df_feat = pd.DataFrame(per_feature_rows_all)
            feat_out_file = here / "cp_vs_ridge_fe_v2_per_feature.csv"
            df_feat.to_csv(feat_out_file, index=False)
            print(f"\n[DONE] Per-feature results saved to {feat_out_file}")

            persist_out = df_feat[["Mode", "L", "Feature", "AR1_rho", "Delta_Test"]]
            persist_out_file = here / "cp_vs_ridge_fe_v2_persistence.csv"
            persist_out.to_csv(persist_out_file, index=False)
            print(f"\n[DONE] Persistence metrics saved to {persist_out_file}")

            # Statistical significance summary
            stats_rows = []
            for (mode, L), grp in df_feat.groupby(["Mode", "L"]):
                plot_df = grp.dropna(subset=["AR1_rho", "Delta_Test"])
                test_df = grp.dropna(subset=["CP_Test_R2", "Ridge_Test_R2"])
                
                # Spearman correlation: persistence vs delta
                if len(plot_df) >= 5:
                    spear_r, spear_p = spearmanr(plot_df["AR1_rho"], plot_df["Delta_Test"])
                else:
                    spear_r, spear_p = np.nan, np.nan
                
                # Paired Wilcoxon: CP vs Ridge test R²
                if len(test_df) >= 5:
                    try:
                        wilcox_stat, wilcox_p = wilcoxon(test_df["CP_Test_R2"], test_df["Ridge_Test_R2"])
                    except Exception:
                        wilcox_stat, wilcox_p = np.nan, np.nan
                else:
                    wilcox_stat, wilcox_p = np.nan, np.nan
                
                mean_delta = test_df["Delta_Test"].mean() if len(test_df) > 0 else np.nan
                
                stats_rows.append({
                    "Mode": mode,
                    "L": L,
                    "N_features": len(plot_df),
                    "Mean_Delta_Test": mean_delta,
                    "Spearman_r": spear_r,
                    "Spearman_p": spear_p,
                    "Wilcoxon_p": wilcox_p
                })
                
                # Scatter plot with regression line and stats
                if plot_df.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(plot_df["AR1_rho"], plot_df["Delta_Test"], alpha=0.7, s=60)
                
                # Regression line
                x = plot_df["AR1_rho"].values
                y = plot_df["Delta_Test"].values
                if len(x) >= 3:
                    slope, intercept, r_val, p_val, std_err = linregress(x, y)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f"OLS: slope={slope:.4f}")
                    
                    # Confidence band (approximate 95% CI) - guard against zero variance
                    n = len(x)
                    y_pred = slope * x + intercept
                    resid_std = np.sqrt(np.sum((y - y_pred)**2) / (n - 2)) if n > 2 else 0
                    x_mean = np.mean(x)
                    x_var = np.sum((x - x_mean)**2)
                    if x_var > 1e-12:  # Only draw CI band if x has variance
                        se_line = resid_std * np.sqrt(1/n + (x_line - x_mean)**2 / x_var)
                        ax.fill_between(x_line, y_line - 1.96*se_line, y_line + 1.96*se_line, 
                                        alpha=0.2, color='red', label="95% CI")
                
                # Annotate with stats (note: raw p-values; see stats CSV for BH-adjusted)
                stats_text = f"Spearman r = {spear_r:.3f}, p = {spear_p:.4f} (raw)\nWilcoxon p = {wilcox_p:.4f} (raw)"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel("AR(1) rho (higher = more persistent)", fontsize=11)
                ax.set_ylabel("Delta_Test (CP - Ridge)", fontsize=11)
                ax.set_title(f"Persistence vs Performance Delta | MODE={mode}, L={L}", fontsize=12)
                ax.legend(loc='lower right')
                plt.tight_layout()
                plot_file = here / f"persistence_scatter_v2_{mode}_L{L}.png"
                plt.savefig(plot_file, dpi=150)
                plt.close()
                print(f"[DONE] Saved scatter plot: {plot_file}")
            
            # Apply Benjamini-Hochberg correction for multiple testing
            stats_df = pd.DataFrame(stats_rows)
            
            # Collect all p-values for correction
            spear_pvals = stats_df["Spearman_p"].values
            wilcox_pvals = stats_df["Wilcoxon_p"].values
            
            # BH correction for Spearman p-values
            valid_spear = ~np.isnan(spear_pvals)
            if valid_spear.sum() > 0:
                _, spear_adj, _, _ = multipletests(spear_pvals[valid_spear], method='fdr_bh')
                stats_df.loc[valid_spear, "Spearman_p_adj"] = spear_adj
            else:
                stats_df["Spearman_p_adj"] = np.nan
            
            # BH correction for Wilcoxon p-values
            valid_wilcox = ~np.isnan(wilcox_pvals)
            if valid_wilcox.sum() > 0:
                _, wilcox_adj, _, _ = multipletests(wilcox_pvals[valid_wilcox], method='fdr_bh')
                stats_df.loc[valid_wilcox, "Wilcoxon_p_adj"] = wilcox_adj
            else:
                stats_df["Wilcoxon_p_adj"] = np.nan
            
            # Reorder columns
            col_order = ["Mode", "L", "N_features", "Mean_Delta_Test", 
                        "Spearman_r", "Spearman_p", "Spearman_p_adj",
                        "Wilcoxon_p", "Wilcoxon_p_adj"]
            stats_df = stats_df[[c for c in col_order if c in stats_df.columns]]
            
            # Save stats summary
            stats_file = here / "cp_vs_ridge_fe_v2_stats.csv"
            stats_df.to_csv(stats_file, index=False)
            print(f"\n[DONE] Statistical summary saved to {stats_file}")
            print("\n--- Statistical Significance Summary (BH-adjusted p-values) ---")
            print(stats_df.to_string(index=False))
    else:
        print("\n[WARN] No results collected.")

if __name__ == "__main__":
    main()