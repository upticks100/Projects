#!/usr/bin/env python3
"""
compare_pure_cp_struct_vs_ridge_L2_L4__ridgeFE_tsCV.py

Pure CP (Structured) vs Ridge baseline, for BOTH lookbacks L=2 and L=4
(and for both MODE=LEVELS/SURPRISE).

Updates:
  - Calculates Mean CV R^2 (on 80% dev set)
  - Calculates Final Hold-Out Test R^2 (fitting on 80% dev, testing on 20% hold-out)
  - Saves all results to 'pure_cp_vs_ridge_results.csv'

Run:
  python compare_pure_cp_struct_vs_ridge_L2_L4__ridgeFE_tsCV.py
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
from sklearn.preprocessing import StandardScaler
from tensorly.regression.cp_regression import CPRegressor

# --- thread safety ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

SEED = 42
MIN_VALID_ENTRIES = 1000
MIN_OBS_PER_FEAT = 25

MODES = ["LEVELS", "SURPRISE"]
LOOKBACKS = [2, 4]

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_L2.pkl", 4: "tensor_levels_L4.pkl"},
    "SURPRISE": {2: "tensor_L2_R40_20_2.pkl", 4: "tensor_L4_R40_20_4.pkl"},
}

RIDGE_ALPHAS = np.array([1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4], dtype=float)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Pooled, mask-aware R²."""
    y_t, y_p, m = y_true.flatten(), y_pred.flatten(), mask.flatten()
    valid = m > 0
    if np.sum(valid) < MIN_VALID_ENTRIES:
        return None
    clean_t, clean_p = y_t[valid], y_p[valid]
    sst = np.sum((clean_t - np.mean(clean_t)) ** 2)
    if sst <= 1e-8:
        return None
    return 1.0 - np.sum((clean_t - clean_p) ** 2) / sst

def get_best_trial_for(study: optuna.Study, mode: str, L: int) -> optuna.trial.FrozenTrial:
    best = None
    for t in study.get_trials(deepcopy=False):
        if t.value is None:
            continue
        if t.params.get("MODE") != mode:
            continue
        if int(t.params.get("L")) != int(L):
            continue
        if (best is None) or (t.value > best.value):
            best = t
    if best is None:
        raise RuntimeError(f"No COMPLETE trials for MODE={mode}, L={L}")
    return best

def _within_firm_means_X(X_tr_2d: np.ndarray, firm_ids: np.ndarray, n_firms: int) -> np.ndarray:
    means = np.zeros((n_firms, X_tr_2d.shape[1]), dtype=float)
    for i in range(n_firms):
        means[i] = X_tr_2d[firm_ids == i].mean(axis=0)
    return means

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

def ridge_structured_fixed_effects_ts_cv(
    X_tr_4d: np.ndarray, Y_tr_3d: np.ndarray, M_tr_3d: np.ndarray,
    X_val_4d: np.ndarray,
    min_obs_per_feat: int = MIN_OBS_PER_FEAT,
    inner_splits: int = 3
) -> np.ndarray:
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

    X_firm_mean = _within_firm_means_X(X_tr_2d, firm_ids_tr, n_f)
    X_tr_dm = X_tr_2d - X_firm_mean[firm_ids_tr]
    X_val_dm = X_val_2d - X_firm_mean[firm_ids_val]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_sc = scaler.fit_transform(X_tr_dm)
    X_val_sc = scaler.transform(X_val_dm)

    inner_tscv = TimeSeriesSplit(n_splits=inner_splits)
    y_pred_val_f = np.zeros((n_val_t * n_f, n_feat), dtype=float)

    for j in range(n_feat):
        yj = Y_tr_f[:, j]
        mj = M_tr_f[:, j]

        y_firm_mean = _within_firm_means_y(
            y_tr=yj, m_tr=mj, firm_ids=firm_ids_tr, n_firms=n_f,
            fallback_global=float(y_global_mean[j])
        )

        obs_all = (mj > 0)
        if obs_all.sum() < min_obs_per_feat:
            y_pred_val_f[:, j] = y_firm_mean[firm_ids_val]
            continue

        yj_dm = yj - y_firm_mean[firm_ids_tr]

        best_alpha = None
        best_score = np.inf

        for alpha in RIDGE_ALPHAS:
            fold_mses = []
            for tr_times, va_times in inner_tscv.split(np.arange(n_tr_t)):
                tr_mask = (mj > 0) & np.isin(time_ids_tr, tr_times)
                va_mask = (mj > 0) & np.isin(time_ids_tr, va_times)

                if tr_mask.sum() < min_obs_per_feat or va_mask.sum() < min_obs_per_feat:
                    continue

                rg = Ridge(alpha=float(alpha), fit_intercept=False, solver="auto", random_state=SEED)
                rg.fit(X_tr_sc[tr_mask], yj_dm[tr_mask])
                pred = rg.predict(X_tr_sc[va_mask])
                mse = float(np.mean((yj_dm[va_mask] - pred) ** 2))
                fold_mses.append(mse)

            if len(fold_mses) == 0:
                continue

            avg_mse = float(np.mean(fold_mses))
            if avg_mse < best_score:
                best_score = avg_mse
                best_alpha = float(alpha)

        if best_alpha is None:
            best_alpha = 100.0

        rg_final = Ridge(alpha=best_alpha, fit_intercept=False, solver="auto", random_state=SEED)
        rg_final.fit(X_tr_sc[obs_all], yj_dm[obs_all])
        y_pred_val_f[:, j] = rg_final.predict(X_val_sc) + y_firm_mean[firm_ids_val]

    return y_pred_val_f.reshape(n_val_t, n_f, n_feat)

def run_one_combo(study: optuna.Study, cache_path: Path, mode: str, L: int) -> Dict:
    cache = joblib.load(cache_path)
    X_all = cache["X"]     # (T, Firms, Features, L)
    Y_all = cache["Y"]     # (T, Firms, Features)
    M_all = cache["Mask"]  # (T, Firms, Features)

    bt = get_best_trial_for(study, mode, L)

    rank_reg = int(bt.params["RANK_REGRESS"])
    reg_w = float(bt.params["REG_W"])
    use_rms = bool(bt.params.get("USE_RMS_SCALING", bt.params.get("RMS_Scaling", False)))

    print(f"\n=== PURE CP STRUCT vs RIDGE (FE+TS-CV Ridge) | MODE={mode} L={L} ===")
    print(f"Best CP Params: rank={rank_reg} reg_w={reg_w:.6g} rms={use_rms}")

    # --- Split Data: 80% Dev (CV), 20% Hold-out Test ---
    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]
    X_test, Y_test, M_test = X_all[split_idx:], Y_all[split_idx:], M_all[split_idx:]

    # 1. Cross-Validation Loop (on Dev set)
    tscv = TimeSeriesSplit(n_splits=3)
    ridge_scores: List[float] = []
    cp_scores: List[float] = []

    print("--- Running Cross-Validation on Dev Set (80%) ---")
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]

        # Ridge
        y_ridge_val = ridge_structured_fixed_effects_ts_cv(X_tr, Y_tr, M_tr, X_val)
        r2_ridge = evaluate_model(Y_val, y_ridge_val, M_val)
        
        # CP
        denom = M_tr.sum(axis=(0, 1)) + 1e-8
        y_mean_tr = (Y_tr * M_tr).sum(axis=(0, 1)) / denom
        Y_tr_cent = (Y_tr - y_mean_tr) * M_tr

        if use_rms:
            y_obs = Y_tr_cent[M_tr > 0]
            if y_obs.size > MIN_VALID_ENTRIES:
                y_rms = float(np.sqrt(np.mean(y_obs ** 2)))
                Y_target_scaled = Y_tr_cent / (y_rms + 1e-8)
            else:
                y_rms = 1.0
                Y_target_scaled = Y_tr_cent
        else:
            y_rms = 1.0
            Y_target_scaled = Y_tr_cent

        cp_model = CPRegressor(weight_rank=rank_reg, reg_W=reg_w, n_iter_max=150, random_state=SEED)
        cp_model.fit(X_tr, Y_target_scaled)
        preds = cp_model.predict(X_val)
        y_cp_final = (preds * y_rms) + y_mean_tr
        r2_cp = evaluate_model(Y_val, y_cp_final, M_val)

        r_sc = max(r2_ridge, -1.0) if r2_ridge is not None else np.nan
        c_sc = max(r2_cp, -1.0) if r2_cp is not None else np.nan
        ridge_scores.append(r_sc)
        cp_scores.append(c_sc)

    ridge_mean_cv = float(np.nanmean(ridge_scores))
    cp_mean_cv = float(np.nanmean(cp_scores))

    # 2. Final Hold-Out Test (Train on full Dev, Predict Test)
    print("--- Running Final Evaluation on Hold-Out Test Set (20%) ---")
    
    # Ridge Final
    y_ridge_test = ridge_structured_fixed_effects_ts_cv(X_cv, Y_cv, M_cv, X_test)
    r2_ridge_test = evaluate_model(Y_test, y_ridge_test, M_test)
    if r2_ridge_test is None: r2_ridge_test = np.nan

    # CP Final
    denom = M_cv.sum(axis=(0, 1)) + 1e-8
    y_mean_cv = (Y_cv * M_cv).sum(axis=(0, 1)) / denom
    Y_cv_cent = (Y_cv - y_mean_cv) * M_cv

    if use_rms:
        y_obs = Y_cv_cent[M_cv > 0]
        y_rms = float(np.sqrt(np.mean(y_obs ** 2))) if y_obs.size > 100 else 1.0
        Y_target_final = Y_cv_cent / (y_rms + 1e-8)
    else:
        y_rms = 1.0
        Y_target_final = Y_cv_cent

    cp_final = CPRegressor(weight_rank=rank_reg, reg_W=reg_w, n_iter_max=150, random_state=SEED)
    cp_final.fit(X_cv, Y_target_final)
    preds_test = cp_final.predict(X_test)
    y_cp_test = (preds_test * y_rms) + y_mean_cv
    r2_cp_test = evaluate_model(Y_test, y_cp_test, M_test)
    if r2_cp_test is None: r2_cp_test = np.nan

    print(f"Test Ridge R2: {r2_ridge_test:.5f}")
    print(f"Test CP R2:    {r2_cp_test:.5f}")

    return {
        "Mode": mode,
        "L": L,
        "Ridge_CV_Mean": ridge_mean_cv,
        "CP_CV_Mean": cp_mean_cv,
        "Ridge_Test_R2": r2_ridge_test,
        "CP_Test_R2": r2_cp_test,
        "Delta_Test": r2_cp_test - r2_ridge_test
    }

def main():
    tl.set_backend("numpy")
    np.random.seed(SEED)

    here = Path(__file__).resolve().parent
    cache_dir = here / "tensor_cache"

    db_path = Path("/student/mcnama53/research_pure_struct.db")
    study_name = "research_pure_struct_v1"
    storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    results_list = []

    for mode in MODES:
        for L in LOOKBACKS:
            cache_file = cache_dir / FILE_MAP[mode][L]
            if not cache_file.exists():
                print(f"[SKIP] Missing cache: {cache_file}")
                continue
            try:
                res = run_one_combo(study, cache_file, mode, L)
                results_list.append(res)
            except Exception as e:
                print(f"[FAIL] MODE={mode} L={L}: {e}")

    # Create DataFrame and save
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Reorder columns for readability
        cols = ["Mode", "L", "Ridge_CV_Mean", "CP_CV_Mean", "Ridge_Test_R2", "CP_Test_R2", "Delta_Test"]
        df = df[cols]
        
        out_file = "pure_cp_vs_ridge_results.csv"
        df.to_csv(out_file, index=False)
        
        print(f"\n[DONE] Results saved to {out_file}")
        print(df.to_string(index=False))
    else:
        print("\n[WARN] No results collected.")

if __name__ == "__main__":
    main()