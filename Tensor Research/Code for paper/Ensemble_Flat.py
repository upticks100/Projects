"""
Ensemble: Dual-Mode Flattened (Final Production Version)
Strategy: Feature-wise Ridge + Flattened (Sample x Feature x Lag) Tensor Residuals
Audit: Corrected Centering, Stability Pruning, A/B Scaling, Full Logging
FIX: Removed [..., 0] slicing to match new Imputer format (Y/Mask are 3D).
"""

import os
import joblib
import numpy as np
import optuna
import warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import tensorly as tl
from tensorly.regression.cp_regression import CPRegressor

# 1. CLUSTER SAFETY
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# --- GLOBAL CONFIG ---
SEED       = 42
N_TRIALS   = 500
CACHE_DIR  = Path(__file__).resolve().parent / "tensor_cache"

LOG_MAP  = {"LEVELS": "log_research_flat_levels.txt",
            "SURPRISE": "log_research_flat_surprise.txt"}

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_L2.pkl", 4: "tensor_levels_L4.pkl"},
    "SURPRISE": {2: "tensor_L2_R40_20_2.pkl", 4: "tensor_L4_R40_20_4.pkl"}
}

def evaluate_ensemble(y_true, y_pred, mask):
    y_t, y_p, m = y_true.flatten(), y_pred.flatten(), mask.flatten()
    valid = m > 0
    if np.sum(valid) < 1000:
        return None  # Prune sparse folds
    clean_t, clean_p = y_t[valid], y_p[valid]
    sst = np.sum((clean_t - np.mean(clean_t))**2)
    return 1.0 - np.sum((clean_t - clean_p)**2) / sst if sst > 1e-8 else None

def objective(trial):
    # --- EXPERIMENTAL SETUP ---
    target_mode = trial.suggest_categorical("MODE", ["LEVELS", "SURPRISE"])
    L = trial.suggest_categorical("L", [2, 4])
    use_rms_scaling = trial.suggest_categorical("USE_RMS_SCALING", [True, False])

    cache = joblib.load(CACHE_DIR / FILE_MAP[target_mode][L])

    # --- CRITICAL FIX: NO [..., 0] SLICING ---
    # Cached shapes:
    #   X:    (T, Firms, Features, L)
    #   Y:    (T, Firms, Features)
    #   Mask: (T, Firms, Features)
    X_all = cache["X"]
    Y_all = cache["Y"]
    Mask_all = cache["Mask"]

    # --- HYPERPARAMETERS ---
    ridge_alpha = trial.suggest_float("RIDGE_ALPHA", 1.0, 1000.0, log=True)
    rank_reg    = trial.suggest_int("RANK_REGRESS", 2, 15)
    reg_w       = trial.suggest_float("REG_W", 1e-5, 10.0, log=True)
    gamma       = trial.suggest_float("GAMMA", 0.0, 2.0)

    tscv = TimeSeriesSplit(n_splits=3)
    scores, rms_vals, density_vals = [], [], []

    # Holdout protection
    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], Mask_all[:split_idx]

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]
        n_tr_t, n_f, n_feat, n_l = X_tr.shape

        # 1. 3D-Global Mask-Aware Centering
        denom_3d = M_tr.sum(axis=(0, 1)) + 1e-8
        y_mean_tr = (Y_tr * M_tr).sum(axis=(0, 1)) / denom_3d

        # 2. Flattening & Global Filtering
        # CP input:   (Samples, Features, Lags)
        # Ridge input:(Samples, Features*Lags)
        X_tr_flat_3d = X_tr.transpose(0, 1, 3, 2).reshape(n_tr_t * n_f, n_feat, n_l)
        X_tr_ridge_2d = X_tr_flat_3d.reshape(n_tr_t * n_f, -1)

        Y_tr_f = Y_tr.reshape(n_tr_t * n_f, n_feat)
        M_tr_f = M_tr.reshape(n_tr_t * n_f, n_feat)

        # Filter: keep rows (firm-quarters) with at least one observed feature
        f_mask_tr = M_tr_f.any(axis=1)

        # Apply Filter
        X_tr_tens_c  = X_tr_flat_3d[f_mask_tr]
        X_tr_ridge_c = X_tr_ridge_2d[f_mask_tr]
        Y_tr_c       = Y_tr_f[f_mask_tr]
        M_tr_c       = M_tr_f[f_mask_tr]

        # 3. Feature-wise Ridge (aligned to filtered rows)
        y_ridge_pred_tr = np.zeros_like(Y_tr_c)
        ridge_models = []
        for j in range(n_feat):
            obs_idx = (M_tr_c[:, j] > 0)
            if obs_idx.sum() < 25:
                ridge_models.append(None)
                continue

            rg = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="cholesky")
            rg.fit(X_tr_ridge_c[obs_idx], Y_tr_c[obs_idx, j] - y_mean_tr[j])
            ridge_models.append(rg)
            y_ridge_pred_tr[obs_idx, j] = rg.predict(X_tr_ridge_c[obs_idx])

        # 4. Residuals & RMS Scaling (on filtered rows)
        Y_tr_centered_c = (Y_tr_c - y_mean_tr) * M_tr_c
        residuals_unscaled = Y_tr_centered_c - (y_ridge_pred_tr * M_tr_c)

        if use_rms_scaling:
            res_obs = residuals_unscaled[M_tr_c > 0]
            if res_obs.size < 1000:
                raise optuna.TrialPruned()

            res_rms = np.sqrt(np.mean(res_obs**2))
            if not np.isfinite(res_rms) or res_rms < 1e-4:
                raise optuna.TrialPruned()

            residuals_scaled = residuals_unscaled / (res_rms + 1e-8)
            rms_vals.append(res_rms)
        else:
            res_rms = 1.0
            residuals_scaled = residuals_unscaled
            rms_vals.append(1.0)

        # 5. Flattened Tensor Boost
        tensor_model = CPRegressor(
            weight_rank=rank_reg,
            reg_W=reg_w,
            n_iter_max=100,
            random_state=SEED
        )
        tensor_model.fit(X_tr_tens_c, residuals_scaled)

        # 6. Prediction Phase
        n_val_t = X_val.shape[0]
        X_val_flat_3d = X_val.transpose(0, 1, 3, 2).reshape(n_val_t * n_f, n_feat, n_l)
        X_val_ridge_2d = X_val_flat_3d.reshape(n_val_t * n_f, -1)

        Y_val_f = Y_val.reshape(n_val_t * n_f, n_feat)
        M_val_f = M_val.reshape(n_val_t * n_f, n_feat)

        f_mask_val = M_val_f.any(axis=1)  # Only evaluate valid rows

        X_val_ridge_c = X_val_ridge_2d[f_mask_val]
        X_val_tens_c  = X_val_flat_3d[f_mask_val]
        Y_val_final   = Y_val_f[f_mask_val]
        M_val_final   = M_val_f[f_mask_val]

        # Ridge Predict
        y_ridge_val = np.zeros((X_val_ridge_c.shape[0], n_feat))
        for j, rg in enumerate(ridge_models):
            if rg is not None:
                y_ridge_val[:, j] = rg.predict(X_val_ridge_c)

        # Tensor Predict & Rescale
        y_tensor_val = tensor_model.predict(X_val_tens_c) * res_rms

        # Combine
        y_final_c = y_ridge_val + (gamma * y_tensor_val) + y_mean_tr

        # Score
        r2 = evaluate_ensemble(Y_val_final, y_final_c, M_val_final)
        if r2 is None or (fold_idx == 0 and r2 < -0.1):
            raise optuna.TrialPruned()

        scores.append(max(r2, -1.0))
        density_vals.append(M_tr.mean())

    # --- LOGGING ---
    final_score = float(np.mean(scores))
    with open(LOG_MAP[target_mode], "a") as f:
        # Trial, Value, L, Alpha, Rank, RegW, Gamma, Scaling, MeanRMS, MeanDensity
        f.write(
            f"{trial.number},{final_score:.5f},{L},{ridge_alpha:.2f},"
            f"{rank_reg},{reg_w:.6f},{gamma:.4f},{use_rms_scaling},"
            f"{np.mean(rms_vals):.6f},{np.mean(density_vals):.4f}\n"
        )

    return final_score

if __name__ == "__main__":
    tl.set_backend("numpy")

    for mode in LOG_MAP:
        if not Path(LOG_MAP[mode]).exists():
            with open(LOG_MAP[mode], "w") as f:
                f.write("Trial,Value,L,Ridge_Alpha,Rank,RegW,Gamma,RMS_Scaling,Avg_RMS,Avg_Density\n")

    study = optuna.create_study(
        study_name="research_ens_flat_v1",
        storage="sqlite:///research_ens_flat.db",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)
