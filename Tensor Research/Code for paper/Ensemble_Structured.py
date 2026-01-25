"""
Ensemble: Dual-Mode Structured (Final Production Version)
Final Audit: Corrected Centering, Stability Pruning, and Full Parameter Logging
FIX: Removed [..., 0] slicing because cached Y/Mask are (T, Firms, Features).
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

LOG_MAP  = {"LEVELS": "log_research_struct_levels.txt",
            "SURPRISE": "log_research_struct_surprise.txt"}

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_L2.pkl", 4: "tensor_levels_L4.pkl"},
    "SURPRISE": {2: "tensor_L2_R40_20_2.pkl", 4: "tensor_L4_R40_20_4.pkl"}
}

def evaluate_ensemble(y_true, y_pred, mask):
    y_t, y_p, m = y_true.flatten(), y_pred.flatten(), mask.flatten()
    valid = m > 0
    # Mandatory threshold for R2 stability
    if np.sum(valid) < 1000:
        return None
    clean_t, clean_p = y_t[valid], y_p[valid]
    sst = np.sum((clean_t - np.mean(clean_t))**2)
    return 1.0 - np.sum((clean_t - clean_p)**2) / sst if sst > 1e-8 else None

def objective(trial):
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

    # Optional: quick sanity check (comment out later)
    # print("X", X_all.shape, "Y", Y_all.shape, "M", Mask_all.shape)

    # --- HYPERPARAMETERS ---
    ridge_alpha = trial.suggest_float("RIDGE_ALPHA", 1.0, 1000.0, log=True)
    rank_reg    = trial.suggest_int("RANK_REGRESS", 2, 15)
    reg_w       = trial.suggest_float("REG_W", 1e-5, 10.0, log=True)
    gamma       = trial.suggest_float("GAMMA", 0.0, 2.0)

    tscv = TimeSeriesSplit(n_splits=3)
    scores, rms_vals, density_vals = [], [], []

    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], Mask_all[:split_idx]

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]
        n_tr_t, n_f, n_feat, n_l = X_tr.shape

        # 1. 3D-Global Mask-Aware Centering
        denom_3d = M_tr.sum(axis=(0, 1)) + 1e-8
        y_mean_tr = (Y_tr * M_tr).sum(axis=(0, 1)) / denom_3d
        Y_tr_centered_3d = (Y_tr - y_mean_tr) * M_tr

        # 2. Feature-wise Ridge
        X_tr_ridge_all = X_tr.transpose(0, 1, 3, 2).reshape(n_tr_t * n_f, -1)
        Y_tr_f = Y_tr.reshape(n_tr_t * n_f, n_feat)
        M_tr_f = M_tr.reshape(n_tr_t * n_f, n_feat)
        f_mask_tr = M_tr_f.any(axis=1)

        y_ridge_pred_tr_flat = np.zeros_like(Y_tr_f)
        ridge_models = []
        for j in range(n_feat):
            obs_idx = (M_tr_f[:, j] > 0) & f_mask_tr
            if obs_idx.sum() < 25:
                ridge_models.append(None)
                continue

            rg = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="cholesky")
            rg.fit(X_tr_ridge_all[obs_idx], Y_tr_f[obs_idx, j] - y_mean_tr[j])
            ridge_models.append(rg)
            y_ridge_pred_tr_flat[obs_idx, j] = rg.predict(X_tr_ridge_all[obs_idx])

        # 3. Structured Residuals & Scaling
        ridge_pred_3d = y_ridge_pred_tr_flat.reshape(n_tr_t, n_f, n_feat)
        res_3d = Y_tr_centered_3d - (ridge_pred_3d * M_tr)

        if use_rms_scaling:
            res_obs = res_3d[M_tr > 0]
            if res_obs.size < 1000:
                raise optuna.TrialPruned()
            res_rms = np.sqrt(np.mean(res_obs**2))
            if not np.isfinite(res_rms) or res_rms < 1e-4:
                raise optuna.TrialPruned()
            res_scaled_3d = res_3d / (res_rms + 1e-8)
            rms_vals.append(res_rms)
        else:
            res_rms = 1.0
            res_scaled_3d = res_3d
            rms_vals.append(1.0)

        # 4. Structured Tensor Boost
        tensor_model = CPRegressor(
            weight_rank=rank_reg,
            reg_W=reg_w,
            n_iter_max=100,
            random_state=SEED
        )
        tensor_model.fit(X_tr, res_scaled_3d)

        # 5. Prediction
        n_val_t = X_val.shape[0]
        X_val_ridge_all = X_val.transpose(0, 1, 3, 2).reshape(n_val_t * n_f, -1)

        y_ridge_val_f = np.zeros((n_val_t * n_f, n_feat))
        for j, rg in enumerate(ridge_models):
            if rg is not None:
                y_ridge_val_f[:, j] = rg.predict(X_val_ridge_all)

        y_ridge_val_3d = y_ridge_val_f.reshape(n_val_t, n_f, n_feat)
        y_tensor_val_3d = tensor_model.predict(X_val) * res_rms
        y_final_3d = y_ridge_val_3d + (gamma * y_tensor_val_3d) + y_mean_tr

        r2 = evaluate_ensemble(Y_val, y_final_3d, M_val)
        if r2 is None: #or (fold_idx == 0 and r2 < -0.1):
            raise optuna.TrialPruned()

        scores.append(max(r2, -1.0))
        density_vals.append(M_tr.mean())

    final_score = float(np.mean(scores))

    # Save trial scores to user_attrs for deep audit later
    trial.set_user_attr("fold_scores", scores)

    # --- LOGGING ---
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
        study_name="research_ens_struct_v2",
        storage="sqlite:///research_ens_struct.db",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)
