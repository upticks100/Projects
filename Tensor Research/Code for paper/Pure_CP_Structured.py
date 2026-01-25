"""
Pure CP Trader: Dual-Mode Structured (Final Production Version)
Strategy: Direct Tensor Regression on 3D (Time x Firm x Feature) Targets
Audit: Corrected Centering, Stability Pruning, A/B Scaling, Full Logging
FIX: Removed [..., 0] slicing to match new Imputer format (Y/Mask are 3D).
"""

import os
import joblib
import numpy as np
import optuna
import warnings
from pathlib import Path
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

LOG_MAP  = {"LEVELS": "log_pure_struct_levels.txt",
            "SURPRISE": "log_pure_struct_surprise.txt"}

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_L2.pkl", 4: "tensor_levels_L4.pkl"},
    "SURPRISE": {2: "tensor_L2_R40_20_2.pkl", 4: "tensor_L4_R40_20_4.pkl"}
}

def evaluate_model(y_true, y_pred, mask):
    """Robust R2 for 3D Structured blocks."""
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
    rank_reg = trial.suggest_int("RANK_REGRESS", 5, 50)
    reg_w    = trial.suggest_float("REG_W", 1e-4, 100.0, log=True)

    tscv = TimeSeriesSplit(n_splits=3)
    scores, rms_vals, density_vals = [], [], []

    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], Mask_all[:split_idx]

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]

        # 1. 3D-Global Mask-Aware Centering (feature-wise mean over time+firms)
        denom_3d = M_tr.sum(axis=(0, 1)) + 1e-8
        y_mean_tr = (Y_tr * M_tr).sum(axis=(0, 1)) / denom_3d
        Y_tr_centered = (Y_tr - y_mean_tr) * M_tr

        # 2. RMS Scaling (A/B Test) on Targets
        if use_rms_scaling:
            y_obs = Y_tr_centered[M_tr > 0]
            if y_obs.size < 1000:
                raise optuna.TrialPruned()

            y_rms = np.sqrt(np.mean(y_obs**2))
            if not np.isfinite(y_rms) or y_rms < 1e-4:
                raise optuna.TrialPruned()

            Y_target_scaled = Y_tr_centered / (y_rms + 1e-8)
            rms_vals.append(y_rms)
        else:
            y_rms = 1.0
            Y_target_scaled = Y_tr_centered
            rms_vals.append(1.0)

        # 3. Pure CP Regression (Structured)
        # X: (Time, Firm, Feature, Lag) ; Y: (Time, Firm, Feature)
        tensor_model = CPRegressor(
            weight_rank=rank_reg,
            reg_W=reg_w,
            n_iter_max=150,
            random_state=SEED
        )
        tensor_model.fit(X_tr, Y_target_scaled)

        # 4. Prediction & Rescale
        preds_scaled = tensor_model.predict(X_val)
        preds_final = (preds_scaled * y_rms) + y_mean_tr

        # 5. Score
        r2 = evaluate_model(Y_val, preds_final, M_val)
        if r2 is None or (fold_idx == 0 and r2 < -0.1):
            raise optuna.TrialPruned()

        scores.append(max(r2, -1.0))
        density_vals.append(M_tr.mean())

    # --- LOGGING ---
    final_score = float(np.mean(scores))
    with open(LOG_MAP[target_mode], "a") as f:
        # Trial, Value, L, Rank, RegW, RMS_Scaling, Avg_RMS, Avg_Density
        f.write(
            f"{trial.number},{final_score:.5f},{L},{rank_reg},"
            f"{reg_w:.6f},{use_rms_scaling},{np.mean(rms_vals):.6f},"
            f"{np.mean(density_vals):.4f}\n"
        )

    return final_score

if __name__ == "__main__":
    tl.set_backend("numpy")

    for mode in LOG_MAP:
        if not Path(LOG_MAP[mode]).exists():
            with open(LOG_MAP[mode], "w") as f:
                f.write("Trial,Value,L,Rank,RegW,RMS_Scaling,Avg_RMS,Avg_Density\n")

    study = optuna.create_study(
        study_name="research_pure_struct_v1",
        storage="sqlite:///research_pure_struct.db",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)
