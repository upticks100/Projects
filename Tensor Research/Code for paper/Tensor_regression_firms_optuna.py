"""
Cluster-Optimized CP Tensor Regression
Strategy: Structured CP (Tensor-on-Tensor) 
Author: Bryan McNamara
Date: Dec 2025
"""

import os
import joblib
import numpy as np
import optuna
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from tensorly.regression.cp_regression import CPRegressor
import tensorly as tl

# --- SERVER CONFIGURATION ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- EXPERIMENT SETTINGS ---
SEED = 42
N_TRIALS = 50
CACHE_DIR = Path(__file__).resolve().parent / "tensor_cache"

# Sticking to Structured DB and Log
DB_URL     = "sqlite:///pure_cp_structured.db"
STUDY_NAME = "pure_cp_structured_v1"
LOG_FILE   = "pure_cp_structured_log.txt"

CACHE_FILES = {
    2: "tensor_L2_R40_20_2.pkl",
    4: "tensor_L4_R40_20_4.pkl"
}

def load_tensor_data(window_size):
    filename = CACHE_FILES.get(window_size)
    file_path = CACHE_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Cache file missing: {file_path}")
    return joblib.load(file_path)

def process_structured_fold(X, Y):
    """
    Sticks to Documentation: (n_samples, I1, I2, I3) -> (n_samples, O1, O2)
    No flattening of Firms or Features.
    """
    # Centering 3D Tensor Y across the Time (sample) dimension
    # Mean is calculated per (Firm, Feature) coordinate
    y_mean = np.mean(Y, axis=0)
    Y_centered = Y - y_mean
    
    return X, Y_centered, y_mean

def evaluate_predictions(y_true, y_pred, mask):
    """Computes R2 score specifically on valid entries within the tensor block."""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    # Use mask to ignore imputed values during scoring
    valid_idx = (mask_flat > 0) & (~np.isnan(y_true_flat))
    
    clean_true = y_true_flat[valid_idx]
    clean_pred = y_pred_flat[valid_idx]
    
    if len(clean_true) < 100:
        raise ValueError(f"Validation set too small ({len(clean_true)} samples)")
        
    sst = np.sum((clean_true - clean_true.mean())**2)
    if sst < 1e-4:
        return -1.0 # Variance collapse
        
    return 1.0 - np.sum((clean_true - clean_pred)**2) / sst

def objective(trial):
    L = trial.suggest_categorical("L", [2, 4])
    rank_reg = trial.suggest_int("RANK_REGRESS", 1, 25)
    reg_w = trial.suggest_float("REG_W", 0.1, 50.0, log=True)
    
    data = load_tensor_data(L)
    X_all, Y_all, Mask_all = data["X"], data["Y"], data["Mask"]
    
    split_idx = int(0.8 * len(X_all))
    tscv = TimeSeriesSplit(n_splits=3)
    
    scores = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_all[:split_idx])):
        X_tr, Y_tr = X_all[tr_idx], Y_all[tr_idx]
        X_val, Y_val = X_all[val_idx], Y_all[val_idx]
        Mask_val = Mask_all[val_idx]

        # Process while keeping the 3D/4D structure
        X_train, Y_train_centered, y_mean_tr = process_structured_fold(X_tr, Y_tr)
        
        model = CPRegressor(
            weight_rank=rank_reg, 
            reg_W=reg_w, 
            n_iter_max=100, 
            tol=1e-3, 
            verbose=0, 
            random_state=SEED
        )
        model.fit(X_train, Y_train_centered)
        
        # Predict directly on the 4D validation tensor
        preds_centered = model.predict(X_val)
        preds_real = preds_centered + y_mean_tr
        
        # Scoring logic
        r2 = evaluate_predictions(Y_val, preds_real, Mask_val)
        
        if fold_idx == 0 and r2 < -0.1:
            raise optuna.TrialPruned()
            
        scores.append(max(r2, -1.0))

    return np.mean(scores)

def logging_callback(study, trial):
    with open(LOG_FILE, "a") as f:
        f.write(f"{trial.number},{trial.value:.5f},"
                f"{trial.params.get('L')},{trial.params.get('RANK_REGRESS')},"
                f"{trial.params.get('REG_W'):.4f}\n")

if __name__ == "__main__":
    try:
        tl.set_backend("numpy")
        print(f"--- Starting Structured CP Tensor Study: {STUDY_NAME} ---")
        
        if not Path(LOG_FILE).exists():
            with open(LOG_FILE, "w") as f:
                f.write("Trial,Value,L,Rank_Reg,Reg_W\n")
        
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=DB_URL,
            direction="maximize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[logging_callback])
        print(f"\nOptimization Complete. Best Value: {study.best_value:.4f}")
        
    except Exception:
        import traceback
        traceback.print_exc()
        exit(1)