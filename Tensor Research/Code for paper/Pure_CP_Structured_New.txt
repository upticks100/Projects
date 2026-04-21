#!/usr/bin/env python3
"""
Pure CP Trader: Dual-Mode Structured (FE-upgraded version)
Strategy: Direct Tensor Regression on 3D (Time x Firm x Feature) Targets
Upgrade: Firm×Feature fixed effects (mask-aware) so CP competes with FE ridge.
Audit: Corrected Centering, Stability Pruning, A/B Scaling, Full Logging
FIX: Removed [..., 0] slicing to match new Imputer format (Y/Mask are 3D).
"""

import os
import joblib
import numpy as np
import optuna
import warnings
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import tensorly as tl
from tensorly.regression.cp_regression import CPRegressor

# 1. CLUSTER SAFETY
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

# --- GLOBAL CONFIG ---
SEED       = 42
N_TRIALS   = 200
CACHE_DIR  = Path(__file__).resolve().parent / "tensor_cache"

LOG_DIR  = Path(__file__).resolve().parent
LOG_MAP  = {"LEVELS": LOG_DIR / "log_cp_fe_v2_levels.txt",
            "SURPRISE": LOG_DIR / "log_cp_fe_v2_surprise.txt"}

FILE_MAP = {
    "LEVELS":   {2: "tensor_levels_v2_L2.pkl", 4: "tensor_levels_v2_L4.pkl"},
    "SURPRISE": {2: "tensor_surprise_v2_L2.pkl", 4: "tensor_surprise_v2_L4.pkl"}
}

# --- ADAPTIVE THRESHOLD HELPERS ---
def get_min_valid_entries(mask: np.ndarray, min_frac: float = 0.05, 
                          floor: int = 100, cap: int = 5000) -> int:
    """
    Adaptive threshold for R² evaluation.
    Require 5% of total possible entries, bounded [100, 5000].
    """
    total_entries = int(mask.size)
    return max(floor, min(cap, int(min_frac * total_entries)))

def evaluate_model(y_true, y_pred, mask):
    """Robust pooled R² for 3D structured blocks (mask-aware, adaptive threshold)."""
    y_t, y_p, m = y_true.flatten(), y_pred.flatten(), mask.flatten()
    valid = m > 0
    min_valid = get_min_valid_entries(mask)
    if np.sum(valid) < min_valid:
        return None
    clean_t, clean_p = y_t[valid], y_p[valid]
    sst = np.sum((clean_t - np.mean(clean_t))**2)
    return 1.0 - np.sum((clean_t - clean_p)**2) / sst if sst > 1e-8 else None

def firm_feature_means(Y_tr: np.ndarray, M_tr: np.ndarray) -> np.ndarray:
    """
    Mask-aware mean over time for each (firm, feature).
    Y_tr, M_tr: (T, Firms, Features)
    Returns mu_ff: (Firms, Features)
    """
    denom = M_tr.sum(axis=0)                 # (Firms, Features)
    mu_ff = (Y_tr * M_tr).sum(axis=0) / (denom + 1e-8)

    # Fallback for firm-feature pairs never observed in train: use global feature mean
    denom_feat = M_tr.sum(axis=(0, 1))       # (Features,)
    mu_feat = (Y_tr * M_tr).sum(axis=(0, 1)) / (denom_feat + 1e-8)
    missing = denom <= 0
    if np.any(missing):
        idx_firm, idx_feat = np.where(missing)
        mu_ff[idx_firm, idx_feat] = mu_feat[idx_feat]
    return mu_ff

def make_objective(target_mode: str):
    """Factory to create mode-specific objective function."""
    def objective(trial):
        # --- EXPERIMENTAL SETUP ---
        L = trial.suggest_categorical("L", [2, 4])
        use_rms_scaling = trial.suggest_categorical("USE_RMS_SCALING", [True, False])

        cache = joblib.load(CACHE_DIR / FILE_MAP[target_mode][L])

        # Cached shapes:
        #   X:    (T, Firms, Features, L)
        #   Y:    (T, Firms, Features)
        #   Mask: (T, Firms, Features)
        X_all = cache["X"]
        Y_all = cache["Y"]
        M_all = cache["Mask"]

        # --- HYPERPARAMETERS ---
        rank_reg = trial.suggest_int("RANK_REGRESS", 5, 50)
        reg_w    = trial.suggest_float("REG_W", 1e-4, 100.0, log=True)

        tscv = TimeSeriesSplit(n_splits=3)
        scores, rms_vals, density_vals = [], [], []

        split_idx = int(0.8 * len(X_all))
        X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
            X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
            X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]

            # 1) Firm×Feature Fixed Effects (mask-aware), like a panel FE intercept
            mu_ff = firm_feature_means(Y_tr, M_tr)                 # (Firms, Features)
            # NOTE: Missing entries become 0 after masking. CPRegressor fits on ALL entries
            # (no mask support), so it implicitly assumes missing residuals = 0. This is a
            # known limitation; evaluation is mask-aware so scoring is unaffected.
            Y_tr_centered = (Y_tr - mu_ff[None, :, :]) * M_tr      # (T, Firms, Features)

            # 2) RMS Scaling (A/B) on FE-demeaned targets
            if use_rms_scaling:
                y_obs = Y_tr_centered[M_tr > 0]
                min_rms_entries = get_min_valid_entries(M_tr)
                if y_obs.size <= min_rms_entries:
                    scores.append(None)  # Mark fold as skipped
                    rms_vals.append(None)
                    density_vals.append(float(M_tr.mean()))
                    continue

                y_rms = float(np.sqrt(np.mean(y_obs**2)))
                if (not np.isfinite(y_rms)) or (y_rms < 1e-4):
                    scores.append(None)  # Mark fold as skipped
                    rms_vals.append(None)
                    density_vals.append(float(M_tr.mean()))
                    continue

                Y_target_scaled = Y_tr_centered / (y_rms + 1e-8)
                rms_vals.append(y_rms)
            else:
                y_rms = 1.0
                Y_target_scaled = Y_tr_centered
                rms_vals.append(1.0)

            # 3) Pure CP Regression (Structured)
            tensor_model = CPRegressor(
                weight_rank=rank_reg,
                reg_W=reg_w,
                n_iter_max=150,
                random_state=SEED
            )
            tensor_model.fit(X_tr, Y_target_scaled)

            # 4) Predict & add FE back (and undo RMS if used)
            preds_scaled = tensor_model.predict(X_val)
            preds_final = (preds_scaled * y_rms) + mu_ff[None, :, :]

            # 5) Score (collect all, prune after loop)
            r2 = evaluate_model(Y_val, preds_final, M_val)
            scores.append(r2 if r2 is not None else None)
            density_vals.append(float(M_tr.mean()))

        # --- ADAPTIVE PRUNING: require ≥2 valid folds, prune on median ---
        valid_scores = [s for s in scores if s is not None and np.isfinite(s)]
        if len(valid_scores) < 2:
            print(f"[WARN] Trial {trial.number}: only {len(valid_scores)} valid fold(s), pruning")
            raise optuna.TrialPruned()
        
        if np.median(valid_scores) < -0.1:
            raise optuna.TrialPruned()
        
        # Clip scores for final average
        scores = [max(s, -1.0) for s in valid_scores]

        # --- LOGGING ---
        final_score = float(np.mean(scores))
        valid_rms = [r for r in rms_vals if r is not None]
        avg_rms = float(np.mean(valid_rms)) if valid_rms else 1.0
        with open(LOG_MAP[target_mode], "a") as f:
            # Trial, Value, L, Rank, RegW, RMS_Scaling, Avg_RMS, Avg_Density, FE_Centering, ValidFolds
            f.write(
                f"{trial.number},{final_score:.5f},{L},{rank_reg},"
                f"{reg_w:.6f},{use_rms_scaling},{avg_rms:.6f},"
                f"{np.mean(density_vals):.4f},FE_FIRMxFEAT,{len(valid_scores)}\n"
            )

        return final_score
    
    return objective

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CP regression Optuna study")
    parser.add_argument("--mode", type=str, choices=["LEVELS", "SURPRISE", "both"], 
                        default="both", help="Which mode to run (default: both)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                        help=f"Number of trials per mode (default: {N_TRIALS})")
    args = parser.parse_args()
    
    tl.set_backend("numpy")
    np.random.seed(SEED)

    from optuna.storages import RDBStorage

    # Determine which modes to run
    modes_to_run = ["LEVELS", "SURPRISE"] if args.mode == "both" else [args.mode]
    
    for mode in modes_to_run:
        # Initialize log file
        log_path = LOG_MAP[mode]
        if log_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = log_path.with_name(f"{log_path.stem}_archived_{timestamp}{log_path.suffix}")
            log_path.rename(archived)
        with open(log_path, "w") as f:
            f.write("Trial,Value,L,Rank,RegW,RMS_Scaling,Avg_RMS,Avg_Density,Centering,ValidFolds\n")
        
        # Create mode-specific study
        db_path = CACHE_DIR.parent / f"research_cp_fe_v2_{mode.lower()}.db"
        storage = RDBStorage(
            url=f"sqlite:///{db_path.as_posix()}",
            engine_kwargs={"connect_args": {"timeout": 120}},
        )

        study = optuna.create_study(
            study_name=f"research_cp_fe_v2_{mode.lower()}",
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )

        print(f"\n{'='*60}")
        print(f"Running Optuna study for MODE={mode}")
        print(f"Database: {db_path}")
        print(f"Trials: {args.n_trials}")
        print(f"{'='*60}\n")

        objective_fn = make_objective(mode)
        study.optimize(objective_fn, n_trials=args.n_trials, n_jobs=3)
