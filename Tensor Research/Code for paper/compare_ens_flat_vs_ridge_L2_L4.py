"""
compare_ens_flat_vs_ridge_L2_L4.py

Flat Ensemble (Ridge + gamma * CPResidual) vs Ridge-only baseline,
for BOTH lookback periods L=2 and L=4 (and for both MODE=LEVELS/SURPRISE),
with the SAME data prep + CV + pooled mask-aware R² used in Ensemble_Flat.py.

No command-line args. Just run:
  python compare_ens_flat_vs_ridge_L2_L4.py
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

import joblib
import numpy as np
import optuna
import tensorly as tl

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from tensorly.regression.cp_regression import CPRegressor

# --- thread safety (match your scripts) ---
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

def evaluate_ensemble(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Pooled, mask-aware R² (same definition used in your ensemble scripts)."""
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

def run_one_combo(study: optuna.Study, cache_path: Path, mode: str, L: int) -> Tuple[float, float, float]:
    cache = joblib.load(cache_path)
    X_all = cache["X"]     # (T, Firms, Features, L)
    Y_all = cache["Y"]     # (T, Firms, Features)
    M_all = cache["Mask"]  # (T, Firms, Features)

    bt = get_best_trial_for(study, mode, L)

    ridge_alpha = float(bt.params["RIDGE_ALPHA"])
    rank_reg    = int(bt.params["RANK_REGRESS"])
    reg_w       = float(bt.params["REG_W"])
    gamma       = float(bt.params["GAMMA"])
    use_rms     = bool(bt.params["USE_RMS_SCALING"])

    print(f"\n=== FLAT ENSEMBLE vs RIDGE | MODE={mode} L={L} ===")
    print(
        f"Best trial #{bt.number} value={bt.value:.5f} | "
        f"ridge_alpha={ridge_alpha:.6g} rank={rank_reg} reg_w={reg_w:.6g} gamma={gamma:.6g} rms={use_rms}"
    )

    # CV setup (same pattern as your scripts)
    tscv = TimeSeriesSplit(n_splits=3)
    split_idx = int(0.8 * len(X_all))
    X_cv, Y_cv, M_cv = X_all[:split_idx], Y_all[:split_idx], M_all[:split_idx]

    ridge_scores: List[float] = []
    ens_scores: List[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_cv)):
        X_tr, Y_tr, M_tr = X_cv[tr_idx], Y_cv[tr_idx], M_cv[tr_idx]
        X_val, Y_val, M_val = X_cv[val_idx], Y_cv[val_idx], M_cv[val_idx]

        n_tr_t, n_f, n_feat, n_l = X_tr.shape

        # 1) mask-aware mean over (time, firm)
        denom = M_tr.sum(axis=(0, 1)) + 1e-8
        y_mean_tr = (Y_tr * M_tr).sum(axis=(0, 1)) / denom  # (Feat,)

        # 2) flatten to (T*F, Feat, L) and ridge 2D (T*F, Feat*L)
        X_tr_flat_3d = X_tr.transpose(0, 1, 3, 2).reshape(n_tr_t * n_f, n_feat, n_l)
        X_tr_ridge_2d = X_tr_flat_3d.reshape(n_tr_t * n_f, -1)

        Y_tr_f = Y_tr.reshape(n_tr_t * n_f, n_feat)
        M_tr_f = M_tr.reshape(n_tr_t * n_f, n_feat)

        # keep firm-quarters with at least one observed feature
        f_mask_tr = M_tr_f.any(axis=1)

        X_tr_tens_c = X_tr_flat_3d[f_mask_tr]
        X_tr_ridge_c = X_tr_ridge_2d[f_mask_tr]
        Y_tr_c = Y_tr_f[f_mask_tr]
        M_tr_c = M_tr_f[f_mask_tr]

        # 3) ridge per feature on filtered rows
        ridge_models: List[Optional[Ridge]] = []
        y_ridge_pred_tr = np.zeros_like(Y_tr_c)

        for j in range(n_feat):
            obs_idx = (M_tr_c[:, j] > 0)
            if obs_idx.sum() < MIN_OBS_PER_FEAT:
                ridge_models.append(None)
                continue

            rg = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="cholesky")
            rg.fit(X_tr_ridge_c[obs_idx], Y_tr_c[obs_idx, j] - y_mean_tr[j])
            ridge_models.append(rg)
            y_ridge_pred_tr[obs_idx, j] = rg.predict(X_tr_ridge_c[obs_idx])

        # 4) residuals (centered & masked) minus ridge_pred (masked)
        Y_tr_centered_c = (Y_tr_c - y_mean_tr) * M_tr_c
        residuals_unscaled = Y_tr_centered_c - (y_ridge_pred_tr * M_tr_c)

        # optional residual RMS scaling
        if use_rms:
            res_obs = residuals_unscaled[M_tr_c > 0]
            if res_obs.size < MIN_VALID_ENTRIES:
                print(f"Fold {fold_idx}: pruned-like (too few residual obs for RMS).")
                ridge_scores.append(np.nan)
                ens_scores.append(np.nan)
                continue
            res_rms = float(np.sqrt(np.mean(res_obs ** 2)))
            if (not np.isfinite(res_rms)) or (res_rms < 1e-4):
                print(f"Fold {fold_idx}: pruned-like (invalid residual RMS).")
                ridge_scores.append(np.nan)
                ens_scores.append(np.nan)
                continue
            residuals_scaled = residuals_unscaled / (res_rms + 1e-8)
        else:
            res_rms = 1.0
            residuals_scaled = residuals_unscaled

        # 5) CP residual model
        tensor_model = CPRegressor(
            weight_rank=rank_reg,
            reg_W=reg_w,
            n_iter_max=100,
            random_state=SEED,
        )
        tensor_model.fit(X_tr_tens_c, residuals_scaled)

        # 6) validation flatten + filter
        n_val_t = X_val.shape[0]
        X_val_flat_3d = X_val.transpose(0, 1, 3, 2).reshape(n_val_t * n_f, n_feat, n_l)
        X_val_ridge_2d = X_val_flat_3d.reshape(n_val_t * n_f, -1)

        Y_val_f = Y_val.reshape(n_val_t * n_f, n_feat)
        M_val_f = M_val.reshape(n_val_t * n_f, n_feat)

        f_mask_val = M_val_f.any(axis=1)

        X_val_tens_c = X_val_flat_3d[f_mask_val]
        X_val_ridge_c = X_val_ridge_2d[f_mask_val]
        Y_val_final = Y_val_f[f_mask_val]
        M_val_final = M_val_f[f_mask_val]

        # 7) ridge predictions on filtered val rows
        y_ridge_val = np.zeros((X_val_ridge_c.shape[0], n_feat), dtype=float)
        for j, rg in enumerate(ridge_models):
            if rg is not None:
                y_ridge_val[:, j] = rg.predict(X_val_ridge_c)

        y_ridge_final = y_ridge_val + y_mean_tr
        r2_ridge = evaluate_ensemble(Y_val_final, y_ridge_final, M_val_final)

        # 8) ensemble prediction
        y_tensor_val = tensor_model.predict(X_val_tens_c) * res_rms
        y_ens_final = y_ridge_val + (gamma * y_tensor_val) + y_mean_tr
        r2_ens = evaluate_ensemble(Y_val_final, y_ens_final, M_val_final)

        ridge_score = max(r2_ridge, -1.0) if r2_ridge is not None else np.nan
        ens_score   = max(r2_ens, -1.0)   if r2_ens is not None else np.nan

        ridge_scores.append(ridge_score)
        ens_scores.append(ens_score)

        print(
            f"Fold {fold_idx}: Ridge R2={ridge_score: .5f}   "
            f"Ensemble R2={ens_score: .5f}   Delta={ens_score - ridge_score: .5f}"
        )

    ridge_mean = float(np.nanmean(ridge_scores))
    ens_mean   = float(np.nanmean(ens_scores))
    delta      = ens_mean - ridge_mean

    print(f"\nSUMMARY | MODE={mode} L={L}")
    print(f"Ridge mean CV R2:    {ridge_mean:.5f}")
    print(f"Ensemble mean CV R2: {ens_mean:.5f}")
    print(f"Delta (Ens-Ridge):   {delta:.5f}")

    return ridge_mean, ens_mean, delta

def main():
    tl.set_backend("numpy")
    np.random.seed(SEED)

    here = Path(__file__).resolve().parent
    cache_dir = here / "tensor_cache"
    db_path = Path("/student/mcnama53/research_ens_flat.db")
    study_name = "research_ens_flat_v1"

    storage = f"sqlite:///{db_path.as_posix()}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    summary = []

    for mode in MODES:
        for L in LOOKBACKS:
            cache_file = cache_dir / FILE_MAP[mode][L]
            if not cache_file.exists():
                print(f"[SKIP] Missing cache: {cache_file}")
                continue
            try:
                ridge_mean, ens_mean, delta = run_one_combo(study, cache_file, mode, L)
                summary.append((mode, L, ridge_mean, ens_mean, delta))
            except Exception as e:
                print(f"[FAIL] MODE={mode} L={L}: {e}")

    print("\n===== FINAL SUMMARY (Flat Ensemble vs Ridge) =====")
    print(f"{'MODE':<9} {'L':<2} {'RIDGE':>10} {'ENS':>10} {'DELTA':>10}")
    print("-" * 45)
    for mode, L, ridge_mean, ens_mean, delta in summary:
        print(f"{mode:<9} {L:<2d} {ridge_mean:>10.5f} {ens_mean:>10.5f} {delta:>10.5f}")

if __name__ == "__main__":
    main()
