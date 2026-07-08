"""Build the rolling-window tensor caches used by the CP regression pipeline.

For each (mode, lookback L) pair, slides a window over the quarterly tensor,
imputes missing entries with a mask-aware Tucker decomposition (observed cells
preserved exactly; only NaNs are filled from the reconstruction), and stacks
the windows into a single cache:

    CACHE_DIR/tensor_{levels,surprise}_L{2,4}.pkl   {X, Y, Mask, ...}
    CACHE_DIR/meta.pkl                              firms/features/quarters

Ported verbatim from prediction_new/build_prediction_caches.py; only the
config import and panel helpers moved (src/data/panel.py).

Run from the replication root:
    python -m src.tensors.build_caches
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

import config
from src.data.panel import build_raw_tensor, load_filtered_panel

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def process_window(t: int, tensor: np.ndarray, L: int, ranks: list[int], mode: str):
    tl.set_backend("numpy")
    H = 1
    X_raw = tensor[:, :, t : t + L]
    Y_raw = tensor[:, :, t + L : t + L + H]
    Y_flat = Y_raw[..., 0]

    if np.isnan(Y_flat).all():
        return None

    obs_indices = ~np.isnan(X_raw)
    rms = float(np.sqrt(np.mean(X_raw[obs_indices] ** 2))) + 1e-8 if obs_indices.any() else 1.0
    X_scaled = X_raw / rms
    mask = obs_indices.astype(int)
    X_filled = np.nan_to_num(X_scaled, nan=0.0)

    tucker_failed = False
    try:
        core, factors = tucker(
            X_filled, rank=ranks, mask=mask,
            n_iter_max=100, tol=1e-5,
            init="svd", random_state=config.SEED,
            verbose=0,
        )
        X_clean_scaled = tucker_to_tensor((core, factors))
        numer = float(np.linalg.norm((X_clean_scaled - X_filled) * mask))
        denom = float(np.linalg.norm(X_filled * mask))
        recon_error = numer / (denom + 1e-8)
    except Exception as exc:
        print(f"      [WARN] Tucker failed at window t={t}: {type(exc).__name__}: {exc}")
        X_clean_scaled = X_filled
        recon_error = 1.0
        tucker_failed = True

    # Preserve observed cells exactly; only fill the missing (NaN) cells with
    # the Tucker reconstruction.
    if mode == "SURPRISE":  # paper "Normalized": (observed  imputed) / r_t
        X_out = np.where(obs_indices, X_scaled, X_clean_scaled)
        Y_out = Y_flat
    else:  # LEVELS == paper "Unscaled": observed and imputed in original units
        X_clean_unscaled = X_clean_scaled * rms
        X_out = np.where(obs_indices, X_raw, X_clean_unscaled)
        Y_out = Y_flat

    mask_out = (~np.isnan(Y_flat)).astype(bool)
    return (
        X_out.astype(np.float32),
        np.nan_to_num(Y_out, nan=0.0).astype(np.float32),
        mask_out,
        recon_error,
        tucker_failed,
    )


def main() -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_names, firms, quarters = load_filtered_panel()
    # capture conm + tic so meta is human-auditable
    universe_meta = (
        df.sort_values("datadate")
          .drop_duplicates("gvkey", keep="last")[["gvkey", "tic", "conm"]]
          .set_index("gvkey").reindex(firms)
          .reset_index().to_dict(orient="records")
    )
    raw_tensor = build_raw_tensor(df, feature_names, firms, quarters)

    joblib.dump(
        {
            "firms": firms,
            "universe_meta": universe_meta,
            "quarters": [str(q) for q in quarters],
            "feature_names": feature_names,
            "ref_quarter_for_universe": config.UNIVERSE_REF_QUARTER,
            "universe_top_n": config.UNIVERSE_TOP_N,
        },
        config.meta_path(),
    )
    print(f"      meta saved to {config.meta_path()}")

    print("[4/4] per-window Tucker imputation (LEVELS + SURPRISE x L=2 + L=4)")
    n_time = raw_tensor.shape[2]

    for mode in config.MODES:
        for L in config.LOOKBACKS:
            ranks = config.IMPUTATION_RANKS[L]
            n_wins = n_time - L
            print(f"\n=== {mode} L={L}: {n_wins} windows, ranks={ranks} ===")
            t0 = time.perf_counter()
            results = Parallel(n_jobs=4, verbose=0)(
                delayed(process_window)(t, raw_tensor, L, ranks, mode=mode)
                for t in range(n_wins)
            )
            valid = [r for r in results if r is not None]
            if not valid:
                print(f"  ! no valid windows for {mode} L={L}; skipping")
                continue

            X_all = np.stack([r[0] for r in valid])
            Y_all = np.stack([r[1] for r in valid])
            Mask_all = np.stack([r[2] for r in valid])
            errors = np.array([r[3] for r in valid], dtype=np.float32)
            tucker_failed = np.array([r[4] for r in valid], dtype=bool)

            elapsed = time.perf_counter() - t0
            fname = config.cache_path(mode, L)
            joblib.dump(
                {
                    "X": X_all, "Y": Y_all, "Mask": Mask_all,
                    "L": L, "Ranks": ranks,
                    "TuckerFailed": tucker_failed,
                    "ReconErrors": errors,
                },
                fname,
            )
            print(f"  windows: {len(valid)}/{n_wins}  avg recon: {errors.mean():.4f}  "
                  f"Tucker fails: {int(tucker_failed.sum())}/{len(valid)}  elapsed: {elapsed:.1f}s")
            print(f"  saved -> {fname}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
