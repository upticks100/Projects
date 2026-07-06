"""prediction_new/build_prediction_caches.py

Rebuild the rolling-window tensor caches used by the CP regression pipeline,
on the v2 fundamentals (40 features, YTD-differenced cash flows, 50 mega-caps).

Output:
    tensor_cache/tensor_levels_L{2,4}.pkl
    tensor_cache/tensor_surprise_L{2,4}.pkl
    tensor_cache/meta.pkl
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR.parent))

from prediction_config import (  # noqa: E402
    END_DATE, FEATURE_SPECS, IMPUTATION_RANKS, LOCAL_FUNDAMENTALS_FILE,
    LOCAL_META_COLUMNS, LOOKBACKS, MODES, SEED, START_DATE,
    TENSOR_CACHE_DIR, UNIVERSE_REF_QUARTER, UNIVERSE_TOP_N,
    cache_path, meta_path, select_universe_gvkeys,
)
from Build_PrePrediction_Exhibits import (  # noqa: E402
    first_available_column, ytd_to_quarterly,
)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

TENSOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_filtered_panel() -> tuple[pd.DataFrame, list[str], list[str], list]:
    print(f"[1/4] reading {LOCAL_FUNDAMENTALS_FILE.name}")
    print(f"      selecting top {UNIVERSE_TOP_N} firms by mkvaltq at {UNIVERSE_REF_QUARTER}")
    universe = select_universe_gvkeys()
    print(f"      universe size: {len(universe)} gvkeys")

    needed = set(LOCAL_META_COLUMNS) | {c for spec in FEATURE_SPECS for c in spec.source_columns}
    df = pd.read_csv(
        LOCAL_FUNDAMENTALS_FILE,
        dtype={"gvkey": str},
        usecols=lambda c: c in needed,
        low_memory=False,
    )
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["gvkey", "datadate"]).copy()
    df = df[df["gvkey"].isin(universe)]
    df = df[(df["datadate"] >= START_DATE) & (df["datadate"] <= END_DATE)]
    df = df.sort_values(["gvkey", "datadate"])
    df["quarter_period"] = df["datadate"].dt.to_period("Q")
    df = df.drop_duplicates(["gvkey", "quarter_period"], keep="last")
    print(f"      filtered rows: {len(df):,}  gvkeys: {df['gvkey'].nunique()}  quarters: {df['quarter_period'].nunique()}")

    print("[2/4] applying 40-feature spec (incl. ytd_to_quarterly for cash-flow YTDs)")
    feature_names = []
    for spec in FEATURE_SPECS:
        values = first_available_column(df, spec.source_columns)
        if spec.transform == "ytd_to_quarterly":
            values = ytd_to_quarterly(df, values)
        df[spec.label] = values
        feature_names.append(spec.label)

    print("      applying log-modulus transform (sign-preserving compression)")
    arr = df[feature_names].to_numpy(dtype=np.float64)
    df.loc[:, feature_names] = np.sign(arr) * np.log1p(np.abs(arr))

    firms = sorted(df["gvkey"].unique())
    quarters = sorted(df["quarter_period"].unique())
    return df, feature_names, firms, quarters


def build_raw_tensor(df: pd.DataFrame, feature_names: list[str],
                     firms: list[str], quarters: list) -> np.ndarray:
    print(f"[3/4] building raw tensor: {len(firms)} firms × {len(feature_names)} features × {len(quarters)} quarters")
    full_idx = pd.MultiIndex.from_product([firms, quarters], names=["gvkey", "quarter_period"])
    df_idx = df.set_index(["gvkey", "quarter_period"]).reindex(full_idx)

    slices = []
    for feat in feature_names:
        wide = df_idx[feat].unstack("quarter_period").reindex(index=firms, columns=quarters)
        slices.append(wide.to_numpy(dtype=np.float32))
    tensor = np.stack(slices, axis=1)
    obs_density = (~np.isnan(tensor)).mean()
    print(f"      raw observed density: {obs_density*100:.2f}%")
    return tensor


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
            init="svd", random_state=SEED,
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

    # Preserve observed cells exactly; only fill the missing (NaN) cells
    # with the Tucker reconstruction. This matches the paper (Section
    # "Tensor construction and imputation", line 591: "imputed values
    # are returned to original units"). The previous implementation
    # replaced observed cells with the low-rank reconstruction too,
    # which silently smoothed ~90% of the regressor's input.
    if mode == "SURPRISE":  # paper "Normalized": (observed ∪ imputed) / r_t
        X_out = np.where(obs_indices, X_scaled, X_clean_scaled)
        Y_out = Y_flat
    else:  # LEVELS == paper "Unscaled": observed in original units, imputed in original units
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
    df, feature_names, firms, quarters = load_filtered_panel()
    # capture conm + tic + 2024Q4 mkvaltq so meta is human-auditable
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
            "ref_quarter_for_universe": UNIVERSE_REF_QUARTER,
            "universe_top_n": UNIVERSE_TOP_N,
        },
        meta_path(),
    )
    print(f"      meta saved to {meta_path()}")

    print("[4/4] per-window Tucker imputation (LEVELS + SURPRISE × L=2 + L=4)")
    n_time = raw_tensor.shape[2]

    for mode in MODES:
        for L in LOOKBACKS:
            ranks = IMPUTATION_RANKS[L]
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
            fname = cache_path(mode, L)
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
            print(f"  saved → {fname}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
