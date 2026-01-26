"""
generate_data_quarterly_final.py
Goal: Generate FINAL Tensor caches with:
  1. Strict Quarterly Grid (Solved Time Gap).
  2. Observed-Only RMS Scaling (Solved Sparsity Inflation).
  3. No Ghost Dimensions (Solved Shape Issues).
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from joblib import Parallel, delayed
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

# --- CONFIGURATION ---
START_DATE = "2005-01-01"
END_DATE   = "2024-12-31"
SEED       = 42

TARGET_TICKERS = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","AVGO","META","TSLA","BRK.B","LLY","WMT",
    "JPM","V","ORCL","JNJ","MA","XOM","NFLX","COST","ABBV","PLTR","BAC","HD","AMD",
    "PG","GE","KO","CSCO","CVX","UNH","IBM","MS","WFC","CAT","MU","MRK","AXP","GS",
    "PM","RTX","TMUS","ABT","MCD","TMO","CRM","PEP","ISRG","APP","AMAT"
]

FEATURES = [
    "aoq", "aqaq", "atq", "ltq", "ceqq", "cheq", "ciq", "cogsq", "dlcq", "dlttq", 
    "epsfxq", "epspxq", "ibq", "invtq", "intanq", "mibq", "nopiq", "oiadpq", 
    "piq", "ppentq", "pstkq", "rectq", "txtq", "xidoq"
]

# L=2 (6 Months), L=4 (1 Year)
CONFIGS = [
    (2, [40, 20, 2]), 
    (4, [40, 20, 4]), 
]

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "tensor_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_quarterly_data():
    print("Loading Raw Data...")
    df = pd.read_csv(ROOT_DIR / "90-25_Q_Fundamentals.csv", low_memory=False)
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    
    # Filter Tickers
    mapping = pd.read_csv(ROOT_DIR / "gvkeys_to_gics.csv")
    valid_ids = mapping[mapping["tic"].isin(TARGET_TICKERS)]["gvkey"].astype(str).unique()
    
    df["gvkey"] = df["gvkey"].astype(str)
    df = df[df["gvkey"].isin(valid_ids)]
    df = df[(df["datadate"] >= START_DATE) & (df["datadate"] <= END_DATE)]
    
    # --- SNAP TO QUARTER ---
    print("Snapping jagged dates to Quarterly Grid...")
    df["quarter_period"] = df["datadate"].dt.to_period("Q")
    
    # Drop duplicates (Keep last report in quarter) - Handles the 0.5% collisions
    df = df.sort_values("datadate").drop_duplicates(subset=["gvkey", "quarter_period"], keep="last")
    
    # Reindex to dense grid
    all_quarters = pd.period_range(start=START_DATE, end=END_DATE, freq="Q")
    all_gvkeys = df["gvkey"].unique()
    idx = pd.MultiIndex.from_product([all_gvkeys, all_quarters], names=["gvkey", "quarter_period"])
    df = df.set_index(["gvkey", "quarter_period"]).reindex(idx)
    
    # Log-Modulus Transform
    print("Applying Log-Modulus Transform...")
    df[FEATURES] = np.sign(df[FEATURES]) * np.log1p(np.abs(df[FEATURES]))
    
    return df.reset_index()

def build_raw_tensor(df):
    firms = sorted(df["gvkey"].unique())
    quarters = sorted(df["quarter_period"].unique())
    print(f"\nFinal Tensor Shape: {len(firms)} Firms x {len(FEATURES)} Features x {len(quarters)} Quarters")
    
    tensor = np.full((len(firms), len(FEATURES), len(quarters)), np.nan, dtype=np.float32)
    f_map = {f: i for i, f in enumerate(firms)}
    q_map = {q: i for i, q in enumerate(quarters)}
    
    for row in df.itertuples():
        f_idx = f_map.get(row.gvkey)
        q_idx = q_map.get(row.quarter_period)
        if f_idx is not None and q_idx is not None:
            vals = [getattr(row, feat) for feat in FEATURES]
            tensor[f_idx, :, q_idx] = vals
    return tensor

def process_window(t, tensor, L, ranks, mode="SURPRISE"):
    tl.set_backend("numpy")
    H = 1 
    
    # Slice Window
    X_raw = tensor[:, :, t : t + L]
    # SQUEEZE Y NOW: (F, Feat, 1) -> (F, Feat)
    # This kills the ghost dimension forever
    Y_raw = tensor[:, :, t + L : t + L + H]
    Y_flat = Y_raw[..., 0] 
    
    if np.isnan(Y_flat).all(): return None
    
    # --- RMS SCALING (OBSERVED ONLY) ---
    # Fix: Calculate RMS only on non-NaN values to avoid "Sparsity Inflation"
    
    # 1. Get Observed Values
    obs_indices = ~np.isnan(X_raw)
    
    # 2. Compute RMS on Observed Data Only
    if np.sum(obs_indices) > 0:
        rms = np.sqrt(np.mean(X_raw[obs_indices]**2)) + 1e-8
    else:
        rms = 1.0 # Fallback for empty windows
    
    # 3. Scale
    X_scaled = X_raw / rms
    
    # 4. Prepare for Tucker
    mask = obs_indices.astype(int)
    X_filled = np.nan_to_num(X_scaled, nan=0.0)
    
    # --- IMPUTE ---
    tucker_failed = False
    try:
        rng = np.random.default_rng(SEED + t)
        core, factors = tucker(X_filled, rank=ranks, mask=mask, 
                               n_iter_max=30, tol=1e-3, 
                               init="random", random_state=int(rng.integers(0, 1e9)), 
                               verbose=0)
        X_clean_scaled = tucker_to_tensor((core, factors))
        
        numer = np.linalg.norm((X_clean_scaled - X_filled) * mask)
        denom = np.linalg.norm(X_filled * mask)
        recon_error = numer / (denom + 1e-8)
    except Exception as e:
        print(f"  [WARN] Tucker failed at window t={t}: {type(e).__name__}: {e}")
        X_clean_scaled = X_filled
        recon_error = 1.0
        tucker_failed = True

    # --- OUTPUT ---
    if mode == "SURPRISE":
        # Trading: Return Scaled Data (Normalized by Density-Aware RMS)
        Y_out = Y_flat
        X_out = X_clean_scaled
        
    elif mode == "LEVELS":
        # Benchmark: Return Unscaled Data
        X_out = X_clean_scaled * rms
        Y_out = Y_flat 

    # Mask is now (Firms, Features) - No ghost dim!
    mask_out = (~np.isnan(Y_flat)).astype(bool)

    return (X_out.astype(np.float32), 
            np.nan_to_num(Y_out, nan=0.0).astype(np.float32), 
            mask_out,
            recon_error,
            tucker_failed)

if __name__ == "__main__":
    df = get_quarterly_data()
    raw_tensor = build_raw_tensor(df)
    n_time = raw_tensor.shape[2]
    
    modes = [("LEVELS", "tensor_levels_v2"), ("SURPRISE", "tensor_surprise_v2")]
    
    for mode_name, file_prefix in modes:
        print(f"\n=== GENERATING {mode_name} DATA (Final Safe Version) ===")
        for L, ranks in CONFIGS:
            print(f"  > Processing L={L}...")
            n_wins = n_time - (L + 1)
            
            results = Parallel(n_jobs=6, verbose=1)(
                delayed(process_window)(t, raw_tensor, L, ranks, mode=mode_name) 
                for t in range(n_wins)
            )
            
            valid = [r for r in results if r is not None]
            
            if not valid:
                print("  !! ERROR: No valid windows found.")
                continue

            X_all = np.stack([r[0] for r in valid])
            Y_all = np.stack([r[1] for r in valid])
            Mask_all = np.stack([r[2] for r in valid])
            Errors = [r[3] for r in valid]
            TuckerFailed_all = np.array([r[4] for r in valid], dtype=bool)
            Tucker_failures = TuckerFailed_all.sum()
            
            fname = OUTPUT_DIR / f"{file_prefix}_L{L}.pkl"
            
            print(f"  > Avg Recon Error: {np.mean(Errors):.4f}")
            print(f"  > Tucker Failures: {Tucker_failures}/{len(valid)} windows")
            if Tucker_failures > 0:
                print(f"  [WARN] {Tucker_failures} windows used fallback (no imputation)")
            print(f"  > Saving {fname}")
            
            # Persist Tucker failure flag so downstream can filter/flag failed windows
            joblib.dump({
                "X": X_all, "Y": Y_all, "Mask": Mask_all,
                "L": L, "Ranks": ranks,
                "TuckerFailed": TuckerFailed_all,
                "ReconErrors": np.array(Errors, dtype=np.float32)
            }, fname)

    print("\n--- ALL DATA REGENERATED SAFELY ---")