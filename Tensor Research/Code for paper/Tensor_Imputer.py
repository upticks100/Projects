# %%
# --- Tensor Imputation --- 
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from joblib import Parallel, delayed
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

# --- CONFIG ---
START_DATE = "2005-01-31"
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

# (Window_Size, [Spatial_Rank_1, Spatial_Rank_2, Time_Rank])
# We use high ranks to guarantee <5% Error (High Fidelity)
CONFIGS_TO_GENERATE = [
    (2, [40, 20, 2]),  
    (4, [40, 20, 4]), 
]

OUTPUT_DIR = Path(__file__).resolve().parent / "tensor_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DATA LOADING ---
def get_clean_data():
    print("Loading Raw Data...")
    root = Path(__file__).resolve().parent
    df = pd.read_csv(root / "90-25_Q_Fundamentals.csv")
    df["datadate"] = pd.to_datetime(df["datadate"])
    mapping = pd.read_csv(root / "gvkeys_to_gics.csv")
    valid_ids = mapping[mapping["tic"].isin(TARGET_TICKERS)]["gvkey"].astype(str).unique()
    df = df[df["gvkey"].astype(str).isin(valid_ids)]
    df = df[(df["datadate"] >= START_DATE) & (df["datadate"] <= END_DATE)]
    df = df[["gvkey", "datadate"] + FEATURES].sort_values(by=["gvkey", "datadate"])
    
    # Log-Modulus & Scaling
    df[FEATURES] = np.sign(df[FEATURES]) * np.log1p(np.abs(df[FEATURES]))
    def rms_scale(x): return x / (np.sqrt(np.mean(x**2)) + 1e-8)
    df[FEATURES] = df.groupby("datadate")[FEATURES].transform(rms_scale)
    df[FEATURES] = df[FEATURES].clip(-5, 5).fillna(0)
    return df

def build_raw_tensor(df):
    firms = sorted(df["gvkey"].unique())
    dates = sorted(df["datadate"].unique())
    tensor = np.full((len(firms), len(FEATURES), len(dates)), np.nan, dtype=np.float32)
    f_map = {f: i for i, f in enumerate(firms)}
    d_map = {d: i for i, d in enumerate(dates)}
    for _, row in df.iterrows():
        tensor[f_map[row["gvkey"]], :, d_map[row["datadate"]]] = row[FEATURES]
    return tensor

# --- WORKER ---
def process_window(t, tensor, L, ranks):
    tl.set_backend("numpy")
    H = 1 
    X = tensor[:, :, t : t + L]
    Y = tensor[:, :, t + L : t + L + H]
    
    if np.isnan(Y).all(): return None
    
    mask = (~np.isnan(X)).astype(int) 
    X_filled = np.nan_to_num(X, nan=0.0)
    
    try:
        rng = np.random.default_rng(SEED + t) 
        core, factors = tucker(X_filled, rank=ranks, mask=mask, 
                               n_iter_max=30, tol=1e-3, 
                               init="random", random_state=int(rng.integers(0, 1e9)), 
                               verbose=0)
        X_clean = tucker_to_tensor((core, factors))
        
        # Calc Reconstruction Error (Targeting < 0.05)
        norm_tensor = np.linalg.norm(X_filled * mask)
        if norm_tensor < 1e-5: 
            recon_error = 0.0
        else:
            recon_error = np.linalg.norm((X_clean - X_filled) * mask) / norm_tensor
            
    except:
        X_clean = X_filled
        recon_error = 1.0

    return (X_clean.astype(np.float32), 
            np.nan_to_num(Y, nan=0.0).astype(np.float32), 
            (~np.isnan(Y)).astype(bool),
            recon_error)

# --- MAIN ---
if __name__ == "__main__":
    df = get_clean_data()
    raw_tensor = build_raw_tensor(df)
    n_time = raw_tensor.shape[2]
    
    print(f"\nTensor Shape: {raw_tensor.shape}")
    print(f"Configurations to run: {len(CONFIGS_TO_GENERATE)}")
    
    for L, ranks in CONFIGS_TO_GENERATE:
        print(f"\n--- PROCESSING L={L}, RANKS={ranks} ---")
        n_wins = n_time - (L + 1)
        
        # Parallel n_jobs=6 is SAFE here (Pure data gen, no nested CV)
        results = Parallel(n_jobs=6, verbose=5)(
            delayed(process_window)(t, raw_tensor, L, ranks) for t in range(n_wins)
        )
        
        valid = [r for r in results if r is not None]
        
        X_all = np.stack([r[0] for r in valid])
        Y_all = np.stack([r[1] for r in valid])
        Mask_all = np.stack([r[2] for r in valid])
        Errors = [r[3] for r in valid]
        
        avg_err = np.mean(Errors)
        print(f"  > Done. Avg Reconstruction Error: {avg_err:.4f}")
        
        if avg_err > 0.05:
            print("  ! WARNING: Error > 5%. Consider increasing spatial ranks.")
        else:
            print("  > SUCCESS: Fidelity > 95%.")
        
        # Save to Disk
        fname = OUTPUT_DIR / f"tensor_L{L}_R{ranks[0]}_{ranks[1]}_{ranks[2]}.pkl"
        print(f"  > Saving to {fname}...")
        joblib.dump({
            "X": X_all,
            "Y": Y_all,
            "Mask": Mask_all,
            "L": L,
            "Ranks": ranks,
            "Avg_Error": avg_err
        }, fname)
        
    print("\n--- FACTORY COMPLETE. READY FOR TRADING. ---")