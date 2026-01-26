# %%
# --- SECTION 1: IMPORTS & CONFIGURATION ---
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV # Added for benchmark
from sklearn.model_selection import TimeSeriesSplit
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
from tensorly.regression.cp_regression import CPRegressor
import tensorly as tl # Added for backend setting

warnings.filterwarnings("ignore")

# --- 1. SETTINGS ---
START_DATE = "2005-01-31"
END_DATE   = "2024-12-31"
SEED       = 42
TEST_SPLIT = 0.2
OUTPUT_FILE = "tensor_grid_results.txt"
OUTPUT_PATH = Path(__file__).resolve().parent / OUTPUT_FILE

# --- 2. THE GRID ---
GRID_SEARCH = {
    # Data Parameters
    "L": [2, 4],             # Lookback
    "RANK_IMPUTE": [10,20],  # High fidelity data reconstruction

    # Model Parameters
    "RANK_REGRESS": [5, 10, 15], 
    "REG_W": [1.0, 10.0, 50.0, 100.0]    
}

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

# --- HELPER: LOGGING ---
def log(message):
    print(message)
    with open(OUTPUT_PATH, "a", encoding="ascii") as f:
        f.write(message + "\n")

# --- SECTION 3: DATA LOADING ---
def get_clean_data():
    log("--- Loading & Cleaning Data ---")
    root = Path(__file__).resolve().parent
    df = pd.read_csv(root / "90-25_Q_Fundamentals.csv")
    df["datadate"] = pd.to_datetime(df["datadate"])

    mapping = pd.read_csv(root / "gvkeys_to_gics.csv")
    valid_ids = mapping[mapping["tic"].isin(TARGET_TICKERS)]["gvkey"].astype(str).unique()
    df = df[df["gvkey"].astype(str).isin(valid_ids)]
    
    df = df[(df["datadate"] >= START_DATE) & (df["datadate"] <= END_DATE)]
    df = df[["gvkey", "datadate"] + FEATURES].sort_values(by=["gvkey", "datadate"])

    log("Normalizing features (Z-Score)...")
    def zscore(x): return (x - x.mean()) / (x.std() + 1e-8)
    df[FEATURES] = df.groupby("datadate")[FEATURES].transform(zscore)
    df[FEATURES] = df[FEATURES].clip(-5, 5).fillna(0)
    return df

def build_raw_tensor(df):
    log("--- Building Raw Tensor ---")
    firms = sorted(df["gvkey"].unique())
    dates = sorted(df["datadate"].unique())
    tensor = np.full((len(firms), len(FEATURES), len(dates)), np.nan, dtype=np.float32)
    f_map = {f: i for i, f in enumerate(firms)}
    d_map = {d: i for i, d in enumerate(dates)}
    for _, row in df.iterrows():
        tensor[f_map[row["gvkey"]], :, d_map[row["datadate"]]] = row[FEATURES]
    return tensor

# --- SECTION 4: IMPUTATION WORKER ---
def clean_window(t, tensor, L, H, rank):
    # FIX #3: Ensure Determinism inside Worker
    tl.set_backend("numpy") 
    
    X = tensor[:, :, t : t + L]
    Y = tensor[:, :, t + L : t + L + H]
    if np.isnan(Y).all(): return None
    
    # FIX #2: Explicit Mask Casting
    # ~np.isnan means: 1=Observed, 0=Missing. This IS correct.
    # We cast to int to be safe.
    mask = (~np.isnan(X)).astype(int) 
    
    X_filled = np.nan_to_num(X, nan=0.0)
    
    # Check if we actually need imputation (if mask has any 0s)
    if np.min(mask) == 0: 
        try:
            # Deterministic RNG for this window
            rng = np.random.default_rng(SEED + t) 
            w, f = parafac(X_filled, rank=rank, mask=mask, n_iter_max=20, tol=1e-3, 
                           init="random", random_state=int(rng.integers(0, 1e9)), verbose=0)
            X = cp_to_tensor((w, f))
        except: 
            X = X_filled
    else: 
        X = X_filled
        
    return X, np.nan_to_num(Y, nan=0.0), ~np.isnan(Y)

# --- SECTION 5: MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure Global reproducibility
    np.random.seed(SEED)
    tl.set_backend("numpy")

    with open(OUTPUT_PATH, "w", encoding="ascii") as f:
        f.write(f"--- Run Start: {pd.Timestamp.now()} ---\n")
    
    df = get_clean_data()
    tensor_raw = build_raw_tensor(df)
    
    # Generate Combinations
    keys, values = zip(*GRID_SEARCH.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    log(f"\n--- Starting Grid Search: {len(all_combos)} Combinations ---")
    
    best_score = -np.inf
    best_combo = None
    best_data = None
    
    data_configs = set((c["L"], c["RANK_IMPUTE"]) for c in all_combos)
    
    for (L, r_imp) in data_configs:
        log(f"\n[DATA BUILD] Building Dataset: Lookback={L}, ImputeRank={r_imp}...")
        
        n_wins = tensor_raw.shape[2] - (L + 1)
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(clean_window)(t, tensor_raw, L, 1, r_imp) for t in range(n_wins)
        )
        valid = [r for r in results if r is not None]
        
        if not valid:
            log("  -> Warning: No valid windows found.")
            continue
            
        X_all = np.stack([r[0] for r in valid])
        Y_all = np.stack([r[1] for r in valid])
        Mask_all = np.stack([r[2] for r in valid])
        
        # Split Data
        split = int((1 - TEST_SPLIT) * len(X_all))
        X_tr, Y_tr = X_all[:split], Y_all[:split]
        Mask_tr    = Mask_all[:split]
        log(f"  Dataset stats: train_windows={len(X_tr)}, test_windows={len(X_all)-split}")
        
        matching_combos = [c for c in all_combos if c["L"] == L and c["RANK_IMPUTE"] == r_imp]
        
        for params in matching_combos:
            r_reg = params["RANK_REGRESS"]
            reg_w = params["REG_W"]
            
            try:
                tscv = TimeSeriesSplit(n_splits=3)
                val_scores = []

                for fold, (tr_idx, v_idx) in enumerate(tscv.split(X_tr), 1):
                    model = CPRegressor(weight_rank=r_reg, reg_W=reg_w, n_iter_max=300, 
                                        tol=1e-4, verbose=0, random_state=SEED)
                    model.fit(X_tr[tr_idx], Y_tr[tr_idx])
                    
                    preds = model.predict(X_tr[v_idx])
                    
                    # FIX #1: Robust Flattening for Scoring
                    mask_val_flat = Mask_tr[v_idx].reshape(-1)
                    y_true_flat   = Y_tr[v_idx].reshape(-1)
                    y_pred_flat   = preds.reshape(-1)
                    
                    # Filter using the boolean mask
                    y_true_valid = y_true_flat[mask_val_flat]
                    y_pred_valid = y_pred_flat[mask_val_flat]
                    
                    sst = np.sum((y_true_valid - y_true_valid.mean())**2)
                    if sst <= 0:
                        log(f"    Fold {fold}: Skipped (zero variance in target).")
                        continue
                    r2_fold = 1.0 - np.sum((y_true_valid - y_pred_valid)**2) / sst
                    val_scores.append(r2_fold)

                if not val_scores:
                    log(f"  Model: Rank={r_reg}, Reg={reg_w} -> Val R2: n/a (no valid folds)")
                    continue

                avg_r2 = np.mean(val_scores)
                log(f"  Model: Rank={r_reg}, Reg={reg_w} -> Avg Val R2: {avg_r2:.4f}")
                
                if avg_r2 > best_score:
                    best_score = avg_r2
                    best_combo = params
                    best_data = (X_all[:split], Y_all[:split], 
                                 X_all[split:], Y_all[split:], Mask_all[split:])
            except Exception as e:
                log(f"  Model Failed: {str(e)}")

    # --- 6. FINAL CHAMPIONSHIP ---
    if best_combo is None:
        log("\nCRITICAL FAILURE: No models trained successfully.")
    else:
        log(f"\n--- CHAMPION SELECTED: {best_combo} ---")
        X_tr, Y_tr, X_te, Y_te, Mask_te = best_data
        
        # 1. Train Champion
        final_model = CPRegressor(weight_rank=best_combo["RANK_REGRESS"], 
                                  reg_W=best_combo["REG_W"],
                                  n_iter_max=500, tol=1e-4, random_state=SEED)
        final_model.fit(X_tr, Y_tr)
        Y_pred = final_model.predict(X_te)
        
        # 2. Score Champion (Fix #1 applied here too)
        mask_te_flat = Mask_te.reshape(-1)
        y_true_te    = Y_te.reshape(-1)[mask_te_flat]
        y_pred_te    = Y_pred.reshape(-1)[mask_te_flat]
        
        sst_total = np.sum((y_true_te - y_true_te.mean())**2)
        r2_tensor = 1.0 - np.sum((y_true_te - y_pred_te)**2) / sst_total
        
        # 3. Naive Baseline
        y_naive_raw = X_te[:, :, :, -1] # Last step of input
        y_naive_te  = y_naive_raw.reshape(-1)[mask_te_flat]
        r2_naive    = 1.0 - np.sum((y_true_te - y_naive_te)**2) / sst_total
        
        # 4. Ridge Baseline (Optimized)
        log("Training Optimized Ridge Benchmark...")
        ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
        
        X_tr_flat = X_tr.reshape(len(X_tr), -1)
        Y_tr_flat = Y_tr.reshape(len(Y_tr), -1)
        X_te_flat = X_te.reshape(len(X_te), -1)
        
        ridge.fit(X_tr_flat, Y_tr_flat)
        y_ridge_raw = ridge.predict(X_te_flat)
        y_ridge_te  = y_ridge_raw.reshape(-1)[mask_te_flat]
        r2_ridge    = 1.0 - np.sum((y_true_te - y_ridge_te)**2) / sst_total
        
        log(f"Best Ridge Alpha: {ridge.alpha_}")
        log(f"{r2_ridge:.4f}")
        
        log("\n" + "="*40)
        log("FINAL TEST RESULTS")
        log("="*40)
        log(f"1. Naive Baseline: {r2_naive:.4f}")
        log(f"2. Best Ridge:     {r2_ridge:.4f}")
        log(f"3. Best Tensor:    {r2_tensor:.4f}")
        log("-" * 40)

        if r2_tensor > r2_naive and r2_tensor > r2_ridge:
            log(f"Success Improvement vs Naive: +{r2_tensor - r2_naive:.4f}")
        else:
            log("FAIL. Check 'reg_W' or 'L' params.")
