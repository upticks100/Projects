import numpy as np 
import pandas as pd 
import tensorly as tn
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for SSH
import matplotlib.pyplot as plt
from datetime import datetime
from tensorly.decomposition import tucker
from tensorly.decomposition import parafac
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor


####################################
# Global Filter Parameters
####################################
START_DATE = "1995-01-01"
END_DATE = "2024-12-31"


####################################
# Building DataFrame
####################################

df_ratios = pd.read_csv(_HERE / "firm_ratios.csv")
df_ratios.head()

df_ratios["public_date"] = pd.to_datetime(df_ratios["public_date"])

df_ratios = df_ratios[
    (df_ratios["public_date"] >= START_DATE) & (df_ratios["public_date"] <= END_DATE)
]

df_ratios.set_index(["gvkey", "public_date"], inplace=True)

df_ratios = df_ratios[["GProf", 
                       "evm", 
                       "roe", 
                       "debt_ebitda", 
                       "fcf_ocf", 
                       "quick_ratio",
                       "curr_ratio",
                       "bm",
                       "rect_act",
                       "gpm",
                       "cash_debt",
                       "invt_act"]]

df_ratios.sort_index(level="public_date", inplace=True)
df_ratios.info()


####################################
# Constructing Tensor
####################################

firms = df_ratios.index.get_level_values("gvkey").unique()
dates = df_ratios.index.get_level_values("public_date").unique()
features = df_ratios.columns

print(len(firms))

firm_to_idx = {s: i for i, s in enumerate(firms)}
date_to_idx = {d: i for i, d in enumerate(dates)}

tensor = np.full((len(firms), len(features), len(dates)), np.nan, dtype=np.float32)

for (firm, date), row in df_ratios.iterrows():
    i = firm_to_idx[firm]
    j = date_to_idx[date]
    for k, feature in enumerate(features):
        tensor[i, k, j] = row[feature]

print("Checking tensor for NaNs...")
total_tensor_nans = np.isnan(tensor).sum()
print(f"Total NaNs in tensor: {total_tensor_nans}")


####################################
# Forward-fill along time
####################################

n_firms, n_feats, n_time = tensor.shape

for i in range(n_firms):
    for j in range(n_feats):
        ts = tensor[i, j, :]
        valid = ~np.isnan(ts)
        if not valid.any():
            continue  # no data at all; leave as NaN
        idx = np.where(valid, np.arange(n_time), 0)
        _ = np.maximum.accumulate(idx, out=idx)  # carry last seen index forward 
        ts_filled = ts[idx]
        # Preserve leading NaNs (before the first valid)
        first_valid = np.argmax(valid)
        ts_filled[:first_valid] = np.nan
        tensor[i, j, :] = ts_filled

# Report remaining NaNs after forward-fill
remaining_nans = np.isnan(tensor).sum()
print(f"NaNs after forward-fill: {remaining_nans}")


####################################
# Filtering Missingness
####################################

APPLY_FIRM_FILTER = True       # set True to drop firms exceeding MAX_FIRM_NAN_FRAC
MAX_FIRM_NAN_FRAC = 0.05        # drop firms with >X% NaNs across all features×dates
MAX_FEAT_NAN_FRAC = 0           # drop firms with >X% NaNs across all dates
TOP_K_REPORT = 200               # how many worst firms to print

n_firms, n_feats, n_time = tensor.shape

# Count NaNs per firm (across all features and dates)
firm_nan_counts = np.isnan(tensor).sum(axis=(1, 2))             
firm_total_cells = np.full(n_firms, n_feats * n_time, dtype=int)   # total cells per firm
firm_nan_frac = firm_nan_counts / firm_total_cells
# Per-(firm, feature) missing fraction across time (denominator = n_time)
firm_feat_nan_frac = np.isnan(tensor).mean(axis=2)  # (n_firms, n_feats)

# Aggregate and per-feature missingness (for reporting only; we do NOT drop features)
nan_report = pd.DataFrame({'gvkey': np.array(firms)})
nan_report['nan_count']   = firm_nan_counts
nan_report['total_cells'] = firm_total_cells
nan_report['nan_frac']    = firm_nan_frac
for idx, feat in enumerate(features):
    nan_report[f'{feat}_nan_frac'] = firm_feat_nan_frac[:, idx]
nan_report = nan_report.set_index('gvkey')
print("Top firms by overall NaN fraction (plus per-feature breakdown):")
# Show full columns without horizontal truncation (but allow vertical rows cutoff)
pd.set_option('display.max_columns', None)
print(nan_report.sort_values('nan_frac', ascending=False).head(TOP_K_REPORT))

if APPLY_FIRM_FILTER:
    feature_ok = (firm_feat_nan_frac <= MAX_FEAT_NAN_FRAC).all(axis=1)  # (n_firms,)
    keep_mask = (firm_nan_frac <= MAX_FIRM_NAN_FRAC) & feature_ok
    dropped = int((~keep_mask).sum())
    print(f"Filtering firms with nan_frac > {MAX_FIRM_NAN_FRAC:.2f}: dropping {dropped} / {n_firms}")

    # Apply filter to firms list and tensor
    firms = np.array(firms)[keep_mask]
    tensor = tensor[keep_mask, :, :]

    # Rebuild firm_to_idx mapping based on the kept firms
    firm_to_idx = {s: i for i, s in enumerate(firms)}

    # Also filter the original df_ratios to only these firms so any later logic aligns
    kept_firm_set = set(firms)
    df_ratios = df_ratios[df_ratios.index.get_level_values('gvkey').isin(kept_firm_set)]

    print("New tensor shape after firm filter:", tensor.shape)
    print(f"Kept firms: {tensor.shape[0]}  | Features untouched: {tensor.shape[1]} | Time: {tensor.shape[2]}")

gvkeys_to_gics = pd.read_csv(_HERE / "gvkeys_to_gics.csv")
gvkeys_to_gics["datadate"] = pd.to_datetime(gvkeys_to_gics["datadate"])
gvkeys_to_gics = gvkeys_to_gics.sort_values('datadate').groupby('gvkey').tail(1)
gvkeys_to_gics = gvkeys_to_gics[['gvkey','tic', 'ggroup', 'gind', 'gsector', 'gsubind']].copy()

_gics = gvkeys_to_gics[gvkeys_to_gics['gvkey'].isin(firms)].copy()
unique_counts = {
    "ggroup": _gics["ggroup"].nunique(),
    "gind": _gics["gind"].nunique(),
    "gsector": _gics["gsector"].nunique(),
    "gsubind": _gics["gsubind"].nunique(),
}
print("GICS unique counts (after full filtering and NaN drop):", unique_counts)
print(np.isnan(tensor).sum())

####################################
# Tucker Decomposition 
####################################

tl.set_backend("numpy")

ranks = [unique_counts["gsector"], unique_counts["gsector"], 12]
print(f"Running Tucker with ranks={ranks} ...")
core, factors = tucker(tensor, rank=ranks, n_iter_max=500, tol=1e-6)
firm_factors, feature_factors, time_factors = factors

# Reconstruct and compute relative Frobenius error
Xhat = tl.tucker_to_tensor((core, factors))
num = np.linalg.norm(tensor - Xhat)
den = np.linalg.norm(tensor)
rel_err = float(num / den) if den > 0 else np.nan

print(f"Relative Frobenius error: {rel_err:.4f}")

####################################
# Plot Average of Time Factors
####################################

out_dir = os.path.join("VScode/Alpha/Research/Code for paper", "figs")
os.makedirs(out_dir, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MFI-style aggregate: average absolute value of Tucker time factors
avg_time_factor = np.mean(np.abs(time_factors), axis=1)
#avg_time_factor = np.linalg.norm(time_factors, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(dates, avg_time_factor, label="Average time factor")
plt.legend()
plt.title(f"Tucker time factors firms | mean(|T|) (ranks={ranks})\nrel err={rel_err:.4f}")
plt.xlabel("Date")
plt.ylabel("Average loading")
plt.xticks(rotation=45)
plt.tight_layout()
fig_path = os.path.join(out_dir, f"avg_time_factor_{stamp}_firms.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"Saved average time factor plot to: {fig_path}")

####################################
# CP Decomposition + Save Plots
####################################

# Rank sweep for CP (PARAFAC) decomposition
# We'll try ranks 1..max_rank and record reconstruction errors

cp_ranks = [1]
cp_errors = []

# Optional: fix seed for reproducibility
rng = np.random.default_rng(0)

for r in cp_ranks:
    try:
        print(f"Running CP rank={r} ...")
        weights, factors = parafac(tensor, rank=r, normalize_factors=False)

        time_factors_cp = factors[2]  # shape: (len(dates), r)
        if time_factors_cp.ndim == 2 and time_factors_cp.shape[1] > 1:
            # Option 1: average of absolutes (mean(|T|)) [default MFI-style]
            # avg_time_cp = np.mean(np.abs(time_factors_cp), axis=1)

            # Option 2: absolute of average (|mean(T)|)
            avg_time_cp = np.abs(np.mean(time_factors_cp, axis=1))
        else:
            # Option 1: just take the raw vector (mean(|T|) vs |mean(T)| are the same for rank=1)
            # avg_time_cp = np.abs(time_factors_cp.reshape(-1))

            # Option 2: if you want the signed vector
            avg_time_cp = time_factors_cp.reshape(-1)

        # Plot and save the (averaged) time factor for this rank
        plt.figure(figsize=(10, 6))
        plt.plot(dates, avg_time_cp, label=f"Avg time factor (CP rank={r})")
        plt.legend()
        plt.title(f"CP time factor — AVERAGE (rank={r})")
        plt.xlabel("Date")
        plt.ylabel("Average loading")
        plt.xticks(rotation=45)
        plt.tight_layout()
        tf_path_avg = os.path.join(out_dir, f"FF302 cp_time_factor_rank{r}_AVG_{stamp}.png")
        plt.savefig(tf_path_avg, dpi=200)
        plt.close()
        print(f"Saved CP average time factor plot for rank {r} to: {tf_path_avg}")

        # Also plot the FIRST time component explicitly (component 1)
        comp1_time_cp = time_factors_cp[:, 0].reshape(-1)
        plt.figure(figsize=(10, 6))
        plt.plot(dates, comp1_time_cp, label=f"First time factor (comp 1) — CP rank={r}")
        plt.legend()
        plt.title(f"CP time factor — COMPONENT 1 (rank={r})")
        plt.xlabel("Date")
        plt.ylabel("Loading (comp 1)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        tf_path_c1 = os.path.join(out_dir, f"FF302 cp_time_factor_rank{r}_COMP1_{stamp}.png")
        plt.savefig(tf_path_c1, dpi=200)
        plt.close()
        print(f"Saved CP first time factor (comp 1) plot for rank {r} to: {tf_path_c1}")

        Xhat_cp = cp_to_tensor((weights, factors))
        num = np.linalg.norm(tensor - Xhat_cp)
        den = np.linalg.norm(tensor)
        rel_err_r = float(num / den) if den > 0 else np.nan
        cp_errors.append(rel_err_r)
        print(f"  rel err: {rel_err_r:.4f}")
    except Exception as e:
        print(f"  CP rank={r} failed: {e}")
        cp_errors.append(np.nan)


