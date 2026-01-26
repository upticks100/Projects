from pathlib import Path
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac, tucker
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
START_DATE = pd.Timestamp("1970-01-31")
END_DATE = pd.Timestamp("2020-01-01")
STANDARDIZE = True
STANDARDIZE_BY_INDUSTRY = False  # False -> standardize across dates instead
USE_MEAN_DATA = True  # toggle between median and mean inputs

FEATURE_BASES = [
    "GProf",
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
    "invt_act",
]

FEATURE_TYPE = "Mean" if USE_MEAN_DATA else "Median"
FEATURES = [f"{name}_{FEATURE_TYPE}" for name in FEATURE_BASES]

DATA_VERSION = "mean" if USE_MEAN_DATA else "median"
DATA_FILE = Path(__file__).resolve().parent / f"FF49_ratios_{DATA_VERSION}.csv"

# --------------------------------------------------------------------
# Load & filter data
# --------------------------------------------------------------------
df = pd.read_csv(DATA_FILE, parse_dates=["public_date"])
df = df[(df["public_date"] >= START_DATE) & (df["public_date"] <= END_DATE)]
df = df.set_index(["FFI49_desc", "public_date"])[FEATURES]


def _to_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        percent = stripped.endswith("%")
        stripped = stripped.rstrip("%").replace(",", "")
        try:
            num = float(stripped)
        except ValueError:
            return np.nan
        return num / 100.0 if percent else num
    return float(value)


df = df.map(_to_numeric)

# quick diagnostics before reindexing/filling
earliest_date = df.index.get_level_values("public_date").min()
has_nans = df.isna().any().any()
print(f"Earliest date in filtered data: {earliest_date.date()}")
print(f"Data contains NaNs before fill: {has_nans}")
first_obs_by_sector = (
    df.reset_index()
    .groupby("FFI49_desc")["public_date"]
    .min()
)

latest_first_obs = first_obs_by_sector.max()
print(f"Latest sector first observation (used as common start): {latest_first_obs.date()}")

# ensure each (sector, date) pair exists by reindexing and forward filling from first real point
df = df[df.index.get_level_values("public_date") >= latest_first_obs]
sectors = sorted(first_obs_by_sector.index)
dates = sorted(
    d for d in df.index.get_level_values("public_date").unique() if d >= latest_first_obs
)
full_index = pd.MultiIndex.from_product(
    [sectors, dates], names=["FFI49_desc", "public_date"]
)
df = (
    df.reindex(full_index)
    .groupby(level=0)
    .apply(lambda frame: frame.ffill())
    .droplevel(0)
)

if STANDARDIZE:
    def _zscore(col: pd.Series) -> pd.Series:
        std = col.std(ddof=0)
        if std == 0 or np.isnan(std):
            return col * 0.0
        return (col - col.mean()) / std

    level = 0 if STANDARDIZE_BY_INDUSTRY else 1
    df = df.groupby(level=level).transform(_zscore)

if df.isna().any().any():
    raise ValueError("Tensor still contains NaNs after filling. Please inspect source.")

df = df.sort_index(level=["FFI49_desc", "public_date"])


# --------------------------------------------------------------------
# Build tensor: (sector, feature, time)
# --------------------------------------------------------------------
array = df.values.reshape(len(sectors), len(dates), len(FEATURES))
tensor = np.transpose(array, (0, 2, 1))  # sectors x features x dates
print(f"Tensor shape: {tensor.shape}")
# per-date energy diagnostic to see if early slices dominate
slice_norms = np.linalg.norm(array, axis=(0, 2))  # sum over sectors and features
norm_series = pd.Series(slice_norms, index=dates)
print(
    "Slice norms (Frobenius): "
    f"first={norm_series.iloc[0]:.4f}, "
    f"median={norm_series.median():.4f}, "
    f"max={norm_series.max():.4f}"
)
print(norm_series.head())


# --------------------------------------------------------------------
# Tucker decomposition
# --------------------------------------------------------------------
tl.set_backend("numpy")
ranks = [
    min(len(sectors), 49),
    min(len(FEATURES), 12),
    min(len(dates), 49),
]
core, factors = tucker(tensor, rank=ranks, n_iter_max=500, tol=1e-6)
sector_factors, feature_factors, time_factors = factors

recon = tl.tucker_to_tensor((core, factors))
rel_err = np.linalg.norm(tensor - recon) / np.linalg.norm(tensor)
print(f"Relative reconstruction error: {rel_err:.4f}")

# MFIX-style temporal index: average absolute loading across components
mfi_like = np.mean(np.abs(time_factors), axis=1)
series = pd.Series(mfi_like, index=dates, name="Industry Index")
print(series.head())
print("Top 5 dates by index value:")
print(series.nlargest(5))
print(f"Min/Max: {series.min():.6f} / {series.max():.6f}")
print("Lowest 5 dates by index value:")
print(series.nsmallest(5))
print(f"Date of min: {series.idxmin()}, value: {series.min():.6f}")


# --------------------------------------------------------------------
# CP decomposition (FCIX-style rank-1 consensus)
# --------------------------------------------------------------------
cp_rank = 1
cp_weights, cp_factors = parafac(
    tensor, rank=cp_rank, n_iter_max=500, tol=1e-6, init="svd"
)
cp_recon = cp_to_tensor((cp_weights, cp_factors))
cp_rel_err = np.linalg.norm(tensor - cp_recon) / np.linalg.norm(tensor)
print(f"CP rank={cp_rank} relative error: {cp_rel_err:.4f}")

# include the CP weight to retain the fitted scale of the rank-1 component
cp_time = cp_factors[2][:, 0] * cp_weights[0]
cp_series = pd.Series(np.abs(cp_time), index=dates, name="Industry Consensus (CP)")
print(f"CP weight: {float(cp_weights[0]):.6f}")
print(
    "CP time factor stats "
    f"(mean={cp_factors[2][:,0].mean():.6e}, "
    f"std={cp_factors[2][:,0].std():.6e}, "
    f"min={cp_factors[2][:,0].min():.6e}, "
    f"max={cp_factors[2][:,0].max():.6e})"
)
print(cp_series.head())
print("Top 5 dates by CP index value:")
print(cp_series.nlargest(5))
print(f"CP Min/Max: {cp_series.min():.6f} / {cp_series.max():.6f}")
print("Lowest 5 dates by CP index value:")
print(cp_series.nsmallest(5))
print(f"CP Date of min: {cp_series.idxmin()}, value: {cp_series.min():.6f}")

# --------------------------------------------------------------------
# Plot both indices
# --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
series.plot(ax=ax, label="Tucker (MFI-style)")
cp_series.plot(ax=ax, label="CP (FCIX-style)")
ax.set_title("Industry ratios: Tucker vs CP indices")
ax.set_xlabel("Date")
ax.set_ylabel("Index value")
ax.legend()
fig.tight_layout()
plot_path = Path(__file__).resolve().parent / "industry_indices.png"
plt.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"Saved comparison plot to {plot_path}")
