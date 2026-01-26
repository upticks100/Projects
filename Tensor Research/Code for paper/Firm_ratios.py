# %%
from pathlib import Path
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from tensorly.cp_tensor import cp_to_tensor
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


####################################
# Global Filter Parameters
####################################
START_DATE = pd.Timestamp("2010-01-01")
END_DATE = pd.Timestamp("2024-12-31")
STANDARDIZE = True
STANDARDIZE_BY_FIRM = True  # False -> standardize across dates
MAX_FIRST_OBS = pd.Timestamp("2015-01-01")  # set to a Timestamp to drop firms that start after this date

# set to None to use all firms
TICKER_FILTER = None #["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
# Alternatively, specify GVKEYs directly (set to None to ignore)
GVKEY_FILTER = None

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
FEATURES = FEATURE_BASES  # firm_ratios.csv uses base names (no _Median/_Mean suffix)

DATA_FILE = Path(__file__).resolve().parent / "firm_ratios.csv"


# %%
####################################
# Building DataFrame
####################################

df_ratios = pd.read_csv(DATA_FILE, parse_dates=["public_date"])
df_ratios = df_ratios[
    (df_ratios["public_date"] >= START_DATE) & (df_ratios["public_date"] <= END_DATE)
]

if TICKER_FILTER is not None and "TICKER" in df_ratios.columns:
    df_ratios = df_ratios[df_ratios["TICKER"].isin(TICKER_FILTER)]
if GVKEY_FILTER is not None:
    df_ratios = df_ratios[df_ratios["gvkey"].isin(GVKEY_FILTER)]

df_ratios = df_ratios.set_index(["gvkey", "public_date"])[FEATURES]

# numeric conversion
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

df_ratios = df_ratios.map(_to_numeric)
df_ratios.sort_index(level="public_date", inplace=True)

print("Dataframe info:")
print(df_ratios.info())
print("Head:")
print(df_ratios.head())
print("Tail:")
print(df_ratios.tail())
print("Describe (numeric):")
print(df_ratios.describe())
print("Non-NaN count by column:")
print(df_ratios.count())

# keep only firms that have at least one non-NaN for every feature in the window
avail = df_ratios.notna().groupby(level=0).any()
keep_firms = avail.index[avail.all(axis=1)]
if keep_firms.empty:
    raise ValueError("No firms have complete feature coverage in the selected window.")
df_ratios = df_ratios[df_ratios.index.get_level_values("gvkey").isin(keep_firms)]

# %%
####################################
# Constructing Tensor
####################################

firms = sorted(df_ratios.index.get_level_values("gvkey").unique())
dates = sorted(df_ratios.index.get_level_values("public_date").unique())
features = df_ratios.columns

# align start to latest first observation across firms and forward-fill within firm
first_obs_by_firm = (
    df_ratios.reset_index()
    .groupby("gvkey")["public_date"]
    .min()
)

if MAX_FIRST_OBS is not None:
    late_mask = first_obs_by_firm > MAX_FIRST_OBS
    if late_mask.any():
        late_firms = first_obs_by_firm[late_mask].index.tolist()
        print(f"Dropping firms starting after {MAX_FIRST_OBS.date()}: {late_firms}")
        df_ratios = df_ratios[~df_ratios.index.get_level_values("gvkey").isin(late_firms)]
        firms = sorted(df_ratios.index.get_level_values("gvkey").unique())
        first_obs_by_firm = (
            df_ratios.reset_index()
            .groupby("gvkey")["public_date"]
            .min()
        )

latest_first_obs = first_obs_by_firm.max()
df_ratios = df_ratios[df_ratios.index.get_level_values("public_date") >= latest_first_obs]
dates = sorted(
    d for d in df_ratios.index.get_level_values("public_date").unique() if d >= latest_first_obs
)
firms = sorted(df_ratios.index.get_level_values("gvkey").unique())
full_index = pd.MultiIndex.from_product(
    [firms, dates], names=["gvkey", "public_date"]
)
df_ratios = (
    df_ratios.reindex(full_index)
    .groupby(level=0)
    .ffill()
)

if STANDARDIZE:
    def _zscore(col: pd.Series) -> pd.Series:
        std = col.std(ddof=0)
        if std == 0 or np.isnan(std):
            return col * 0.0
        return (col - col.mean()) / std

    level = 0 if STANDARDIZE_BY_FIRM else 1
    df_ratios = df_ratios.groupby(level=level).transform(_zscore)

# drop firms that still have any NaNs after fill/standardize
nan_by_firm = df_ratios.isna().groupby(level=0).apply(lambda g: g.values.any())
if nan_by_firm.to_numpy().any():
    bad_firms = nan_by_firm[nan_by_firm].index.tolist()
    print(f"Dropping firms with unresolved NaNs: {bad_firms}")
    keep_set = set(nan_by_firm[~nan_by_firm].index)
    df_ratios = df_ratios[df_ratios.index.get_level_values("gvkey").isin(keep_set)]
    firms = sorted(df_ratios.index.get_level_values("gvkey").unique())

# if any NaNs remain, iteratively drop affected firms
while df_ratios.isna().any().any():
    bad_firms = df_ratios[df_ratios.isna().any(axis=1)].index.get_level_values(0).unique()
    print(f"Dropping firms still containing NaNs: {list(bad_firms)}")
    df_ratios = df_ratios[~df_ratios.index.get_level_values(0).isin(bad_firms)]
    firms = sorted(df_ratios.index.get_level_values("gvkey").unique())
    if not len(firms):
        break

if df_ratios.empty or df_ratios.isna().any().any():
    raise ValueError("Tensor still contains NaNs after filling.")

df_ratios = df_ratios.sort_index(level=["gvkey", "public_date"])

array = df_ratios.values.reshape(len(firms), len(dates), len(features))
tensor = np.transpose(array, (0, 2, 1))  # firms x features x dates
print("Tensor shape:", tensor.shape)
print(f"Date span: {dates[0].date()} to {dates[-1].date()} ({len(dates)} periods)")

# %%
####################################
# Tucker Decomposition 
####################################

tl.set_backend("numpy")

ranks = [
    min(len(firms), len(FEATURES)),
    min(len(FEATURES), 12),
    min(len(dates), 49),
]
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

avg_time_factor = np.mean(np.abs(time_factors), axis=1)
series = pd.Series(avg_time_factor, index=dates, name="Firm Index (Tucker)")
print(series.head())
print("Top 5 dates by Tucker index value:")
print(series.nlargest(5))
print(f"Min/Max: {series.min():.6f} / {series.max():.6f}")


# %%
####################################
# CP Decomposition + Save Plots
####################################

cp_rank = 1
weights, factors = parafac(tensor, rank=cp_rank, n_iter_max=500, tol=1e-6, init="svd")
Xhat_cp = cp_to_tensor((weights, factors))
num = np.linalg.norm(tensor - Xhat_cp)
den = np.linalg.norm(tensor)
cp_rel_err = float(num / den) if den > 0 else np.nan
print(f"CP rank={cp_rank} relative error: {cp_rel_err:.4f}")

cp_time = factors[2][:, 0] * weights[0]
cp_series = pd.Series(np.abs(cp_time), index=dates, name="Firm Consensus (CP)")
print(cp_series.head())
print("Top 5 dates by CP index value:")
print(cp_series.nlargest(5))
print(f"CP Min/Max: {cp_series.min():.6f} / {cp_series.max():.6f}")

# plot both series
fig, ax = plt.subplots(figsize=(10, 5))
series.plot(ax=ax, label="Tucker (firms)")
cp_series.plot(ax=ax, label="CP (rank-1)")
ax.set_title("Firm ratios: Tucker vs CP indices")
ax.set_xlabel("Date")
ax.set_ylabel("Index value")
ax.legend()
fig.tight_layout()
plot_path = Path(__file__).resolve().parent / "mag7_indices.png"
plt.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"Saved comparison plot to {plot_path}")
