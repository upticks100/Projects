import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import AffinityPropagation
from datetime import datetime

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEC_THRESH = 0.25 # filter rate for secturities missing (dates x feature)
# FEAT_THRESH = 0.40  

#### Missingness Functions ####

def compute_worst_features(df: pd.DataFrame,
                           id_cols=("gvkey","public_date")) -> pd.DataFrame:
    """Given the raw df, return features ranked by missingness (count and rate)."""
    # Identify universe and feature columns
    _df = (df.sort_values(list(id_cols))
             .drop_duplicates(subset=list(id_cols), keep='last'))
    feature_cols = [c for c in _df.columns if c not in id_cols]

    all_gvkeys = _df[id_cols[0]].dropna().unique()
    all_dates  = _df[id_cols[1]].dropna().sort_values().unique()

    full_index = pd.MultiIndex.from_product([all_gvkeys, all_dates], names=id_cols)
    df_full = (_df.set_index(list(id_cols))[feature_cols]
                 .reindex(full_index)
                 .sort_index())

    n_gvkeys = len(all_gvkeys)
    n_dates  = len(all_dates)
    total_cells = n_gvkeys * n_dates

    feat_nan_counts = df_full[feature_cols].isna().sum(axis=0)
    return (
        pd.DataFrame({
            'nan_count': feat_nan_counts,
            'missing_rate': feat_nan_counts / total_cells,
        })
        .sort_values('missing_rate', ascending=False)
    )


def compute_worst_securities(df: pd.DataFrame,
                             id_cols=("gvkey","public_date")) -> pd.DataFrame:
    """Given the raw df, return gvkeys ranked by
    missingness rate and count."""
    _df = (df.sort_values(list(id_cols))
             .drop_duplicates(subset=list(id_cols), keep='last'))
    feature_cols = [c for c in _df.columns if c not in id_cols]

    all_gvkeys = _df[id_cols[0]].dropna().unique()
    all_dates  = _df[id_cols[1]].dropna().sort_values().unique()

    full_index = pd.MultiIndex.from_product([all_gvkeys, all_dates], names=id_cols)
    df_full = (_df.set_index(list(id_cols))[feature_cols]
                 .reindex(full_index)
                 .sort_index())

    n_dates = len(all_dates)
    n_features = len(feature_cols)
    total_cells_per_security = n_dates * n_features

    sec_nan_counts = df_full.isna().sum(axis=1).groupby(level=id_cols[0]).sum()
    return (
        sec_nan_counts.reset_index(name='nan_count')
                      .assign(missing_rate=lambda d: d['nan_count'] / total_cells_per_security)
                      .sort_values('missing_rate', ascending=False)
    )


########## 

df = pd.read_csv('VScode/Alpha/Research/Code for paper/firm_ratios.csv')
# Ensure date column is present and parsed from THIS file

df['public_date'] = pd.to_datetime(df['public_date'])

df_worst_securities = compute_worst_securities(df)
print(df_worst_securities.head(100))

num_good_firms = (df_worst_securities['missing_rate'] < SEC_THRESH).sum()
print(f"Number of firms with <{int(SEC_THRESH*100)}% missing: {num_good_firms}")

# Keep only firms with missing rate below threshold (on the RAW df)
good_gvkeys = df_worst_securities.loc[df_worst_securities['missing_rate'] < SEC_THRESH, 'gvkey']
df = df[df['gvkey'].isin(good_gvkeys)]

df_worst_features = compute_worst_features(df)
print(df_worst_features.head(50))

df = df.drop(columns=
             ['adate', 
              'qdate', 
              'TICKER', 
              'PEG_trailing', 
              'sale_nwc', 
              'divyield' ])

##### DESIGN MATRIX #####

id_cols = ['gvkey', 'public_date']
feature_cols = [c for c in df.columns if c not in id_cols]

mat = (
    df.groupby('gvkey')[feature_cols]
      .mean(numeric_only=True)
)

X_df = mat.dropna(axis=0, how='any')

print("Design matrix (rows x features):", X_df.shape)

X = X_df.to_numpy()
row_index = X_df.index
col_index = X_df.columns.tolist()

selected_features = ['GProf','evm','roe','debt_ebitda','fcf_ocf','quick_ratio']
use_features = [f for f in selected_features if f in X_df.columns]

X_sub = X_df[use_features].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sub)

ap = AffinityPropagation(random_state=0)
labels = ap.fit_predict(X_scaled)

n_clusters = len(np.unique(labels))
print(f"Affinity Propagation with {len(use_features)} features found {n_clusters} exemplars/clusters.")


gvkeys_to_gics = pd.read_csv('VScode/Alpha/Research/Code for paper/gvkeys_to_gics.csv')
gvkeys_to_gics["datadate"] = pd.to_datetime(gvkeys_to_gics["datadate"])
gvkeys_to_gics = gvkeys_to_gics.sort_values('datadate').groupby('gvkey').tail(1)
gvkeys_to_gics = gvkeys_to_gics[['gvkey','tic', 'ggroup', 'gind', 'gsector', 'gsubind']].copy()

final_gvkeys = X_df.index.unique()
_gics = gvkeys_to_gics[gvkeys_to_gics['gvkey'].isin(final_gvkeys)].copy()
unique_counts = {
    "ggroup": _gics["ggroup"].nunique(),
    "gind": _gics["gind"].nunique(),
    "gsector": _gics["gsector"].nunique(),
    "gsubind": _gics["gsubind"].nunique(),
}
print("GICS unique counts (after full filtering and NaN drop):", unique_counts)

out_path = "ap_random_hits.txt"

BASE = ['GProf','evm','roe','debt_ebitda','fcf_ocf','quick_ratio']
#BASE = []
all_feats = [c for c in X_df.columns if c not in BASE]


TARGETS = np.array([21, 50, 10, 84])
TOL = 2               
N_TRIALS = 300000        
rng = np.random.default_rng(0)

true_labels = _gics.set_index('gvkey').reindex(X_df.index)['ggroup']

with open(out_path, "a") as f:  # append mode
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n===== RUN {ts} | TOL=±{TOL} | N_TRIALS={N_TRIALS} | TARGETS={TARGETS.tolist()} =====\n")
    f.write(f"BASE features: {BASE}\n")
    f.write("Filter: AMI > 0.2 and |k - TARGETS| <= TOL\n")
    hit_count = 0
    for t in range(N_TRIALS):
        # sample and run AP (existing code remains inside loop)

        extra = rng.choice(all_feats, size=6, replace=False)
        use = BASE + list(extra)
        X_use = X_df[use].to_numpy()
        X_scaled = StandardScaler().fit_transform(X_use)#
        ap = AffinityPropagation(random_state=0, verbose=False)
        pred = ap.fit_predict(X_scaled)
        k = len(np.unique(pred))
        mask = ~true_labels.isna()
        ami = adjusted_mutual_info_score(true_labels[mask], pred[mask.values]) if mask.any() else np.nan
        
        if (not np.isnan(ami)) and (ami > 0.25) and np.any(np.abs(k - TARGETS) <= TOL):
            hit_count += 1
            f.write(f"[hit] trial {t}: k={k}, AMI={ami:.3f}, features={use}\n")
        if t % 10000 == 0 and t > 0:
            print(f"Progress: trial {t}/{N_TRIALS}, hits so far = {hit_count}")
    
    f.write(f"Recorded {hit_count} hits within ±{TOL} of targets {TARGETS.tolist()}.\n")

print(f"Random search finished. Results appended to {out_path}")
