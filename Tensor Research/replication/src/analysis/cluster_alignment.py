"""Affinity-propagation cluster alignment metrics vs GICS.

This is the clean, non-search version of the original `affinity_groups.py`.
It evaluates the canonical six-feature design used in the draft and reports
ARI, NMI, AMI, and purity against GICS levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

import config


SEC_THRESH = 0.25
FEATURES = ["GProf", "evm", "roe", "debt_ebitda", "fcf_ocf", "quick_ratio"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--firm-ratios", type=Path,
                   default=config.DATA_DIR / "firm_ratios.csv")
    p.add_argument("--gics", type=Path, default=config.GICS_FILE)
    p.add_argument("--out-dir", type=Path,
                   default=config.RESULTS_DIR / "cluster_alignment")
    return p.parse_args()


def _missing_by_security(df: pd.DataFrame,
                         id_cols=("gvkey", "public_date")) -> pd.DataFrame:
    d = (df.sort_values(list(id_cols))
           .drop_duplicates(subset=list(id_cols), keep="last"))
    feature_cols = [c for c in d.columns if c not in id_cols]
    all_gvkeys = d[id_cols[0]].dropna().unique()
    all_dates = d[id_cols[1]].dropna().sort_values().unique()
    full_index = pd.MultiIndex.from_product([all_gvkeys, all_dates],
                                            names=id_cols)
    full = (d.set_index(list(id_cols))[feature_cols]
              .reindex(full_index).sort_index())
    total = len(all_dates) * len(feature_cols)
    counts = full.isna().sum(axis=1).groupby(level=id_cols[0]).sum()
    return (counts.reset_index(name="nan_count")
                  .assign(missing_rate=lambda x: x["nan_count"] / total))


def purity_score(true: np.ndarray, pred: np.ndarray) -> float:
    total = 0
    for label in np.unique(pred):
        vals, counts = np.unique(true[pred == label], return_counts=True)
        if len(counts):
            total += counts.max()
    return float(total / len(true))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.firm_ratios, dtype={"gvkey": str})
    df["public_date"] = pd.to_datetime(df["public_date"])
    miss = _missing_by_security(df)
    good = miss.loc[miss["missing_rate"] < SEC_THRESH, "gvkey"]
    df = df[df["gvkey"].isin(good)]
    drop_cols = [c for c in [
        "adate", "qdate", "TICKER", "PEG_trailing", "sale_nwc", "divyield",
    ] if c in df.columns]
    df = df.drop(columns=drop_cols)
    id_cols = ["gvkey", "public_date"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    mat = df.groupby("gvkey")[feature_cols].mean(numeric_only=True)
    use_features = [f for f in FEATURES if f in mat.columns]
    X_df = mat[use_features].dropna(axis=0, how="any")
    X = StandardScaler().fit_transform(X_df.to_numpy())
    pred = AffinityPropagation(random_state=0).fit_predict(X)

    gics = pd.read_csv(args.gics, dtype={"gvkey": str})
    gics["datadate"] = pd.to_datetime(gics["datadate"])
    gics = gics.sort_values("datadate").groupby("gvkey").tail(1)
    gics = gics.set_index("gvkey").reindex(X_df.index)

    rows = []
    for level in ["gsector", "ggroup", "gind", "gsubind"]:
        true = gics[level]
        mask = true.notna().to_numpy()
        y = true[mask].astype(str).to_numpy()
        p = pred[mask]
        rows.append({
            "gics_level": level,
            "n_firms": int(mask.sum()),
            "n_gics_classes": int(pd.Series(y).nunique()),
            "n_ap_clusters": int(len(np.unique(p))),
            "ari": float(adjusted_rand_score(y, p)),
            "nmi": float(normalized_mutual_info_score(y, p)),
            "ami": float(adjusted_mutual_info_score(y, p)),
            "purity": purity_score(y, p),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "affinity_gics_alignment.csv", index=False)
    meta = {
        "features": use_features,
        "sec_missing_threshold": SEC_THRESH,
        "design_shape": list(X_df.shape),
        "n_ap_clusters": int(len(np.unique(pred))),
    }
    (args.out_dir / "affinity_gics_alignment.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(out.to_string(index=False))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
