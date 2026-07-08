"""Fundamentals panel assembly: CSV -> filtered firm-quarter panel.

Shared by the tensor cache builder (src/tensors/build_caches.py) and the MFI
rebuild (src/mfi/build_mfi.py). Ported verbatim from the original pipeline
(Build_PrePrediction_Exhibits.py helpers + build_prediction_caches.py loader).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def first_available_column(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Coalesce the first non-null value across candidate Compustat columns."""
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column in df.columns:
            values = values.where(values.notna(), pd.to_numeric(df[column], errors="coerce"))
    return values


def ytd_to_quarterly(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Convert year-to-date cash-flow fields to quarterly flows by
    within-fiscal-year differencing (Q1 keeps the reported YTD value)."""
    work = df[["gvkey", "datadate", "fyearq", "fqtr"]].copy()
    work["value"] = values
    work["fyearq"] = pd.to_numeric(work["fyearq"], errors="coerce")
    work["fqtr"] = pd.to_numeric(work["fqtr"], errors="coerce")
    work = work.sort_values(["gvkey", "fyearq", "fqtr", "datadate"])

    previous = work.groupby(["gvkey", "fyearq"])["value"].shift(1)
    quarterly = work["value"] - previous
    quarterly = quarterly.where(work["fqtr"] != 1, work["value"])
    quarterly = quarterly.where(work["fqtr"].between(1, 4), np.nan)
    return quarterly.reindex(df.index)


def load_filtered_panel() -> tuple[pd.DataFrame, list[str], list[str], list]:
    """Load the fundamentals CSV, restrict to the configured universe and date
    range, snap to a strict quarterly grid (last report per firm-quarter),
    apply the 40-feature spec, and log-modulus transform.

    Returns (df, feature_names, firms, quarters)."""
    print(f"[1/4] reading {config.FUNDAMENTALS_FILE.name}")
    print(f"      selecting top {config.UNIVERSE_TOP_N} firms by mkvaltq at "
          f"{config.UNIVERSE_REF_QUARTER}")
    universe = config.select_universe_gvkeys()
    print(f"      universe size: {len(universe)} gvkeys")

    needed = set(config.LOCAL_META_COLUMNS) | {
        c for spec in config.FEATURE_SPECS for c in spec.source_columns}
    df = pd.read_csv(
        config.FUNDAMENTALS_FILE,
        dtype={"gvkey": str},
        usecols=lambda c: c in needed,
        low_memory=False,
    )
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["gvkey", "datadate"]).copy()
    df = df[df["gvkey"].isin(universe)]
    df = df[(df["datadate"] >= config.START_DATE) & (df["datadate"] <= config.END_DATE)]
    df = df.sort_values(["gvkey", "datadate"])
    df["quarter_period"] = df["datadate"].dt.to_period("Q")
    df = df.drop_duplicates(["gvkey", "quarter_period"], keep="last")
    print(f"      filtered rows: {len(df):,}  gvkeys: {df['gvkey'].nunique()}  "
          f"quarters: {df['quarter_period'].nunique()}")

    print("[2/4] applying 40-feature spec (incl. ytd_to_quarterly for cash-flow YTDs)")
    feature_names = []
    for spec in config.FEATURE_SPECS:
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
    """Pivot the panel into a (firms, features, quarters) float32 tensor with
    NaN for missing entries."""
    print(f"[3/4] building raw tensor: {len(firms)} firms x {len(feature_names)} "
          f"features x {len(quarters)} quarters")
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
