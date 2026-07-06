from __future__ import annotations

import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tensorly.cp_tensor import cp_to_tensor
from tensorly.decomposition import parafac

from pre_prediction_config import (
    CACHE_DIR,
    CP_ERROR_FILE,
    CP_RANK_MAX,
    END_QUARTER,
    FEATURE_SPECS,
    FIGURE_2_EPS_PATH,
    FIGURE_2_PDF_PATH,
    LOCAL_FUNDAMENTALS_FILE,
    LOCAL_META_COLUMNS,
    SEED,
    START_QUARTER,
)


MAX_CP_RANK = CP_RANK_MAX


def first_available_column(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = pd.Series(np.nan, index=df.index, dtype="float64")
    for column in columns:
        if column in df.columns:
            values = values.where(values.notna(), pd.to_numeric(df[column], errors="coerce"))
    return values


def ytd_to_quarterly(df: pd.DataFrame, values: pd.Series) -> pd.Series:
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


def load_raw_fundamentals() -> pd.DataFrame:
    source_columns = {col for spec in FEATURE_SPECS for col in spec.source_columns}
    needed_columns = set(LOCAL_META_COLUMNS) | source_columns
    df = pd.read_csv(
        LOCAL_FUNDAMENTALS_FILE,
        usecols=lambda col: col in needed_columns,
        dtype={"gvkey": str},
        low_memory=False,
    )
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["gvkey", "datadate"]).copy()

    df = df.sort_values(["gvkey", "datadate"])
    df["quarter_period"] = df["datadate"].dt.to_period("Q")
    return df.drop_duplicates(["gvkey", "quarter_period"], keep="last")


def build_tensor() -> tuple[np.ndarray, np.ndarray]:
    df = load_raw_fundamentals()
    feature_names = [spec.label for spec in FEATURE_SPECS]

    for spec in FEATURE_SPECS:
        values = first_available_column(df, spec.source_columns)
        if spec.transform == "ytd_to_quarterly":
            values = ytd_to_quarterly(df, values)
        df[spec.label] = values

    has_data = df.groupby("gvkey")[feature_names].apply(lambda x: x.notna().to_numpy().any())
    firms = sorted(has_data[has_data].index.astype(str))
    quarters = pd.period_range(START_QUARTER, END_QUARTER, freq="Q")
    full_index = pd.MultiIndex.from_product([firms, quarters], names=["gvkey", "quarter_period"])
    df = df.set_index(["gvkey", "quarter_period"]).sort_index()

    slices = []
    for feature in feature_names:
        wide = (
            pd.to_numeric(df[feature], errors="coerce")
            .reindex(full_index)
            .unstack("quarter_period")
            .reindex(index=firms, columns=quarters)
        )
        slices.append(wide.to_numpy(dtype=np.float32))

    tensor = np.stack(slices, axis=1)
    mask = np.isfinite(tensor)
    print(
        f"Raw tensor: {tensor.shape[0]} firms x {tensor.shape[1]} features x "
        f"{tensor.shape[2]} quarters; observed density {mask.mean():.2%}",
        flush=True,
    )
    return tensor, mask


def cp_error(tensor: np.ndarray, mask: np.ndarray, rank: int) -> float:
    observed = tensor[mask]
    scale = float(np.sqrt(np.mean(observed**2))) if observed.size else 1.0
    filled = np.nan_to_num(tensor / max(scale, 1e-8), nan=0.0)
    base_norm = np.linalg.norm(filled[mask])

    weights, factors = parafac(
        filled,
        rank=rank,
        init="random",
        random_state=SEED,
        mask=mask.astype(np.float64),
    )
    reconstructed = cp_to_tensor((weights, factors))
    return float(np.linalg.norm((filled - reconstructed) * mask) / (base_norm + 1e-12))


def run_cp_sweep(tensor: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    rows = []
    if CP_ERROR_FILE.exists():
        old = pd.read_csv(CP_ERROR_FILE)
        if {"rank", "observed_relative_error"}.issubset(old.columns):
            rows = old[old["rank"] <= MAX_CP_RANK].to_dict("records")

    done = {int(row["rank"]) for row in rows}
    for rank in range(1, MAX_CP_RANK + 1):
        if rank in done:
            print(f"CP rank {rank}/{MAX_CP_RANK} cached", flush=True)
            continue

        start = time.perf_counter()
        error = cp_error(tensor, mask, rank)
        rows.append(
            {
                "rank": rank,
                "observed_relative_error": error,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        pd.DataFrame(rows).sort_values("rank").to_csv(CP_ERROR_FILE, index=False)
        print(f"CP rank {rank}/{MAX_CP_RANK}: error={error:.6f}", flush=True)

    return pd.DataFrame(rows).sort_values("rank")


def plot_cp_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(curve["rank"], curve["observed_relative_error"], color="#1f4e79", linewidth=2)
    ax.set_xlabel("CP Rank")
    ax.set_ylabel("Observed Relative Error")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    FIGURE_2_EPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_2_EPS_PATH, format="eps")
    fig.savefig(FIGURE_2_PDF_PATH)
    plt.close(fig)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tensor, mask = build_tensor()
    curve = run_cp_sweep(tensor, mask)
    plot_cp_curve(curve)
    print(curve.sort_values("observed_relative_error").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
