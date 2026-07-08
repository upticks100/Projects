"""Regenerate the two pre-prediction paper figures from the clean MFI v2 rebuild.

The original generator lived in the full Build_PrePrediction_Exhibits.py (since
truncated); the v1 figures in Paper_Draft/Figures/ were built from the polluted
April v1 tensor (audit Finding 1). This script reads the v2 series produced by
rebuild_mfi_tensor_v2.py and writes NEW files (suffix _v2) so nothing is
overwritten:

  Paper_Draft/Figures/Fig_QMFI_v2.pdf              <- mfi_v2/mfi_quarterly_v2.csv
  Paper_Draft/Figures/Fig_Cross_Corr_Quarters_v2.pdf <- mfi_v2/mfi_fcix_quarterly_v2.csv

Run:  <research-env python> regen_prepred_figures_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MFI_V2_DIR = HERE / "pre_prediction_cache" / "mfi_v2"
FIG_DIR = HERE.parent / "Paper_Draft" / "Figures"

MAX_LAG = 10  # quarters, matches the v1 exhibit (-10..+10)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def quarter_to_ts(q: pd.Series) -> pd.Series:
    return pd.PeriodIndex(q, freq="Q").to_timestamp(how="end")


def fig_qmfi(out: Path) -> None:
    df = pd.read_csv(MFI_V2_DIR / "mfi_quarterly_v2.csv")
    mfi_col = [c for c in df.columns if c.lower().startswith("mfi")][0]
    t = quarter_to_ts(df["quarter"])

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(t, df[mfi_col], color="black", lw=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"$\mathrm{MFI}_t$")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample cross-correlation corr(x_t, y_{t+lag}) for lag in [-max_lag, max_lag]."""
    out = np.empty(2 * max_lag + 1)
    n = len(x)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            a, b = x[: n - lag], y[lag:]
        else:
            a, b = x[-lag:], y[: n + lag]
        out[i] = np.corrcoef(a, b)[0, 1]
    return out


def fig_cross_corr(out: Path) -> None:
    df = pd.read_csv(MFI_V2_DIR / "mfi_fcix_quarterly_v2.csv").dropna(
        subset=["MFI_v2", "FCIX"])
    fcix = df["FCIX"].to_numpy(float)
    mfi = df["MFI_v2"].to_numpy(float)
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    ccf = cross_correlation(fcix, mfi, MAX_LAG)
    conf = 1.96 / np.sqrt(len(df))

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.stem(lags, ccf, linefmt="k-", markerfmt="ko", basefmt="k-")
    ax.axhline(conf, color="0.4", ls="--", lw=0.9)
    ax.axhline(-conf, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel("Lag (quarters)")
    ax.set_ylabel("Cross-correlation")
    ax.set_xticks(lags[::2])
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    peak = int(lags[np.argmax(np.abs(ccf))])
    print(f"wrote {out} | peak lag={peak} value={ccf[np.argmax(np.abs(ccf))]:.3f} "
          f"| conf band ±{conf:.3f} | n={len(df)}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_qmfi(FIG_DIR / "Fig_QMFI_v2.pdf")
    fig_cross_corr(FIG_DIR / "Fig_Cross_Corr_Quarters_v2.pdf")


if __name__ == "__main__":
    main()
