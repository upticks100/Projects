"""Figure: per-feature CP booster ΔR² vs train-set stationarity (vr_stat)."""
from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
AGG = (
    ROOT.parent
    / "Code for paper"
    / "prediction_new"
    / "results"
    / "v3_holdout_20260620_084220"
    / "per_feature_20260629_161824"
    / "per_feature_aggregate.csv"
)
OUT = ROOT / "Figures" / "Fig_Booster_Stationarity.pdf"

# Short leaders in display points (visual angles).
# L=4: LT debt at 60°, Other assets at 0° (horizontal).
LABEL_OFFSET_PTS = {
    (2, "LT debt"): (28, 0),
    (2, "Sales"): (28, 0),
    (2, "Other assets"): (28, -10),
    (4, "Sales"): (28, 12),
    (4, "LT debt"): (
        32 * math.cos(math.radians(60)),
        32 * math.sin(math.radians(60)),
    ),
    (4, "Other assets"): (32, 0),
}
SHORT = {
    "Long-Term Debt - Total": "LT debt",
    "Sales/Turnover (Net)": "Sales",
    "Assets - Other - Total": "Other assets",
}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.sqrt(np.dot(x_c, x_c) * np.dot(y_c, y_c)))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x_c, y_c) / denom)


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_c = x - x.mean()
    slope = float(np.dot(x_c, y - y.mean()) / np.dot(x_c, x_c))
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def main() -> None:
    df = pd.read_csv(AGG)
    ridge = df[df["objective"] == "ridge_delta_v3"].copy()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)

    for ax, L in zip(axes, (2, 4)):
        sub = ridge[ridge["L"] == L]
        g = (
            sub.groupby("feature_name", as_index=False)
            .agg(delta=("delta", "mean"), vr_stat=("vr_stat", "first"))
            .dropna(subset=["delta", "vr_stat"])
        )
        x = g["vr_stat"].to_numpy(dtype=float)
        y = g["delta"].to_numpy(dtype=float)
        r = _corr(x, y)

        ax.scatter(
            x,
            y,
            s=34,
            facecolors="white",
            edgecolors="0.15",
            linewidths=0.9,
            zorder=3,
        )
        slope, intercept = _ols_line(x, y)
        x_line = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(x_line, intercept + slope * x_line, color="0.2", lw=1.5, zorder=2)
        ax.axhline(0.0, color="0.65", lw=0.7, ls="--", zorder=1)

        g_idx = g.set_index("feature_name")
        for feat, short in SHORT.items():
            if feat not in g_idx.index:
                continue
            row = g_idx.loc[feat]
            px = float(row["vr_stat"])
            py = float(row["delta"])
            dx, dy = LABEL_OFFSET_PTS[(L, short)]
            ax.annotate(
                short,
                xy=(px, py),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color="0.2",
                ha="left",
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.55",
                    "lw": 0.55,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=4,
                clip_on=False,
            )

        ax.set_title(rf"Ridge booster, $L={L}$  ($r={r:.2f}$)", pad=8)
        ax.set_xlabel(
            r"$\mathrm{vr\_stat}=\mathrm{std}(\Delta x_f)/\mathrm{std}(x_f)$"
            "\n(lower $=$ more trending)"
        )
        if L == 2:
            ax.set_ylabel(r"Per-feature $\Delta R^{2}$")
        ax.set_xlim(0.05, 1.55)
        ax.set_ylim(-0.12, 0.65)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
