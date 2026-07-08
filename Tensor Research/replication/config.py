"""Single configuration module for the tensor replication package.

Everything the pipeline needs lives here: the 40-feature Compustat spec,
universe selection, date ranges, the calendar-fixed train/test split, Tucker
imputation ranks, and all filesystem paths.

Paths
-----
Three environment variables control where the package reads and writes; all
have sensible defaults so the package runs out of the box on the original lab
data without copying it:

    REPL_DATA_DIR    directory holding the input CSVs (fundamentals, GICS,
                     event-study pulls). Default: the original "Code for paper"
                     directory next to this package.
    REPL_CACHE_DIR   where tensor caches are written. Default: replication/cache
    REPL_RESULTS_DIR where results are written. Default: replication/results

Panel variants (development 50-firm, extended, 499, HY) are selected with:

    REPL_FUNDAMENTALS  fundamentals CSV filename within REPL_DATA_DIR
                       (default: the extended panel used by the paper)
    REPL_TOP_N         universe size, top-N by mkvaltq at 2024Q4 (default 50)
    REPL_END_DATE      panel end date (default 2026-06-30, the extended panel)
    REPL_GVKEYS_FILE   optional explicit gvkey list (one per line), overriding
                       top-N selection (used for the curated HY universe)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("REPL_DATA_DIR", str(PROJECT_DIR / "Code for paper")))
CACHE_DIR = Path(os.environ.get("REPL_CACHE_DIR", str(PACKAGE_DIR / "cache")))
RESULTS_DIR = Path(os.environ.get("REPL_RESULTS_DIR", str(PACKAGE_DIR / "results")))

FUNDAMENTALS_FILE = DATA_DIR / os.environ.get(
    "REPL_FUNDAMENTALS", "90-26_Q_Fundamentals_v2_extended.csv")
GICS_FILE = DATA_DIR / "gvkeys_to_gics.csv"

# Event-study data pulls (CRSP daily, market index, link table, OptionMetrics
# IV, FF3 factors, Markit CDS) live in per-universe subdirectories of the
# original cache; see src/data/README note in the package README.
EVENT_DIR_EXTENDED = DATA_DIR / "pre_prediction_cache" / "event_study_extended"
EVENT_DIR_499 = DATA_DIR / "pre_prediction_cache" / "event_study_499"
EVENT_DIR_HY = DATA_DIR / "pre_prediction_cache" / "event_study_hy"

LOCKED_CELLS_FILE = PACKAGE_DIR / "locked_cells.csv"

# --------------------------------------------------------------------------- #
# Panel definition
# --------------------------------------------------------------------------- #
START_DATE = os.environ.get("REPL_START_DATE", "2005-01-01")
# NOTE on END_DATE vintages (verified bit-identical against the originals):
#   50-firm extended cache (paper's T=21 holdout): REPL_END_DATE=2026-03-31
#     (capped at 2026Q1 because 2026Q2 was barely reported at build time)
#   499-firm and HY caches:                        REPL_END_DATE=2026-06-30
END_DATE = os.environ.get("REPL_END_DATE", "2026-06-30")
SEED = 42

# Calendar-fixed train/test boundary: the first quarter whose prediction
# target belongs to the TEST block. Frozen at 2021Q1 so extending the panel
# only appends test quarters and never shifts the training window.
TEST_START_TARGET_QUARTER = os.environ.get("REPL_TEST_START_Q", "2021Q1")

UNIVERSE_TOP_N: int = int(os.environ.get("REPL_TOP_N", "50"))
UNIVERSE_REF_QUARTER: str = "2024Q4"  # market-cap snapshot quarter
GVKEYS_FILE = os.environ.get("REPL_GVKEYS_FILE", "")

# Tucker imputation ranks for the per-window pre-processing pass; tensor shape
# per window is (firms, features, L). One-standard-error picks from the
# stratified hidden-cell CV (sweep documented in the paper and RESEARCH_LOG):
#   L=2: [2, 2, 2]   L=4: [4, 4, 4]
IMPUTATION_RANKS: dict[int, list[int]] = {2: [2, 2, 2], 4: [4, 4, 4]}

LOOKBACKS: tuple[int, ...] = (2, 4)
MODES: tuple[str, ...] = ("LEVELS", "SURPRISE")


def cache_path(mode: str, L: int) -> Path:
    return CACHE_DIR / f"tensor_{mode.lower()}_L{L}.pkl"


def meta_path() -> Path:
    return CACHE_DIR / "meta.pkl"


def select_universe_gvkeys(top_n: int | None = None,
                           ref_quarter: str = UNIVERSE_REF_QUARTER) -> list[str]:
    """Universe = top-N gvkeys by market cap (mkvaltq) at the reference quarter,
    or an explicit list from REPL_GVKEYS_FILE (curated HY universe)."""
    if GVKEYS_FILE:
        return [ln.strip() for ln in Path(GVKEYS_FILE).read_text().splitlines()
                if ln.strip()]

    import pandas as pd

    df = pd.read_csv(
        FUNDAMENTALS_FILE,
        dtype={"gvkey": str},
        usecols=["gvkey", "tic", "conm", "datadate", "mkvaltq"],
        low_memory=False,
    )
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df["quarter_period"] = df["datadate"].dt.to_period("Q")
    ref = pd.Period(ref_quarter, freq="Q")
    snap = df[(df["quarter_period"] == ref) & df["mkvaltq"].notna()]
    n = top_n if top_n is not None else UNIVERSE_TOP_N
    return (snap.sort_values("mkvaltq", ascending=False)
                .head(n)["gvkey"].astype(str).tolist())


# --------------------------------------------------------------------------- #
# Feature spec: 40 quarterly accounting features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    label: str
    source_columns: tuple[str, ...]
    transform: str = "as_reported"        # or "ytd_to_quarterly"
    combine: str = "first_non_null"


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("Acquisitions", ("aqcy",), transform="ytd_to_quarterly"),
    FeatureSpec("Assets - Other - Total", ("aoq",)),
    FeatureSpec("Assets and Liabilities", ("anoq",)),
    FeatureSpec("Capital Expenditures", ("capxy",), transform="ytd_to_quarterly"),
    FeatureSpec("Cash and Short-Term Investments", ("cheq",)),
    FeatureSpec("Cash Dividends", ("dvy",), transform="ytd_to_quarterly"),
    FeatureSpec("Common/Ordinary Equity - Total", ("ceqq",)),
    FeatureSpec("Comprehensive Income - Total", ("ciq",)),
    FeatureSpec("Cost of Goods Sold", ("cogsq",)),
    FeatureSpec("Debt in Current Liabilities", ("dlcq",)),
    FeatureSpec("Earnings Per Share (Basic)", ("epspxq",)),
    FeatureSpec("Earnings Per Share (Diluted)", ("epsfxq",)),
    FeatureSpec("Excess Tax Benefit of Stock Options", ("txbcofy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Extraordinary Items", ("xidoq", "xiq")),
    FeatureSpec("Financing Activities - Net Cash Flow", ("fincfy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Funds from Operations - Other", ("fopoy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Quarterly Income Before Extraordinary Items", ("ibq",)),
    FeatureSpec("Annual Income Before Extraordinary Items", ("ibadj12",)),
    FeatureSpec("Income Taxes", ("txtq",)),
    FeatureSpec("Intangible Assets - Total", ("intanq",)),
    FeatureSpec("Inventories - Total", ("invtq",)),
    FeatureSpec("Investing Activities - Net Cash Flow", ("ivncfy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Investing Activities - Other", ("ivacoy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Long-Term Debt - Total", ("dlttq",)),
    FeatureSpec("Long-Term Debt - Issuance", ("dltisy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Liabilities Netting Other Adjustments", ("lnoq",)),
    FeatureSpec("Noncontrolling Interest", ("mibtq", "mibq")),
    FeatureSpec("Non-Operating Income (Expense) - Total", ("nopiq",)),
    FeatureSpec("Operating Activities - Net Cash Flow", ("oancfy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Operating Income Before Depreciation", ("oibdpq", "oiadpq")),
    FeatureSpec("Preferred/Preference Stock (Capital) - Total", ("pstkq",)),
    FeatureSpec("Pretax Income", ("piq",)),
    FeatureSpec("Receivables - Total", ("rectq",)),
    FeatureSpec("Sale of Common and Preferred Stock", ("sstky",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Sale of Investments", ("sivy",), transform="ytd_to_quarterly"),
    FeatureSpec("Sale of PPE and Investments - Gain/Loss", ("sppivy",),
                transform="ytd_to_quarterly"),
    FeatureSpec("Sales/Turnover (Net)", ("saleq",)),
    FeatureSpec("Short-Term Investments - Total", ("ivstq",)),
    FeatureSpec("Special Items", ("spiq",)),
    FeatureSpec("Stockholders Equity", ("seqq",)),
)

LOCAL_META_COLUMNS: tuple[str, ...] = (
    "gvkey", "datadate", "tic", "conm", "fyearq", "fqtr",
    "ggroup", "gind", "gsector", "gsubind",
)

# MFI (pre-prediction) construction constants: full 1990-2024 panel, all firms.
# The MFI deliberately uses the 2024-vintage fundamentals (not the extended
# panel) because its FCIX comparison series ends in 2024Q4.
MFI_FUNDAMENTALS_FILE = DATA_DIR / "90-25_Q_Fundamentals_v2.csv"
MFI_START_QUARTER = "1990Q1"
MFI_END_QUARTER = "2024Q4"
MFI_START_DATE = "1990-01-01"   # datadate window of the MFI fundamentals pull
MFI_END_DATE = "2024-12-31"
MFI_TUCKER_RANKS = (67, 40, 20)   # (firms, features, time) for the 499x40x140 tensor
MFI_NUM_BINS = 6                  # equal-frequency bins for independence tests
MFI_NUM_PERMUTATIONS = 10_000

# Series produced by the original pipeline, used as comparison inputs by the
# MFI rebuild (FCIX quarterly series; v1 MFI for the old-vs-new correlation).
PRE_PRED_CACHE = DATA_DIR / "pre_prediction_cache"
FCIX_QUARTERLY_FILE = PRE_PRED_CACHE / "fcix_quarterly.csv"
MFI_V1_QUARTERLY_FILE = PRE_PRED_CACHE / "mfi_quarterly.csv"
