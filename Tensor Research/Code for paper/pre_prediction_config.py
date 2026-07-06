from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureSpec:
    label: str
    source_columns: tuple[str, ...]
    transform: str = "as_reported"
    combine: str = "first_non_null"


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent
PAPER_DRAFT_DIR = PROJECT_DIR / "Paper_Draft"
FIGURES_DIR = PAPER_DRAFT_DIR / "Figures"
FUNDAMENTALS_FIGURES_DIR = FIGURES_DIR / "Fundamentals"
CACHE_DIR = ROOT_DIR / "pre_prediction_cache"
AUDIT_DIR = CACHE_DIR / "audit"

LOCAL_FUNDAMENTALS_FILE = ROOT_DIR / "90-25_Q_Fundamentals_v2.csv"
LOCAL_GICS_FILE = ROOT_DIR / "gvkeys_to_gics.csv"
LOCAL_CRSP_FILE = PROJECT_DIR / "Masoud" / "CRSP_Data.csv"

REFRESHED_CRSP_FILE = CACHE_DIR / "CRSP_Data_1990_2024.csv"
CRSP_EXTENSION_FILE = CACHE_DIR / "CRSP_Data_2024_extension.csv"
FUNDAMENTALS_PANEL_FILE = CACHE_DIR / "fundamentals_panel_40_features.csv"
FCIX_DAILY_FILE = CACHE_DIR / "fcix_daily.csv"
FCIX_QUARTERLY_FILE = CACHE_DIR / "fcix_quarterly.csv"
MFI_QUARTERLY_FILE = CACHE_DIR / "mfi_quarterly.csv"
MERGED_QUARTERLY_FILE = CACHE_DIR / "mfi_fcix_quarterly.csv"
CP_ERROR_FILE = CACHE_DIR / "cp_relative_error.csv"
SUMMARY_FILE = CACHE_DIR / "pre_prediction_summary.json"
PERMUTATION_FILE = CACHE_DIR / "independence_permutation_summary.csv"

FIGURE_1_PATH = FIGURES_DIR / "Fundamentals_Tensor.pdf"
FIGURE_2_EPS_PATH = FUNDAMENTALS_FIGURES_DIR / "Relative_Error.eps"
FIGURE_2_PDF_PATH = FUNDAMENTALS_FIGURES_DIR / "Relative_Error.pdf"
FIGURE_3_PATH = FIGURES_DIR / "Fig_QMFI.pdf"
FIGURE_4_PATH = FIGURES_DIR / "Fig_Cross_Corr_Quarters.pdf"

START_DATE = "1990-01-01"
END_DATE = "2024-12-31"
START_QUARTER = "1990Q1"
END_QUARTER = "2024Q4"

SEED = 42
NUM_BINS = 6
NUM_PERMUTATIONS = 10_000
CP_RANK_MAX = 100
TUCKER_R1 = 67
TUCKER_R3 = 20
CRSP_PRICE_THRESHOLD = 10.0
CRSP_VOLUME_THRESHOLD = 1000.0
CRSP_ROW_THRESHOLD = 0.01
GICS_LEVEL = "gind"

WRDS_CRSP_LIBRARY = "crsp"
WRDS_CRSP_TABLE = "dsf"
WRDS_FUND_LIBRARY = "comp"
WRDS_FUND_TABLE = "fundq"

# Each feature is mapped to one coherent quarterly accounting signal.
# YTD cash-flow fields are differenced within gvkey/fyearq before tensor construction.
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
    FeatureSpec(
        "Excess Tax Benefit of Stock Options",
        ("txbcofy",),
        transform="ytd_to_quarterly",
    ),
    FeatureSpec("Extraordinary Items", ("xidoq", "xiq")),
    FeatureSpec("Financing Activities - Net Cash Flow", ("fincfy",), transform="ytd_to_quarterly"),
    FeatureSpec("Funds from Operations - Other", ("fopoy",), transform="ytd_to_quarterly"),
    FeatureSpec("Quarterly Income Before Extraordinary Items", ("ibq",)),
    FeatureSpec("Annual Income Before Extraordinary Items", ("ibadj12",)),
    FeatureSpec("Income Taxes", ("txtq",)),
    FeatureSpec("Intangible Assets - Total", ("intanq",)),
    FeatureSpec("Inventories - Total", ("invtq",)),
    FeatureSpec("Investing Activities - Net Cash Flow", ("ivncfy",), transform="ytd_to_quarterly"),
    FeatureSpec("Investing Activities - Other", ("ivacoy",), transform="ytd_to_quarterly"),
    FeatureSpec("Long-Term Debt - Total", ("dlttq",)),
    FeatureSpec("Long-Term Debt - Issuance", ("dltisy",), transform="ytd_to_quarterly"),
    FeatureSpec("Liabilities Netting Other Adjustments", ("lnoq",)),
    FeatureSpec("Noncontrolling Interest", ("mibtq", "mibq")),
    FeatureSpec("Non-Operating Income (Expense) - Total", ("nopiq",)),
    FeatureSpec("Operating Activities - Net Cash Flow", ("oancfy",), transform="ytd_to_quarterly"),
    FeatureSpec("Operating Income Before Depreciation", ("oibdpq", "oiadpq")),
    FeatureSpec("Preferred/Preference Stock (Capital) - Total", ("pstkq",)),
    FeatureSpec("Pretax Income", ("piq",)),
    FeatureSpec("Receivables - Total", ("rectq",)),
    FeatureSpec("Sale of Common and Preferred Stock", ("sstky",), transform="ytd_to_quarterly"),
    FeatureSpec("Sale of Investments", ("sivy",), transform="ytd_to_quarterly"),
    FeatureSpec("Sale of PPE and Investments - Gain/Loss", ("sppivy",), transform="ytd_to_quarterly"),
    FeatureSpec("Sales/Turnover (Net)", ("saleq",)),
    FeatureSpec("Short-Term Investments - Total", ("ivstq",)),
    FeatureSpec("Special Items", ("spiq",)),
    FeatureSpec("Stockholders Equity", ("seqq",)),
)

LOCAL_META_COLUMNS: tuple[str, ...] = (
    "gvkey",
    "datadate",
    "tic",
    "conm",
    "fyearq",
    "fqtr",
    "ggroup",
    "gind",
    "gsector",
    "gsubind",
)

GICS_COLUMNS: tuple[str, ...] = (
    "gvkey",
    "tic",
    "ggroup",
    "gind",
    "gsector",
    "gsubind",
)
