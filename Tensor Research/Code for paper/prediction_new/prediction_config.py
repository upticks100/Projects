"""prediction_new/prediction_config.py

Configuration for the prediction redo:
  - 50 mega-cap tickers (computational tractability + index-weight coverage)
  - 40-feature spec (matches paper Table; imported from pre_prediction_config)
  - v2 fundamentals data
  - 2005-2024 date range (current prediction-section convention)

This file is the single source of truth for the prediction_new pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from pre_prediction_config import (  # noqa: E402
    FEATURE_SPECS,
    LOCAL_FUNDAMENTALS_FILE,
    LOCAL_GICS_FILE,
    LOCAL_META_COLUMNS,
)

# --- Environment overrides for the EXTENDED ("refresh") build ---------------
# Defaults reproduce the Part 1 (2005-2024) run exactly. Set these env vars to
# build the extended panel into VERSIONED paths without ever touching Part 1:
#   PRED_FUNDAMENTALS_FILE  -> extended fundamentals CSV
#   PRED_END_DATE           -> e.g. "2026-06-30"
#   PRED_START_DATE         -> e.g. "2005-01-01"
#   PRED_CACHE_DIR          -> e.g. ".../tensor_cache_ext"
#   PRED_TEST_START_Q       -> first TEST target quarter (calendar-fixed split)
if os.environ.get("PRED_FUNDAMENTALS_FILE"):
    LOCAL_FUNDAMENTALS_FILE = Path(os.environ["PRED_FUNDAMENTALS_FILE"])

START_DATE = os.environ.get("PRED_START_DATE", "2005-01-01")
END_DATE = os.environ.get("PRED_END_DATE", "2024-12-31")
SEED = 42

# Calendar-fixed train/test boundary: the first quarter that belongs to the
# TEST block (its prediction target). Frozen so extending the sample only
# appends new quarters to the test side. For the original 80-quarter panel
# this reproduces the old int(0.8*n_windows) split exactly (L=2->62, L=4->60).
TEST_START_TARGET_QUARTER = os.environ.get("PRED_TEST_START_Q", "2021Q1")

# Universe is selected programmatically from v2 fundamentals: top-N firms by
# market cap (mkvaltq) at the reference quarter. Reproducible and avoids
# survivorship bias from a hand-curated list.
UNIVERSE_TOP_N: int = 50
UNIVERSE_REF_QUARTER: str = "2024Q4"  # quarter at which we measure market cap


def select_universe_gvkeys(top_n: int = UNIVERSE_TOP_N,
                           ref_quarter: str = UNIVERSE_REF_QUARTER) -> "list[str]":
    """Return the top-N gvkeys by market cap (mkvaltq) at the reference quarter.

    Reads the v2 fundamentals lazily so this stays a pure config helper.
    """
    import pandas as pd  # local import to keep module load light

    df = pd.read_csv(
        LOCAL_FUNDAMENTALS_FILE,
        dtype={"gvkey": str},
        usecols=["gvkey", "tic", "conm", "datadate", "mkvaltq"],
        low_memory=False,
    )
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df["quarter_period"] = df["datadate"].dt.to_period("Q")
    ref = pd.Period(ref_quarter, freq="Q")
    snap = df[(df["quarter_period"] == ref) & df["mkvaltq"].notna()]
    return (
        snap.sort_values("mkvaltq", ascending=False)
            .head(top_n)["gvkey"].astype(str).tolist()
    )

# Tucker imputation ranks for the per-window pre-processing pass.
# Tensor shape per window is (Firms, Features, L). Ranks must be <= each dim.
#
# Re-derived for the new universe (top-50 by mkvaltq, 40 features, v2 data) via
# sweep_imputer_ranks_cv.py. The CV hides 10% of observed cells, stratified by
# feature within each rolling window, and scores only those hidden cells.
# Observed cells are preserved exactly in build_prediction_caches.py; Tucker only
# fills NaNs. The old values [40, 20, L] over-fit per-window tensors badly.
#
# One-standard-error picks from sweep_results/imputer_rank_cv_stratified_summary.csv:
#   L=2: [2, 2, 2] is statistically tied with the best [3, 3, 2]
#   L=4: [4, 4, 4] is the best one-SE choice
IMPUTATION_RANKS: dict[int, list[int]] = {
    2: [2, 2, 2],
    4: [4, 4, 4],
}

LOOKBACKS: tuple[int, ...] = (2, 4)
MODES: tuple[str, ...] = ("LEVELS", "SURPRISE")

# CPRegressor search space for the Optuna study.
RANK_RANGE: tuple[int, int] = (5, 80)          # was (5, 50) in legacy code
REG_W_RANGE: tuple[float, float] = (1e-5, 1e3)  # was (1e-4, 100) in legacy code
N_ITER_MAX = 150

# Calibrated standalone CP follow-up search (`residual_delta_v2`).
# Designed to address the weak/flat residual-delta scores from the first pass:
#   - tighter, smaller-rank focus (residual signal likely low-rank),
#   - stronger upper regularization bound,
#   - GAMMA scaling so CP can shrink its contribution toward 0
#     (the optimizer can pick gamma=0 as a safety net equal to FE-only),
#   - optional per-feature target standardization so high-variance features
#     do not dominate the pooled residual objective.
RDV2_RANK_RANGE: tuple[int, int] = (1, 25)
RDV2_REG_W_RANGE: tuple[float, float] = (1e-2, 1e5)
RDV2_GAMMA_RANGE: tuple[float, float] = (0.0, 2.0)

TENSOR_CACHE_DIR = Path(os.environ.get("PRED_CACHE_DIR", str(ROOT_DIR / "tensor_cache")))
JOURNAL_DIR = ROOT_DIR / "optuna_journal"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"


def cache_path(mode: str, L: int) -> Path:
    return TENSOR_CACHE_DIR / f"tensor_{mode.lower()}_L{L}.pkl"


def meta_path() -> Path:
    return TENSOR_CACHE_DIR / "meta.pkl"


def journal_path(mode: str, L: int) -> Path:
    return JOURNAL_DIR / f"study_{mode.lower()}_L{L}.log"


def study_name(mode: str, L: int) -> str:
    return f"cp_pred_{mode.lower()}_L{L}"


# Warm-start hints from the legacy Optuna run (for enqueueing as known-good seeds).
LEGACY_WARM_START: dict[tuple[str, int], dict] = {
    ("LEVELS", 2):   {"RANK_REGRESS": 8,  "REG_W": 98.1655, "USE_RMS_SCALING": True},
    ("LEVELS", 4):   {"RANK_REGRESS": 5,  "REG_W": 67.3035, "USE_RMS_SCALING": True},
    ("SURPRISE", 2): {"RANK_REGRESS": 43, "REG_W": 52.4373, "USE_RMS_SCALING": True},
    ("SURPRISE", 4): {"RANK_REGRESS": 49, "REG_W": 85.3185, "USE_RMS_SCALING": True},
}
