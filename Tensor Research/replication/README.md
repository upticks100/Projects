# Tensor Research Replication Package

This directory is the clean, readable replication surface for the paper. It keeps
the historical `Code for paper/` directory untouched and ports only the canonical
pipeline: clean data pulls, 40-feature tensor caches, locked-cell refits,
universe transfer, MFI rebuild, event studies, veer signals, and CDS translation.

The Optuna search is intentionally not part of normal replication. The four
paper cells are frozen in `locked_cells.csv`, copied from
`prediction_new/results/v3_holdout_ext_20260629_230144/aggregate_summary.csv`.
Replication begins from those locked hyperparameters.

## Layout

```text
replication/
  config.py                  # single source of paths, features, dates, ranks
  locked_cells.csv           # four frozen objective x L cells
  environment.yml            # conda/mamba environment
  scripts/                   # thin numbered CLIs
  src/data/                  # WRDS fetchers and panel/universe helpers
  src/tensors/               # cache builder and LowMemCPRegressor
  src/model/                 # locked-cell refit and transfer check
  src/mfi/                   # MFI rebuild, figures, robustness checks
  src/analysis/              # event study, veer, CDS, cluster alignment
  tests/                     # low-memory CP equivalence test
  verify/                    # targeted verification artifacts
```

## Data Expectations

Raw licensed data is not copied into this package. By default, `config.py`
points `REPL_DATA_DIR` at the existing `../Code for paper/` directory, which
contains the WRDS/Markit/OptionMetrics-derived files used in the paper.

The main environment variables are:

```bash
export REPL_DATA_DIR="/path/to/Code for paper"
export REPL_CACHE_DIR="/path/to/cache"
export REPL_RESULTS_DIR="/path/to/results"
export REPL_TOP_N=50              # or 499
export REPL_END_DATE=2026-03-31   # extended 50-firm cache
export REPL_CP_LOWMEM=1           # exact low-memory CP fitter
```

Use `REPL_GVKEYS_FILE` to run a curated explicit universe such as HY/crossover
issuers instead of selecting top-N by `mkvaltq`.

## Pipeline

1. Fetch/update data if needed:

```bash
python scripts/01_fetch_data.py --help
```

2. Build rolling tensor caches:

```bash
REPL_END_DATE=2026-03-31 python scripts/02_build_caches.py
```

3. Refit locked cells and dump predictions:

```bash
REPL_CP_LOWMEM=1 python scripts/03_refit_and_dump.py \
  --objective residual_delta_v3 --L 2 --out-dir results/refit
```

4. Check transfer at wider universe:

```bash
python scripts/04_transfer_check.py results/v3_holdout_499
```

5. Rebuild MFI and figures:

```bash
python scripts/05_build_mfi.py
python -m src.mfi.figures
```

6. Run event-study analysis:

```bash
python scripts/06_event_study.py --help
```

7. Run veer and CDS analyses:

```bash
python scripts/07_veer_and_cds.py veer <holdout_dir> --tag 499
python scripts/07_veer_and_cds.py cds  <holdout_dir> --tag hy \
  --event-dir "../Code for paper/pre_prediction_cache/event_study_hy" \
  --cds-file cds_markit_hy.csv.gz
```

HY CDS robustness:

```bash
python -m src.analysis.cds_translation <hy_holdout_dir> \
  --tag hy --event-dir "../Code for paper/pre_prediction_cache/event_study_hy" \
  --cds-file cds_markit_hy.csv.gz --robust-battery
```

## Verification Performed

Artifacts are under `verify/`.

- Cache equivalence: rebuilt the 50-firm extended cache in `verify/cache_scratch`
  and compared all four `tensor_{levels,surprise}_L{2,4}.pkl` files to
  `prediction_new/tensor_cache_ext`. `X`, `Y`, masks, recon errors, and failure
  flags match exactly (`max_abs=0`).
- Low-memory CP equivalence: `tests/test_cp_lowmem_equiv.py` passes in the
  research environment for block sizes 1, 4, and 100.
- One-cell refit: `residual_delta_v3`, L=2 was refit with `REPL_CP_LOWMEM=1`
  from the verified scratch cache. It exactly reproduces the locked holdout
  numbers (`base_R2=0.72063`, `ensemble_R2=0.76823`, `delta=+0.04760`) and
  writes `verify/refit_lowmem/predictions_residual_delta_v3_L2_rank1.pkl`.
- Transfer/analyzer checks: `verify/analyzers_499/transfer_check_499.csv`
  reproduces transfer-positive deltas in all four cells; veer panels and CDS
  translation artifacts are present under `verify/analyzers_499/`.
- HY CDS robustness: `verify/hy_cds_robustness.csv` shows the expected sign is
  stable across raw/log spreads, event windows, winsorization, quarter drops,
  leave-one-quarter-out checks, and spread terciles, while some conservative
  variants attenuate t-statistics.
- MFI/FCIX block permutation: `verify/mfi_block_permutation/` rejects
  independence at 1% for 4-quarter circular blocks and 5% for 8-quarter blocks.
- Cluster alignment: `verify/cluster_alignment/affinity_gics_alignment.csv`
  reports ARI/NMI/AMI/purity against GICS levels.

The initial stock-CP one-cell refit was stopped after it remained quiet in the
CP step; the completed verification uses `LowMemCPRegressor`, which is
algebraically equivalent and the solver used for the 499-firm scale-up.

## Study Materials

Use `../STUDY_GUIDE.md` first for reading order and paper-section-to-code
mapping. Use `../MODEL_HISTORY.md` for the history of why the prediction model
changed from Pure CP to baseline-plus-CP residual boosters.
