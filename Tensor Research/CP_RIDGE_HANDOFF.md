# CP vs Fixed-Effects Ridge Prediction Handoff

Date: 2026-04-30 (initial), 2026-07-07 (prior addenda), **2026-07-14 (current)**
Project root: `/student/mcnama53/Projects/Tensor Research`
Primary readable history: `MODEL_HISTORY.md` (send this, not the full log)
Full audit trail: `RESEARCH_LOG.md`
Current prediction code: `Code for paper/prediction_new/`
**Active paper draft:** `Paper_Draft/main_v2.tex` / `main_v2.pdf`
Stale paper narrative: `Paper_Draft/main.tex` (historical only — do not edit)

---

## 2026-07-14 CURRENT STATUS (READ THIS FIRST)

### Paper
`main_v2.tex` is the draft to share. Relative to `main.tex`:

- Prediction section rewritten as **baseline + CP booster** (FE or ridge),
  selected on incremental \(\Delta R^2\), not pooled \(R^2\).
- Old headline table (CP beats ridge by up to +0.028) is a **mirage** —
  see `MODEL_HISTORY.md` Phase 2. Honest holdout: FE cells ~+0.043–0.048;
  ridge cells ~+0.011–0.019; transfer to ~498 firms preserves the sign.
- Economic arc locked in the draft:
  - **Size** of CP revision → post-announcement vol in mega-caps, **subsumed by ATM IV**
  - **Sign** / cash-flow drift (veers) → HY CDS tightening ~**6.5 bp per 1 SD of
    cash-flow drift per quarter**, 4/4 frozen setups; IG CDS clean null
- FE defined explicitly as firm–feature training-window means \(\mu_{ij}\)
- Dual corresponding authors (Masoud first, Bryan second); no “Contributing author” label
- Prose readability pass; Hadamard/low-mem ALS solver detail **removed from the paper**
  (still in code; algebraically equivalent)
- Float placement: soft `\FloatBarrier` at major sections only (no `\clearpage` spam)

### One-line ELI5 for coauthors
Firms moving away from what the tensor expects → credit signal (HY CDS).
How hard the tensor revises the baseline → vol signal, already in options IV.

### Where to read what
| Need | File |
|---|---|
| Why the model changed (readable) | `MODEL_HISTORY.md` |
| Day-by-day / numbers | `RESEARCH_LOG.md` |
| Paper | `Paper_Draft/main_v2.pdf` |
| Operational prediction code | `Code for paper/prediction_new/` |

### Empirics still locked (do not retune without a new pre-registration)
- Four frozen setups: FE vs ridge × \(L\in\{2,4\}\); FEATURE_TARGET_SCALE on,
  FEATURE_X_SCALE off; USE_RMS on except residual \(L=4\) False
- Holdout / transfer / HY CDS / H1 DD numbers as in `main_v2` tables
- HY CDS: **4/4** cells (not 3/4 — older notes were wrong)

### MFI/FCIX
Macro exhibits in `main_v2` use the clean 40-feature rebuild (earlier handoff
“polluted April v1 / provisional” caveat is **obsolete** for the current draft).

### Auth / backup
Push target remains github.com/upticks100/Projects (licensed caches untracked).

---

## 2026-07-07 WHY THE MODEL TYPE CHANGED (consolidated — detail in MODEL_HISTORY)

Full readable narrative: `MODEL_HISTORY.md`. Short version:

The original (Jan 2026, pre-log) design was Pure CP + fixed effects on
`N=49`/`F=24`, selected by pooled R², benchmarked against ridge — this is
still what `main.tex`'s prediction section describes. The Apr 26–29
40-feature/499-firm data rebuild triggered a rerun of that same search at
scale, which exposed three separate, successively-deeper problems, each
caught by a dedicated audit rather than assumed away:

1. **Apr 30** — pooled R² is degenerate: CP could "win" by shrinking to zero
   and reproducing pure FE. Fixed with the `residual_delta` objective
   (`R²(FE+CP) − R²(FE-only)`).
2. **May 25** — `residual_delta` v1 (7 days of search) still gravitated to
   FE-imitation via high rank + heavy regularization. Fixed with
   `residual_delta_v2` (explicit `GAMMA` floor + per-feature scaling). Also:
   a real inner-CV leakage bug in the CP-matched Ridge baseline's alpha
   selection was found and patched the same day.
3. **Jun 19** — the Ridge-booster track's headline delta was found to be
   partly an OOF-fallback artifact (CP training against nothing in early
   windows lacking real out-of-fold Ridge predictions). Fixed by dropping
   un-initialized rows from booster training.

The two-track split (FE-residual CP vs. Ridge-booster CP, `residual_delta_v3`
/ `ridge_delta_v3`) was introduced May 28 and is the architecture that
survives into `main_v2.tex`, validated on the Jun 20 v3 holdout (CP wins all
4 locked cells). **The 40-feature expansion did not break the old model
mechanically — it triggered the rerun that surfaced a pooled-R² flaw latent
since January.**

## 2026-07-06 SUPERSEDING ADDENDUM (historical operational snapshot — superseded by 2026-07-14 above)

**The section titled "CURRENT EXECUTION PLAN (Authoritative)" further down is the
APRIL state and is NO LONGER AUTHORITATIVE** (external audit Finding 5). It is kept
for provenance only. Snapshot as of 2026-07-06/07 (still true unless noted):

1. **Part 1 is final and forward-validated.** CP/booster beats an alpha-tuned Ridge
   on OOS next-quarter fundamentals across all 4 locked cells (ridge_delta_v3 /
   residual_delta_v3 × L2/L4). The 21-quarter calendar-fixed extension added five
   truly out-of-sample quarters (2025Q1–2026Q1) — ensemble delta positive in all
   4 cells on them. Stationarity gradient confirmed in every cell.
2. **Part 2 (event study) is final: an honest, heavily-fortified null + one real
   but subsumed signal.** Event-return alpha: 0 headline survivors at BY<0.1 in
   all 4 cells (cell-invariant null). Vol signal (|cp_increment| forecasts
   post-event vol beyond lagged realized vol): strong in ridge cells (75/80,
   69/80), weak in residual cells → per-architecture claim. Option-implied vol
   subsumes it everywhere (ivctrl survivors 0/29/13/9 of 80; the ridge-L4 29 does
   not replicate) → "beats lagged vol, subsumed by IV"; no straddle alpha (VRP
   mechanics only). OptionMetrics ATM 30d IV is wired into the builder (pre_iv).
3. **Veer anomaly + HY CDS (2026-07-07):** H1 drift_cashflow→ΔDD confirmed;
   H2 veers→ΔIV formally positive but economically negligible; **H-HY
   drift_cashflow→Δlog(CDS) confirmed 4/4** (~−6.5 bp per 1 SD cash-flow drift).
4. **MFI/FCIX:** rebuilt on clean 40-feature tensor for `main_v2` (July write-up).
5. **499-firm scale-up: DONE.** Transfer PASS 4/4. Low-memory CP fitter used at
   scale (`cp_regressor_lowmem.py`); not discussed in the paper body.
6. **Backup:** logs, handoff, prediction_new code, Optuna journals, result
   summaries → github.com/upticks100/Projects (licensed caches untracked).

Master ranked idea list + anomaly catalog: top of `RESEARCH_LOG.md`.

## 2026-06-29 Addendum — Part 2 (event study) pipeline: Codex audit fixes applied

The Part 2 economic-exercise pipeline was audited (Codex) before being
run. Six concrete issues were identified; all six are now patched in
the scripts. Part 1 R² results are **not** affected by these — they
exist purely in the unrun event-study path.

| # | Severity        | File                              | Fix |
|---|-----------------|-----------------------------------|-----|
| 1 | Confirmed bug   | `dump_test_predictions.py`        | `quarters_test` was L quarters too early; cache window w predicts `quarters[w+L]`. Also dumps `input_quarters` per window for audit. |
| 2 | Likely bug      | `build_event_study_dataset.py`    | PERMNO link is now resolved on `ann_date` (announcement-date link), not `datadate`. Old result preserved as `permno_datadate` diagnostic. |
| 3 | Methodology     | `build_event_study_dataset.py`    | Now emits parallel `*_raw_units` columns via `inv_log_modulus`; analyzer adds a raw-dollar robustness row. |
| 4 | Likely bug      | `analyze_event_study.py`          | `two_way_demean` replaces the old `demean_by` cell-level mistake. |
| 5 | Methodology     | `analyze_event_study.py`          | `pooled_ols` supports two-way clustered SEs (firm + quarter); regressions now report `*_t_cl2way` instead of HC0. |
| 6 | Methodology     | builder + analyzer                | Carry `mktcap_pre` (pre-event `|prc|×shrout`); analyzer emits `ls_ew_*` and `ls_vw_*` long-short rows. |

Not yet acted on (deliberate):
- Universe-by-2024Q4 framing — paper writing fix, not code.
- Holm / BH multiple-testing discipline — applied at result-interpretation
  time over the per-feature × per-cell table.
- FF3-adjusted abnormal returns and drift / pre-leakage windows —
  follow-up robustness pass once the headline numbers land.

Next step: rerun `dump_test_predictions.py` → `build_event_study_dataset.py`
→ `analyze_event_study.py` end-to-end on the rank-1 booster L=2
predictions once the per-feature jobs finish. No Optuna search needed.

### Part 2 FIRST RUN result (2026-06-29) — directionally consistent, UNDERPOWERED

Ran end-to-end on `ridge_delta_v3` L=2 rank-1
(`results/v3_holdout_20260620_084220/event_study_analysis/`).
Off-by-L verified fixed (targets 2021Q1–2024Q4, each fed by prior 2
quarters). Permno-on-ann_date mattered (22 rows re-routed).

- **Regression** `CAR ~ base + ensemble surprise`, firm-clustered SEs
  (primary; two-way clustering is non-PSD at T=16 — eigen-cleaning was
  tried and **reverted** because it manufactured fake t-stats):
  **33/40 features positive CP coefficient, 0/40 with |t|>1.96.**
- **Economic test** = quarterly long-short on ex-ante CP signal, 16
  quarterly returns, df=15: best Sharpe ≈ 0.60 (VW), **0/40 with
  |t|>2.13.** Top names are the same trending items that won on R².
- **Verdict:** underpowered exactly as the audit predicted (50 firms ×
  16 quarters). Frame Part 2 as suggestive/directional, NOT a headline
  economic result; do not claim a tradeable strategy. Part 1 unaffected
  and remains the contribution.

### Per-feature stationarity result (2026-06-29) — hypothesis CONFIRMED

All 20 per-feature cells complete
(`per_feature_20260629_161824/per_feature_summary.txt`). CP's gain
concentrates on non-stationary / trending features in **every** cell:
`vr_stat` corr −0.45 to −0.48 (low vr_stat = non-stationary → big
delta), `trend_slope` corr +0.44 to +0.50. Non-stationary tercile
delta is 6-12× the stationary tercile. Top features are Long-Term Debt,
Sales/Turnover, Assets-Other, Operating Income — all large trending
items, pos_rate 1.0 across top-5 trials. Sharpened paper claim is
supported: "CP captures latent structure specifically where linear
models break down." Suggested figure: per-feature delta vs vr_stat
scatter with fit, one panel per cell.

## 2026-06-20 Addendum — v3 holdout COMPLETE; CP wins all 4 cells, paper story is set

Full top-5 holdout (`results/v3_holdout_20260620_084220/`) finished
across 4 lab hosts. All 20 trials produce a positive ensemble test
delta over their baseline. Per-cell rank-1:

| Cell                              | Baseline          | base R² | ensemble R² | **delta** |
|-----------------------------------|-------------------|---------|-------------|-----------|
| `residual_delta_v3` LEVELS L=2    | FE (firm-feature) | 0.7253  | 0.7712      | +0.0459   |
| `residual_delta_v3` LEVELS L=4    | FE                | 0.7267  | 0.7711      | +0.0444   |
| `ridge_delta_v3`    LEVELS L=2    | Ridge (α-tuned)   | 0.7665  | 0.7845      | +0.0181   |
| `ridge_delta_v3`    LEVELS L=4    | Ridge             | 0.7684  | 0.7853      | +0.0169   |

The +0.017 to +0.018 booster delta over a properly alpha-tuned Ridge
baseline is the **paper headline number**. The +0.044 to +0.046 FE
delta is real but the FE baseline is less competitive.

### Regime story (best trial per cell)

The strongest single regime indicator is cross-sectional Y dispersion
(`y_disp`, mean per-feature std across firms in the test window):

| Cell                       | Q1 (low disp) | Q2       | Q3 (high disp) | corr   |
|----------------------------|---------------|----------|----------------|--------|
| `ridge_delta_v3`    L=2    | +0.0227       | +0.0171  | +0.0137        | −0.84  |
| `ridge_delta_v3`    L=4    | +0.0212       | +0.0159  | +0.0128        | −0.80  |
| `residual_delta_v3` L=2    | +0.0501       | +0.0408  | +0.0459        | −0.21  |
| `residual_delta_v3` L=4    | +0.0509       | +0.0387  | +0.0422        | −0.40  |

**CP helps most when firms move together** — the rank-K factorization
recovers shared cross-firm structure that linear-in-features Ridge
cannot capture. Strongest for the booster (booster CP only has to
explain Ridge-orthogonal signal, so the firm-firm component dominates).

Within-test window index also positively correlates with delta in all
4 cells (booster corr +0.69, FE corr +0.27 to +0.48): CP benefits from
training-history length.

### Verdict

v3 is paper-defensible. The OOF-fallback fix amplified the booster
signal from CV's +0.005 to holdout's +0.017 (3.5×), proving the
contaminated training was actively hurting CP.

### What is next, in priority order

1. Paper figures (per-window scatter, per-cell bar chart, regime
   table) → 1-2 h.
2. Per-feature R² breakdown on rank-1 model per cell.
3. Draft the v3 results paragraph for `Paper_Draft/main.tex`.

### What is explicitly NOT next

- v4 Optuna search. Top-5 plateau (std < 0.001 R² in L=2 cells) says
  the search has converged.
- Masked CP. Out of scope.
- Re-defining the baseline to make CP look better. Current Ridge
  baseline is correctly alpha-tuned with `ridge_structured_cp_matched_zero_filled_ts_cv`.

## 2026-06-19 Night Addendum — OOF-fallback fix landed in code; smokes running, holdout next

Acting on the audit results and the user's "apply the fixes
autonomously" instruction. Verbose detail in `RESEARCH_LOG.md`
("2026-06-19 (night)" entry); summary for the next AI / next reader:

### What changed in code

1. `prediction_new/worker.py::_compute_ridge_predictions_for_fold`
   - Returns `(ridge_oof_tr, ridge_va, initialized)`. `initialized` is a
     per-training-row boolean from the inner-TimeSeriesSplit pass.
   - **Removed** the silent fallback that wrote firm-feature means into
     `ridge_oof_tr` for un-initialised rows. That fallback was the
     mechanism behind the +0.005 OOF artifact.
   - Inner-TSS skip threshold (`inner_tr_idx.size < 5`) hoisted into a
     named constant `MIN_INNER_TR_SIZE`; the evaluator imports the same
     concept so train/test conventions can't drift.
2. `prediction_new/worker.py::make_objective`
   - Booster trials filter `X_tr / Y_tr / M_tr / base_tr` down to
     `pack["ridge_oof_valid"]` before CP fit. Per-feature X and Y
     scaling are fit on the filtered set.
   - If any booster fold has zero honest OOF rows, the whole trial
     returns NaN (Optuna prunes). Conservative; avoids selection bias.
   - Logs `honest_oof_rows=<kept>/<total>` per fold after Ridge
     precompute.
3. `prediction_new/audit_one_fit.py` — unpacks the new 3-tuple and
   replays the drop-fallback convention. Logs `honest_oof_rows`.
4. `prediction_new/evaluate_top_trials_test.py` — full rewrite (backup
   left as `.pre_v3_backup_*`):
   - Supports all 5 objective names. Honours `GAMMA`,
     `FEATURE_TARGET_SCALE`, `FEATURE_X_SCALE`, `USE_RMS_SCALING`.
   - `ridge_delta_v3` path replays the post-fix worker convention end
     to end: Ridge OOF on dev with the same skip rule, drop un-init
     rows, fit CP on filtered set, test = `Ridge_test + GAMMA *
     CP_test` with `Ridge_test` from
     `ridge_structured_cp_matched_zero_filled_ts_cv` on the full dev
     set.
   - Emits two CSVs: pooled summary and per-window deltas. Per-window
     is what supports the regime analysis hypothesis (CP wins in
     high-dispersion windows, ties elsewhere).
   - Incremental `.partial` writes so a crash mid-run doesn't lose
     completed trials.

### What is deliberately NOT done in this pass

- Persisting unclipped per-fold deltas as Optuna user-attrs (only
  matters for the next search; current top-K analysis uses the new
  CSV).
- Removing the `max(score, -1.0)` clip (same reason).
- Moving baseline helpers out of `CP_struct_test_new.py` (cosmetic).
- `prediction_config.py` survivorship-bias wording (paper edit pass).
- Building a masked CP estimator (explicitly out of scope per this
  handoff).

### Status as of writing

- Worker, audit script, and evaluator patched. No linter errors.
- **Both smokes PASSED with large positive deltas** — see numbers
  below; full discussion in `RESEARCH_LOG.md` ("2026-06-19 (night)").
- Full top-5 holdout fanned out to 4 lab hosts via
  `prediction_new/launch_v3_holdout.sh`. Output dir:
  `results/v3_holdout_20260620_084220/`.

### Smoke results (this is the big news)

| Smoke                              | CV delta | base test R² | ensemble test R² | **test delta** |
|------------------------------------|----------|--------------|------------------|----------------|
| `ridge_delta_v3` L=2 trial 7853    | +0.0046  | 0.7665       | 0.7841           | **+0.0176**    |
| `ridge_delta_v3` L=2 trial 2696    | +0.0046  | 0.7665       | 0.7845           | **+0.0181**    |
| `residual_delta_v3` L=2 trial 2170 | +0.0152  | 0.7253       | 0.7701           | **+0.0448**    |
| `residual_delta_v3` L=2 trial 2138 | +0.0149  | 0.7253       | 0.7685           | **+0.0415**    |

Test deltas are 3-4× the CV deltas in both code paths. Per-window CSV
inspection: **every single test window has a positive ensemble delta**
for both top-2 booster trials (range +0.003 to +0.024). The
OOF-fallback fix unmasked a real Ridge-orthogonal signal that
contaminated CV had been diluting.

### What is now actually in scope

1. Wait for full top-5 holdout (~3 h, dominated by L=4 FE on
   utmlab10-02).
2. Build a regime-tagged per-window analysis (cross-sectional return
   IQR or fundamentals dispersion as the regime indicator). Goal: is
   the gain uniform or concentrated? Either answer is a paper story.
3. Pick a single best `(objective, L, trial)` per cell as the headline
   model. Run per-feature / per-firm-size-bucket diagnostics on it.
4. Write the v3 verdict section of the paper. Only relaunch v4 if the
   per-window analysis reveals a specific failure mode worth fixing.

## 2026-06-19 Late-Evening Addendum — per-fold audit + fix plan

Quick decisive update on top of the morning addendum below.

**The +0.005 v3 booster CV signal is structurally an OOF-fallback
artifact.** A distributed per-fold audit (60 fits across 29 lab hosts;
47/60 returned, the missing are slow L=4 FE rank-12 trials that hit the
SSH timeout) decomposes the headline as follows:

| Cell                     | Fold 1                    | Fold 2          | Fold 3          | Mean        |
|--------------------------|---------------------------|-----------------|-----------------|-------------|
| `ridge_delta_v3` L=2     | **0.0000000000** (exact)  | 0.00345         | 0.01027         | 0.00461     |
| `ridge_delta_v3` L=4     | **0.0000000000** (exact)  | 0.00383         | 0.00939         | 0.00448     |
| `residual_delta_v3` L=2  | 0.00522 (real)            | 0.01635         | 0.02181         | 0.01522     |

Two takeaways:

1. **Real booster Ridge-orthogonal CP signal is `~+0.010 R²` in fold 3,
   not `+0.005`.** Fold 1 is an artifact (CP trained on FE residuals due
   to inner-TSS skip rule → contributes literally zero on top of Ridge).
   The mean is dragged down by half by this artifact.
2. **The booster search has plateaued.** Across-trial std on fold 2 and
   3 deltas is < 1% of the mean. Different `(rank, reg_w, gamma)`
   combinations produce essentially identical predictions. More search
   budget on the current space buys nothing.

The FE-residual cells (`residual_delta_v3`) do **not** show the
artifact — fold-1 deltas are real small positives, and the pattern
across folds is consistent with time-series learning rather than
structural collapse.

### Fix plan (approved, pending apply)

Three deliverables came back from a parallel audit pass and have been
reviewed and approved:

1. **OOF-fallback fix (option a)**: drop fallback rows from CP training
   in worker.py. Return a third element (`initialized`) from
   `_compute_ridge_predictions_for_fold`. In `make_objective`, slice
   `(X_tr, Y_tr, M_tr)` by that mask before fitting CP. Removes the
   artifact instead of modeling around it.
2. **Evaluator extension**: full rewrite of
   `evaluate_top_trials_test.py` to support all five objectives,
   replay all v2/v3 scaling toggles, score `Ridge_test + γ·CP_test` for
   the booster using the patched
   `ridge_structured_cp_matched_zero_filled_ts_cv`, and emit a
   per-window CSV for regime analysis.
3. **Regime indicator (primary)**: cross-sectional IQR of robust-
   standardized fundamentals per quarter, threshold at q75 of training
   windows. Robustness: cross-sectional return IQR from existing CRSP
   cache. Fold-local cuts for CV, dev-only cut for test.

### Sequence to apply

1. Apply worker.py fix.
2. Apply evaluator rewrite.
3. Run new evaluator on existing v3 journals to get a fair holdout
   under the corrected convention.
4. Decide:
   - Booster beats Ridge on test → paper main result, supported by
     per-window/regime analysis from the new evaluator output.
   - Neutral/borderline → short 24h v4 Optuna search with the fixed
     worker to see if the search ceiling moves.
   - Clearly negative → pivot the paper to FE-residual CP as the
     primary framing (it has no artifact, has real time-series learning
     signal, and the CV story is cleaner).
5. Build regime indicator as a small utility module once column
   conventions are agreed.

### Operational notes for whoever picks this up

- The audit confirms the diagnosis already documented in this file's
  morning addendum (below) — the booster CV result was real in the
  journal, not paper-ready as headline. Now we have hard per-fold
  evidence.
- The fix is small but load-bearing. Three pinning concerns:
  - The `inner_tr_idx.size < 5` skip threshold in
    `_compute_ridge_predictions_for_fold` becomes the operational
    definition of "honest Ridge OOF." Pin it and document it.
  - The same skip rule must appear identically in both worker.py and
    evaluator. Add a comment in both linking them.
  - Some booster trials may now produce NaN folds. Log
    `n_completed_folds` per trial so we can detect selection bias.
- All v3 journals are durable on NFS. No active workers.

## 2026-06-19 Morning Addendum — v3 results landed

The 72h v3 distributed search finished around 2026-05-31. Two clusters ran
in parallel: `residual_delta_v3` (v2 FE-residual + per-feature X-scaling
toggle) and `ridge_delta_v3` (CP fits Ridge residuals; score is
`R²(Ridge_va + γ·CP_va) − R²(Ridge_va)`). 28 hosts × 8 workers each.

Headline CV results:

| Study                          | n_trials | best Δ   | rank | gamma | fX (winner) |
|--------------------------------|---------:|---------:|-----:|------:|-------------|
| `residual_delta_v3` LEVELS L2  |    3,072 | 0.01522  |  13  | 1.263 | **False**   |
| `residual_delta_v3` LEVELS L4  |      880 | 0.01169  |  12  | 1.352 | **False**   |
| `ridge_delta_v3`   LEVELS L2   |    9,598 | 0.00461  |   4  | 0.751 | **False**   |
| `ridge_delta_v3`   LEVELS L4   |    2,297 | 0.00448  |   5  | 0.843 | **False**   |

Two things to internalize before reading anything else:

1. **Per-feature X scaling did not help.** All four winners selected
   `FEATURE_X_SCALE = False`. The v3-FE cluster essentially confirms v2.
   Do not relaunch this hypothesis without a new motivating argument.

2. **The Ridge booster found Ridge-orthogonal signal in CV.** Positive
   delta ≈ +0.005 R² at both lookbacks, achieved with low CP rank (4 and
   5) and sub-unit gamma (0.75-0.85). This is the right shape for an
   honest booster: small contribution that genuinely adds rather than
   re-deriving Ridge.

**Critical open question (must be resolved before any new compute spend):
does the booster CV delta survive the test holdout?**

- `prediction_new/evaluate_top_trials_test.py` does not currently know how
  to score the booster (it expects `FE + γ·CP`, not `Ridge + γ·CP`).
- The right next ~1 hour of work is to extend that evaluator and run the
  top-5 per booster study against the test split, using the patched
  `ridge_structured_cp_matched_zero_filled_ts_cv` (the same Ridge the
  booster trained against). Output CSV in the same schema as the v1/v2
  holdout tables for direct comparison.

Decision tree based on holdout result:

- Booster beats Ridge alone on test in ≥ 1 cell → this is the paper's
  prediction-section main result. Next compute spend should be a
  per-feature γ booster (40-dim γ vector trained jointly with CP) to
  exploit the per-feature persistence heterogeneity we already documented.
- Booster fails to beat Ridge on test → v3 search space is exhausted; the
  paper pivots to the Defensible Conditional / Fallback outcomes in the
  Enhancement success hierarchy below, and CP is reported as a methodological
  diagnostic + factor-analysis tool rather than a head-to-head predictor.

Operational notes:

- All v3 workers and watchdogs have exited. Journals are durable on NFS:
  `prediction_new/optuna_journal/study_levels_L{2,4}_{residual,ridge}_delta_v3.log`.
- The booster pipeline is in `worker.py::_compute_ridge_predictions_for_fold`
  and the `is_booster` branch of `make_objective`. Ridge OOF predictions are
  produced once per worker startup via nested `TimeSeriesSplit(3)` to avoid
  leakage. Early training windows without OOF Ridge fall back to FE
  residual targets — this is consistent ("subtract best baseline available")
  but worth re-confirming if booster holdout fails unexpectedly.

## 2026-05-02 Status Addendum

- A focused transformation audit of the active `prediction_new` path found no
  duplicate residual/surprise/imputation transform bug in the current code.
- Current data flow is:
  1) input-window imputation on `X` only (observed `X` preserved exactly, Tucker
     fills NaNs),
  2) mode-specific `X` representation (`SURPRISE`: RMS-normalized inputs;
     `LEVELS`: original-unit inputs with imputed missing cells),
  3) FE-centering + optional RMS scaling on training `Y` residuals inside CV
     folds,
  4) `residual_delta` used only as a scoring objective
     (`R²(FE+CP)-R²(FE)`), not as an extra data transform.
- Practical implication: if we choose to prioritize `LEVELS` for the next run,
  that should be treated as a modeling priority decision, not as a correction
  for a discovered preprocessing defect.

## Executive Conclusion

After reading the full paper draft, the prediction section should be treated as an
enhancement and validation layer for the paper's main contribution: a tensor
fundamentals framework and the Market Fundamentals Index (MFI). The paper is not
a Ridge paper, and the prediction section should not be rewritten as if Ridge is
the protagonist. Ridge is a stringent persistence control.

The completed legacy CP evidence still cannot support the current `main.tex`
claim that standalone CP globally beats Ridge. The old CP Optuna objective
optimized pooled R2 and collapsed to fixed-effect-only behavior. The top
completed L=2 CP trials selected by that objective reproduce the FE-only test
result exactly. That is a selection-objective failure, not a reason to abandon
the tensor prediction claim.

The goal now is to give CP the right statistical job. Standalone CP should be
reported as a diagnostic, but the paper-preserving CP question is whether
low-rank tensor structure adds signal after firm-feature fixed effects and
simple persistence controls have removed the easiest accounting dynamics:

```text
prediction = FE + Ridge_persistence + gamma * CP_residual
```

This is the useful idea in the old flat-ensemble script path, especially
`Code for paper/compare_ens_flat_vs_ridge_L2_L4.py`, but it is **not** evidence
that the old ensemble already worked. The old ensemble used suboptimal/stale
features and, by memory, did not perform well. Treat it as a design sketch to
re-evaluate inside `prediction_new` with the current v2 data, observed-cell-
preserving Tucker input imputation, leakage-safe development selection, and one
untouched holdout evaluation.

Do not try to make CP win by weakening Ridge, changing test splits, selecting favorable features post hoc, or chasing a custom target-mask-aware CP estimator during this draft cycle.

Enhancement success hierarchy:

1. Best outcome: calibrated standalone CP or Ridge+CP residual beats strong Ridge
   globally on the untouched holdout.
2. Strong paper outcome: CP beats the exact nested or CP-matched baseline and
   shows positive median/per-feature value, even if the per-feature Ridge control
   remains the strongest pooled persistence benchmark.
3. Defensible conditional outcome: CP wins in pre-specified low-persistence,
   low-FE-fit, high residual-correlation, or economically coherent feature
   groups defined using development data only.
4. Fallback outcome: prediction becomes a rigorous boundary condition on
   one-quarter fundamentals forecasting, while MFI construction and MFI/FCIX
   dependence remain the paper's central empirical contributions.

### Scope Decision (Critical)

For this paper timeline, **target-mask-aware CP regression is dropped**.
Treat any custom weighted-loss CP regressor as follow-up-methodology work, not a blocker for the current paper.

- We do **not** have bandwidth to build, validate, and robustly benchmark a custom
  masked CP optimizer (PyTorch/ALS/custom objective) to publication quality.
- Any major new estimator implementation would add substantial methodological and
  debugging risk and could derail paper completion.
- Therefore, the current paper should proceed with:
  - standalone residual-delta CP as a diagnostic,
  - FE+Ridge+CP-residual as the main CP attempt,
  - transparent strong/matched Ridge reporting,
  - clear limitations text stating that the CP regression implementation is not
    fitted with a target-observation mask.

## Agreement / Disagreement Snapshot

### Agreed

- Legacy pooled-R2 CP selection collapsed to FE-only behavior and is not suitable
  for selecting incremental CP signal.
- Residual-delta objective is the correct immediate fix for **standalone** CP model selection.
- The strongest CP path is probably Ridge residual augmentation, not standalone CP.
- Ridge comparisons must remain strong and transparent (no weakening baselines).
- Paper claims should be stated at the level supported by corrected objective
  and holdout evidence: global, nested-baseline, or pre-specified conditional
  CP value.

### Nuance / Open Questions

- "CP cannot beat Ridge" is **not** proven; it is currently "not demonstrated"
  under the completed legacy objective.
- CP may still show value conditionally (e.g., pre-specified lower-persistence or
  lower-FE-fit regimes), but this must be defined using development data only.
- Current CP-matched Ridge is useful for fairness diagnostics, but should not be
  overinterpreted until inner-fold alpha selection is fully leakage-safe.

### Implementation-Owner Context Notes (Added 2026-04-30)

- **Comment:** The currently running distributed studies are the standalone
  `worker.py --objective residual_delta` studies (not a ridge-residual CP worker).
  Any section below proposing a new `ridge_residual_worker.py` is a forward
  proposal, not current production status.
- **Comment:** The active relaunch is already live across 21 hosts with 168
  workers and watchdog coverage. Treat "start residual-delta studies" steps as
  completed unless explicitly marked as a fresh rerun.
- **Comment:** A true masked-target CP estimator remains out of scope for this
  paper cycle. Suggestions that imply a major new estimator build should be read
  as follow-up work, not current deliverables.
- **Comment:** The objective pivot to residual-delta is implemented in code and
  should remain the primary model-selection path until we have completed
  residual-delta holdout evaluations.
- **Comment:** Keep distinction between:
  1) confirmed empirical findings (legacy pooled collapse to FE-only),
  2) implemented fixes (residual-delta worker),
  3) speculative enhancements (ridge-residual CP architecture).
- **Read-only monitor update:** A later monitor check found only one completed
  residual-delta trial so far:
  - `cp_pred_levels_L2_residual_delta`: 22 total trials, 1 complete, 21 running,
    best delta `-0.45136512634752374` with `RANK_REGRESS=7`,
    `REG_W=0.00019499601303143508`, `USE_RMS_SCALING=True`.
  - `cp_pred_levels_L4_residual_delta`: 63 running, 0 complete.
  - `cp_pred_surprise_L2_residual_delta`: 21 running, 0 complete.
  - `cp_pred_surprise_L4_residual_delta`: 63 running, 0 complete.
  This is not enough evidence to conclude CP fails, but it shows that weakly
  regularized residual CP can severely degrade FE residual performance.

## CURRENT EXECUTION PLAN (Authoritative)

This section overrides ambiguity elsewhere in this document.

### What Is Running Right Now

- Active distributed studies are:
  - `cp_pred_levels_L2_residual_delta`
  - `cp_pred_levels_L4_residual_delta`
  - `cp_pred_surprise_L2_residual_delta`
  - `cp_pred_surprise_L4_residual_delta`
- Objective: `residual_delta = R²(FE + CP residual) - R²(FE-only)`.
- Infrastructure:
  - 21 hosts
  - 168 workers total
  - watchdog coverage enabled on all active hosts
  - 24-hour worker budgets

### Next 3 Tasks (In Order)

1. Finish (or time-box) the current residual-delta runs and collect completed
   trial distributions by study.
2. Evaluate top-k residual-delta-selected CP trials on holdout for both
   `L=2` and `L=4`, with FE and Ridge baselines in the same output table.
3. Rewrite the prediction section in `Paper_Draft/main.tex` using corrected
   methodology/results language (no legacy pooled-objective claims).

### Do Not Do During This Cycle

- Do not start a masked-target CP implementation.
- Do not pivot to a large new model architecture before finishing the current
  residual-delta evidence pipeline.
- Do not weaken Ridge, alter chronological splits, or perform post-hoc feature
  subset selection to force a CP win.

### Optional Only If Current Run Is Clearly Weak

- If residual-delta standalone CP remains weak after completed holdout checks,
  then run a small calibrated follow-up (`residual_delta_v2`) with tighter rank
  and stronger regularization controls.
- Treat ridge-residual CP worker ideas as secondary follow-up, not the primary
  path until the current residual-delta cycle is fully summarized.

## Key Files

- Current pipeline:
  - `Code for paper/prediction_new/worker.py`
  - `Code for paper/prediction_new/prediction_config.py`
  - `Code for paper/prediction_new/build_prediction_caches.py`
- Baselines and evaluation:
  - `Code for paper/prediction_new/evaluate_l2_top3_test.py`
  - `Code for paper/prediction_new/imputer_sensitivity.py`
  - future shared helpers should live in `Code for paper/prediction_new/baselines.py`
- Current dependency leak:
  - `evaluate_l2_top3_test.py` and `imputer_sensitivity.py` currently import Ridge/FE helper functions from `Code for paper/CP_struct_test_new.py`.
  - Treat that as implementation debt. Final pipeline scripts should import those helpers from `prediction_new/baselines.py` instead.
- Current caches:
  - `Code for paper/prediction_new/tensor_cache/`
- Current Optuna journals:
  - `Code for paper/prediction_new/optuna_journal/`
- Logs:
  - `Code for paper/prediction_new/logs/`
  - `Code for paper/distributed_logs/`
- Research chronology:
  - `RESEARCH_LOG.md`
- Paper draft needing correction:
  - `Paper_Draft/main.tex`
- Historical only:
  - `Code for paper/CP_struct_test_new.py`
  - `Code for paper/compare_ens_flat_vs_ridge_L2_L4.py`

## Current Data and Pipeline State

The current `prediction_new` pipeline uses:

- Top 50 firms by `mkvaltq` at `2024Q4`.
- 40-feature v2 fundamentals.
- Date range 2005Q1 through 2024Q4.
- Lookbacks L=2 and L=4.
- Modes:
  - `LEVELS`: observed inputs retained in original log-modulus units; imputed values unscaled back to those units.
  - `SURPRISE`: input windows divided by observed-window RMS.
- Tucker window imputation:
  - L=2 ranks `[2, 2, 2]`.
  - L=4 ranks `[4, 4, 4]`.
  - Observed cells are preserved exactly; Tucker fills only missing input cells.
- Target evaluation:
  - Mask-aware on observed target cells only.
  - 80% chronological development block, 20% chronological holdout.

`prediction_new/prediction_config.py` is the active source of truth:

- `FEATURE_SPECS` is imported from `pre_prediction_config.py`, so current feature identity should come from the v2 feature spec or `tensor_cache/meta.pkl`, not any old hard-coded feature list.
- `cache_path(mode, L)` points to `prediction_new/tensor_cache/tensor_{mode}_L{L}.pkl`.
- `meta_path()` points to `prediction_new/tensor_cache/meta.pkl`.
- `journal_path(mode, L)` and `study_name(mode, L)` define the current Optuna namespace.
- Current standalone CP range is `RANK_RANGE=(5,80)` and `REG_W_RANGE=(1e-5,1e3)`.

Config note:

The comment in `prediction_config.py` says the 2024Q4 market-cap universe "avoids survivorship bias from a hand-curated list." That is too strong. The design avoids manual cherry-picking, but it is still a retrospective fixed-universe design. The paper should call it:

```text
a reproducible fixed 2024Q4 large-cap universe
```

not an investable no-look-ahead universe.

Cache rebuild log reports:

```text
raw tensor shape: 50 firms x 40 features x 80 quarters
raw observed density: 90.32%
LEVELS L=2: 78 windows, avg recon 0.4043
LEVELS L=4: 76 windows, avg recon 0.3626
SURPRISE L=2: 78 windows, avg recon 0.4043
SURPRISE L=4: 76 windows, avg recon 0.3626
```

## Leakage Audit

### Split Logic

No direct test leakage was found in the rolling-window split:

- Windows are ordered chronologically.
- The test block is the last 20% of windows.
- TimeSeriesSplit is used inside the development block.
- CP and Ridge both train on development windows only before holdout scoring.

### Universe Selection Caveat

The universe is selected by market cap at `2024Q4`. That is retrospective. It is not leakage into the target values, but it is a survivorship/look-ahead design choice if the paper frames the experiment as historically investable.

Defensible framing:

> We evaluate predictability conditional on a fixed 2024Q4 large-cap universe.

Do not frame this as a live trading universe unless the universe is rebuilt using only information available at each forecast date.

### Imputation

No target leakage found in the Tucker imputation:

- Each input window uses only lagged quarters `t : t+L`.
- The target quarter `t+L` is not used in the imputer.
- Observed cells are preserved exactly.

### Fixed Effects

FE construction is mask-aware and uses training data in the current outer fold or development block. This is correct for CP and the main observed-label Ridge baseline.

### Scaling

CP target RMS scaling is computed from observed training residual cells only. That is correct.

Input-window RMS scaling in `SURPRISE` is computed per input window from observed input cells only. That is also non-leaky.

## Root Cause of CP Collapse

The old CP Optuna objective selected by pooled R2:

```text
score = R2(FE + CP residual)
```

But CP is trained on:

```text
Y_tr_cent = (Y_tr - FE_mean) * target_mask
```

TensorLy `CPRegressor` does not support a target mask. Missing target residuals are therefore set to zero and included in the training loss.

Under pooled R2, a strongly regularized or ineffective CP residual map can shrink toward zero. After adding FE back, the model becomes:

```text
prediction ~= FE_mean
```

Because FE-only already explains most pooled variance, this can be optimal under the legacy objective.

Observed evidence:

```csv
mode,L,completed_trials,at_FE_CV,share_at_FE
LEVELS,2,112,62,0.554
LEVELS,4,45,45,1.000
SURPRISE,2,55,46,0.836
SURPRISE,4,43,43,1.000
```

The exact FE-only CV values match legacy Optuna best values:

```csv
L,FE_CV_R2
2,0.7318296596109586
4,0.7336781355701287
```

The top L=2 legacy CP test result exactly matches FE-only test:

```csv
L,FE_test_R2,legacy_CP_test_R2
2,0.7253239306500501,0.7253239306500501
```

## Current Baseline Results

Current test-set results from the current `prediction_new` caches:

```csv
mode,L,FE_test_R2,strong_Ridge_test_R2,global_alpha_Ridge_test_R2,global_alpha,CP_matched_Ridge_test_R2
LEVELS,2,0.725324,0.766939,0.767397,1000,0.764757
LEVELS,4,0.726703,0.768934,0.767081,10000,0.767631
SURPRISE,2,0.725324,0.764074,0.763794,100,0.755351
SURPRISE,4,0.726703,0.766447,0.767114,100,0.757671
```

Interpretation:

- FE-only is strong.
- Strong observed-label Ridge adds about 3.9-4.2 R2 points over FE.
- Global-alpha Ridge is about as strong as per-feature Ridge, so Ridge's strength is not mainly from per-feature alpha tuning.
- CP-matched zero-filled Ridge remains strong, especially in `LEVELS`.
- Legacy CP does not beat any Ridge baseline; it does not beat FE under L=2.

## Full-Context Reassessment: What We Were Missing

### Why The Old Positive Prediction Results Are Not Paper-Ready

The old paper draft and old CP/Ridge scripts should be treated as historical context, not as final evidence. They were useful for discovering the research direction, but the current redo exists for concrete reasons:

- The original fundamentals extract missed the YTD cash-flow columns needed by the paper feature table. Several intended features were structurally empty or nearly empty in the old tensor.
- The live feature spec and paper spec diverged (`39` versus `40` features). The current v2 pipeline restores the 40-feature specification and uses the corrected v2 Compustat extract.
- The prediction cache builder previously replaced whole input windows with Tucker reconstructions. The current `prediction_new` builder preserves observed accounting cells exactly and uses Tucker only to fill missing lagged inputs.
- A prediction-window off-by-one bug was corrected before the current distributed search.
- Old scripts such as `compare_ens_flat_vs_ridge_L2_L4.py` and `Ensemble_Flat.py` load the old `Code for paper/tensor_cache/` files, including `tensor_L2_R40_20_2.pkl` and `tensor_L4_R40_20_4.pkl`. Those are not the current v2 prediction caches.
- `Paper_Draft/main.tex` still contains stale language claiming CP matches or exceeds Ridge across all configurations. That claim is invalid under the corrected pipeline until re-established.

This is why the paper should not cite old positive CP-versus-Ridge tables as final results. They can motivate the redesign, but all final prediction claims must come from `prediction_new`.

### CP Geometry: Why Standalone CP Is The Wrong Fight

Ridge has a structural advantage that is real, not a bug: it captures persistence cheaply. A per-feature Ridge model can put weight directly on a firm's own lagged features, especially the own-feature lag that dominates many accounting variables.

Standalone structured CP must learn this through a low-rank 5D coefficient tensor over:

```text
input firm x input feature x lag x output firm x output feature
```

The easy predictive map in fundamentals is close to diagonal:

```text
same firm, same feature, recent lag -> same firm, same feature, next quarter
```

That diagonal map is not naturally low CP rank across 50 firms and 40 features. A standalone CP model can spend most of its rank budget approximating persistence that Ridge gets almost for free. Under a pooled R2 objective, the optimizer then has a safe escape: shrink the CP residual toward zero and let FE dominate. This explains both observed failures:

- pooled-objective CP collapses to FE-only behavior,
- strong Ridge's apparent edge comes from modeling the diagonal persistence
  component directly after FE.

This does **not** mean CP has no value. It means standalone CP is being asked to solve the wrong first-order problem.

### Old Ensemble Code: A Hypothesis To Re-Test, Not Evidence

`compare_ens_flat_vs_ridge_L2_L4.py` and `Ensemble_Flat.py` contain the right conceptual move:

```text
1. Fit feature-wise Ridge on observed labels.
2. Compute Ridge residuals.
3. Fit CPRegressor to the residual tensor.
4. Predict FE + Ridge + gamma * CP_residual.
```

This is worth re-testing because the CP component no longer has to learn own-feature persistence from scratch; it only needs to capture residual cross-firm/cross-feature structure that Ridge misses. But this should be treated as a hypothesis, not as a known winning route. The old ensemble work does **not** provide paper-ready support:

- The old pipeline used suboptimal pre-v2 features, including the cash-flow-column problem that motivated the redo.
- The old caches were built before the observed-cell-preserving Tucker correction.
- The old script loads `Code for paper/tensor_cache/`, not `Code for paper/prediction_new/tensor_cache/`.
- The old ensemble reportedly did not perform well, so there is no reason to assume the residual booster will rescue CP.

A paper-ready re-test must be ported to `prediction_new` and must use:

- v2 40-feature caches,
- current `LEVELS` and `SURPRISE` construction,
- current observed-preserving Tucker input imputation,
- leakage-safe inner selection of Ridge and CP hyperparameters,
- a CV objective equal to ensemble improvement over Ridge,
- one untouched chronological holdout evaluation.

The stopping rule should be strict: if development CV does not show a positive,
stable `delta_over_ridge`, do not spend the holdout on that architecture. Move
to the pre-specified conditional diagnostics instead of treating the result as a
global CP rejection.

The final question should be:

```text
Does CP residual augmentation improve FE+Ridge out-of-sample?
```

not:

```text
Can standalone CP replace Ridge?
```

## Paper-Preserving Enhancement Strategy After Reading `main.tex`

The full paper's center of gravity is:

1. construction of a 40-feature fundamentals representation,
2. tensor smoothing and MFI construction,
3. statistical dependence between MFI and FCIX,
4. prediction as an out-of-sample validation of whether the tensor structure in
   fundamentals carries forecasting information.

The prediction section should therefore be rewritten around **incremental tensor
structure under stringent controls**, not around a winner-take-all CP/Ridge
contest. The current `main.tex` result table is too strong because it reports
old positive CP deltas from the stale pipeline. The replacement should keep the
same ambition but use corrected evidence:

```text
Firm-feature fixed effects explain persistent levels.
Ridge controls for direct lag persistence.
CP is tested as low-rank tensor residual structure.
```

Recommended paper table structure:

```text
FE-only
strong observed-label Ridge
global-alpha Ridge
CP-matched zero-filled Ridge
standalone CP selected by residual_delta
calibrated standalone CP selected by residual_delta_v2, if run
FE + Ridge + CP residual, if run
```

The key paper question becomes:

```text
After accounting for fixed effects and direct persistence, does a low-rank
multilinear tensor component explain remaining predictable fundamentals?
```

That question is publishable even if the answer is conditional. A conditional CP
result can still enhance the paper if the regimes are defined before holdout and
are economically meaningful:

- lower AR(1) persistence features,
- low FE-only R2 features,
- features with high cross-feature residual correlation,
- restored cash-flow flow variables versus balance-sheet stock variables,
- cases where CP beats the exact CP-matched zero-filled Ridge baseline.

Do not write the revised section as "Ridge dominates CP." Write it as a
calibrated test of tensor residual information. Ridge is the control that makes
any CP gain credible.

## Existing L=2 Legacy CP Test Diagnostic

File:

`Code for paper/prediction_new/results/l2_top3_test_vs_ridge_20260430_111636_l2_top3_test.csv`

Results:

```csv
mode,L,rank_order,trial_number,cv_r2,RANK_REGRESS,REG_W,USE_RMS_SCALING,ridge_test_r2,cp_test_r2,delta_cp_minus_ridge
LEVELS,2,1,0,0.7318296596109586,8,98.1655,True,0.7669390648137895,0.7253239306500501,-0.04161513416373941
LEVELS,2,2,1,0.7318296596109586,33,403.3800832600389,True,0.7669390648137895,0.7253239306500501,-0.04161513416373941
LEVELS,2,3,3,0.7318296596109586,33,403.3800832600389,True,0.7669390648137895,0.7253239306500501,-0.04161513416373941
SURPRISE,2,1,0,0.7318296596109586,43,52.4373,True,0.7640741637791535,0.7253239306500501,-0.03875023312910342
SURPRISE,2,2,1,0.7318296596109586,33,403.3800832600389,True,0.7640741637791535,0.7253239306500501,-0.03875023312910342
SURPRISE,2,3,3,0.7318296596109586,33,403.3800832600389,True,0.7640741637791535,0.7253239306500501,-0.03875023312910342
```

Interpretation:

- Different CP hyperparameters yield identical CP test R2.
- That identical value is FE-only test R2.
- Legacy pooled CV selected residual collapse, not incremental CP signal.

## Residual-Delta Objective

`worker.py` now supports:

```text
--objective residual_delta
```

The residual-delta fold score is:

```text
R2(FE + CP residual) - R2(FE-only)
```

This is aligned with the real research question: does CP add predictive value beyond fixed effects?

Current status from the first audit snapshot:

```csv
study,completed,running
cp_pred_levels_L2_residual_delta,0,21
cp_pred_levels_L4_residual_delta,0,63
cp_pred_surprise_L2_residual_delta,0,21
cp_pred_surprise_L4_residual_delta,0,63
```

Later read-only monitor snapshot:

```csv
study,total_trials,completed,running,best_delta,best_rank,best_reg_w,best_rms
cp_pred_levels_L2_residual_delta,22,1,21,-0.45136512634752374,7,0.00019499601303143508,True
cp_pred_levels_L4_residual_delta,63,0,63,NA,NA,NA,NA
cp_pred_surprise_L2_residual_delta,21,0,21,NA,NA,NA,NA
cp_pred_surprise_L4_residual_delta,63,0,63,NA,NA,NA,NA
```

Interpretation:

- One completed trial is not enough to judge the residual-delta objective.
- The completed `LEVELS L=2` trial is strongly negative, which means the current
  broad search space includes CP fits that are much worse than FE-only.
- Do not launch a second architecture yet. First let the current residual-delta
  studies either produce enough completed trials for top-k evaluation or reach
  the time budget.

Minimum useful checkpoint:

```text
At least 20 completed residual-delta trials per mode/L,
or the 24-hour budget expires,
then evaluate top-k residual_delta trials on holdout.
```

## Important Fairness Issue Still Remaining

The CP-matched zero-filled Ridge baseline is useful, but its inner alpha CV currently computes zero-filled residuals using firm means from the full outer-training block. That is not test leakage, but it is inner-CV leakage for alpha selection.

Patch before using CP-matched Ridge as reviewer-facing evidence:

- For each inner fold, compute firm-feature means only on inner-training rows.
- Build zero-filled residuals from those inner-training means.
- Score only observed validation targets.
- For the final fit, use full outer-training means.

This mirrors the leakage-safe pattern already used in the strong observed-label Ridge baseline.

## What Is Out Of Scope (This Paper)

- Building a new masked-CP estimator from scratch (custom weighted loss over
  observed target cells only) is explicitly out of scope for this draft cycle.
- Adding major framework dependencies solely to implement masked CP is out of scope.
- Re-architecting the full pipeline around a new estimator class is out of scope.

This should be documented as future work, not silently attempted mid-paper.

## How To Enhance CP Fairly

The right goal is to remove avoidable disadvantages in CP while keeping Ridge
strong enough to make a positive tensor result credible. CP should be given the
task that matches its structure: explaining shared residual variation across
firms, features, and lags after simpler fixed-effect and persistence effects are
controlled.

### 1. Time-Box The FE + Ridge + CP Residual Re-Test

The main CP re-test should be a Ridge-residual booster, rebuilt from the idea in:

`Code for paper/compare_ens_flat_vs_ridge_L2_L4.py`

Target model:

```text
Y_hat = FE + Ridge(X) + gamma * CP_residual(X)
```

Development objective:

```text
delta_over_ridge = R2(FE + Ridge + gamma * CP_residual) - R2(FE + Ridge)
```

This is the most defensible way to make CP competitive and paper-relevant
because:

- Ridge handles the diagonal persistence component.
- CP is asked to model residual cross-firm/cross-feature/lags structure.
- `gamma = 0` is the nested Ridge-only model, so CP must earn positive weight.
- The holdout comparison remains against strong Ridge, not a weakened baseline.

But this is **not** a guaranteed CP win. The old ensemble reportedly did not perform well, and the corrected v2 feature set may make Ridge even harder to beat if the newly restored cash-flow features are persistent. The reason to re-test the booster is methodological cleanliness, not optimism about the old result.

Recommended search space:

```text
RIDGE_ALPHA: selected inside fold or chosen from the same grid as strong Ridge
RANK_REGRESS: 1..20
REG_W: 1e-5..1e5
GAMMA: 0..2
USE_RMS_SCALING: True/False
residual feature scaling: on/off, selected by CV
```

A conservative one-SE rule should be used for the paper table: among models within one standard error of the best development delta, choose the smallest CP rank, then the larger `REG_W`, then the smaller `gamma`.

Pre-specified go/no-go:

- If mean development `delta_over_ridge <= 0`, stop that architecture and move
  to the pre-specified CP diagnostics/regime tests.
- If mean development delta is positive but fold signs are unstable, treat the result as diagnostic only.
- If development delta is positive and stable, evaluate once on holdout and report the result regardless of sign.

### 2. Finish Residual-Delta Standalone CP As A Diagnostic

Do not evaluate more legacy pooled-R2-selected CP models except as failure-mode evidence.

Use residual-delta-selected trials:

```bash
cd "/student/mcnama53/Projects/Tensor Research"

"/student/mcnama53/.local/share/mamba/envs/research/bin/python" \
  "Code for paper/prediction_new/worker.py" \
  --mode LEVELS --L 2 --objective residual_delta --n-trials 100 --time-budget-s 86400
```

Repeat for:

- `LEVELS L=2`
- `LEVELS L=4`
- `SURPRISE L=2`
- `SURPRISE L=4`

Interpretation:

- If standalone residual-delta CP improves over FE but not Ridge, it is useful diagnostics but not the main paper win.
- If standalone CP also beats Ridge, report it, but still show the Ridge-residual booster because it answers the cleaner incremental-structure question.
- If standalone CP fails, that is not the end of CP; it confirms the geometric diagnosis above.

### 3. Evaluate Residual-Delta-Selected CP On Holdout

`evaluate_l2_top3_test.py` should be generalized:

- Accept `--objective residual_delta`.
- Accept `--lookbacks 2,4`.
- Load objective-specific journals via `worker.objective_journal_path()` and `worker.objective_study_name()`.
- Report FE, strong Ridge, global-alpha Ridge, CP-matched Ridge, and CP in the same table.

Required columns:

```text
mode
L
objective
rank_order
trial_number
cv_residual_delta
RANK_REGRESS
REG_W
USE_RMS_SCALING
fe_test_r2
cp_test_r2
cp_residual_delta_test
ridge_per_feature_test_r2
ridge_global_alpha_test_r2
ridge_global_alpha
ridge_cp_matched_test_r2
delta_cp_minus_per_feature_ridge
delta_cp_minus_global_alpha_ridge
delta_cp_minus_cp_matched_ridge
```

### 4. Do Not Chase Target-Mask-Aware CP In This Paper

TensorLy `CPRegressor` is not target-mask-aware. That is a limitation, but it should not become the main paper project.

For this draft cycle:

- Keep target evaluation mask-aware on observed cells.
- Keep CP training exactly reproducible and transparent.
- Keep CP-matched zero-filled Ridge as a fairness diagnostic.
- State the limitation clearly.
- Do not implement a custom masked CP estimator unless the paper timeline changes.

This avoids turning the empirical paper into an unvalidated estimator paper.

### 5. Optimize Incremental Residual Value, Not Pooled R2

Because FE explains most pooled variance, pooled R2 is insensitive to residual quality.

Better objectives:

- `residual_delta`: current best immediate objective.
- `ridge_residual_delta`: recommended objective for the Ridge-residual booster.
- Observed residual MSE improvement:
  ```text
  1 - SSE_residual_CP / SSE_residual_zero
  ```
- Weighted per-feature residual delta:
  ```text
  mean_j [R2_j(FE + CP) - R2_j(FE)]
  ```
  with a minimum observation threshold.
- Winsorized or median per-feature residual delta to avoid one high-variance feature dominating.

Recommended next objective:

```text
mean over features of residual_delta_j, weighted by sqrt(n_observed_j), clipped below at -0.25
```

Reason:

- Keeps the objective incremental.
- Prevents pooled high-variance features from dominating.
- Still rewards broad feature-level improvement.

For the Ridge-residual booster, the analogous feature-weighted objective is:

```text
mean_j [R2_j(FE + Ridge + CP) - R2_j(FE + Ridge)]
```

with feature weights and clipping fixed before holdout evaluation.

### 6. Pre-Specify Regime Tests

If CP wins only conditionally, make that the claim.

Potential pre-specified regimes:

- Low FE-only R2 features.
- Low Ridge residual predictability features.
- Features with high cross-feature residual correlation.
- Low or medium AR(1) persistence features, if justified before test evaluation.
- Cash-flow features versus balance-sheet stock variables.

Do not pick these after seeing holdout CP wins. Use development data only to define strata, then evaluate once on holdout.

### 7. Strengthen CP Search Space

Current search:

```text
RANK_REGRESS: 5..80
REG_W: 1e-5..1e3
USE_RMS_SCALING: True/False
```

Potential additions:

- Smaller ranks: include 1, 2, 3, 4. Residual signal may be very low-rank.
- Larger regularization: allow `1e4` or `1e5`.
- Residual target standardization by feature, not only pooled RMS.
- Feature-weighted target loss.
- Multiple CP random restarts per trial or fixed ensemble seeds.

Important: these should be selected by development CV only.

### 8. Try CP Ensembles

A top-k residual ensemble can be defensible if selected by CV and reported alongside the parsimonious one-SE model:

```text
standalone: prediction = FE + average CP residual prediction from top-k residual-delta trials
booster:    prediction = FE + Ridge + average gamma_k * CP_residual_k from top-k ridge-residual trials
```

Report:

- best single model
- one-SE parsimonious model
- top-3 ensemble
- top-5 ensemble

Use the same holdout once.

### 9. Compare Against Ridge Honestly

Keep both:

- Strong Ridge:
  - observed-label only
  - per-feature alpha
  - no target-zero-fill
- Matched Ridge:
  - same zero-filled residual target as current TensorLy CP
  - same FE restoration

For the booster experiment, the essential baseline is the exact nested Ridge-only model produced by setting `gamma = 0`.

## Recommended Paper Completion Plan

1. Let the current residual-delta standalone CP studies finish or hit the 24-hour budget.
2. Before final tables, make `prediction_new` self-contained by moving active Ridge/FE helpers into `prediction_new/baselines.py`.
3. Generalize `evaluate_l2_top3_test.py` into an all-lookback residual-delta evaluator.
4. Evaluate residual-delta-selected standalone CP on holdout for all completed mode/lookback cells.
5. If standalone residual-delta CP is weak, run a small calibrated `residual_delta_v2` study with `gamma` and feature residual scaling.
6. Only after that, time-box a `prediction_new` Ridge-residual CP re-test of the old flat-ensemble idea.
7. Produce one table with FE, Ridge, standalone CP, calibrated standalone CP if run, Ridge+CP residual if tested, and deltas over Ridge.
8. Produce per-feature deltas and AR(1) persistence diagnostics using pre-specified strata.
9. Rewrite `main.tex` around the corrected enhancement story: CP is a
   low-rank tensor residual layer tested under strong persistence controls, with
   the final claim stated as global, nested-baseline, or pre-specified
   conditional value depending on the corrected results.

Do not let the prediction table consume the paper's identity. The core
contribution is the tensor fundamentals/MFI construction and its FCIX dependence
result. The prediction section should strengthen that contribution by showing
where tensor structure survives strong persistence controls. If the cleanest CP
evidence is conditional rather than global, the paper should say so and use the
conditional result to explain what kind of fundamentals contain tensor residual
signal.

## Publication Claim Guidance

Current defensible framing:

> The initial pooled-R2 CP selection was invalid for the incremental tensor
> prediction question because it selected fixed-effect-only behavior. The
> corrected pipeline treats Ridge as a persistence control and tests whether
> low-rank CP structure explains residual cross-firm/cross-feature variation
> beyond fixed effects and direct lag persistence.

Preferred global claim if calibrated standalone CP succeeds:

> When selected on an incremental residual objective, low-rank CP regression
> improves out-of-sample prediction of quarterly fundamentals beyond
> firm-feature fixed effects and the strong Ridge persistence benchmark. The
> gain is robust to matched zero-filled Ridge controls, parsimonious one-SE
> selection, and per-feature diagnostics.

Preferred claim if the Ridge-residual booster succeeds:

> CP does not need to replace Ridge as the persistence model. Instead, a
> low-rank tensor residual layer adds incremental out-of-sample value after
> Ridge removes firm-feature persistence, indicating that the remaining
> predictable component has cross-firm/cross-feature structure.

Preferred conditional claim if CP is not globally positive:

> In the corrected v2 prediction pipeline, direct persistence controls explain a
> large share of pooled one-quarter fundamentals predictability. Tensor CP adds
> value in pre-specified regimes where persistence is weaker or residual
> cross-feature dependence is stronger, supporting the view that tensor
> structure is most consequential outside the near-diagonal persistence
> component.

Fallback language if no CP variant produces a stable positive result:

> The prediction experiment provides a stringent boundary condition on
> one-quarter forecasting in a highly persistent fundamentals panel. The result
> does not weaken the paper's main MFI contribution; instead, it clarifies that
> the MFI/FCIX dependence result captures market-wide fundamentals variation
> that is not reducible to a simple global CP forecasting gain over direct
> persistence controls.

Claims to avoid:

- `CP beats Ridge globally`, unless the corrected holdout results support it.
- `Ridge dominates CP`, as a paper-level conclusion. Ridge is a control, not the
  research contribution.

The revised text should preserve the paper's constructive thesis: tensor
fundamentals matter, and prediction is the stress test that determines where
low-rank CP structure adds forecastable information.

## `main.tex` Rewrite Warning

`Paper_Draft/main.tex` currently contains stale language claiming CP beats Ridge across all configurations. Specifically, the section around the old CP/Ridge results says CP “matches or exceeds” Ridge and reports positive deltas. That should be replaced.

Replacement direction:

- Keep the tensor construction and Tucker imputation description, with minor updates for current ranks and density.
- Rewrite the CP result section as an audit-driven enhancement of the original
  prediction experiment.
- Present the legacy pooled objective as a failed selection criterion.
- Present FE and Ridge results as current baselines.
- State that residual-delta CP evaluation is ongoing or report it only after
  completed residual-delta trials are tested.
- Add the Ridge+CP residual experiment if it is run, because it is the most
  paper-aligned CP design: Ridge removes direct persistence, CP tests remaining
  tensor structure.
- Keep the conclusion language constructive: tensor methods expose and summarize
  market fundamentals, while prediction identifies where that structure is
  forecastable at the firm-feature level.

Suggested paragraph:

> In the first CP selection pass, hyperparameters were chosen by pooled out-of-sample R2. This criterion proved inappropriate for the incremental prediction question because the firm-feature fixed effects explain most pooled variation. Several CP configurations selected by the pooled objective produced predictions numerically indistinguishable from the FE-only baseline, yielding identical cross-validation and holdout scores despite different ranks and regularization weights. We therefore treat the pooled-objective CP results as a diagnostic failure rather than evidence of tensor predictive value. The revised selection criterion is the residual improvement over FE-only, evaluated on observed target cells only.

Suggested enhancement paragraph if Ridge+CP residual is added:

> We use ridge regression as a stringent persistence control rather than as the
> object of the paper. Accounting fundamentals contain strong own-firm,
> own-feature persistence, which a per-feature ridge model captures directly.
> The tensor question is whether the remaining forecastable component has
> multilinear structure across firms, features, and lags. We therefore evaluate a
> residual tensor augmentation in which CP is fitted to the errors left by the
> fixed-effect ridge model and receives positive weight only if it improves
> development performance under time-series cross-validation.

## Proposed Patch Set

Do not apply blindly. These are the code changes that are justified by the audit and by the current `prediction_new` state. The immediate highest-impact change is **evaluation and baseline cleanup**, not a new model architecture. The Ridge-residual CP experiment remains the best second-stage attempt to make CP beat Ridge, but it should not preempt the live residual-delta studies.

### Priority Order

1. Finish or time out the currently running standalone `residual_delta` studies.
2. Move reusable Ridge/FE helpers into `prediction_new/baselines.py` so final runs do not depend on stale old-script imports.
3. Generalize top-trial evaluation to all objectives and both lookbacks.
4. Evaluate residual-delta-selected standalone CP on holdout before launching a new architecture.
5. Keep `prediction_config.py` as the source of truth and add objective-specific constants for future studies without changing active running-study constants.
6. If residual-delta CP is weak, launch a smaller calibrated standalone `residual_delta_v2` study with `gamma` and feature residual scaling.
7. Only after that, time-box a Ridge-residual CP worker/evaluator under `prediction_new`.
8. Fix feature names and summary outputs using `meta_path()` or `FEATURE_SPECS`.

### Patch -1: Tighten `prediction_config.py`

File:

`Code for paper/prediction_new/prediction_config.py`

Do not disrupt the running residual-delta studies by changing `RANK_RANGE` midstream unless the current jobs are stopped and relaunched intentionally. Instead, add new objective-specific constants for the next studies.

Recommended additions:

```python
# Standalone CP studies already launched with RANK_RANGE/REG_W_RANGE.
# Keep these for backward compatibility with existing journals.
STANDALONE_CP_RANK_RANGE: tuple[int, int] = RANK_RANGE
STANDALONE_CP_REG_W_RANGE: tuple[float, float] = REG_W_RANGE

# Calibrated standalone residual CP search, only if current residual_delta is weak.
RESIDUAL_DELTA_V2_RANK_RANGE: tuple[int, int] = (1, 20)
RESIDUAL_DELTA_V2_REG_W_RANGE: tuple[float, float] = (1e-2, 1e5)
RESIDUAL_DELTA_V2_GAMMA_RANGE: tuple[float, float] = (0.0, 2.0)

# Ridge-residual CP booster search, phase 2.
RIDGE_RESIDUAL_CP_RANK_RANGE: tuple[int, int] = (1, 20)
RIDGE_RESIDUAL_REG_W_RANGE: tuple[float, float] = (1e-5, 1e5)
RIDGE_RESIDUAL_GAMMA_RANGE: tuple[float, float] = (0.0, 2.0)
RIDGE_RESIDUAL_ALPHA_GRID: tuple[float, ...] = (100.0, 1000.0, 10000.0)

# Shared Ridge grids for reporting.
RIDGE_ALPHA_GRID: tuple[float, ...] = (
    1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4
)
```

Add path/study helpers for the Ridge-residual booster:

```python
def ridge_residual_journal_path(mode: str, L: int) -> Path:
    base = journal_path(mode, L)
    return base.with_name(f"{base.stem}_ridge_residual_delta{base.suffix}")


def ridge_residual_study_name(mode: str, L: int) -> str:
    return f"{study_name(mode, L)}_ridge_residual_delta"
```

Also add optional helpers for calibrated standalone residual CP if that study is launched:

```python
def residual_delta_v2_journal_path(mode: str, L: int) -> Path:
    base = journal_path(mode, L)
    return base.with_name(f"{base.stem}_residual_delta_v2{base.suffix}")


def residual_delta_v2_study_name(mode: str, L: int) -> str:
    return f"{study_name(mode, L)}_residual_delta_v2"
```

Soften the universe-selection comment:

```python
# Universe is selected programmatically from v2 fundamentals: top-N firms by
# market cap (mkvaltq) at the reference quarter. This creates a reproducible
# fixed large-cap universe for the prediction exercise. It is not a live
# investable no-look-ahead universe.
```

Do not use `LEGACY_WARM_START` for residual-delta, residual-delta-v2, or Ridge-residual studies. Those values came from the old pooled-R2 behavior and can anchor the search toward FE-collapse regimes. If warm starts are used at all, label them explicitly as pooled-objective diagnostics.

### Patch 0: Create Shared Prediction Helpers

New file:

`Code for paper/prediction_new/baselines.py`

Reason:

- The active code path is `Code for paper/prediction_new/`.
- Final prediction code should not import old scripts such as `CP_struct_test_new.py` that live outside `prediction_new`.
- The Ridge and FE implementations should be shared by the worker, evaluator, and summarizer.

Current state:

```python
# prediction_new/evaluate_l2_top3_test.py
from CP_struct_test_new import (
    _within_firm_means_y,
    evaluate_model,
    firm_feature_means,
    get_min_obs_per_feat,
    ridge_structured_cp_matched_zero_filled_ts_cv,
    ridge_structured_fixed_effects_ts_cv,
)
```

`prediction_new/imputer_sensitivity.py` has the same kind of dependency. This does not mean the final pipeline should run `CP_struct_test_new.py`; it means those helper functions need to be moved into the active package.

Implement these functions directly in `prediction_new/baselines.py` using the current `prediction_new` cache conventions:

```text
get_min_valid_entries
get_min_obs_per_feat
evaluate_model
evaluate_model_per_feature
firm_feature_means
_within_firm_means_y
ridge_structured_fixed_effects_ts_cv
ridge_structured_cp_matched_zero_filled_ts_cv
ar1_persistence_per_feature
```

Add these new helpers:

```python
def flatten_ridge_design(X_4d: np.ndarray) -> np.ndarray:
    n_t, n_f, n_feat, n_l = X_4d.shape
    return X_4d.transpose(0, 1, 3, 2).reshape(n_t * n_f, n_feat * n_l)


def fit_predict_fe_ridge(
    X_tr_4d: np.ndarray,
    Y_tr_3d: np.ndarray,
    M_tr_3d: np.ndarray,
    X_pred_4d: np.ndarray,
    alpha: float,
    per_feature_alpha: bool = False,
) -> np.ndarray:
    """Fit fixed-effect Ridge on observed labels and predict X_pred."""
```

The `fit_predict_fe_ridge` helper should match the current strong Ridge convention:

- compute firm-feature means using training rows only,
- fit each target feature only on observed labels,
- add firm-feature means back at prediction,
- use `alpha` globally unless `per_feature_alpha=True` is explicitly requested.

For the first Ridge-residual booster, prefer a simple train-residual helper over mandatory cross-fitting:

```python
def fit_predict_fe_ridge_train_and_pred(
    X_tr_4d: np.ndarray,
    Y_tr_3d: np.ndarray,
    M_tr_3d: np.ndarray,
    X_pred_4d: np.ndarray,
    alpha: float,
    per_feature_alpha: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit FE+Ridge once and return predictions for training rows and X_pred.

    The training-row prediction is used to construct CP residual targets:
    residual_train = (Y_tr - ridge_train_pred) * M_tr
    """
```

Reason:

- The outer validation fold already tests whether CP residuals generalize.
- Mandatory residual cross-fitting would reduce usable training signal and slow the first booster experiment.
- Cross-fitted residuals can be a robustness check after a simple booster shows positive development value.

Optional robustness helper:

```python
def crossfit_ridge_residuals(...) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use time-series cross-fitting inside the outer training block."""
```

### Patch 1: Add Calibrated Standalone `residual_delta_v2` Worker If Needed

Do this only after evaluating the current residual-delta studies. The monitor already showed one strongly negative completed `LEVELS L=2` trial, so if the broader current search remains weak, the first corrective experiment should be a smaller calibrated standalone CP search before the Ridge-residual architecture.

Implementation options:

- Extend `worker.py` with a new objective name `residual_delta_v2`, or
- create `Code for paper/prediction_new/residual_delta_v2_worker.py`.

Recommended model:

```text
Y_hat = FE + gamma * CP_residual
```

Recommended trial parameters:

```python
rank_reg = trial.suggest_int("RANK_REGRESS", 1, 20)
reg_w = trial.suggest_float("REG_W", 1e-2, 1e5, log=True)
gamma = trial.suggest_float("GAMMA", 0.0, 2.0)
use_rms = trial.suggest_categorical("USE_RMS_SCALING", [True, False])
feature_scale = trial.suggest_categorical("FEATURE_TARGET_SCALE", [True, False])
```

Objective:

```text
R2(FE + gamma * CP_residual) - R2(FE)
```

Why this comes before Ridge-residual CP:

- It directly addresses the first completed residual-delta failure mode: unstable, under-regularized CP residuals.
- `gamma=0` lets the search choose the FE-only null model instead of forcing harmful residual predictions.
- Feature target scaling prevents dense/high-variance features from dominating the residual objective.
- It is much smaller than a new two-stage Ridge-residual architecture.

### Patch 2: Add Ridge-Residual CP Worker As Phase 2

New file:

`Code for paper/prediction_new/ridge_residual_worker.py`

Do **not** launch this before the current residual-delta studies are evaluated. A separate worker keeps objective semantics and Optuna journals clean.

Objective:

```text
delta_over_ridge = R2(FE + Ridge + gamma * CP_residual) - R2(FE + Ridge)
```

Study naming:

```python
def ridge_residual_journal_path(mode: str, L: int) -> Path:
    base = journal_path(mode, L)
    return base.with_name(f"{base.stem}_ridge_residual_delta{base.suffix}")


def ridge_residual_study_name(mode: str, L: int) -> str:
    return f"{study_name(mode, L)}_ridge_residual_delta"
```

Suggested trial parameters:

```python
ridge_alpha = trial.suggest_categorical("RIDGE_ALPHA", [100.0, 1000.0, 10000.0])
rank_reg = trial.suggest_int("RANK_REGRESS", 1, 20)
reg_w = trial.suggest_float("REG_W", 1e-5, 1e5, log=True)
gamma = trial.suggest_float("GAMMA", 0.0, 2.0)
use_rms = trial.suggest_categorical("USE_RMS_SCALING", [True, False])
feature_scale = trial.suggest_categorical("FEATURE_RESID_SCALE", [True, False])
```

Outer-fold logic:

```python
for tr_idx, va_idx in TimeSeriesSplit(n_splits=3).split(X_dev):
    X_tr, Y_tr, M_tr = X_dev[tr_idx], Y_dev[tr_idx], M_dev[tr_idx]
    X_va, Y_va, M_va = X_dev[va_idx], Y_dev[va_idx], M_dev[va_idx]

    ridge_tr, ridge_va = fit_predict_fe_ridge_train_and_pred(
        X_tr, Y_tr, M_tr, X_va, alpha=ridge_alpha
    )
    ridge_r2 = evaluate_model(Y_va, ridge_va, M_va)

    X_cp = X_tr
    resid_cp = (Y_tr - ridge_tr) * M_tr
    M_cp = M_tr

    if feature_scale:
        feat_sse = np.sum((resid_cp ** 2) * M_cp, axis=(0, 1))
        feat_n = np.sum(M_cp, axis=(0, 1))
        feat_scale = np.sqrt(feat_sse / (feat_n + 1e-8))
        feat_scale = np.where(np.isfinite(feat_scale) & (feat_scale > 1e-8), feat_scale, 1.0)
    else:
        feat_scale = np.ones(Y_tr.shape[2])

    target = resid_cp / feat_scale[None, None, :]

    if use_rms:
        obs = target[M_cp > 0]
        target_rms = np.sqrt(np.mean(obs ** 2))
        target = target / (target_rms + 1e-8)
    else:
        target_rms = 1.0

    cp = CPRegressor(
        weight_rank=rank_reg,
        reg_W=reg_w,
        n_iter_max=N_ITER_MAX,
        random_state=SEED,
    )
    cp.fit(X_cp, target * M_cp)

    cp_va = cp.predict(X_va) * target_rms * feat_scale[None, None, :]
    ens_va = ridge_va + gamma * cp_va

    ens_r2 = evaluate_model(Y_va, ens_va, M_va)
    score = ens_r2 - ridge_r2
```

Important details:

- `gamma=0` is the exact nested Ridge-only model.
- The worker should maximize `delta_over_ridge`, not pooled ensemble R2.
- Store `ridge_r2`, `ens_r2`, per-fold deltas, and fold-sign count with `trial.set_user_attr`.
- Prune if fewer than two outer folds produce finite scores.
- Do not spend holdout unless the completed study has positive and stable development deltas.
- Cross-fitted Ridge residuals are a robustness extension, not the default first booster.

Minimum stability rule:

```text
mean(delta_over_ridge) > 0
and at least 2 of 3 outer folds have delta_over_ridge > 0
```

Reviewer-safe one-SE rule:

```text
Among trials within one standard error of the best CV delta:
1. choose smallest RANK_REGRESS,
2. then largest REG_W,
3. then smallest GAMMA.
```

### Patch 3: Add Ridge-Residual Holdout Evaluator

New file:

`Code for paper/prediction_new/evaluate_ridge_residual_top_trials.py`

Purpose:

- load top completed `ridge_residual_delta` trials,
- fit final FE+Ridge on full development block,
- train CP residual model on full-development Ridge residuals,
- evaluate once on holdout,
- compare against exact nested Ridge-only and strong Ridge.

Required output:

`Code for paper/prediction_new/results/ridge_residual_top_trials_test.csv`

Required columns:

```text
mode
L
rank_order
trial_number
cv_delta_over_ridge
cv_delta_fold_mean
cv_delta_fold_min
cv_delta_positive_folds
RIDGE_ALPHA
RANK_REGRESS
REG_W
GAMMA
USE_RMS_SCALING
FEATURE_RESID_SCALE
fe_test_r2
nested_ridge_test_r2
strong_ridge_test_r2
ridge_cp_residual_test_r2
delta_over_nested_ridge_test
delta_over_strong_ridge_test
elapsed_seconds
evaluated_at
```

Interpretation:

- `delta_over_nested_ridge_test` tells whether CP helped relative to the Ridge model used inside the ensemble.
- `delta_over_strong_ridge_test` tells whether the final ensemble actually beats the strongest standalone Ridge baseline.
- Both must be reported. Do not hide the strong Ridge comparison.

### Patch 4: Generalize Existing Top-Trial Evaluator

File:

`Code for paper/prediction_new/evaluate_l2_top3_test.py`

Recommended replacement:

`Code for paper/prediction_new/evaluate_top_trials_test.py`

Changes:

- Add args:
  - `--objective pooled_r2|residual_delta`
  - `--lookbacks 2,4`
  - `--top-k 1,3,5`
  - `--modes LEVELS,SURPRISE`
- Replace direct `journal_path()` / `study_name()` calls with:
  - `objective_journal_path()`
  - `objective_study_name()`
- Generalize `load_split(mode)` to `load_split(mode, L)`.
- Include FE-only baseline and residual deltas.

Required columns:

```text
mode
L
objective
rank_order
trial_number
cv_score
RANK_REGRESS
REG_W
USE_RMS_SCALING
fe_test_r2
cp_test_r2
cp_residual_delta_test
ridge_per_feature_test_r2
ridge_global_alpha_test_r2
ridge_global_alpha
ridge_cp_matched_test_r2
delta_cp_minus_per_feature_ridge
delta_cp_minus_global_alpha_ridge
delta_cp_minus_cp_matched_ridge
```

### Patch 5: Implement CP-Matched Ridge Cleanly Inside `prediction_new`

File:

`Code for paper/prediction_new/baselines.py`

Function:

`ridge_structured_cp_matched_zero_filled_ts_cv`

Current issue:

- Alpha selection builds zero-filled residuals using firm means from the full outer-training block.
- This is not test leakage, but it is inner-CV leakage for selecting alpha.
- This should be fixed in the new `prediction_new` helper, not by continuing to patch `CP_struct_test_new.py`.

Patch inner alpha-selection loop so each inner fold:

```python
for tr_time_idx, va_time_idx in inner_tscv.split(np.arange(n_tr_t)):
    tr_rows = np.isin(time_ids_tr, tr_time_idx)
    va_rows = np.isin(time_ids_tr, va_time_idx)

    obs_inner = (mj[tr_rows] > 0)
    if obs_inner.sum() == 0:
        continue

    inner_global = float(yj[tr_rows][obs_inner].mean())
    inner_mean = _within_firm_means_y(
        y_tr=yj[tr_rows],
        m_tr=mj[tr_rows],
        firm_ids=firm_ids_tr[tr_rows],
        n_firms=n_f,
        fallback_global=inner_global,
    )

    y_zero_tr = (yj[tr_rows] - inner_mean[firm_ids_tr[tr_rows]]) * mj[tr_rows]

    rg = Ridge(alpha=float(alpha), fit_intercept=False, solver="auto", random_state=SEED)
    rg.fit(X_tr_2d[tr_rows], y_zero_tr)

    pred_va = rg.predict(X_tr_2d[va_rows]) + inner_mean[firm_ids_tr[va_rows]]
```

Score only observed validation cells:

```python
obs_va = mj[va_rows] > 0
y_true_va = yj[va_rows][obs_va]
y_pred_va = pred_va[obs_va]
```

Final fit can still use full outer-training means.

### Patch 6: Fix Feature Names In `prediction_new` Per-Feature Output

Current issue:

- Current prediction tensors have 40 features.
- Final per-feature output should use labels from `prediction_new/tensor_cache/meta.pkl`, not any stale hard-coded list.

Add `get_feature_names` to `prediction_new/baselines.py`:

```python
def get_feature_names(n_feat: int) -> list[str]:
    from prediction_config import meta_path

    meta_file = meta_path()
    if meta_file.exists():
        meta = joblib.load(meta_file)
        names = meta.get("feature_names")
        if isinstance(names, list) and len(names) == n_feat:
            return names
    return [f"feat_{i}" for i in range(n_feat)]
```

If import paths make that awkward from scripts, use an explicit local path under `prediction_new`:

```python
def get_feature_names(n_feat: int) -> list[str]:
    meta_path = Path(__file__).resolve().parent / "tensor_cache" / "meta.pkl"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        names = meta.get("feature_names")
        if isinstance(names, list) and len(names) == n_feat:
            return names
    return [f"feat_{i}" for i in range(n_feat)]
```

### Patch 7: Retire `CP_struct_test_new.py` From The Active Pipeline

Do not patch or run `Code for paper/CP_struct_test_new.py` for final paper results.

If a useful baseline routine currently exists only there, reimplement it in `prediction_new/baselines.py` and make all final scripts import from `prediction_new`.

Active final scripts should use:

```text
Code for paper/prediction_new/worker.py
Code for paper/prediction_new/ridge_residual_worker.py
Code for paper/prediction_new/evaluate_top_trials_test.py
Code for paper/prediction_new/evaluate_ridge_residual_top_trials.py
Code for paper/prediction_new/summarize_prediction_results.py
Code for paper/prediction_new/baselines.py
```

### Patch 8: Add Completed-Study Summarizer

New file:

`Code for paper/prediction_new/summarize_prediction_results.py`

Purpose:

- read completed Optuna journals,
- compute FE-only CV/test,
- compute Ridge baselines,
- compute standalone CP top-k holdout metrics,
- compute Ridge-residual CP top-k holdout metrics,
- produce final CSVs for paper tables.

Outputs:

```text
Code for paper/prediction_new/results/RESULTS_SUMMARY.csv
Code for paper/prediction_new/results/PER_FEATURE_DELTA.csv
Code for paper/prediction_new/results/PERSISTENCE_SUMMARY.csv
```

Required `RESULTS_SUMMARY.csv` rows:

```text
FE-only
strong per-feature Ridge
global-alpha Ridge
CP-matched zero-filled Ridge
standalone CP selected by residual_delta
Ridge + CP residual selected by ridge_residual_delta
```

Required result columns:

```text
mode
L
model
selection_objective
cv_score
test_r2
delta_over_fe
delta_over_strong_ridge
delta_over_cp_matched_ridge
rank
reg_w
gamma
ridge_alpha
use_rms_scaling
feature_resid_scale
notes
```

### Patch 9: Add Feature-Weighted Objectives Only After Core Results

Optional after the above:

```text
residual_delta_feature_mean
ridge_residual_delta_feature_mean
residual_mse_ratio
```

Feature-weighted standalone objective:

```text
mean_j [R2_j(FE + CP) - R2_j(FE)]
```

Feature-weighted booster objective:

```text
mean_j [R2_j(FE + Ridge + CP) - R2_j(FE + Ridge)]
```

Use:

```text
weights = sqrt(n_observed_j)
clip each feature delta below at -0.25
require minimum observations per feature
```

This reduces domination by high-variance/high-density features, but it should not be introduced before the basic pooled residual-delta results are reproducible.

## Exact Repro Commands

Rebuild caches:

```bash
cd "/student/mcnama53/Projects/Tensor Research"

"/student/mcnama53/.local/share/mamba/envs/research/bin/python" \
  "Code for paper/prediction_new/build_prediction_caches.py"
```

Monitor Optuna journals:

```bash
"/student/mcnama53/.local/share/mamba/envs/research/bin/python" \
  "Code for paper/prediction_new/monitor.py" \
  --objective residual_delta
```

Run a residual-delta worker:

```bash
"/student/mcnama53/.local/share/mamba/envs/research/bin/python" \
  "Code for paper/prediction_new/worker.py" \
  --mode LEVELS \
  --L 2 \
  --objective residual_delta \
  --n-trials 100 \
  --time-budget-s 86400
```

Legacy L2 top-k diagnostic:

```bash
"/student/mcnama53/.local/share/mamba/envs/research/bin/python" \
  "Code for paper/prediction_new/evaluate_l2_top3_test.py" \
  --top-k 3 \
  --modes LEVELS,SURPRISE
```

This legacy diagnostic should be replaced by `evaluate_top_trials_test.py` before final paper tables. Use it only for reproducing the known pooled-objective failure.

## Decision Tree After Residual-Delta Search Completes

1. If CP residual-delta is positive in CV but negative on holdout:
   - Report CP as unstable.
   - Do not claim global incremental value from that model.
   - Use the result to motivate calibrated CP or pre-specified regime analysis.
   - Do not pivot to masked CP in this paper.

2. If CP beats FE but not Ridge:
   - Claim CP has residual signal beyond fixed effects.
   - Treat Ridge as the persistence control that CP has not yet exceeded
     globally.
   - Analyze whether CP wins any pre-specified feature groups.

3. If CP beats CP-matched Ridge but not strong Ridge:
   - Claim CP helps under the same target-missingness convention.
   - This is a meaningful fairness win because both models share the zero-filled
     target convention.
   - Also report observed-label Ridge as the strongest persistence control.

4. If CP beats strong Ridge globally:
   - Report global pooled win, per-feature win count, Wilcoxon test, persistence/regime diagnostics, and one-SE/parsimony robustness.

5. If CP wins only in a pre-specified regime:
   - Claim conditional value only.
   - Explain why the regime is economically or statistically coherent.
   - Keep global baselines visible, but do not make them the paper's thesis.

6. If both standalone CP and Ridge-residual CP fail:
   - Do not write a paper-level "Ridge dominates CP" conclusion.
   - Use the prediction section as a rigorous boundary condition on short-horizon
     firm-feature forecasting.
   - Keep the MFI/FCIX dependence result and tensor fundamentals construction as
     the main contribution.

## Bottom Line For Next Researcher

The next useful work is not another pooled-R2 CP run. The next useful work is:

1. Finish or time out the live residual-delta studies.
2. Create `prediction_new/baselines.py` and remove imports from `CP_struct_test_new.py`.
3. Generalize the residual-delta holdout evaluator to both L=2 and L=4.
4. Evaluate residual-delta-selected standalone CP before launching a new architecture.
5. If weak, run `residual_delta_v2` with smaller ranks, stronger regularization, feature scaling, and `gamma`.
6. Only then time-box Ridge-residual CP.
7. If all CP variants are weak globally, preserve the paper by centering MFI and
   report prediction as a stringent short-horizon forecasting stress test, with
   any CP result stated at the supported global, nested-baseline, or conditional
   level.

The current paper text overclaims because it uses old positive prediction
numbers from a stale pipeline. The fix is not to downgrade the paper into a
Ridge result. The fix is to complete the corrected `prediction_new` evidence,
give CP the residual tensor task it is suited for, and then write the prediction
section as an enhancement of the tensor fundamentals story.
