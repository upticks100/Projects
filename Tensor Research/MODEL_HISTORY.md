# Model History: Every Architecture We Tried, What We Found, and Why We Changed It

This is the consolidated story behind the prediction section, from the original
January design through the final architecture reported in `main_v2.tex`. It
exists because the full story is scattered across `RESEARCH_LOG.md` (3000+
lines), `CP_RIDGE_HANDOFF.md`, and old committed scripts — this document is
the readable version.

**One-sentence version:** the original model's headline result (CP beats
Ridge by up to +0.028 R²) was a mirage caused by a broken selection
objective; fixing that properly took three rounds of audits over four months
and, once fixed, unlocked everything that makes the paper interesting now
(499 firms, the credit-risk result, the event study).

---

## Timeline at a glance

| When | Phase | Headline finding |
|---|---|---|
| Jan 25–26, 2026 | Original design | Pure CP + FE, N=49/F=24, pooled R², claims CP beats Ridge by +0.008 to +0.028 |
| Apr 26–29 | Data audit + rebuild | 12/24 features were 0%-dense; rebuilt on 40 clean features, 499 firms |
| Apr 30 | **Bug #1** | Pooled R² is degenerate — CP was winning by imitating FE exactly |
| May 2 | Audit (clean) | Checked for a duplicate-transform bug; none found |
| May 2–25 | **Bug #2 (subtler)** | `residual_delta` v1 search still gravitated to FE-imitation via high rank + heavy shrinkage; separately, a real Ridge-baseline leakage bug found & patched |
| May 25 | Fix | `residual_delta_v2`: explicit `GAMMA` floor + per-feature scaling |
| May 28 | Architecture split | Added parallel Ridge-booster track (`ridge_delta_v3`) alongside FE-residual track (`residual_delta_v3`) |
| Jun 19 | **Bug #3** | Booster's `+0.005` headline was partly an OOF-fallback artifact (fold 1 = exact zero) |
| Jun 20 | **First honest result** | v3 holdout: CP beats baseline in all 4 locked cells, for real |
| Jun 29–Jul 7 | Economic content | Event study (return null), IV subsumption, per-feature stationarity result, veer anomaly framework |
| Jul 6–7 | Scale-up | 499 firms (transfer confirmed), HY/crossover CDS universe (credit result confirmed 4/4) |
| Jul 7 | Write-up | `main_v2.tex` — full rewrite of prediction section onward |

---

## Phase 0 — Original design (Jan 25–26, 2026, pre-dates the research log)

**FE, defined once:** throughout this document and the paper, *firm--feature
fixed effects* (FE) means the mask-aware training-window mean \(\mu_{ij}\) for
each firm \(i\) and Compustat feature \(j\) — not firm-only dummies, not time
FE, and not a fitted LSDV panel regression. CP then fits residuals after that
mean (or after ridge, which itself includes the same FE).

**What we built:** `Pure_CP_Structured.py` — low-rank CP tensor regression
with firm-by-feature FE subtracted first, RMS normalization,
optionally combined with a per-feature ridge baseline for comparison.
Selection objective was **pooled R²** (just the overall $R^2$ of the fitted
model, no comparison built into the score itself).

- Universe: $N=49$ large-cap firms, $F=24$ Compustat fundamentals, 2005Q1–2024Q4.
- This is what `Paper_Draft/main.tex`'s prediction section still describes
  (its exhibit numbers were later bumped to N=50/F=40 without updating the
  architecture prose — one of the inconsistencies `main_v2.tex` fixes).
- Alongside `Pure_CP_Structured.py`, a handful of exploratory variants were
  also written but never became the paper's model: `Ensemble_Flat.py`,
  `Ensemble_Structured.py` (already experimenting with a Ridge+CP ensemble
  idea, years before it became the final architecture — but on stale
  features and, by all indications, not performing well), `Decomposition_firms.py`,
  and several `compare_*_vs_ridge*.py` scripts.
- `predictive.tex` (Jan 26) is a **placeholder draft** — its results tables
  are literally blank (`--`). No committed empirical numbers yet.

**What we found (claimed, later shown false):** by the time `main.tex` had
real numbers filled in (some point before April), it reported:

| Mode | $L$ | Ridge Test $R^2$ | CP Test $R^2$ | $\Delta$ |
|---|---|---|---|---|
| Unscaled | 2 | 0.760 | 0.778 | +0.018 |
| Unscaled | 4 | 0.758 | 0.766 | +0.008 |
| Normalized | 2 | 0.750 | 0.768 | +0.018 |
| Normalized | 4 | ~0.756 | 0.784 | +0.028 |

This is the table still sitting in `main.tex` today. **These numbers are the
mirage this whole saga is about** — see Phase 2.

---

## Phase 1 — Data quality audit and 40-feature rebuild (Apr 26–29)

Root-caused that **12 of the original 24 features were 0%-dense**: Compustat
only reports cash-flow line items on a fiscal-year-to-date (YTD) basis, and
the local extract never had the corresponding quarterly conversion, so those
columns were silently all-missing the entire time. This also explained why
the MFI's Tucker reconstruction error was stuck around 22%.

**Fix:** pulled a full `comp.fundq` snapshot (648 columns instead of 378),
finalized a clean 40-feature spec, converted YTD cash-flow fields to
quarterly via within-fiscal-year differencing, and rebuilt the panel/tensor.
Observed density jumped from ~50% to ~74–90%. This is also where the
499-firm universe (vs. the original 49/50) starts being tracked, though the
full 499-firm prediction rerun didn't happen until July (Phase 7).

Immediately after this, the *old* Optuna search (still using the pooled-R²
objective from Phase 0) was relaunched on the new 40-feature data at scale
across 16 lab machines — which is what surfaced the next problem.

---

## Phase 2 — Bug #1: the pooled-R² objective is degenerate (Apr 30)

**What we found:** the re-run's top CP trials had test $R^2$ that matched
the FE-only (no-CP) baseline to 6 decimal places. That's not a coincidence —
it's proof the "winning" CP models were contributing nothing.

**Why it happened:** quarterly fundamentals are extremely persistent — a
firm's debt level next quarter looks like this quarter's. So FE alone (each
firm-feature's own historical mean) already explains most of the variance.
Under plain pooled R² (the overall fit of `FE + CP`), **CP could score
exactly as well by shrinking its own contribution toward zero** as by trying
to add real signal — and a half-baked attempt at the latter could only
*hurt* the score by adding noise. The objective never required CP to earn
anything; "do nothing safely" was the rational, pooled-R²-optimal move. This
means the Phase 0 results table above (CP beating Ridge by up to +0.028)
was, at best, not actually testing what it claimed to test, and at worst
was measuring noise in how close CP's near-zero contribution happened to
land relative to Ridge on that particular split.

**Fix:** new objective, `residual_delta`:
$$\Delta = R^2(\text{FE} + \text{CP residual}) - R^2(\text{FE-only})$$
CP now only scores points for improvement *beyond* the trivial baseline.
Direction: maximize. Relaunched the distributed search under this objective
across 21 hosts, 168 workers, 24h budget.

---

## Phase 3 — Bug #2: the search still finds a way to hide (May 2–25)

**May 2:** a separate concern was raised — could residual optimization +
imputation + the "SURPRISE" (RMS-normalized) input mode be silently
double-applying a transform? Audited the full data flow end to end;
**came back clean, no bug found.**

**May 2–25 (~7 days of unattended distributed search):** `residual_delta`
v1 finally converged, but only to **tiny** improvements (+0.002 to +0.005
$R^2$) over FE-only. Digging into *why* the improvement was so small:
Optuna kept selecting **high CP rank combined with heavy regularization**,
which shrinks the CP factor matrices toward zero — a subtler recurrence of
the exact same failure mode as Phase 2 (CP imitating FE), this time via the
search space finding a loophole rather than the objective itself being
broken. CP wasn't being dishonest about its score anymore, but the search
still wasn't finding much of a reason to try.

**Also found May 25 (real bug, unrelated to the above):** the CP-matched
Ridge baseline (`ridge_structured_cp_matched_zero_filled_ts_cv` in
`CP_struct_test_new.py`) computed firm-feature means for its **inner-CV
alpha selection** using the *entire* outer-training block — meaning
inner-validation rows leaked into the means used to pick their own
validation target. Magnitude was likely small (Ridge variants all sit
within ~0.001 of each other), but it was a genuine correctness issue.
Patched to use inner-training-only means.

**Fix for the search-space problem:** `residual_delta_v2` — added an
explicit `GAMMA` scaling parameter on the CP residual, searched over
$[0, 2]$, with $\text{GAMMA}=0$ as a **guaranteed safety floor** ("fall back
to FE-only" is now a valid, honestly-scored option instead of something CP
has to fake via shrinkage). Also added optional per-feature target
standardization and tightened the rank search range. Relaunched across 28
hosts; all 4 cells showed 3–10x v1's best deltas within 18 hours.

---

## Phase 4 — Architecture split: Ridge as the booster's baseline (May 28)

**What changed:** introduced a second, parallel objective track,
`ridge_delta_v3`, alongside the existing FE-residual track
(renamed `residual_delta_v3` with a per-feature X-scaling toggle added).

- `residual_delta_v3`: baseline = firm-feature means (FE); CP fits on
  $Y - \mu_{ff}$.
- `ridge_delta_v3`: baseline = the CP-matched Ridge regression itself (which
  can already exploit each firm's own lagged values, not just its mean); CP
  fits on Ridge's **out-of-fold residuals**, and the final prediction is
  $\hat{Y}_{\text{Ridge}} + \gamma \cdot \widehat{\text{CP}}$.

**Why:** FE is a very weak baseline (just a mean). Ridge is a much stronger,
harder-to-beat baseline. If CP can add value *on top of Ridge specifically*
(not just on top of a naive mean), that's a much more convincing case that
the tensor structure is doing real work rather than mopping up what any
decent persistence model would already get. This two-track split
(FE-residual CP vs. Ridge-booster CP) is the architecture that survives
into the final paper.

---

## Phase 5 — Bug #3: the booster's headline number is partly an artifact (Jun 19)

**What we found:** a per-fold audit of the top-5 trials in each cell showed
the booster's celebrated `+0.005` R² delta was **structurally front-loaded**:
fold 1 was *exactly* `0.0000000000` across all 5 top trials, fold 2
contributed ~+0.0035, fold 3 contributed ~+0.0103. The mean matched the
journal's reported best almost to the decimal — meaning the whole headline
number was really just folds 2–3 diluted by a fold that contributed nothing.

**Why:** early outer-training windows didn't have enough history to produce
a real out-of-fold (OOF) Ridge prediction (the inner time-series split
was skipped whenever `inner_tr_idx.size < 5`). In those windows, CP was
being trained against a filled-in FE residual instead of an honest Ridge
residual, and then added on top of Ridge at validation time — contributing
literally nothing there. The "real" Ridge-orthogonal CP signal, once you
throw out the contaminated fold-1 windows, was **~+0.010 R² (L=2)** and
**~+0.009 R² (L=4)** — actually *better* than the reported headline, just
honestly measured on fewer, cleaner rows.

**Fix:** `_compute_ridge_predictions_for_fold` now returns an
`initialized` mask; booster training **drops** un-initialized rows entirely
instead of silently back-filling them with FE residuals.

---

## Phase 6 — First fully honest result: v3 holdout (Jun 20)

With all three bugs fixed, the v3 holdout (`residual_delta_v3` /
`ridge_delta_v3` × $L \in \{2,4\}$) ran clean: **CP wins in all 4 locked
cells** on the calendar-fixed 2021Q1+ holdout. This is the architecture,
objective, and result set that `main_v2.tex` reports:

$$\hat{Y}_{\text{ensemble}} = \hat{Y}_{\text{baseline}} + \gamma \cdot \big(\text{CP.predict}(X_{\text{test}}) \times y_{\text{rms}} \times s_{\text{feat}}\big)$$

where baseline is either firm-feature FE or the CP-matched Ridge regression,
and the whole thing is fit with a `LowMemCPRegressor` at scale (Gram-matrix
identity, avoids materializing the full design matrix — needed once the
universe grew past ~500 firms; see Phase 7).

Headline holdout deltas (rank-1 trial per cell, extended holdout):
FE cells **+0.048 / +0.043** ($L=2/L=4$), Ridge cells **+0.019 / +0.011**;
20/20 top trials positive across cells.

---

## Phase 7 — Scale to 499 firms + HY/crossover credit universe (Jul 6–7)

With the architecture finally trustworthy, locked the winning hyperparameters
(no re-tuning) and refit on the full 499-firm universe. **Transfer confirmed
in all 4 cells** — CP's incremental value survives well beyond the original
50-name universe, at roughly 40% of the mega-cap-sample effect size.

This unlocked the economic-content work (Phase 8), including a dedicated
**high-yield/crossover CDS universe** (298 Markit reference entities with
median spread ≥150bp, linked to CRSP/Compustat, 113 survive the pipeline's
universe gate) to test whether the tensor's forecast errors say anything
about credit risk specifically where default risk is priced.

---

## Phase 8 — Economic content: does any of this matter for markets? (Jun 29–Jul 7)

Once Part 1 (predicting fundamentals) was validated, Part 2 asked whether
the *errors* in that prediction (forecast "surprises") carry information
that prices react to:

- **Return event study:** null, robustly, everywhere (0 headline survivors
  at BH<0.1 across all 4 cells) — forecast surprises don't predict raw
  stock returns.
- **Volatility signal:** the CP-increment does forecast post-event realized
  volatility beyond lagged vol (strong in Ridge cells), **but** option-implied
  volatility (IV) already subsumes almost all of it — the market is pricing
  this information already. No straddle alpha at mega-caps (pure VRP
  mechanics).
- **Veer anomaly framework:** built a studentized, firm-de-biased forecast
  "surprise" z-score, aggregated into 5 themed scores (leverage, earnings,
  investment, liquidity, cashflow). Pre-registered test: **H1**
  (`drift_cashflow → ΔDefault-Distance`) confirmed in all 4 cells at 499
  firms (slope ≈ +0.020, p<0.05 in both residual cells) — this is the one
  genuinely new, economically coherent, statistically confirmed result.
  H2 (veers → ΔImplied-vol) technically confirmed but economically tiny
  (ΔR²≈0.003).
- **HY/crossover CDS:** re-ran the H1-style test
  (`drift_cashflow → Δlog(CDS spread)`) on the high-yield universe —
  **confirmed in all 4 cells**, ~−6.5bp of CDS tightening per 1 SD of
  cash-flow drift (per quarter), the strongest and most
  economically interpretable result in the paper. A pre-registered options
  tradeability test at HY found partial, architecture-dependent straddle
  alpha (2/4 cells, L4-only) — reported as a subordinate, not headline, result.

---

## Phase 9 — Paper rewrite (Jul 7)

`main_v2.tex` was written to replace everything from the prediction section
onward with the validated Phase 6–8 architecture and results, while leaving
the MFI/pre-prediction sections largely in place except for exhibit updates
(feature count, Tucker error, MFI figures regenerated on the clean 40-feature
data). `main.tex` is left as-is / historical.

---

## Files worth knowing about

| File | What it is |
|---|---|
| `RESEARCH_LOG.md` | Full reverse-chronological log; see the "CONSOLIDATED" entry (2026-07-07 evening) for the same story with file/line citations |
| `CP_RIDGE_HANDOFF.md` | Living design-decision doc; top section is current status |
| `Code for paper/prediction_new/worker.py` | Optuna objective + booster training logic (all 3 bug fixes live here) |
| `Code for paper/prediction_new/cp_regressor_lowmem.py` | Gram-identity low-memory CP fitter, needed for 499 firms |
| `Code for paper/CP_struct_test_new.py` | CP-matched Ridge baseline (leakage fix from Phase 3) |
| `Code for paper/prediction_new/veer_anomaly_experiment.py` | Veer z-score construction + pre-registered H1/H2 tests |
| `Code for paper/prediction_new/transfer_check_499.py` | 499-firm transfer gate |
| `Paper_Draft/main_v2.tex` | The paper as it stands now |
