# Tensor Research — Research Log

Hand-curated journal of decisions, actions, and open questions. Append new
entries at the **top** (reverse chronological). Keep entries focused and cite
scripts / result files by relative path.

For verbose context (full chat transcripts, every tool call), see
`~/.cursor/projects/student-mcnama53/agent-transcripts/`. This file is the
distilled version that's actually worth reading later.

---

## 2026-07-07 — H1 ECONOMIC MAGNITUDE: PD translation DONE + CDS extension PRE-REGISTERED

Question (user): what does the H1 slope (+0.020 DD per unit drift_cashflow)
mean in spread / default-probability terms? Two layers.

### Layer 1 — model-implied PD translation (no new data) — DONE
Naive Merton maps DD → physical PD via PD = N(−DD). Per 1-sd move in
drift_cashflow (panel sd 2.76–3.33 across cells), ΔDD = slope × sd =
+0.054..+0.067. Evaluated at the 499-panel dd_pre distribution
(p25/med/p75 = 6.04 / 8.71 / 11.99), the RELATIVE PD change per +1sd signal:
  p25 (DD 6.0):  −29% .. −34%
  med (DD 8.7):  −38% .. −45%
  p75 (DD 12.0): −48% .. −56%
Absolute PDs are astronomically small for these large caps (1e−10..1e−33), so
the honest paper framing is the relative hazard change (for large DD,
dlogPD/dDD ≈ −DD, hence the growing relative effect). Caveat for the draft:
naive-Merton PD is a physical-measure model quantity; market spreads embed
risk premia — hence Layer 2.

### Layer 2 — PRE-REGISTERED CDS extension (before touching the data)
WRDS has Markit single-name CDS: markit_cds.cds2001..cds2026 (parspread by
date × redcode × tenor × tier × docclause). Naive ticker match vs our 556
permnos: ~221 US 5Y USD SNRFOR reference entities (2018 sample year) — real
cross-sectional power.

Declared BEFORE running (this is a magnitude/robustness extension of the
already-confirmed H1, on an independent market measurement of the same
construct — NOT a new headline hypothesis):
  - Spec: 5Y tenor, USD, tier SNRFOR, docclause XR/XR14 (XR14 preferred when
    both), country = United States. Daily parspread, 2021-07..2026-07.
  - Link: CRSP stocknames ticker ↔ Markit ticker, names date-valid in the
    panel window; ambiguous multi-matches resolved to the redcode with the
    most spread observations.
  - Target: d_logcds = log spread(+63td) − log spread(−2td) around the same
    announcement dates as the veer panels; controls log(spread_pre) and the
    pre-delta over [−65td, −2td] (exactly parallel to the d_iv construction;
    trading-day offsets on the CRSP market calendar, spreads ffilled ≤5d).
  - Test: FM slope of drift_cashflow → d_logcds with controls, all 4 cells.
  - EXPECTED SIGN: NEGATIVE (cash-flow over-performance → spreads tighten).
  - Deliverable: bp and % tightening per 1-sd drift_cashflow at the sample
    median spread.
Scripts: fetch_cds_markit.py (pull), prediction_new/cds_h1_translation.py
(target build + FM + translation). Results entry to follow.

### RESULTS (same day, ~30 min after pre-registration)

**Pull**: 206,166 spread-days, 189 gvkeys matched (of 497; ~38% coverage —
CDS trades on the larger/levered half of the universe). Link audit: 84% exact
name-prefix agreement; all flagged disagreements are abbreviation artifacts
("Gen Mls" = General Mills) EXCEPT ticker BR, where Markit's BR is Burlington
Resources (dead ticker reused by Broadridge) — excluded. Median spread 56 bp
(investment-grade universe, as expected).

**CDS test (pre-registered sign: negative): CLEAN NULL, 4/4 cells.**
  FM slope on d_logcds ≈ 0 in every cell (t between −0.08 and +0.14, p>0.88);
  per +1sd drift_cashflow: ±0.03 bp at the 56 bp median spread. Partial
  rank-IC has the right sign (−0.014..−0.028) but t ≈ −0.5..−0.9, ns.
  n=1,961 events, 169 firms, 14 FM quarters per cell.

**Composition check (crucial)**: re-ran the pre-registered d_dd FM on the
CDS-covered subsample only — slope is INTACT and slightly larger
(+0.024..+0.026, t +2.05..+2.29 vs +0.020 full panel). So the null is not
"CDS firms lack the effect"; the same firm-events show DD improving while
their traded spread does not move.

**Verdict / paper framing.** H1's economic magnitude:
  (a) In model units: +1sd cash-flow drift → ΔDD ≈ +0.05..+0.07, i.e. a
      29–56% RELATIVE reduction in naive-Merton PD (Layer 1, artifact
      h1_pd_translation_499.csv). Sounds big, but
  (b) absolute PDs for these investment-grade firms are 1e−10..1e−33, and
  (c) the market price of credit confirms the irrelevance: 5Y CDS does not
      respond (±0.03 bp per 1sd).
  This RHYMES with the IV-subsumption result: every TRADED price we test
  (options IV, now CDS) already embeds what the veer signal knows; only
  MODEL-IMPLIED quantities (naive DD) respond. Honest paper sentence: "the
  effect is statistically robust in the model-implied credit metric but
  economically invisible in traded credit prices for this investment-grade
  universe." Follow-up idea (NOT pre-registered): decompose d_dd into its
  equity-value / equity-vol / debt components to locate the channel; and the
  effect might price in HY/crossover names where PDs are non-trivial — our
  top-499-by-mktcap universe is the wrong place to see spread moves.

Artifacts: results/v3_holdout_499_20260706/{cds_h1_translation_499.csv,
h1_pd_translation_499.csv}, pre_prediction_cache/event_study_499/
{cds_markit.csv.gz, cds_link_audit.csv}.

### EXPLORATORY addendum (same evening) — spread-tercile heterogeneity
Q (user): are the nulls large-cap-IG artifacts? Within-quarter terciles of
pre-event spread level (medians 35 / 56 / 94 bp), all 4 cells:
- d_logcds slope: FLAT across terciles, interaction sig×spread ns (t −0.3..
  −1.1). No gradient — but note the "high" tercile median is 94 bp, still
  BBB-ish; genuine HY (300bp+) is essentially absent from a top-499-by-mktcap
  universe, so this does NOT refute "prices in HY", it only says the gradient
  within IG is flat.
- d_dd slope: CONCENTRATES in the high-spread tercile in all 4 cells
  (+0.017..+0.024, t +1.8..+2.2) and is ns in the low tercile (t +0.5..+1.1).
  The credit channel lives among the riskier half of the universe —
  consistent with the effect strengthening down the quality spectrum.
NOT pre-registered; treat as motivation for a future HY/crossover extension,
not a claim. Paper framing: state the universe boundary explicitly (results
are for the mega/large-cap IG segment; return null, IV subsumption, and CDS
null may all reflect the most-arbitraged corner of the market).

### DECLARED (same evening, before running) — veer → EQUITY RETURNS test
Gap check: Part 2's return null was for the CP-increment signal around
announcements; the veer/drift signals were routed only to d_dd/d_logpe/d_iv.
drift_cashflow → forward equity returns is UNTESTED. Economic motivation:
equity is a call on the firm; if drift_cashflow moves credit (confirmed,
d_dd), levered/low-DD equity is where it should show up — and the d_dd
effect concentrates in the risky tercile.
Design (EXPLORATORY, declared before running; expected sign POSITIVE):
  target   fwd market-adjusted buy-hold return +2td..+63td after ann_date
  control  pre-event abnormal return −65td..−2td (momentum guard)
  tests    (a) FM slope w/ control, (b) within-quarter tercile LS portfolio
           (top−bottom drift_cashflow), mean/t/Sharpe over quarters
  slices   full panel; within-quarter dd_pre terciles (risky = low DD)
  cells    all 4. Script: prediction_new/veer_return_test.py
If null: tradeability verdict for this universe is closed; HY extension is
the remaining route. Results below.

RESULTS (n=6,380 events/cell, 14 quarters; one bug fixed before results were
read: initial run silently used the 50-firm link table via the imported
module default -> only 660 events; veer_return_test.py now loads the 499
link table directly):
- FM slope: ~0 everywhere (t −0.9..+0.7). No linear cross-sectional return
  predictability from drift_cashflow. Matches Part 2's return null.
- Tercile LS: positive in all 4 cells (+0.34..+0.90%/q, t +0.7..+1.7,
  SR 0.4..0.9) but nothing crosses t=2 on the full panel. NOT tradeable
  evidence: 14 quarters, correlated cells, and the FM says the linear
  signal is zero.
- Slices: effect is NOT in the risky-DD tercile (t ~0.1-0.5 there, contra
  the levered-equity story); the single t=+2.22 (ridge L4, SAFE tercile,
  +1.17%/q, SR 1.18) is 1 of 12 slices = multiple-testing noise until it
  replicates. Do NOT chase without a pre-registered replication design.
VERDICT: equity-return route in the mega/large-cap universe is CLOSED
(now tested for both CP-increment and veer signals). Tradeability options
remaining: HY/crossover credit extension (where d_dd concentrates), or a
small-cap universe. Artifact: veer_return_test_499.csv.

---

## 2026-07-07 — ★ 499 SCALE-UP COMPLETE: transfer PASS, H1 partially confirmed, H2 confirmed-but-negligible

Timeline: caches 19:28 → OOM incident + gram fitter (entries below) → all 4
dumps by 22:42 → gate PASS → veer test ran same night. One fix on 07-07: the
elastic-net stage skipped every fold at 499 (control-only vs control+veer
models scored DIFFERENT dropna subsets; coincided at 50 mega-caps, diverged on
the sparser 499 panel). Fixed to a common frame (both models fit/scored on
identical rows); only the EN section changed on re-run — FM/IC/clustering
numbers untouched.

### Transfer check (locked hyperparams, 498 firms, 22 test windows) — PASS 4/4
  ridge L2:    base .745 → ens .759  (delta +.014 | 50-firm +.019)
  ridge L4:    base .751 → ens .762  (delta +.011 | 50-firm +.011)
  residual L2: base .691 → ens .709  (delta +.018 | 50-firm +.048)
  residual L4: base .693 → ens .713  (delta +.020 | 50-firm +.043)
CP structure is NOT a mega-cap artifact — delta positive everywhere — but the
residual-cell edge is ~2.5x smaller on the wide universe (mega-cap tilt).

### ★ H1 drift_cashflow → d_dd (pre-registered): CONFIRMED 2/4, directional 4/4
Slope ≈ +0.020 in ALL four cells (remarkably stable); one-sided p<0.05 in both
residual cells (L2 p=.040, L4 p=.033), near-miss in ridge (L2 p=.060, L4
p=.071); partial-IC t +2.1..+2.6 in all 4. vs 50-firm: slope halves (+.039 →
+.020) but n grows 10x (6,889 events). Verdict: real, replicating directional
effect — persistent cash-flow over-performance vs model forecast precedes
credit improvement. Paper framing: pre-registered confirmation in the residual
architecture, consistent sign everywhere.

### H2 veers → d_iv (pre-registered): formally 3/4, economically NEGLIGIBLE
EN OOS dR2: resid L2 +.0020, resid L4 +.0032, ridge L2 +.0026, ridge L4
−.0006 (n=2,436; IV targets end 2025Q2 — OptionMetrics stops 2025-08). The
50-firm +.027 collapses ~10x at scale. Honest verdict: DO NOT headline —
consistent with Part 2's IV-subsumption story (options already price what the
veers know). Fold into that narrative.

### Exploratory (BY<0.1 over the 33-pair grid per cell)
- earnings → d_logpe (negative, huge t): the KNOWN mechanical denominator
  artifact; persists at 499; ignore per first-look verdict.
- NEW: veer_leverage → d_dd, slope ≈ −0.09, BY≈.03-.04 in BOTH ridge cells
  (debt surprises above forecast → DD deterioration). Economically sensible.
- NEW: veer_investment → d_dd, slope ≈ −0.008..−0.010, BY≈.03-.05 in BOTH
  residual cells (capex overshoot → credit deterioration).
  Ridge cells see the leverage channel, residual cells the investment channel —
  candidates for the next pre-registration, NOT claims.
- Error clustering: STILL NULL at 499 (ARI vs GICS ≈ .006, top-PC share 3.8%,
  mean |error corr| .064) — errors are idiosyncratic; kills anomaly idea #3
  more decisively than the 50-firm null.

Artifacts: results/v3_holdout_499_20260706/ (dumps, transfer_check_499.csv,
veer_report/grid/panel_*_499). Closeout: handoff pointer updated, summaries
pushed to GitHub.

---

## 2026-07-06 (evening) — 499-FIRM SCALE-UP: data landed + PRE-REGISTRATION (read before running)

### WRDS pull for the full universe — DONE (fetch_universe_499.py, one Duo tap)
Output: `pre_prediction_cache/event_study_499/` (append-only; 50-firm caches untouched).
- link_table.csv: 649 rows; 497/499 gvkeys linked (2 without CRSP link) → 556 permnos.
- daily_returns.csv.gz: 2,442,083 rows, 525 permnos, 2005-01-01..2026-07-06
  (crsp.dsf legacy through 2024-12-31 + wrds_dsfv2 2025+), ret density 99.98%.
- daily_market.csv: 5,283 rows across both eras.
- optionmetrics_iv.csv.gz: 2,387,383 secid-dates, 527 permnos, iv_30d/iv_60d,
  2005..2025-08 (optionm.vsurfd2026 does not exist yet on WRDS — expected).

### ★ PRE-REGISTRATION (locked BEFORE any 499-firm test is run)
Headline family for the 499 veer test — exactly two hypotheses, both generated
by the 50-firm first look (2026-07-06 entry below) and now to be falsified:
  H1: drift_cashflow → d_dd. FM-with-controls slope POSITIVE (firms persistently
      beating cash-flow forecasts see distance-to-default improve). Primary stat:
      FM t with controls (dd_pre, d_dd_pre); support: partial rank-IC.
  H2: veers → d_iv. Elastic-net OOS ΔR² > 0 for d_iv (controls iv_pre, d_iv_pre
      vs controls+veers), expanding folds.
EVERYTHING ELSE (full grid, asymmetry, other themes/targets, error-clustering)
is EXPLORATORY, reported with BY correction and labeled as such.

### Locked-hyperparameter rationale (NO re-tuning at 499)
Refits reuse the 4 cells' frozen hyperparams (e.g. ridge_delta_v3 L2: CP rank 4,
REG_W≈23.3, gamma≈0.75; imputation ranks [2,2,2]/[4,4,4]). Re-searching at 499
would reopen the forking-paths problem the Part 2 discipline killed. Suboptimal
transfer only makes forecast errors noisier → biases AGAINST the veer signal
(conservative). GATE: after refit, check base/ensemble R² + delta vs the 50-firm
values; if CP delta collapses on the wider universe, STOP and report ("CP
structure is a mega-cap phenomenon" is a finding, not a bug).

### Execution notes
- Universe knob: PRED_UNIVERSE_TOP_N env override (config edit 2026-07-06);
  caches to prediction_new/tensor_cache_499/ via PRED_CACHE_DIR.
- MFI rebuild folded in (audit Finding 1): full 1990Q1-2024Q4 tensor on v2/40
  spec, Tucker [67,40,20], persist S/F/T/core/D̂ (unlocks anomaly B/C/D/E gate),
  re-run MFI↔FCIX ρ + independence tests. Script: rebuild_mfi_tensor_v2.py →
  pre_prediction_cache/mfi_v2/.
- All long jobs launched detached (setsid nohup, logs on NFS) — robust to user
  disconnect. Orchestrator: prediction_new/run_499_scaleup.sh (idempotent).

### PARKED PROPOSAL (2026-07-06 21:30) — MASKED CP regression arm ("v4"); decide AFTER 499 run
Idea (user): while the 499 refits grind, run a parallel arm testing whether a
properly MASKED CP regression beats the locked unmasked Optuna trials in the
4-cell design. Context: the current fit multiplies (Y - baseline) by the mask,
so unobserved target cells enter as fake zero-residual targets → shrinkage of
CP toward the baseline. Cheap to build now: cp_regressor_lowmem.py assembles
the normal equations row-by-row, so masking = dropping unobserved rows for the
X-side factor updates (+ per-slice (r×r) solves on the y-side); ~50 lines + an
all-ones-mask equivalence test.
Assessment / pre-set decision rules (BEFORE any result is seen):
1. QUARANTINE: the in-flight 499 pre-registered test runs on the locked
   UNMASKED cells no matter what the masked arm shows. Masked is v4/appendix
   ("robust to proper missing-data treatment"), never a mid-run swap —
   otherwise the pre-registration is dead.
2. Two-step design at 50 firms first (cheap; trials are minutes, not hours):
   (a) refit the 4 locked cells with masked fit, SAME hyperparams — isolates
   the estimator change; (b) fresh masked Optuna search, same budget/CV
   protocol/journal machinery — tests whether masking shifts the optimum.
   Ridge/FE baselines stay exactly as-is (they define the delta objectives).
3. Honest prior: masked is NOT guaranteed to win. Zero-residual cells act as
   regularization toward baseline; if missingness is informative (mega-caps
   report almost everything; 50-firm density ~97%), dropping them can hurt.
   Effect should be LARGER at 499 (88.4% density) — which is where it matters.
4. Success metric: same CV objective as the locked search; winner gets ONE
   test-set evaluation. Report either way (a masked loss is also a finding:
   the zero-fill convention is not just harmless, it helps).
STATUS: parked by user decision — reconsider once the 499 scale-up closes out.

### CP FITTER VALIDATION + SECOND OOM MODE (2026-07-06 22:00) — Gram fitter ADOPTED for all 4 cells
The first low-mem fitter (sample-blocked design matrix) fixed OOM #1 but died
at a SECOND memory bomb inherited from tensorly: the small-factor (lookback)
update materializes the joint Khatri-Rao over all other factors — ~400M rows
(~41 GB) at 498 firms. Rewrote cp_regressor_lowmem.py to the classic CP-ALS
Gram identity: phi'phi = (Z'Z) ∘ expanded (K'K), phi'y assembled via einsum —
neither the design matrix nor the joint KR is ever formed. Bonus: the firms²
flop factor drops out; the 50-firm ridge-L2 refit went 641s → 10s.

VALIDATION (before deploying):
1. Small-scale: gram fitter == stock tensorly EXACTLY (0.0 diff, same iters).
2. Real 50-firm scale, few iterations: rel weight diff 4.8e-9 after 1 iter,
   ~1e-6 after 3-50 iters → update equations mathematically identical; larger
   long-run differences are fp-rounding amplified by ALS, not bias.
3. ★ FINDING — CP-ALS at this scale is fp-CHAOTIC: refitting the SAME locked
   ridge-L2 cell with STOCK tensorly today gives max prediction diff 4.4-4.6
   vs the June dump (R2 0.78389-0.78392 vs 0.78301); OMP2 vs OMP8 stock differ
   by 0.63. Gram runs scatter similarly (R2 0.7791-0.7828). Across ALL seven
   validation runs: ensemble R2 in [0.7791, 0.7839], base identical at 0.7637,
   delta always positive (+0.015..+0.020). Implication: "locked" can only mean
   locked hyperparameters + estimator family — exact predictions were never
   bit-reproducible across BLAS thread counts / library builds, even in June.
   Cell-level R2 claims carry ~±0.003 fitter-noise; the Part 1 deltas (2-5x
   that) are unaffected.
DECISION: gram fitter (PRED_CP_LOWMEM=1) adopted for ALL four 499 cells —
uniform fitter removes a cross-cell confound; ridge cells restarted too (sunk
cost ~2.5h, CP fit now seconds; ridge-OOF baseline unchanged and dominates).
All 4 relaunched ~22:12 with OMP=20. Equivalence tests: test_cp_lowmem_equiv.py.

### 499 REFIT INCIDENT + FIX (2026-07-06 ~21:00) — residual cells OOM'd; exact low-mem CP fitter
The two residual_delta_v3 refits (CP rank 12-13) crash-looped on utmlab10-05/07:
stock tensorly CPRegressor materializes a design matrix whose size scales with
firms², ≈65 GB at 498 firms — over the hosts' 62 GB. Silent SIGKILL (OOM), 4
relaunches, no traceback. Fix: prediction_new/cp_regressor_lowmem.py —
LowMemCPRegressor accumulates the normal equations (φ'φ, φ'y) over sample
blocks; line-for-line port of the tensorly fit (same init, same reg, same
convergence). test_cp_lowmem_equiv.py: weight tensors + predictions agree
EXACTLY (rel diff 0.0) with stock at block sizes 1/4/100 → estimator unchanged,
locked-hyperparameter discipline intact. Enabled via PRED_CP_LOWMEM=1 in
run_499_scaleup.sh (ridge cells untouched — rank 4-5 fits in RAM and they were
already 1.5h into healthy runs). RSS after fix: ~3 GB. NOTE: utmlab10-07 went
unresponsive from the OOM thrashing; residual L4 moved to idle utmlab10-04.
Threads raised 8→20 (hosts are 24-core and otherwise idle). Expect ridge dumps
in hours, residual in ~1-3 days (cost scales ~firms² vs the 50-firm 1-2 h runs).

### MFI v2 REBUILD — DONE (2026-07-06 19:30): Tier-0 result SURVIVES, un-provisional
rebuild_mfi_tensor_v2.py → pre_prediction_cache/mfi_v2/. Clean v2/40 tensor:
499 firms × 40 features × 140 quarters (1990Q1-2024Q4), 74.0% observed density
(v1 was polluted by 12 structurally-empty features). Mask-aware Tucker
[67,40,20], RMS-scaled, SVD init: observed relative error **5.45%** (the old
22.2% figure reflected the polluted tensor + random init). Artifacts persisted:
S/F/T factors + core + D̂ (tucker_v2_decomposition.joblib) — anomaly ideas
B/C/D/E now unlocked.
- MFI_v2 ↔ MFI_v1: ρ = 0.60 (same object, cleaner).
- MFI_v2 ↔ FCIX: **ρ = 0.318** (paper claimed ≈0.33 — holds).
- Gretton–Györfi (6 equal-freq bins, 10,000 perms): L_n = 0.546 (crit 1% =
  0.459), I_n = 0.237 (crit 1% = 0.174), perm p ≈ 1e-4 both → **independence
  REJECTED at 1%** — matches the paper's Table (0.544 / 0.242) almost exactly.
VERDICT: audit Finding 1 resolved; MFI↔FCIX is now grounded in the clean
tensor and the "provisional" flag on Tier-0 item 2 is lifted. Paper exhibits
should cite the v2 numbers (error 5.45%, ρ 0.318, L_n 0.546 / I_n 0.237).

---

## 2026-07-06 (audit intake) — external audit verified core; CORRECTION to feature-count note

External LLM audit (full log + handoff + draft vs artifacts): 4-cell numbers, leakage
fixes, frozen snapshot all VERIFIED correct. Also confirmed the 5 fresh OOS quarters
(2025Q1–2026Q1) show positive ensemble deltas in all 4 cells (real forward validation
— highlight in paper). Five findings, triaged below.

### CORRECTION (audit Finding 2) — the 2026-06-30 "FEATURE-COUNT RECONCILIATION"
### note is WRONG. Do not act on it.
That note claimed `len(FEATURE_SPECS)` "is 39 now (dropped iby)" and advised
"update paper text to 39... for MVP use 39". VERIFIED FALSE on 2026-07-06 by
importing the live config: `len(FEATURE_SPECS) = 40`, including
"Annual Income Before Extraordinary Items" (ibadj12). The prediction dumps are
already (21, 50, 40). The 39-feature tensor (496×39×140 in
`fresh_clean_tucker_validation.json`) is a STALE Apr 25–26 v1 artifact predating
the Apr 29 spec finalization — the same build the Apr 29 entry itself declared
polluted (12 structurally-empty features). Correct fix = REBUILD the MFI tensor
on the v2 40-feature spec, NOT a paper edit to "39". A [CORRECTED 2026-07-06]
marker was added at the original note.

### Audit triage
1. (Finding 1, MAJOR) MFI/FCIX exhibits built from the polluted v1 tensor, never
   rebuilt → **RESOLVED 2026-07-06 (evening entry above)**: v2/40 rebuild done
   (rebuild_mfi_tensor_v2.py); ρ = 0.318 (vs claimed ≈0.33), L_n/I_n reject at
   1% (0.546/0.237 vs paper 0.544/0.242), observed error 5.45%. Result survives.
2. (Finding 3) main.tex still headlines discredited legacy results (pre-v2, pooled-R²
   era); main.pdf is a Jan vintage — write-up track, with the honest v3 story.
3. (Finding 4) NO off-site backup of journals/results/logs. **EXECUTED 2026-07-06
   evening**: untracked tensor_cache/*.pkl (licensed), committed RESEARCH_LOG,
   CP_RIDGE_HANDOFF, prediction_new code, all 16 Optuna journals, result
   summaries/reports (t-stats only, no licensed row-level data), veer experiment;
   rebased onto remote (Mac-side deletions) and pushed → github.com/upticks100/
   Projects @ 37ddde6. Push auth: new passphrase-less ~/.ssh/id_ed25519_github
   key (old lab keys were passphrase-locked), ssh config Host github.com block.
   REMAINING: licensed .pkl still in OLD commit history (private repo; purge =
   history rewrite + force-push, deferred); OneDrive copy of _part1_frozen not
   yet made.
4. (Finding 5) Handoff "authoritative plan" section is stale (April state) —
   needs superseding addendum. Write-up track.
5. (Minor) ridge-L2 ext report exists only as FLAGGED_volrisk_...txt (naming);
   analyze_per_feature.py / audit_one_fit.py still hard-code 0.8 splits — guard
   before pointing anything at tensor_cache_ext; paper should say 499 gvkeys →
   496 realized after filters.

---

## 2026-07-06 (later) — VEER ANOMALY EXPERIMENT (Master List #2 + #3) — first look DONE

### Design (all data in hand; no Tucker persist needed)
`prediction_new/veer_anomaly_experiment.py`, run on all 4 cells in
`results/v3_holdout_ext_20260629_230144/` (outputs `veer_report_<obj>_L<L>.txt`,
`veer_panel_*.csv`, `veer_grid_*.csv`).
- **Veer** = studentized forecast error z = (e − center)/(1.4826·MAD), e =
  realized − predicted_ensemble; center = expanding FIRM median (persistent-bias
  removal), scale = expanding pooled per-feature MAD, PAST test quarters only
  (4-q burn-in → usable panel 17q 2022Q1–2026Q1, 50 firms, 850 firm-quarters).
- **Themes** (40 feats → 5): leverage, earnings, investment, liquidity_bs,
  cashflow; signed mean-z + veer_rms; **drift_<theme>** = 3-q rolling mean
  (persistence variant).
- **Targets** (+63td vs −2td around rdq; per-channel controls = −2td level +
  pre-delta [−65,−2]): `d_dd` naive Bharath-Shumway Merton distance-to-default
  (F = dlcq + 0.5·dlttq, as-of ANNOUNCED; σ_E 252d; E from CRSP); `d_logpe`
  log(mktcap/trailing-4q ibq); `d_iv` OptionMetrics ATM 30d.
- **Tests**: FM-with-controls (primary, BH/BY/Holm over 33-pair grid), partial
  rank-IC, neg/pos asymmetry split, expanding elastic-net OOS ΔR² (controls vs
  controls+veers), error-clustering vs GICS + top-PC common-factor gauge.

### Results (cross-cell)
1. **drift_cashflow → d_dd REPLICATES IN ALL 4 CELLS** (FM t≈+2.4..+2.9 AND
   partial-IC p<0.05 in every cell; the only pair that does). Firms persistently
   generating more cash than the model forecasts see their distance-to-default
   IMPROVE over the next quarter, beyond DD level + pre-trend. Sign economic;
   it is the PERSISTENCE (drift) variant that works, not the one-shot veer —
   run-length is where the signal lives, as hypothesized. Caveat: q_bh≈0.18
   within a single cell; 4-cell replication is suggestive, not independent.
   THIS IS THE CREDIT-CHANNEL HYPOTHESIS TO PRE-REGISTER FOR THE 499 SCALE-UP.
2. **veer_earnings → d_logpe (t≈−7, survives BY/Holm in ridge cells) is
   MECHANICAL — do not headline.** d_logpe's denominator updates to include the
   announced quarter; positive earnings veer ⇒ trailing E jumps ⇒ multiple
   compresses because price doesn't respond (consistent w/ the return null).
   It's the denominator effect, not anomaly info. (Honest-caveat twin of Part 2.)
3. **Elastic net**: d_iv gains OOS ΔR² +0.024..+0.041 in ALL 4 cells (on top of
   R²≈0.31 from controls) — veers carry some info about NEXT-QUARTER IV CHANGES.
   Given the level-subsumption verdict, this hints options price the level but
   absorb veer-info with a lag. Hypothesis-generating only. d_logpe ΔR² positive
   (mechanical suspect); d_dd ΔR² negative (11-signal EN overfits; the single
   drift_cashflow FM is the right-sized test).
4. **Error-clustering (#3): errors are essentially IDIOSYNCRATIC.** ARI vs GICS
   sector ≈ 0.00–0.07 (clusters do NOT recover industry), mean |error corr|
   ≈ 0.06, top-PC share ≈ 6–9%. Good: common-factor threat to firm-specific
   veer claims is small. Bad: no nameable omitted factor / contagion structure
   at 50 names — idea #3 comes up empty (may differ at 499).
5. Asymmetry: scattered weak hints (|t|≈2.2, p≈0.04), nothing multiplicity-proof.

### Verdict / next
The veer object is NOT dead-on-arrival (unlike returns): one economically-sane
credit-channel pair replicates across all 4 cells, and the ΔIV elastic-net
increment is uniform. Both are exactly the "slow/non-return channels" predicted.
Next escalation when chosen: pre-register {drift_cashflow→ΔDD, veers→ΔIV} and
scale to the 499-firm universe (Master List L1) — that's the falsification test.

---

## 2026-07-06 — 4-CELL SYNTHESIS: confirmation COMPLETE, verdicts locked

All 4 cells finished overnight Jun 30 (residual L4 panel ran via auto follow-on
driver `driver_residL4_followon.log` — no manual re-trigger was needed). Reports in
`prediction_new/results/v3_holdout_ext_20260629_230144/multitarget_report_ext_*.txt`.

### Verdicts (the "did the story hold?" answers)
1. **EVENT-RETURN NULL IS CELL-INVARIANT. LOCKED.** Pre-registered headline family
   survivors at BY<0.1 = **0 in all 4 cells** (ridge L2/L4, residual L2/L4).
   Strongest possible form of the honest null.
2. **VOL SIGNAL IS CELL-DEPENDENT (ridge cells carry it).** |t|≥2 after lagged-vol
   control (vctrl): ridge L2 **75/80**, ridge L4 **69/80**, residual L2 **13/80**
   (though partial rank-IC 39/80), residual L4 **6/80**. The "CP increment forecasts
   vol beyond lagged vol" claim is robust in the ridge_delta cells, weak in
   residual_delta cells → phrase paper claim per-architecture, not universal.
3. **IV SUBSUMPTION: effectively universal.** ivctrl |t|≥2: L2-ridge 0/80,
   L4-ridge 29/80, L2-resid 13/80, L4-resid 9/80. No cell shows majority survival;
   ridge-L4's 29/80 (the June nuance) does NOT replicate in any other cell → treat
   as cross-sectional-dependence inflation, not robust incremental-to-IV info.
   Headline stays: **beats lagged realized vol, subsumed by option-implied vol.**
4. **No straddle alpha anywhere.** Straddle-proxy LS vs implied benchmark
   boot p<0.1: 80/69/42/40 of 80 across cells but with negative/uniform or
   mixed-sign means = the mechanical variance risk premium, not tradeable signal.

### Status change
4-cell confirmation = DONE. Part 2 empirics are final on the 21Q extended panel.
Open next-step fork (Master Idea List below): (a) write up Part 1 + MFI/FCIX +
honest Part 2, and/or (b) cheap dump-based experiments (veer+B(X) #2,
error-clustering #3) now that all 4 prediction dumps exist. Optional robustness:
walk-forward refit vs fixed-origin on the headline cell.

---

## ★ MASTER IDEA LIST — single source of truth (2026-06-30)
Consolidated ranking of every live idea (proven spine + honest nulls + untested
bets). Two cross-cutting levers gate most of this: **(L1) the 499-firm universe**
(pipeline uses only 50; MFI tensor already spans ~496 — the power lever) and
**(L2) the Tucker-factor PERSIST** (`S/F/T/H/D̂` not saved anywhere yet — unlocks
all structural ideas). Only the dump-based ideas run free today.

### Tier 0 — RESULTS IN HAND (the paper's spine)
1. CP beats Ridge on next-q fundamentals, esp. less-stationary features (Part 1).
2. MFI↔FCIX: aggregate fundamentals-vol depends on price-vol (ρ≈0.33, indep.
   rejected 1%). Macro result. **CONFIRMED on clean v2/40 rebuild 2026-07-06:
   ρ=0.318, L_n/I_n reject at 1%, observed error 5.45% (see MFI v2 REBUILD
   entry). Provisional flag lifted.**
3. Micro-analog (honest): CP increment concordant w/ option-implied vol — REAL but
   SUBSUMED by IV; event returns a clean NULL. "Vol structure lives in
   fundamentals; prices already reflect it."

### Tier 1 — TOP UNTESTED BETS (ranked)
1. **Credit-migration via peer-group credit quality (STRONGEST).** Label each latent
   peer-cluster w/ avg credit quality (rating/CDS/Merton DD). Compare credit quality
   of where the model PLACES a firm vs where its realized trajectory is HEADING; the
   gap+direction → forecast spread widening / vol. Slow-to-price, tradeable (CDS,
   vol), fundamentals-native peers. Fuses Codex #5 (credit early-warning) + neighbor-
   migration. GATE: L2 persist S + credit label.
2. **Veer + B(X) routing — FIRST LOOK DONE 2026-07-06 (see entry above).** Best
   survivor: drift_cashflow→ΔDD replicates in all 4 cells; ΔIV elastic-net gains
   OOS ΔR²≈+0.03 uniformly; veer_earnings→ΔP/E is mechanical (denominator).
   NEXT: pre-register + scale to 499 (L1).
3. **Error-clustering — FIRST LOOK DONE 2026-07-06: NULL at 50 names.** Errors
   idiosyncratic (ARI vs GICS ≈0, top-PC ≈6-9%). Retry only at 499 scale.
4. **Internal incoherence = forensic/earnings-quality (tensor-only).** Residual
   violating normal feature co-movement in core H (rev up + cash collapses) →
   restatements/credit/downside. GATE: L2 persist D̂/H.
5. **Neighbor migration / latent-regime drift (engine behind #1).** Rotation-
   invariant neighbor-set turnover in S (who model thinks you're like vs who you're
   becoming). MUST beat a static industry dummy to be novel. GATE: L2 persist S
   over EXPANDING fits.
6. **Tensor-Peer ECM stat-arb (Codex #1; highest effort).** Peers from S; trade
   price vs fundamental-fair-value divergence; beat price-pairs/GICS after costs.
   GATE: L2 (S+D̂) + price data + L1.
7. **Fundamental Fair-Value ECM (Codex #2).** FV=a_i+τ_q+b′D̂; z=log(ME)−FV; long
   cheap/short expensive; beat B/M,E/Y,ridge-FV. GATE: L2 D̂, OOS-applied.
Supporting/lower: analyst-revision channel (needs I/B/E/S); MFI regime-timing
(mostly in hand: ΔMFI→market vol/drawdown); option-surface SKEW/term relative value
(ATM level already failed); model-uncertainty confidence weight (anomaly F).

### Two anomaly axes (framing for #2–#5)
TEMPORAL (did firm follow its own expected path? → veer/forecast error) vs
CROSS-SECTIONAL/JOINT (does firm fit low-rank panel + normal feature co-movement?
→ U=D−D̂, incoherence, neighbor divergence). A firm can be anomalous on one, not
the other. EVERY bet must clear: predicts downstream INCREMENTAL to priced info,
AND is firm-specific (not a common factor — the trap that killed drawdown/abn-vol).

### Sequencing
#2/#3 now (dumps) → persist Tucker (L2) → #1 + #4/#5 → stat-arb #6/#7; scale to
499 (L1) as early as data allows. CURRENT priority still = finish 4-cell
confirmation before starting any of this.

---

## 2026-06-30 (snapshot) — STATE before context compression (read this first)

### Where we are in one paragraph
Part 1 (CP improves OOS next-quarter fundamentals R²) is intact/frozen. Part 2
(event study) is being confirmed across the 4 locked cells on the extended 21Q
panel with the OptionMetrics implied-vol benchmark wired in. Findings so far:
event-return alpha is a clean robust NULL; the CP "vol signal" beats lagged
realized vol but is (mostly) subsumed by option-implied vol. We then explored
extensions (stat-arb / anomaly detection) but DEFERRED them to finish the 4-cell
confirmation first.

### HOW WE "PREDICT VOL" (clarification logged for future me)
We do NOT run a volatility model. The model forecasts FUNDAMENTALS. The vol signal
is the MAGNITUDE of CP's forecast adjustment:
  cp_increment = predicted_ensemble − predicted_base  (per firm-feature-quarter,
  EX-ANTE / known before earnings). For vol (a magnitude outcome) the predictor is
  the std-scaled |increment| (`incr_z.abs()` in analyze_event_study_multi.py
  `_add_signals`/`analyze_pair`). Outcome = post-earnings realized vol
  `realized_vol_p2_p30` / `idio_vol_p2_p30` (build_event_study_dataset.py). Test =
  cross-sectional alignment of |increment| with realized vol, controlling for
  lagged vol (`pre_vol`) and implied vol (`pre_iv`). Intuition: big model
  disagreement ⇒ hard-to-pin-down firm-quarter ⇒ more volatile.

### POINT OF OPTIONMETRICS
Implied vol = the market's forward-looking EXPECTED vol = the correct benchmark to
decide if CP's vol signal is NEW. `fetch_optionmetrics_iv.py` (probe-confirmed
`optionm.vsurfd<yyyy>`, link `wrdsapps.opcrsphist`) pulled ATM 30/60d IV, 51 secids,
2005-2025 → `pre_prediction_cache/event_study_extended/optionmetrics_iv.csv`.
Builder reads it (pre_iv at day -2, annualized→daily /sqrt(252)). Verdict: CP vol
signal beats lagged vol but is subsumed by implied vol → no straddle alpha, but
external validation that the signal is economically real.

### 4-CELL CONFIRMATION STATUS (the current priority; "did the story hold?")
Results dir: `prediction_new/results/v3_holdout_ext_20260629_230144/`
Driver: `prediction_new/run_cell_part2.sh <objective> <L>` (rebuild
aggregate_summary → dump ext preds → build panel w/ --iv → outlier-robust analyzer
→ multitarget_report_ext_<obj>_L<L>.txt). Now passes --iv.
- ridge_delta_v3 L2: DONE (+IV). headline survivors BY<0.1 = 0 (return null);
  vctrl(lagged) 75/80; **ivctrl(implied) 0/80 (fully subsumed)**; straddle-proxy
  (implied) LS negative+uniform 80/80 = mechanical VRP. Flagged exhibit:
  `FLAGGED_volrisk_report_ridge_delta_v3_L2.txt`.
- ridge_delta_v3 L4: DONE (+IV). headline 0; vctrl 69/80;
  **ivctrl 29/80 (only PARTIAL subsumption — NOT 0 like L2)**; straddle implied
  LS 69/80. → NUANCE TO RESOLVE: is L4's 29/80 real incremental vol info or
  cross-sectional-dependence inflation (80 = 40 feats × 2 targets, correlated)?
- residual_delta_v3 L2: BUILDING NOW (cross-cell driver pid 3396554, dump stage).
- residual_delta_v3 L4: STILL SCORING (evaluate_top_trials_test pid 3041091,
  3/5 trials, ~80 min left; base R²≈0.722, ensemble≈0.765, delta≈+0.043 — healthy).
  ⚠️ Its panel is NOT auto-queued (my waiter command was interrupted). After its
  scoring CSV `residual_delta_v3_L4.csv` appears, run:
  `bash run_cell_part2.sh residual_delta_v3 4`.

### USER'S ANOMALY IDEA (#8) — capture, distinct from the subsumed vol result
Hypothesis: use the tensor model to DETECT ANOMALIES (a firm suddenly breaking),
not forecast expected vol. Anomalies = the UNEXPECTED part, which options may not
price (they price expected symmetric vol). Signal objects:
  (1) structural residual  U_i,q = D_i,q − D̂_i,q (firm stops fitting low-rank
      pattern); (2) forecast error realized − predicted (model blindsided).
Test: does an anomaly flag (large |U|, or a firm spiking while tensor-peers don't)
predict subsequent TAIL/drawdown, vol surprises, or analyst revisions INCREMENTAL
to implied vol + recent moves? Honest caveats: residual known only at rdq, and our
post-earnings RETURN null says fast repricing — so target slower/non-return
channels (multi-quarter drift, revisions, downside/tail, credit). Distinct from the
vol test because that used the ex-ante INCREMENT (disagreement, priced) whereas this
uses the EX-POST ERROR/RESIDUAL (the break).

REFINEMENT (user, strong): route each veering feature to its NATURAL downstream
target instead of generic returns — a pre-registered FEATURE→TARGET map:
  leverage / coverage / cash-flow veer → CREDIT SPREAD change
  earnings / sales / margin veer       → forward P/E re-rating + analyst EPS revisions
  broad dispersion / uncertainty veer  → implied-vol CHANGE / SKEW (not level)
Why it's the best version: (1) economically disciplined; (2) DEFENDS against the
common-factor critique that killed drawdown/abn-vol — if different features hit
DIFFERENT, economically-correct targets, that is the signature of genuine
feature-specific information (publishable, fits the paper's EMH framing).
Discipline: always test the veer vs the CHANGE-in / FORWARD value of the target,
INCREMENTAL to its current level + what's priced (same rule that exposed the vol
signal as subsumed). Ranking by chance of surviving "already priced":
  1) credit spreads (best econ fit, slower market; needs Markit CDS/FISD; mega-cap
     sparsity risk) 2) analyst revisions (clean info target; needs I/B/E/S)
  3) forward P/E re-rating (in hand; watch mechanical P/E=price/earnings + price
     efficiency) 4) future IV/skew (in hand; current IV already subsumes level →
     use skew/term-structure).
In-hand now: forward P/E re-rating (CRSP+Compustat), future IV-change/skew
(OptionMetrics), in-house autoregressive "does the veer persist" test. Needs pull:
I/B/E/S revisions, credit. Build = small routing table, test in-hand targets first,
pull credit/revisions only if feature-specific structure appears.

MODEL FORM (user, adopt): instead of a hand-routing map, LEARN it —
  Δtarget_{i, q→q+h} = α + γ'Controls + B' z_{i,·,q} + ε,
z = the 39-vector of per-feature veer z-scores, B fit by ELASTIC-NET (not plain
lasso: the 39 errors are collinear → lasso selection unstable; or regress on the
~3-5 themed GROUP-veers for max interpretability). B's nonzero pattern IS the
result: if CDS loads on leverage/coverage veers and P/E on earnings/margin veers,
the routing is DEMONSTRATED not assumed (uniform loading = common-factor warning).
Three make-or-break rules: (1) predict the FORWARD CHANGE, with current level +
obvious factors (rating, leverage level, equity vol, distance-to-default) IN the
controls so the veer earns its keep incrementally; (2) elastic-net / group-veers
for collinearity; (3) POWER — 39 predictors on 50 firms × 21q is thin → use the
499-firm universe, time-respecting CV, firm+quarter-clustered SEs; per-feature
selection only trustworthy at full scale. Quantify veer = studentized robust
forecast error: z_{i,k,q} = (e − med_k)/(1.4826·MAD_k), e = realized −
predicted_ensemble, scale from PAST quarters only (expanding, leakage-safe),
firm-demeaned; themed signed aggregation for direction-ful targets, RMS magnitude
for uncertainty. All inputs already in the 4 dump pickles (no Tucker-factor
persist needed). Build order: (1) compute z 39-vector from dumps, (2) elastic-net
B(X) on in-hand targets ΔIV/skew + ΔP/E w/ controls+CV, (3) read off B, (4) pull
CDS only if structure appears. GATED behind 4-cell confirmation.

### ANOMALY CATALOG — other detectors from the SAME model (brainstorm)
Two axes: TEMPORAL (did firm follow its own expected trajectory?) vs
CROSS-SECTIONAL/JOINT (does firm fit the low-rank panel + normal feature
co-movement?). A firm can be anomalous on one axis, not the other. Each is only a
RESULT if it predicts something downstream INCREMENTAL to priced info AND is
firm-specific (not a common factor — the trap that killed drawdown/abn-vol).
  A. Temporal surprise = veer = realized − predicted_ensemble (dumps; the plan).
  B. Cross-sectional misfit U = D − D̂ (firm doesn't fit low-rank panel THIS q).
     Free complement to A once D̂ persisted.
  C. Latent-regime DRIFT in firm loadings S_q over EXPANDING fits (firm migrating
     through business-model space = "divergent dynamics" proper; bridges to the
     peer/cointegration thread). MOST NOVEL + tensor-only.
  D. Internal INCOHERENCE — residual violates the feature co-movement encoded in
     core H/F (rev up + cash collapses) = forensic / earnings-quality red flag;
     routes to restatements/credit/downside. Tensor-only, economically juicy.
  E. Peer divergence: U_i vs Σ w_ij U_j (peers from S). Relational; ties to stat-arb.
  F. Model uncertainty: imputation instability across ranks/seeds
     (imputer_sensitivity.py) = "can't pin firm down" confidence flag.
Best NEW bets: C (latent drift) and D (incoherence) — both ONLY possible with the
tensor (justify the machinery). A + B(X) run off dumps; B/C/D/E need the Tucker
factor persist (C needs expanding fits) = SAME GATE as stat-arb below.

### STAT-ARB EXTENSION (Codex) — deferred; gate identified
Codex ranked #1 Tensor-Peer ECM stat-arb (peers from Tucker firm loadings S) and
#2 Fundamental Fair-Value ECM (fair value from D̂) as the only paths not already
killed; enriched Codex prompt (exact MFI/FCIX defs + cointegration threads) was
delivered. GATE: Tucker factor matrices S/F/T/D̂ are NOT persisted anywhere (cache
has only validation JSONs). To pursue, write a persist script reusing
`Sweep_Tucker_Ranks.observed_relative_error` recipe (RMS-scale →
`tucker(filled, rank=[67,n_feat,20], mask, init="random", SEED)` → D̂=recon*rms)
and `Build_PrePrediction_Exhibits.build_tensor()`.
FEATURE-COUNT RECONCILIATION: **[CORRECTED 2026-07-06 — THIS NOTE IS WRONG; see
the 2026-07-06 audit-intake entry.** Live `len(FEATURE_SPECS)` = 40 incl. ibadj12;
the 39-feature tensor is the stale polluted Apr v1 artifact. Fix = rebuild MFI
tensor on v2/40, not a paper edit.] Original (erroneous) text: realized tensor is
496×39×140 (fresh_clean_tucker_validation.json, rank [67,39,20], 22.2% err) but
paper main.tex §7.2 and method_audit.md say 40 → "it's 39 now (dropped iby);
update paper text to 39 OR re-add; for MVP use 39."

### NEXT ACTIONS (ordered)
1. Let cross-cell driver finish residual_delta_v3 L2 (+IV).
2. When residual L4 scoring done → `run_cell_part2.sh residual_delta_v3 4`.
3. Synthesize all 4 cells: confirm return-null cell-invariant; resolve whether
   implied-vol subsumption is universal (L2=0/80) or cell-dependent (L4=29/80).
4. (A) optional walk-forward refit robustness on headline cell.
5. Then decide: write up (Part1 + MFI/FCIX + honest Part2 uncertainty/efficiency
   null) vs pursue anomaly #8 / stat-arb #1-#2 (needs Tucker factor persist).

---

## 2026-06-30 (late²) — Strategy brainstorm (Codex) + MFI/FCIX micro-analog framing

### Framing locked (the Part 2 spine)
The paper's macro result — aggregate fundamentals-vol (MFI) is statistically
dependent on aggregate price-vol (FCIX) — now has a **firm-level micro-analog**:
a firm's CP increment is concordant with its option-implied vol (real, but
subsumed). One thesis at two scales: *vol structure lives in the fundamentals,
and prices already reflect it.* Returns remain a clean null throughout.

Exact grounding (Paper_Draft/main.tex §7.2, §8):
- MFI(t) = (1/R3) Σ_k |T[t,k]|, Tucker rank **[67,40,20]**, design tensor
  **N=496 firms × 40 features × 140 quarters (1990Q1–2024Q4)**, 22.2% obs error.
  CP fails to approximate D even at R=100 → motivates Tucker.
- MFI↔FCIX: contemporaneous ρ≈0.33 (lag 0); Gretton nonparametric L_n/I_n tests
  reject independence at the 1% level.
- **Universe lever:** the MFI tensor already spans ~496 firms; the prediction /
  event-study pipeline uses only 50. The 499-firm universe is the power lever for
  any cross-sectional anomaly work.
- The draft already contains **cointegration threads** (commented §: VIX–FCIX
  cointegration, policy-uncertainty→FCIX Granger conditioning on MFI) → a
  fundamentals-anchored stat-arb is a natural extension of the paper's own
  unfinished work, not a new direction.

### Codex idea ranking (from the first brainstorm prompt)
Priority constraint: avoid anything already killed by the event-return null or
subsumed by implied vol; prefer ideas that use tensor structure standard
return/option benchmarks do not dominate.
1. **Tensor-Peer Error-Correction Stat Arb (TOP).** Peers from fundamental
   loadings S (w_ij ∝ exp(−||S_i−S_j||²/h), top-K, normalized). ECM residual
   e_i = (p_i − Σw p_j) − β_i(f_i − Σw f_j), f = fundamentals fair value from
   D_hat (announced data only); trade −z(e) iff residual stationary + plausible
   half-life. Beat price-only pairs, GICS baskets, B/M, 1-mo reversal, after
   costs. MVP: 50 names, tensor vs GICS peers, weekly, no new data.
2. **Fundamental Fair-Value ECM (TOP).** FV_i,q = a_i+τ_q+b′D_hat (dev-fit,
   OOS-applied); z = log(ME) − FV; long cheap / short expensive. Beat B/M, S/P,
   E/Y, residualized value, non-tensor ridge FV. MVP on 50 then expand to 499.
3. Analyst-revision / forecast-error channel (needs I/B/E/S): CP−Ridge forecast
   improvements predict future consensus revisions/errors.
4. MFI / T-factor regime timing (mostly in hand): ΔMFI predicts next-q
   market/factor vol/drawdown; beat VIX / realized vol / FCIX.
5. Credit / CDS early-warning (needs CDS/bond data; MVP via Merton DD or IV skew).
6. Option-surface relative value (skew / term-structure, not level; higher risk —
   ATM level already failed).
7. Tensor-peer fundamental shock propagation (info test): peer Tucker residual
   shocks predict own next-q fundamentals / revisions; beat GICS/corr/random peers.
Blunt priority: **#1 and #2** are the only ideas with a tradeable path not already
killed; use the 499-firm universe ASAP (50 is too thin for stat-arb discovery).

### Status / next
- Cross-cell (+IV) re-runs in flight: ridge L4 + residual L2 through the updated
  `run_cell_part2.sh` (now passes --iv); residual L4 still scoring. Goal: confirm
  the IV-subsumption + return null are cell-invariant.
- MVP decision pending: tensor-peer ECM (#1) vs fair-value ECM (#2) first-look on
  the current 50 names, then expand to 499.
- Codex prompt enriched with the exact MFI/FCIX grounding + cointegration threads
  for the next (deeper) pass.

---

## 2026-06-30 (late) — OptionMetrics verdict: CP vol-info is SUBSUMED by implied vol

### Pull (1 Duo push, append-only)
`fetch_optionmetrics_iv.py` (probe-confirmed schema `optionm.vsurfd<yyyy>`,
link `wrdsapps.opcrsphist`). ATM 30d & 60d implied vol (mean of |delta|=50
call+put), 51/52 secids, **243k secid-dates, 2005-01-03..2025-08-29** →
`pre_prediction_cache/event_study_extended/optionmetrics_iv.csv`. Builder now
merges it (95.7% daily-grid coverage, ffill≤5d, strictly past) and emits
`pre_iv` (annualized→daily via /sqrt(252)) read at day -2 (ex-ante). Analyzer
runs the incremental battery against BOTH benchmarks (v=lagged, iv=implied).

### Result — ridge_delta_v3 L=2, extended 21Q
- **vs LAGGED realized vol: CP wins** (as before) — 75/80 keep |t|≥2 (vctrl),
  75/80 (partial rank-IC).
- **vs option-IMPLIED vol: CP adds NOTHING.** `ivctrl_t` ≈ +1.0–1.2,
  **0/80** reach |t|≥2; implied-vol partial rank-IC **1/80**. The increment is a
  consistent weak-positive (+1) but not significant at T=21.
- **Straddle-PnL proxy (realized − implied) LS is significantly NEGATIVE and
  uniform across all 40 features** (ivsurp_ls_t ≈ −3 to −4.6, 80/80 boot p<0.1,
  ls_mean ≈ −0.002 for nearly every feature). The uniformity is the same
  common-factor tell as max_drawdown: it's the variance-risk-premium scaling with
  vol level (high-|incr| ⇒ high-vol ⇒ larger |VRP|), NOT a feature-specific edge.

### Interpretation (honest)
- The "beats lagged vol" result is real but the bar was too low: **option-implied
  vol already prices the earnings-window uncertainty that CP's increment detects.**
  No straddle alpha. The OptionMetrics test did its job — a referee would have
  demanded exactly this, and it stops us overclaiming.
- The silver lining is genuine **external validation**: CP's increment lines up
  with the option market's independent, forward-looking uncertainty measure. So
  the increment is economically meaningful (it agrees with what sophisticated
  traders price), even though it is not *new* information beyond options.
- Defensible Part 2 claim now: CP's incremental fundamentals signal is a real
  **ex-ante uncertainty indicator** (forecasts vol beyond backward-looking
  measures; concordant with implied vol), but it is **impounded in option
  prices** → an informative-null on tradeable vol alpha, a positive on
  signal validity + market efficiency. Returns remain a clean null throughout.

### Next
- Re-run all 4 cells through `run_cell_part2.sh` (now passes --iv) to confirm the
  implied-vol subsumption isn't cell-specific.
- Decide framing with user: "uncertainty indicator, priced by options" (honest
  positive + null) vs continue hunting (D) for a feature-specific channel.

---

## 2026-06-30 (later) — VOL-RISK REFRAME: CP increment forecasts vol incremental to lagged vol

### Why
After the audit fixes the only robust grid signals were risk/vol (returns null).
User's point: a **risk signal is itself valuable** and vol is tradeable (options,
VIX, variance). Reframe the Part 2 claim from "direction" → "uncertainty
forecasting." To be credible it must clear Bar 1 (incremental to KNOWN vol
predictors) and Bar 2 (not a repackaged common factor).

### What I built (no new data)
- `build_event_study_dataset.py`: added strictly ex-ante **pre-event vol**
  `pre_vol` / `pre_idio_vol` over `PRE_VOL_WINDOW=(-31,-2)` (same ~30d length as
  the post `VOL_WINDOW=(2,30)`, ends before the [-1,+1] window).
- `analyze_event_study_multi.py`: `VOL_CONTROL` map drives an
  incremental-to-lagged-vol battery on `realized_vol_p2_p30` / `idio_vol_p2_p30`:
  (a) pooled OLS `vol ~ pre_vol + |incr|` firm-clustered → `vctrl_t`;
  (b) **partial rank-IC** (lagged vol partialled out of both, outlier-robust) →
  `pic_t`; (c) straddle-PnL proxy LS on `(realized − lagged)` → `vsurp_ls_*`.

### Result — ridge_delta_v3 L=2, extended 21Q (fixed-origin)
- **Bar 1 PASSES (forecasting level).** |CP increment| forecasts realized & idio
  vol **beyond the firm's own recent vol**: `vctrl_t` ≈ +4.6 to +5.6 (p≈0.000),
  and the outlier-robust **partial rank-IC** `pic_t` ≈ +3.9 to +5.5 (p<0.001),
  consistent across the 21 quarters. **75/80** feature×vol-target pairs keep
  |t|≥2 after the lagged-vol control; **75/80** keep |partial rank-IC t|≥2.
  Notably `vctrl_t` (+5) > raw `within_t` (+2) — firm-demeaning over only ~21
  quarters is noisy; pre_vol absorbs the persistent firm-vol component better,
  leaving a cleaner *conditional* (time-varying, firm-specific) vol signal.
- **The naive tradeable version is weak/underpowered.** Quintile LS on
  `(realized − lagged)` vol: `vsurp_ls_t` ≈ +0.1 to +0.6, only **32/80** at boot
  p<0.10. The spread is positive but not reliably significant — (realized−lagged)
  is a noisy target and the quintile LS discards most of the cross-section.
- **Bar 2 (common factor?):** broad across many fundamentals, but it is
  *incremental to lagged vol* and *conditional*, so it is NOT the max_drawdown
  common-beta artifact. Framing: "CP's incremental fundamentals information
  broadly proxies firm-quarter **uncertainty**," not "feature X predicts vol."

### Read / next
- We have a real **conditional-volatility forecasting** result (publishable as an
  uncertainty/risk contribution), but the *tradeable PnL* claim is only suggestive
  because lagged realized vol is a noisy expected-vol benchmark.
- This directly motivates **OptionMetrics implied vol** (WRDS `optionm`): the
  correct expected-vol benchmark and the real tradeability test — does CP predict
  `realized − implied` (≈ delta-hedged straddle PnL)? Forecasting bar already
  passed; implied-vol is what turns it into an alpha claim.

---

## 2026-06-30 — Part 2 audit fixes applied + outlier-robustness layer; 16Q→21Q results

### Context
Codex did a read-only audit of the whole pipeline (Foundation / Part 1 / Part 2).
Verdict: **none of it invalidates Part 1 R²** (calendar split is in the main
evaluator; cache imputation uses only each window's past X). The findings were
all in **Part 2** (the event study) plus a few correctness bugs. User said
"C then A then D, add to the log", then suggested adding outlier detection.
Applied the fixes before letting Part 2 drive any claim.

### Tier-1 correctness fixes (`build_event_study_dataset.py`)
- **Stale-data hard fail.** If any `quarters_test` quarter is beyond the max
  fundamentals quarter, `sys.exit` with a message telling you to pass the
  extended `--fundamentals` / `--daily-*` / `--ff3`. (The first extended build
  silently emitted blank-rdq / zero-target rows for 2025Q1+ because the builder
  still defaulted to `90-25_..._v2.csv` + legacy `event_study/`.) Plus a soft
  warning for any test quarter with 0 matched fundamentals rows.
- **Downside / drawdown sign bugs.** `_running_trough` now floors at
  `min(0, trough)` (a path that never dips below entry = downside 0, matching the
  documented `<=0` semantics; was returning a positive number). `_max_drawdown`
  now prepends entry wealth 1.0 so a first-day drop registers (peak previously
  started at the first in-window gross return).
- **Abnormal-volume window + source break.** `_abn_volume` now requires a full,
  finite, positive EVENT window (consistent with the CAR/vol `_slice` rule;
  baseline may be partial but needs ≥10 valid days). Added
  `VOL_SOURCE_BOUNDARY=2025-01-01` and `_vol_source_straddle`: any abn-vol whose
  baseline/event straddle the legacy-`dsf`→CIZ-v2 turnover-unit break is set NaN
  and flagged `abn_vol_*_src_straddle`.

### Tier-2 statistics/methodology fixes (`analyze_event_study_multi.py`)
- **Clean incremental-CP parameterization.** Regress `target ~ base_z + incr_z`
  where `incr = predicted_ensemble − predicted_base` (the ex-ante CP increment).
  The old `base + ensemble` spec was collinear since
  `surprise_ens − surprise_base = −incr`.
- **Confound-robust magnitude tests.** Firm-demean magnitude targets & signals
  before sorting/regressing — this **killed the spurious magnitude long-short**
  (e.g. idio_vol LS t≈12-15 → ~0). Those huge t's were "high-vol firms are
  high-|signal| firms," not CP info.
- **Honest inference for small T.** Added **Fama-MacBeth** slope (primary for
  signed targets), **rank-IC** (Spearman, outlier-robust), and **block-bootstrap**
  p for long-short. Firm-clustered pooled OLS is now labeled descriptive only.
- **Multiplicity.** BH (exploratory) + **BY/Holm** (robustness) q-values, over
  the full grid and within a small **pre-registered headline family**
  (5 canonical fundamentals × 5 focused targets).
- **No look-ahead in magnitudes.** Signals scaled by std only (uncentered), so
  `|incr|` doesn't peek at the test-panel mean.
- Added `ff3_car_p2_p10` to default targets (builder emitted it; defaults omitted).

### Outlier-robustness layer (user suggestion) — `analyze_event_study_multi.py`
Added per-pair robustness of the PRIMARY test and a detection report:
- `primary_t_wins` — winsorize y + signals at 1/99% (pooled) and re-estimate.
- `primary_t_trim` — drop the most-extreme 1% |y| obs and re-estimate.
- `y_top1_share` — share of |y| mass in the single most-extreme obs.
- `n_mad_outliers` — count beyond `mad_k=5` robust (median/MAD) units.
- New report sections: "OUTLIER ROBUSTNESS of grid survivors (raw vs winsor vs
  trimmed)" and "per-target outlier detection" (worst gvkey/quarter/value).

### Results — extended 21Q panel, ridge_delta_v3 L=2 (fixed-origin)
- **Pre-registered headline: 0 survivors at BY<0.10.** Best is idio_vol_p2_p30
  (within t≈+2.5–2.7, hl_by≈0.33). Return targets ~0 (FM t≈±0.1, IC t≈−0.1).
- **Returns remain a clean null** — and it's outlier-robust: rank-IC agrees, and
  there's nothing to winsorize away.
- **Exploratory grid** has 89 BY survivors, all magnitude/risk:
  - `realized_vol_p2_p30` within t up to +5.7 — **robust to outliers**
    (winsor +4.6, trim +5.9, top-obs share 0.4%) BUT firm-demeaned LS ≈ 0
    (t≈0.2) → a real *continuous, mechanical* "big model adjustment ⇒ volatile
    quarter" link, not tradeable.
  - `max_drawdown_p2_p60` FM t≈−4 to −5 — **robust to outliers** AND has a real
    LS (t≈−4, boot p≈0.00) BUT fires **near-identically for all 40 features**
    (same t, same ls_mean) ⇒ a **common factor** (size/beta in the 2022
    drawdown), not feature-specific CP information.
- **Takeaway:** outliers do NOT explain the vol/drawdown signals (user's
  hypothesis tested & rejected for those); the signals are genuine but either
  mechanical (vol) or non-CP-specific (drawdown). Return predictability is a
  robust null. Leaning toward framing Part 2 as an **informative null** unless
  the redesign (D) finds a composite/feature-specific channel.

### Plumbing
- `run_cell_part2.sh <objective> <L>` — one-shot driver: rebuild
  `aggregate_summary.csv` → dump extended (21Q) preds → build multi-target panel
  (fixed builder + extended data) → outlier-robust analyzer → report txt. Lets
  every locked cell get identical treatment for the cross-cell null check (C).

### Next
- (C) Run the other 3 cells through `run_cell_part2.sh` (ridge L4 running;
  residual_delta_v3 L2/L4 still scoring) to confirm the null isn't cell-specific.
- (A) Locked-hyperparam **walk-forward** refit scorer as robustness vs the
  current fixed-origin OOS; confirm null holds under both.
- (D) Rethink Part 2 (composite signal / different economic question) before
  settling on the informative-null framing.

---

## 2026-06-29 (night) — DATA REFRESH executed: T 16->21, Part 1 preserved & verified

### What ran (all append-only; Part 1 sources untouched)

1. **WRDS pulls (2 Duo pushes).**
   - `fetch_fundamentals_wrds.py --start-date 2025-01-01 --end-date 2026-06-30
     --output 90-25_Q_Fundamentals_v2_ext_2025_2026.csv` → 2,539 rows, all 15
     YTD cash-flow cols healthy (83-100% density). New fiscal quarters
     2025Q1..2026Q2.
   - **CRSP discovery:** legacy `crsp.dsf`/`dsi` are the ANNUAL-update product,
     frozen at 2024-12-31. 2025+ daily data lives in the CIZ v2 views
     (`crsp.wrds_dsfv2_query`, `crsp.wrds_dailyindexret_query`), max date
     **2025-12-31** (no 2026 daily yet). Wrote `fetch_crsp_v2_append.py` to pull
     2025 from v2 with legacy-compatible columns (dlycaldt→date, dlyret→ret,
     dlyprc→prc, dlyvol→vol, shrout; +open/high/low). 12,750 daily rows,
     100% ret density.

2. **Append-merge to VERSIONED files** (`/tmp/merge_extended.py`):
   - `90-26_Q_Fundamentals_v2_extended.csv` — 61,576 rows = 59,037 historical
     (preserved exactly, no dedup) + 2,539 new.
   - `pre_prediction_cache/event_study_extended/` — daily_returns
     2005-01-03..2025-12-31 (254,627 rows), market, link. Legacy event_study/
     left frozen.

3. **Calendar-fixed split (the load_split blocker).** `prediction_config.py`
   now env-overridable (`PRED_FUNDAMENTALS_FILE/END_DATE/CACHE_DIR/TEST_START_Q`,
   defaults = Part 1). `evaluate_top_trials_test._calendar_split_idx` and
   `analyze_v3_holdout` now anchor the first TEST target quarter to
   **2021Q1** (split_idx = index(2021Q1) − L) instead of int(0.8·n_windows).
   `worker.py` left on 0.8 (search not re-run).

4. **Extended caches** built into `tensor_cache_ext/` (env-overridden), capped
   at **2026Q1** (clean coverage; 2026Q2 barely reported). 85 quarters
   (2005Q1..2026Q1), same 50 firms, 90.68% density.

### Verifications (both PASS)

- **Calendar split reproduces Part 1 on default cache:** split_idx 62 (L=2) /
  60 (L=4), first test = 2021Q1, 16 windows — identical to the old 0.8 split.
- **Extended caches byte-identical to Part 1 for all historical windows**
  (np.array_equal on X/Y/Mask), with exactly **+5 new windows per cell**
  (targets 2025Q1..2026Q1). Extension is provably additive.

### Scope reality

- **Part 1 (prediction R²):** test 2021Q1..2026Q1 = **T=21** (+5 OOS quarters).
  Does not need CRSP.
- **Part 2 (event study):** CRSP daily ends 2025-12-31 → confirmation set is
  +2-3 quarters (horizon-dependent), 2026 daily not yet released.

### Disk note

Hit the per-user quota mid-merge (`Errno 122`; `df` shows shared-FS free space
but the personal quota is enforced and not user-queryable). Reclaimed ~1.9GB:
4 stale `.cursor-server` server installs + caches (~1.2GB), and 3 orphan
re-pullable CRSP CSVs with no live consumer (`backup_.../CRSP_Data_1990_2024`,
`pre_prediction_cache/CRSP_Data_1990_2024`, `Masoud/CRSP_Data.csv`; ~702MB).

### Next

Re-score the 4 locked cells (residual_delta_v3 = FE+CP, ridge_delta_v3 =
Ridge+CP booster; L=2,4) on `tensor_cache_ext` → extended holdout. Expect the
first 16 test windows to reproduce Part 1 and 5 fresh OOS quarters to append.

---

## 2026-06-29 — Part 2 REDESIGN: multi-target outcome panel (pre-registration)

### Why

Part 2's first run (3-day CAR ~ CP surprise) was directionally consistent but
underpowered at T=16. Rather than chase significance on one outcome, we
broaden to a **pre-registered panel of economically-motivated outcomes** and
report which channels CP's signal lights up — transparently, with multiple-
testing control. This converged from two independent design passes (me +
Codex); the agreement across passes is a good sign the panel is right. This
is exploration, not p-hacking, *provided* we (a) pre-register the panel and
estimators here, (b) FDR/Holm-correct across targets, and (c) keep a clean
confirmation set (the new quarters from the refresh) untouched until the end.

### Where the two passes agreed (core panel)

Multi-horizon drift returns; realized vol + |CAR|; downside/drawdown;
I/B/E/S SUE + revisions; future-fundamentals as validation; deprioritize
macro rates / CDS.

### What I had that Codex missed — keeping

- **Abnormal trading volume.** Codex's list had no volume channel — a genuine
  gap. Turnover (vol/shrout, both already in our CRSP pull) vs a pre-event
  baseline is the canonical attention / information-content measure: free,
  magnitude-like (easier than sign), and captures a distinct dimension (a
  stock can barely move yet trade enormous volume = disagreement/information).
  Directly matches CP's "this firm-quarter matters" strength. Added to pass 1.
- **Factor-adjusted returns (FF3 minimum) as a purification layer.** Not a new
  target — a *reporting requirement* on the return targets. Any return signal
  must survive FF3 adjustment or a referee calls it momentum/size in disguise.

### What Codex had that I'm adopting

- `idiosyncratic_vol_p2_p30` — residual vol after market/factor adjustment;
  strictly better than raw realized vol for a firm-specific claim.
- Explicit multi-horizon grid (`p2_p10` / `p2_p30` / `p2_p60`) — beats a vague
  "longer horizon."
- Explicit `max_drawdown` and `downside_car` / left-tail — concrete downside.

### Dropping

- `earnings_gap_m1_p1` — needs open prices; our CRSP pull only has close
  (`prc`), so it collapses to CAR. Skip unless we refetch `openprc`.
- CDS / macro rates — both passes agree; out of scope for a 50-name panel.

### Unified first-pass batch (all FREE from existing CRSP pull)

- `car_m1_p1` — 3-day, kept only as baseline/consistency check
- `car_p2_p10`, `car_p2_p30`, `car_p2_p60` — delayed incorporation / drift
- `abs_car_p2_p30` — repricing intensity
- `realized_vol_p2_p30`, `idiosyncratic_vol_p2_p30` — uncertainty (raw + firm-specific)
- `abnormal_volume` (announcement window + `p2_p30`) — attention/info content (my add)
- `max_drawdown_p2_p60`, `downside_car_p2_p30` — downside risk

### Phase 2 (one WRDS fetch, only if pass 1 shows coherent signal)

- I/B/E/S SUE + revisions.
- Future-fundamentals / F-score–Z-score (data already in hand) as
  internal-consistency, framed honestly given the circularity.

### Practical note (affects the refresh)

The `p60` targets need ~60 trading days *after* the announcement, so when we
extend the sample the most recent ~quarter can't be scored on long windows —
usable T shrinks at the recent end for those targets specifically. Minor, but
it determines which targets the newest quarters can contribute to.

### Agreed execution order

1. Calendar-fixed split fix (see preservation entry below — blocker before any
   `END_DATE` change).
2. Extend `build_event_study_dataset.py` to emit this whole target panel in
   one pass.
3. First-look across all targets on the **current 16 quarters** to see which
   channels light up *before* committing to the refresh.

---

## 2026-06-29 — Data refresh plan + PRESERVATION strategy (don't clobber Part 1)

### Context

Part 2 (event study) came back directionally consistent but underpowered at
T=16 quarters (50 firms × 16). The lever with the highest EV is **more
quarters**: extend the panel to ~2025Q4/2026Q1 (T≈21) without re-running the
Optuna search (hyperparameters stay locked). Before touching anything we
pinned down (a) how NOT to overwrite Part 1, and (b) a known split bug that
makes a naive `END_DATE` bump silently shift the test window.

### Preservation decision (simplified)

- **Raw data is NOT precious — it is re-pullable.** We have the WRDS API, so
  Compustat fundamentals and CRSP daily/market/link pulls can be regenerated
  on demand. No need to back them up to GitHub/OneDrive.
- **What IS irreproducible (the crown jewels):**
  - `Code for paper/prediction_new/optuna_journal/` — days of distributed
    TPE search across the lab machines. Cannot be cheaply regenerated.
  - Result tables: `results/v3_holdout_*/` (holdout summaries, per-window,
    per-feature R², stationarity outputs, event-study outputs + analysis).
  - The two logs: `RESEARCH_LOG.md`, `CP_RIDGE_HANDOFF.md`.
- **Backup routing (if/when we do off-site):** code + scripts + logs +
  derived result CSVs → private GitHub repo (note: GitHub rejects files
  >100 MB and CRSP/Compustat are *licensed* — raw pulls must NOT go to
  GitHub). Any large/licensed snapshot → OneDrive (1 TB UofT-private,
  license-compatible). But per above, the raw data does not need backing up.
- **No-overwrite rule for the refresh:** every refreshed artifact must land
  in a NEW, version-suffixed path (e.g. `*_ext2026q1`, new holdout dir).
  The refresh must never write over an existing Part 1 file. Freeze a small
  snapshot of the irreproducible derived artifacts (journals + results +
  logs) BEFORE starting.

### Known blocker: split is length-based, not calendar-fixed

`prediction_new/evaluate_top_trials_test.py` (`load_split`, ~line 205) does
`split_idx = int(0.8 * len(X_all))`. So if we simply extend `END_DATE`, the
80/20 boundary slides forward and the *test window changes identity* — the
held-out quarters that defined Part 1's headline R² would no longer be the
same quarters. Before extending we must switch to a **calendar-fixed split**
(freeze the train cutoff to the exact quarter used for Part 1; new quarters
append to the test side only). Until that's done, do NOT bump `END_DATE`.

### Status

Nothing executed on the refresh yet. Next: (1) freeze snapshot of
journals+results+logs, (2) calendar-fix `load_split`, (3) re-pull
fundamentals+CRSP through the new end date into versioned paths, (4) rebuild
caches + re-score locked hyperparameters on the extended test side only.

---

## 2026-06-29 — Part 2 (event study) FIRST RUN launched (post-audit-fix)

### Trigger

Audit fixes applied + per-feature result locked. Running the economic
exercise end-to-end for the first time on the headline cell.

### Cell + config

- **Headline cell:** `ridge_delta_v3` LEVELS L=2, rank_order=1 (the
  booster over a competitive Ridge baseline; +0.018 panel R² headline).
- Holdout dir: `results/v3_holdout_20260620_084220/`.
- CRSP inputs present:
  `pre_prediction_cache/event_study/{daily_returns,daily_market,link_table}.csv`
  (daily_returns has prc + shrout → value-weighting works).

### Pipeline (3 scripts, no Optuna)

1. `dump_test_predictions.py --objective ridge_delta_v3 --L 2 --rank-order 1`
   → `predictions_ridge_delta_v3_L2_rank1.pkl` (now with corrected
   `quarters_test` and `input_quarters`).
2. `build_event_study_dataset.py --features ALL --pre -1 --post 1`
   → 3-day CAR event dataset; permno resolved on ann_date; raw-unit +
   transformed-unit surprise columns; `mktcap_pre` for VW sorts.
3. `analyze_event_study.py` → clustered-SE regressions + EW & VW
   long-short sorts, per feature.

### RESULT (2026-06-29) — directionally consistent but UNDERPOWERED (honest null)

Ran all three stages on `ridge_delta_v3` L=2 rank-1. Pickle, event
dataset (32,000 rows, 95% CAR coverage), and analysis all in
`results/v3_holdout_20260620_084220/` + `.../event_study_analysis/`.

**Inference method note (important).** First pass used two-way
(firm+quarter) clustered SEs as the audit requested. At T=16 quarters
the Cameron-Gelbach-Miller two-way estimator is non-PSD for most
features → NaN SEs. I tried eigenvalue-cleaning the covariance; that
**manufactured absurd t-stats** (328, 72, 16…) by collapsing variance
in the indefinite directions, so I reverted it (documented as a dead
end — do NOT eigen-clean CGM here). Final design:
- **Regression primary inference = one-way firm-clustered** (50 clusters,
  always PSD; handles within-firm serial correlation). Two-way kept only
  as an unstable secondary diagnostic column.
- **Economic test = quarterly long-short portfolio** on the ex-ante
  predictable CP signal (`predicted_ensemble − predicted_base`), 16
  quarterly returns, small-sample t (df=15, 95% crit ≈ 2.13). This
  correctly aggregates within-quarter cross-correlation, which is the
  right unit of inference for an economic claim (per the audit).

**Findings:**

1. Incremental regression `CAR ~ surprise_base + surprise_ensemble`:
   **33/40 features have a positive CP-surprise coefficient** (right
   sign — CP's residual leans the same direction as the announcement
   CAR). But **0/40 reach |t_firm| > 1.96** (max t ≈ 1.63,
   Sales/Turnover). Directionally consistent, individually
   insignificant.
2. Portfolio long-short (Q5−Q1) on the predictable CP signal:
   - Equal-weight: best annualized Sharpe ≈ 0.58, max |t| ≈ 1.15.
   - Value-weight (pre-event mktcap): best Sharpe ≈ 0.60, max |t| ≈ 1.19.
   - **0/40 features clear |t| > 2.13.** Top names are the trending
     balance-sheet items (Capital Expenditures, Debt in Current
     Liabilities, Stockholders Equity) — same family that dominated the
     per-feature R² gains, which is at least internally consistent.

**Honest read.** Exactly what the audit predicted: with 50 firms × 16
quarters the economic exercise is **underpowered**. The CP signal points
the right way (positive incremental coefficients on 33/40 features;
positive Sharpes concentrated on trending features) but nothing survives
proper small-sample / clustered inference. This is a genuine null on
*statistical* economic significance, not a sign-flip.

**Implication for the paper.**
- Part 1 (R² gains + non-stationary concentration) remains the
  contribution and is unaffected.
- Part 2 should be framed as *"the CP signal is directionally consistent
  with announcement returns but statistically inconclusive at this
  panel size"* — i.e. motivation / suggestive evidence, NOT a headline
  economic result. Do not claim a tradeable strategy.
- For IJF / Journal of Forecasting (Part-1-led), this framing is fine.
  Making Part 2 a *headline* would need a much larger cross-section or a
  longer sample (more quarters) for power — flagged as future work.

Possible follow-ups if we want to push Part 2 (all power-limited by
T=16, manage expectations): PEAD drift window [+2,+30], FF3-adjusted
abnormal returns, pooling features into a single composite surprise to
gain power, Holm/BH across the 40 features (won't help — nothing is
near significant to begin with).

---

## 2026-06-29 — Part 2 (event-study) pipeline: applied Codex audit fixes

### Trigger

External audit (Codex, prompt covering Foundation/Part 1/Part 2) flagged
six concrete issues in the Part 2 pipeline before it had been run. User
instruction: "kay apply the fixes then." Part 1 R² results are
unaffected by these issues; the fixes affect the *unrun* economic
exercise only.

### Fixes applied (all to scripts in `prediction_new/`)

1. **`dump_test_predictions.py` — quarter label off-by-L (CONFIRMED BUG).**
   Cache builder uses `X[w] = tensor[:, :, w:w+L]`,
   `Y[w] = tensor[:, :, w+L]`, so the *target* calendar quarter of cache
   window `w` is `quarters[w + L]`, not `quarters[w]`. The old script
   wrote `quarters[split_idx : split_idx + W_test]`, which is L quarters
   too early — for L=2 that's 2020Q3 instead of 2021Q1 at the start of
   the test block. Fixed to `quarters[split_idx + L : split_idx + L + W_test]`
   and now also dumps `input_quarters` (the L quarters that fed X[w])
   for downstream audit. This was the audit's #1 priority because
   joining mis-labeled quarters with `rdq` would have routed every
   surprise to the wrong announcement window.

2. **`build_event_study_dataset.py` — PERMNO lookup keyed on `ann_date`
   (LIKELY BUG).** Old code looked up the gvkey→permno link active at
   `datadate` (quarter-end). For CAR computation the link active on the
   announcement date matters; if a permno change happens between
   quarter-end and rdq, returns route to the wrong security. Fixed to
   prefer `ann_date`, fall back to `datadate` only when rdq is missing.
   Kept old result as `permno_datadate` diagnostic column and prints
   how many rows disagree.

3. **`build_event_study_dataset.py` — emit raw-dollar surprise columns
   (METHODOLOGY HOLE).** Targets are sign-preserving log-modulus
   transformed (`sign(x)*log1p(|x|)`), so `realized − predicted` in
   training units is not a standard SUE. Added `inv_log_modulus` and
   now emit a parallel set of `*_raw_units` columns
   (`predicted_*_raw_units`, `realized_raw_units`,
   `surprise_*_raw_units`, `surprise_*_scaled_raw_units`). Headline
   regression still uses transformed units (matches model training
   space); robustness row in `analyze_event_study.py` re-runs the joint
   regression in raw-dollar-units.

4. **`analyze_event_study.py` — proper two-way FE absorption (LIKELY BUG).**
   Old `demean_by(df, cols, by=["gvkey","quarter"])` did cell-level
   demeaning (each row is its own (firm, quarter, feature) cell) which
   removed all variation. Replaced with `two_way_demean(df, cols)`
   implementing `x − firm_mean − time_mean + grand_mean`.

5. **`analyze_event_study.py` — two-way clustered standard errors
   (METHODOLOGY HOLE).** Old SEs were HC0; for ~50 firms × 16 quarters
   this is liberal. `pooled_ols` now accepts `cluster1`/`cluster2`
   arguments and returns Cameron-Gelbach-Miller two-way clustered SEs
   (V_firm + V_time − V_firm×time). All regression rows now report
   `*_t_cl2way` instead of HC0 t-stats.

6. **`build_event_study_dataset.py` + `analyze_event_study.py` —
   value-weighted portfolio sort (METHODOLOGY HOLE).** Builder now
   carries `mktcap_pre = |prc| × shrout` on the trading day BEFORE the
   event window (so the weight is known ex-ante of the announcement).
   Analyzer emits both `ls_ew_*` (equal-weight) and `ls_vw_*`
   (value-weight) long-short stats.

### Not yet acted on (deliberate)

- **Universe-by-2024Q4 look-ahead.** Writing-only fix; will be addressed
  in paper framing ("retrospective 2024 mega-cap forecasting panel").
  No code change.
- **Multiple testing (Holm/BH).** Will be applied when interpreting the
  per-feature × per-cell table after results land.
- **FF3-adjusted abnormal returns, drift/pre-leakage windows.** Will be
  added as robustness in a follow-up pass once the headline numbers are
  in; current `ret - vwretd` matches what was already implemented.

### Verification

`ast.parse` clean on all three files. No leftover references to
`demean_by` or `se_hc0` labels. Next step: re-dump predictions with the
corrected `quarters_test`, rebuild the event dataset, and run the
analyzer. Will run after the per-feature jobs finish (don't want to
contend for the same NFS / lab hosts mid-run).

---

## 2026-06-29 — per-feature R² + stationarity analysis: launched across 20 lab hosts

### Trigger

Holdout result is set. Asking the harder question: does CP's gain over
Ridge concentrate on **non-stationary** features (where linear-in-X
should struggle most), or is it spread evenly? If concentrated, that
sharpens the paper claim from "CP captures latent factor structure"
to "CP captures latent factor structure *specifically where linear
models break down*."

### What is running

`prediction_new/launch_per_feature_distributed.sh` fanned out 20 tasks
(4 cells × top-5 trials) across 20 lab hosts. Each task refits a
single (objective, L, rank) configuration on the full dev set with
the post-2026-06-19 patched evaluator (`evaluate_per_feature.py`),
predicts on test, and writes per-feature R² for both baseline and
ensemble to a unique NFS CSV.

- Output: `results/v3_holdout_20260620_084220/per_feature_20260629_161824/`
  - `cell_<objective>_L<L>_rank<r>.csv` per trial (20 files)
- Logs: `prediction_new/logs/per_feature_20260629_161824/`
- Estimated wall: ~60 min, dominated by FE L=4 rank-1 (slowest CP fit).

### Reframe (per user, 2026-06-29)

Dropped the "FE+CP vs Ridge_alone" cross-pair comparison from the
paper — the 4-of-5 / 2-of-5 robustness issue at L=4 is the only
weak spot in the whole v3 result and the comparison is structurally
asymmetric anyway. Headline paper claims are now only the
**within-pair** ones:

| Pair | Baseline | + CP | Delta | Trials positive |
|------|----------|------|-------|-----------------|
| FE L=2 | 0.7253 | 0.7712 | +0.0459 | 5/5 |
| FE L=4 | 0.7267 | 0.7711 | +0.0444 | 5/5 |
| Ridge L=2 | 0.7665 | 0.7845 | +0.0181 | 5/5 |
| Ridge L=4 | 0.7684 | 0.7853 | +0.0169 | 5/5 |

Every cell, every top-5 trial: positive. Booster is the headline
(+1.7-1.8 pp over a competitive Ridge baseline). L=2 in main text;
L=4 in appendix as corroboration.

### What the per-feature analyzer will report

`analyze_per_feature.py` (already written, lint-clean) aggregates the
20 CSVs and merges in three per-feature stationarity metrics
computed from the **train-only** portion of each cache (no test
leakage):

- `vr_stat = std(diff(x_f)) / std(x_f)`, where x_f[t] is the
  cross-sectional mean of feature f at time t. Low → trending /
  non-stationary; high → stationary. **Primary metric.**
- `cv = std(x_f) / |mean(x_f)|`. Variability around level.
- `trend_slope = |OLS slope| / std`. Strength of monotonic drift.

For each (objective, mode, L), reports: per-bucket mean delta,
regression slope of delta on metric, win/loss counts, top-10 and
bottom-5 features by mean delta across the top-5 trials.

### Hypothesis being tested

If CP's value comes from latent cross-firm factor structure, the
gains should be larger on features where linear-in-X regression is
fundamentally mis-specified — i.e. trending / non-stationary
features. Predicted sign: **`vr_stat` correlation NEGATIVE with
booster delta** (low `vr_stat` = non-stationary → larger delta).

If this hypothesis is wrong (correlation flat or positive), the
paper claim shrinks to the conservative "CP captures co-movement
structure" without the "specifically where linear breaks" sharpening.

### RESULT (2026-06-29, all 20 cells complete) — hypothesis CONFIRMED

All 20 per-cell CSVs landed clean (40 features each). Aggregated via
`analyze_per_feature.py`; full output in
`per_feature_20260629_161824/per_feature_summary.txt`.

**Per-cell mean feature-level delta (across top-5 trials):**

| Cell | mean Δ per feature | win/loss (trial×feat) |
|------|--------------------|-----------------------|
| residual_delta_v3 L=2 | +0.190 ± 0.014 | 165 / 30 |
| residual_delta_v3 L=4 | +0.158 ± 0.061 | 159 / 36 |
| ridge_delta_v3 L=2    | +0.098 ± 0.003 | **183 / 12** |
| ridge_delta_v3 L=4    | +0.071 ± 0.020 | 149 / 46 |

(These per-feature deltas are individual-feature R² gains; they're
larger than the panel-pooled +0.018 headline because pooling averages
the big winners against the ~40% of features CP barely touches.)

**Stationarity gradient — confirmed and consistent across all 4 cells:**

- `vr_stat` (low = non-stationary/trending): corr **−0.45 to −0.48**.
  Non-stationary tercile (Q1) delta is 6-12× the stationary tercile.
  e.g. residual L=2: Q1=+0.488 vs Q3=+0.075.
- `trend_slope` (high = strong drift): corr **+0.44 to +0.50**.
  Strong-trend tercile (Q3) delta is ~10× the flat tercile.
  e.g. residual L=2: Q3=+0.438 vs Q1=+0.036.
- `cv` (variability around level): corr −0.12 to −0.18 (weak, same
  direction). Not a stationarity metric per se; report as secondary.

The two genuine stationarity metrics (`vr_stat`, `trend_slope`) both
point the same way with |corr| ≈ 0.45-0.50 in **every** cell. This is
the sharpened claim: **CP's gain concentrates on exactly the
non-stationary / trending features where linear-in-X regression is
mis-specified.**

**Top features by delta (booster, both L):** Long-Term Debt - Total,
Sales/Turnover (Net), Assets - Other - Total, Operating Income Before
Depreciation, Debt in Current Liabilities, Receivables - Total — all
large-magnitude, trending balance-sheet / income items, `pos_rate`
1.0 (every top-5 trial agrees). Bottom features are either already
near-perfectly predicted (Inventories r²≈0.90, Liabilities Netting
r²≈0.92) or near-degenerate (Extraordinary Items), where there's no
residual structure left for CP to find.

**Paper impact:** keep the sharpened framing. Suggested figure: scatter
of per-feature booster delta vs `vr_stat` (or `trend_slope`) with the
fitted line, one panel per cell; plus the top-10 feature table. This
turns "CP captures latent factor structure" into "CP captures latent
factor structure specifically where linear models break down," which is
the more defensible and more interesting claim.

---

## 2026-06-20 — v3 holdout complete: CP wins on all 4 cells, all 20 trials, all 3 regime axes

### Headlines

All 20 evaluated trials (top-5 per cell × 4 cells) produce a **positive
ensemble test delta** over the matching baseline. Per-cell rank-1
trial:

| Cell                              | Baseline (Ridge or FE) | base test R² | ensemble test R² | **test delta** |
|-----------------------------------|------------------------|--------------|------------------|----------------|
| `residual_delta_v3` LEVELS L=2    | FE                     | 0.72532      | 0.77122          | **+0.04590**   |
| `residual_delta_v3` LEVELS L=4    | FE                     | 0.72670      | 0.77109          | **+0.04439**   |
| `ridge_delta_v3`    LEVELS L=2    | Ridge (alpha-tuned)    | 0.76647      | 0.78454          | **+0.01808**   |
| `ridge_delta_v3`    LEVELS L=4    | Ridge (alpha-tuned)    | 0.76836      | 0.78525          | **+0.01689**   |

Top-5 ranges per cell (smallest – largest delta):

| Cell                              | Top-5 range          | Mean   | Std    |
|-----------------------------------|----------------------|--------|--------|
| `residual_delta_v3` LEVELS L=2    | +0.04098 to +0.04590 | 0.0437 | 0.0022 |
| `residual_delta_v3` LEVELS L=4    | +0.02494 to +0.04439 | 0.0353 | 0.0084 |
| `ridge_delta_v3`    LEVELS L=2    | +0.01626 to +0.01808 | 0.0173 | 0.0007 |
| `ridge_delta_v3`    LEVELS L=4    | +0.00459 to +0.01689 | 0.0124 | 0.0049 |

L=2 cells are extremely tight (top-5 std < 0.002 R²). L=4 cells have
wider spread but every trial is still positive.

### Regime breakdown (best trial per cell)

Cross-sectional Y dispersion (`y_disp`, mean of per-feature std across
firms in the test window), mask density, and within-test window index
were each split into terciles. Mean delta per tercile plus linear
regression slope/correlation:

| Cell                       | Indicator       | Q1 (low)  | Q2        | Q3 (high) | slope    | corr   |
|----------------------------|-----------------|-----------|-----------|-----------|----------|--------|
| `residual_delta_v3` L=2    | `y_disp`        | +0.05006  | +0.04084  | +0.04591  | −0.01106 | −0.214 |
| `residual_delta_v3` L=4    | `y_disp`        | +0.05089  | +0.03870  | +0.04220  | −0.01947 | −0.404 |
| `ridge_delta_v3`    L=2    | `y_disp`        | +0.02268  | +0.01705  | +0.01367  | −0.02140 | −0.841 |
| `ridge_delta_v3`    L=4    | `y_disp`        | +0.02123  | +0.01588  | +0.01280  | −0.02032 | −0.799 |
| `residual_delta_v3` L=2    | `window_index`  | +0.04442  | +0.04218  | +0.05135  | +0.00052 | +0.267 |
| `residual_delta_v3` L=4    | `window_index`  | +0.04153  | +0.04261  | +0.04952  | +0.00087 | +0.484 |
| `ridge_delta_v3`    L=2    | `window_index`  | +0.01476  | +0.01720  | +0.02304  | +0.00066 | +0.691 |
| `ridge_delta_v3`    L=4    | `window_index`  | +0.01336  | +0.01643  | +0.02169  | +0.00066 | +0.687 |

Mask density showed a weak negative relationship in 3 of 4 cells, but
slopes are noisy. Not headline-worthy.

### Interpretation

1. **Cross-sectional dispersion is the strongest single predictor of
   CP value** (booster |corr| ≈ 0.80 in both L=2 and L=4). CP helps
   most when firms are co-moving (low `y_disp`) — i.e. when there is
   a common factor that linear-in-features Ridge cannot capture
   without firm-firm interaction terms. This is the actual mechanism
   CP exploits: rank-K factorization recovers latent shared
   structure across firms.
2. **The within-test trend is positive in all 4 cells** (window-index
   correlation +0.27 to +0.69). CP benefits from more training
   history, matching the per-fold audit signal that fold-3 had
   ~2× the fold-2 delta. The holdout is a strict extrapolation
   forward in time from the dev set, so this is the strongest test
   of "does the gain shrink as you forecast further forward" — and
   the answer is no, it grows slightly.
3. **The Optuna plateau holds at test time.** Top-5 trials within an
   L=2 cell give nearly identical per-window predictions and pooled
   test R². The "real" CP signal is robust to hyperparameter choice
   within a top-K window. **More search is not the bottleneck.**
4. **The booster delta of +0.017 is the more demanding number for the
   paper.** It is over a properly alpha-tuned Ridge with the same
   structured-zero-fill convention. The +0.044 FE delta uses a
   weaker baseline (just firm-feature means), so it is a less
   competitive comparison even though the number is larger.

### Verdict

The v3 holdout is paper-defensible. The OOF-fallback fix was load
bearing — it amplified the booster signal from CV's +0.005 to
test's +0.017 (3.5×), confirming the audit hypothesis that the
contaminated training rows were actively hurting CP.

### What I am NOT doing next

- **Not launching a v4 search.** The top-5 plateau within each L=2
  cell (std < 0.0009) says the search has converged. More budget
  will not move the headline number. A v4 search would only be
  justified if we change the model class (per-feature gamma,
  multi-seed averaging, etc.), and the EV of those changes versus
  paper-writing time is poor.
- **Not building masked CP.** Out of scope per the handoff document.
- **Not weakening Ridge baseline or selecting favorable subsets.**
  The current Ridge baseline uses inner-CV alpha selection on the
  full dev set, same `ridge_structured_cp_matched_zero_filled_ts_cv`
  helper as the worker.

### What is next

1. Build paper-ready figures: per-window delta scatter (with
   `y_disp` color), per-cell summary bar chart, regime tercile
   table. Probably 1-2 hours.
2. Per-feature R² breakdown for the rank-1 trial in each cell, to
   see if CP gains are concentrated in specific accounting line
   items (working capital? earnings? balance sheet?). This is a
   "value-add" diagnostic for the paper, not a go/no-go.
3. Draft the v3 results paragraph for `Paper_Draft/main.tex`.

Result files:

- `prediction_new/results/v3_holdout_20260620_084220/aggregate_summary.csv`
  — 20 rows, one per (objective, L, trial).
- `prediction_new/results/v3_holdout_20260620_084220/aggregate_per_window.csv`
  — 320 rows (20 trials × 16 test windows), with regime indicators
  and quantile tags.
- `prediction_new/results/v3_holdout_20260620_084220/regime_summary.csv`
  — regime breakdown table by (cell, trial, indicator).

---

## 2026-06-19 (night) — OOF-fallback fix landed in worker + evaluator; smokes running

User authorised autonomous application of the audit fixes and a 2-4 day
follow-on experiment if EV-positive. Documenting what changed in code
before any new compute hits the cluster.

### Code changes

1. **`prediction_new/worker.py` — `_compute_ridge_predictions_for_fold`**

   - Function now returns `(ridge_oof_tr, ridge_va, initialized)` instead
     of `(ridge_oof_tr, ridge_va)`. `initialized` is a per-training-row
     boolean: True iff the row was in some inner_va_idx of a non-skipped
     inner-TimeSeriesSplit block.
   - Removed the silent FE-residual fallback for un-initialized rows.
     Previously those rows had `ridge_oof_tr[idx] = mu_ff_inner`, which
     caused CP to learn from FE residuals on early training windows —
     the exact pattern the per-fold audit (above) identified as the
     +0.005 CV artifact.
   - Inner-TSS skip threshold pulled into a named constant
     `MIN_INNER_TR_SIZE = 5`. The skip rule itself is unchanged but is
     now documented and reused by the evaluator, so the train/test
     convention can't drift silently.

2. **`prediction_new/worker.py` — `make_objective` inner fold loop**

   - For booster trials only: filters `X_tr / Y_tr / M_tr / base_tr`
     down to `cp_train_rows = pack["ridge_oof_valid"]` (the new
     `initialized` mask) before any per-feature scaling or CP fit.
     FE-residual trials are unchanged.
   - If a booster fold has `ridge_oof_valid.sum() == 0` the whole trial
     returns NaN (Optuna prunes). Conservative: avoids selection bias
     toward configs that happen to survive on smaller folds.
   - Per-feature X scaling, RMS scaling, and CP fit now run on the
     filtered tensors. Validation tensors are untouched (Ridge_va is
     still fit on the full outer-training set; CP just doesn't get
     contaminated training targets).
   - Added `honest_oof_rows=<kept>/<total>` diagnostic after the Ridge
     precompute loop so future logs make the row-count loss visible.

3. **`prediction_new/audit_one_fit.py`**

   - Unpacks the new 3-tuple from `_compute_ridge_predictions_for_fold`
     and replays the same drop-fallback convention before fitting CP.
     Logs `honest_oof_rows`.

4. **`prediction_new/evaluate_top_trials_test.py` — full rewrite**

   - Previous version only accepted `pooled_r2 | residual_delta`, did
     not pass through `GAMMA / FEATURE_TARGET_SCALE / FEATURE_X_SCALE`,
     and always evaluated `FE + CP`. It could not score the v3 booster
     at all and would silently mis-score v2/v3 standalone trials.
     Backed up as `evaluate_top_trials_test.py.pre_v3_backup_*`.
   - New version supports all five objective names and dispatches:
     - **`ridge_delta_v3`**: replays the post-fix worker convention —
       computes Ridge OOF on dev with the same `MIN_INNER_TR_SIZE`
       skip rule, drops un-initialised rows, fits CP on the filtered
       set, predicts test as `Ridge_test + GAMMA * CP_test` where
       `Ridge_test` is fit on the full dev set with
       `ridge_structured_cp_matched_zero_filled_ts_cv`.
     - **Other objectives**: FE baseline. Honours `GAMMA`,
       `FEATURE_TARGET_SCALE`, `FEATURE_X_SCALE`, `USE_RMS_SCALING`
       exactly as the worker does.
   - Emits **two** CSVs: pooled summary (`<output>.csv`) and per-window
     deltas (`<output>_per_window.csv`). Per-window file is what enables
     the regime analysis the user hinted at (CP should help on some
     regimes, little on others).
   - Both CSVs are written incrementally to `.partial` files between
     trials so a crash doesn't lose work.

### What is NOT done in this pass (deliberately deferred)

- Per-trial unclipped fold-delta persistence (second-AI item: store
  `fold_deltas`, `positive_folds`, `base_r2`, `ensemble_r2` as Optuna
  user-attrs). Useful but only matters for the next search; the
  current top-K analysis is now driven by the new holdout CSV +
  per-window CSV.
- Removing the `max(score, -1.0)` clip. Same reasoning — only matters
  if we relaunch search.
- Moving baseline helpers out of `CP_struct_test_new.py`. Cosmetic.
- `prediction_config.py` survivorship-bias wording. Will fix in the
  paper edit pass, not in code right now.

### Smoke results (both passed, both are paper-changing)

**Booster L=2 top-2** (`smoke_booster_20260619_223225.csv`):

| Trial | rank | gamma  | CV delta | base_test R² | ensemble_test R² | **test delta** | dropped_oof_rows | cp_train_windows |
|-------|------|--------|----------|--------------|------------------|----------------|------------------|------------------|
| 7853  | 4    | 0.751  | 0.00461  | 0.76647      | 0.78408          | **+0.01761**   | 17               | 45               |
| 2696  | 5    | 0.796  | 0.00456  | 0.76647      | 0.78454          | **+0.01808**   | 17               | 45               |

**FE L=2 top-2** (`smoke_eval_20260619_222634.csv`):

| Trial | rank | gamma  | CV delta | base_test R² | ensemble_test R² | **test delta** | cp_train_windows |
|-------|------|--------|----------|--------------|------------------|----------------|------------------|
| 2170  | 13   | 1.263  | 0.01522  | 0.72532      | 0.77010          | **+0.04477**   | 62               |
| 2138  | 10   | 1.070  | 0.01487  | 0.72532      | 0.76854          | **+0.04153**   | 62               |

Key observations:

1. **Test deltas are 3-4× bigger than CV deltas in both code paths.** The
   pessimism comes from two structural facts: (a) for the booster, the
   OOF-fallback artifact zeroed out fold 1 (now fixed in training but
   the CV mean still averaged it in); (b) for both, the CV folds use
   ~20-45 training windows while the holdout uses all 62 dev windows —
   CP genuinely scales with training history (confirmed by the
   per-fold audit's monotonic fold-1 → fold-2 → fold-3 pattern, and
   by the per-window holdout pattern below).
2. **Per-window pattern is uniformly positive.** Booster L=2 trial 7853
   shows positive delta in ALL 16 test windows (range +0.003 to
   +0.024). FE L=2 similar. The gain is largest in late-test windows.
   This is the "more history → more CP signal" story playing out at
   eval time.
3. **Two different hyperparam configs give nearly identical results.**
   Booster: trials 7853 and 2696 produce window-by-window correlated
   deltas; FE: same for trials 2138 and 2170. The Optuna plateau in
   the audit is reflected at test time. This is *good news for the
   paper* — the result is robust to hyperparameter choice within the
   top-K. It's *bad news for further search* — more Optuna budget
   will recapitulate the same plateau.
4. **The booster fix is firing.** `dropped_oof_rows=17 / cp_train=45`
   means we're now training CP on 45 honest Ridge-OOF rows instead of
   62 contaminated rows. The +0.018 holdout delta vs +0.005 CV
   strongly suggests the contaminated training was actively hurting.

### Full holdout (top-5 per cell × 4 cells, in progress)

Fanned out to 4 lab hosts via `prediction_new/launch_v3_holdout.sh`:

| Host           | Cell                          | Expected wall |
|----------------|-------------------------------|---------------|
| utmlab10-02    | residual_delta_v3 LEVELS L=4  | ~3 h (slowest)|
| utmlab10-03    | residual_delta_v3 LEVELS L=2  | ~80 min       |
| utmlab10-05    | ridge_delta_v3 LEVELS L=4     | ~45 min       |
| utmlab10-07    | ridge_delta_v3 LEVELS L=2     | ~25 min       |

Output dir: `results/v3_holdout_20260620_084220/`.

### Why I am NOT launching v4 right now

The smokes show top-K trials already produce paper-worthy holdout
deltas. The audit showed the booster search has plateaued (<1%
across-trial std on fold deltas). Per-window analysis is the next
high-EV experiment, not more Optuna search.

### Next steps (in this order)

1. Wait for full holdout (~3 h). Verify top-5 deltas are not noise
   (i.e. consistently positive across the K trials per cell).
2. Build `analyze_v3_holdout.py`: regime-tagged per-window analysis
   (cross-sectional return IQR + fundamentals dispersion as regime
   proxies). The hypothesis to test: where does CP help most?
3. Pick a single best `(objective, L, trial)` per cell as the
   "headline" model for the paper. Compute additional diagnostics
   (per-feature R², per-firm-size-bucket R²) on that one model.
4. Write up v3 verdict in `RESEARCH_LOG.md` and
   `CP_RIDGE_HANDOFF.md`. Only relaunch v4 if step 3 reveals an
   obvious failure mode worth fixing.

---

## 2026-06-19 (late evening) — per-fold audit decomposes the +0.005 booster signal; OOF-fallback fix approved, pending apply

### Distributed per-fold audit

Reran the top-5 Optuna trials of each v3 cell with per-fold delta logging.
First 5 fits of L=2 booster ran locally (sequential, ~55 min). Remaining
45 fits fanned out across 29 lab hosts via
`prediction_new/launch_distributed_audit.sh` + `audit_one_fit.py`; 47 of
60 fits returned, 13 hit the 1800s SSH timeout on the slowest L=4 FE
rank-12 trials. Enough data to call it. Aggregated CSV:
`results/distributed_audit_20260619_202328/_aggregated.csv` (+ the local
sequential CSV).

### Findings

**Per-trial per-fold deltas, top-5 per cell** (rank order = journal rank):

| Cell                     | Fold-1 mean (std)        | Fold-2 mean (std)        | Fold-3 mean (std)        | Across-trial std / mean |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| `ridge_delta_v3` L=2     | **0.0000000 (0)**        | 0.0034506 (0.0000160)    | 0.0102659 (0.0000738)    | fold2 0.5%, fold3 0.7%  |
| `ridge_delta_v3` L=4     | **0.0000000 (0)**        | 0.0038307 (0.0001276)    | 0.0093853 (0.0000459)    | fold2 3.3%, fold3 0.5%  |
| `residual_delta_v3` L=2  | 0.0052210 (0.0002532)    | 0.0163458 (0.0004537)    | 0.0218137 (0.0017588)    | fold1 4.8%, fold2 2.8%, fold3 8.1% |
| `residual_delta_v3` L=4  | mixed (some exact zero)  | ~0.008                   | (only 1 trial returned)  | inconsistent             |

**Three things this proves:**

1. **The L=2 booster `+0.005` journal headline is structurally artifact-driven.**
   Fold 1 is exact `0.0000000000` to 10 decimal places across all 5 top
   trials. Fold 2 contributes ~`+0.0035`. Fold 3 contributes ~`+0.0103`.
   Mean = `+0.0046` matches the journal best (`0.00461`). The artifact
   pattern is the OOF-fallback issue identified in the second-AI audit:
   CP is trained on FE residuals in early outer-training rows (where
   inner-TSS skips for `inner_tr_idx.size < 5`) and contributes literally
   nothing when added on top of Ridge at validation. **Real fold-3
   Ridge-orthogonal CP signal is `~+0.010` R² (L=2) and `~+0.009` R² (L=4).**

2. **The booster search is plateaued.** Across-trial std on fold 2/3 deltas
   is < 1% of the mean in both L=2 and L=4 booster cells. Different
   hyperparameters (`rank=1, 4, 5, 7`; `reg_w 10-30`; `gamma 0.75-0.84`)
   produce essentially identical per-fold predictions. The Optuna search
   has converged onto a near-degenerate solution family. More search
   budget on the current space is wasted.

3. **The FE-residual CP cells do NOT show the artifact.** L=2 FE fold-1
   deltas range `0.0049-0.0055` across the 5 top trials — small but real
   positive. Folds 2 and 3 grow monotonically (`~0.016`, `~0.022`). This
   is consistent with the "time-series learning" story: CP needs enough
   training history to fit FE residuals well, and the contribution scales
   with available data. **This is also why L=2 FE looks like the
   strongest standalone CP story in pure CV terms** — it doesn't suffer
   the booster's OOF artifact.

### Second-AI fix designs (approved, pending apply)

Three deliverables came back from a parallel audit pass:

- **OOF-fallback fix (a)** — modifies `_compute_ridge_predictions_for_fold`
  to return a third element (`initialized` mask); `make_objective` drops
  un-initialized rows from CP training instead of filling with FE.
  Rejected alternatives: (b) expanding-window Ridge OOF (10-30x slower,
  introduces time-varying Ridge alpha convention), (c) explicit mixed
  baseline (preserves artifact, just labels it).
- **Evaluator extension** — full replacement of
  `evaluate_top_trials_test.py`. Adds `residual_delta_v2`,
  `residual_delta_v3`, `ridge_delta_v3` objectives; replays GAMMA,
  FEATURE_TARGET_SCALE, FEATURE_X_SCALE, USE_RMS_SCALING faithfully; for
  booster, test prediction is `Ridge_test + γ·CP_test` using the patched
  `ridge_structured_cp_matched_zero_filled_ts_cv`; per-window output CSV;
  `.partial` crash-recovery.
- **Regime indicator design** — primary: cross-sectional fundamentals
  IQR per quarter, threshold at q75 of training-window dispersion.
  Robustness: cross-sectional return IQR from existing CRSP cache.
  Cut points fold-local for CV, dev-only for test. Both pre-specified
  on training data, no leakage.

Caveats noted: (i) the fix may force fold 1 to NaN for some booster
trials due to insufficient honest Ridge OOF rows — must log
`n_completed_folds` per trial to spot selection bias; (ii) the
`inner_tr_idx.size < 5` threshold becomes load-bearing and should be
pinned + documented; (iii) the inner-TSS convention is now duplicated
between worker.py and evaluator — must stay in sync (comment in both).

### Operational status

- 28 v3 workers + 28 watchdogs were already done as of 2026-05-31; no
  workers running anywhere on the fleet.
- v3 journals on NFS unchanged, durable.
- One-time L=2 booster sequential audit CSV + 45-fit distributed audit
  CSV both on disk in `prediction_new/results/`.
- Worker.py and evaluate_top_trials_test.py NOT YET patched — pending
  this entry.

### Next steps (in order)

1. Apply the worker.py `_compute_ridge_predictions_for_fold` + booster
   training-row drop patch (option (a)).
2. Apply the evaluator replacement.
3. Run the new evaluator on existing v3 journals (top-K = 5) for both
   `residual_delta_v3` and `ridge_delta_v3`. This is the first apples-
   to-apples holdout under the fixed convention.
4. Based on the result:
   - If booster beats Ridge on test in ≥1 cell → write the paper around
     that, plus the per-window/regime analysis using the per-window CSV.
   - If neutral or modestly negative → relaunch a short (24h, ~60 trials)
     v4 search with the fixed worker to see if the fix moves the search
     ceiling. Sometimes the search plateau is conditional on the
     artifact and goes away once the objective is corrected.
   - If clearly negative → pivot to FE-residual CP as the primary paper
     framing (it doesn't have the artifact, has real per-fold signal,
     and shows the cleanest time-series learning story).
5. Build out regime indicator as a small utility module once we agree
   on column conventions; use it to compute conditional deltas on the
   per-window CSV from step 3.

---

## 2026-06-19 (evening) — v3 results: per-feature X scaling is a non-issue; Ridge booster found Ridge-orthogonal signal

The 72h v3 run completed around 2026-05-31. All 28 hosts × 8 workers ran to the
time budget without crashes. Final journal counts and winners (monitor.py):

| Study                          | n_trials | best Δ   | rank | reg_w  | gamma | rms   | fY   | **fX** |
|--------------------------------|---------:|---------:|-----:|-------:|------:|-------|------|--------|
| `residual_delta_v3` LEVELS L2  |    3,072 | 0.01522  |  13  | 27.19  | 1.263 | True  | True | **False** |
| `residual_delta_v3` LEVELS L4  |      880 | 0.01169  |  12  | 24.03  | 1.352 | False | True | **False** |
| `ridge_delta_v3`   LEVELS L2   |    9,598 | 0.00461  |   4  | 23.34  | 0.751 | True  | True | **False** |
| `ridge_delta_v3`   LEVELS L4   |    2,297 | 0.00448  |   5  | 30.20  | 0.843 | True  | True | **False** |

### Two clean findings

1. **Per-feature X scaling does not help.** Every winning trial across both
   objectives and both lookbacks selected `FEATURE_X_SCALE = False`. Optuna
   had ~3k-10k trials to explore the toggle and consistently rejected it.
   The X-side normalization hypothesis from the v3 plan is empirically dead.
   The v3-FE cluster therefore just confirms v2 with a marginal L=4
   improvement (+0.00086) and a basically-flat L=2 result (+0.00016).

2. **The Ridge booster found Ridge-orthogonal signal.** Both `ridge_delta_v3`
   winners have *positive* deltas against the patched CP-matched Ridge
   baseline. CV score `R²(Ridge_va + γ·CP_va) − R²(Ridge_va) ≈ +0.005` at
   both lookbacks. Booster winners use **low CP rank (4 and 5)** and a
   sub-unit gamma (0.75-0.85), which is what an honest booster should look
   like: small parsimonious CP residual model contributing a moderate
   fraction of its raw signal on top of Ridge.

### Open question: does this survive the holdout?

CV deltas are necessary but not sufficient. Recall the v1 holdout had CP
losing to per-feature Ridge by 0.001-0.008 in three of four cells despite
strong CV deltas in v1/v2. The booster CV delta (+0.005) is the right sign
and the right size to overturn that holdout gap *if* it transfers — but we
do not know yet because `evaluate_top_trials_test.py` does not currently
handle the booster prediction `Ridge_test + γ·CP_test`.

### What this means for the paper narrative

If the booster delta survives the holdout in even one cell, the story
becomes: "CP captures structured residual variation that linear Ridge
misses, and the right way to expose it is as a Ridge booster, not as a
standalone competitor." That is a defensible methods contribution and
matches the per-feature persistence pattern we documented earlier (CP's
incremental value concentrates on less-persistent features).

If the booster delta does *not* survive the holdout, we have ruled out
both the standalone-CP and CP-as-booster framings for this dataset, and
the paper should pivot to the methodological framing in
`CP_RIDGE_HANDOFF.md` (CP as an exploratory factor model for fundamental
heterogeneity, with Ridge as the right benchmark to defer to for pure
prediction).

### Operational status

- All 224 v3 workers and 28 v3 watchdogs have exited cleanly (workers
  killed by time budget, watchdogs trailed on their `--match` flag).
- Local `pgrep` shows zero `_delta_v3` processes still running on
  `dh2010pc08`. Have not yet swept the full fleet for stragglers; no
  symptoms suggest any.
- Journals are durable on NFS at
  `prediction_new/optuna_journal/study_levels_L{2,4}_{residual,ridge}_delta_v3.log`.

### Next steps (in order)

1. **Extend `prediction_new/evaluate_top_trials_test.py`** to score top-K
   trials of both v3 objectives. For `ridge_delta_v3`, test prediction is
   `Ridge_test + γ·CP_test` using the same patched CP-matched Ridge that
   the booster trained against. Output the same CSV schema as the prior
   v1/v2 holdouts for direct comparison.
2. **Run top-5 per cell** (same K as prior holdouts) — 20 trial × CP test
   evaluations plus the trusted Ridge baselines.
3. **Decide based on holdout numbers** whether to (a) run a per-feature γ
   booster as v4, (b) attempt a stacking variant, or (c) pivot the paper
   narrative as above.

### Notes for whoever picks this up

- `worker.py` is the canonical reference for what each scaling toggle means
  (`FEATURE_TARGET_SCALE`, `FEATURE_X_SCALE`, `USE_RMS_SCALING`, `GAMMA`)
  and for the booster pipeline (`_compute_ridge_predictions_for_fold` and
  the `is_booster` branch in `make_objective`).
- The booster reuses `ridge_structured_cp_matched_zero_filled_ts_cv` from
  `CP_struct_test_new.py`, which was patched on 2026-05-25 to fix inner-CV
  leakage. The Ridge OOF baseline the booster trains against and the Ridge
  baseline you compare to in the holdout must be the same patched function.
- The v3 SURPRISE cluster was dropped on purpose — see the v3 plan entry
  immediately below. Don't re-introduce it without a new hypothesis.

---

## 2026-05-28 (early afternoon) — v3 plan: per-feature X scaling + parallel Ridge-booster track

### Triggering observations
- v2 search finished cleanly (all 28 hosts exited on time budget). Best CV deltas:
  LEVELS L2 0.01506, LEVELS L4 0.01083, SURPRISE L2 0.01414, SURPRISE L4 0.01102.
- v1 top-5 holdout completed (`results/residual_delta_top5_holdout_patched.csv`).
  Per-cell summary versus the CP-matched Ridge baseline (the fair one):

  | Cell          | FE       | Ridge_pf | Best CP test    | CP − Ridge_pf |
  |---------------|----------|----------|-----------------|----------------|
  | LEVELS L=2    | 0.72532  | 0.76694  | 0.76304 (t37)   | −0.0039        |
  | LEVELS L=4    | 0.72670  | 0.76893  | 0.76064 (t203)  | −0.0083        |
  | SURPRISE L=2  | 0.72532  | 0.76407  | **0.76480** (t259) | **+0.0007** |
  | SURPRISE L=4  | 0.72670  | 0.76645  | 0.76472 (t323)  | −0.0017        |

  CP narrowly wins one cell (SURPRISE L=2). In the others it captures ~3.8 of
  Ridge's ~4.2 R² points of gain over FE, but still loses by 0.001-0.008.

### Diagnosis raised in chat
1. **The "RMS scaling on X" path is conceptually weak.**
   `build_prediction_caches.py:process_window` computes one scalar
   `r_t = sqrt(mean(X_obs²))` per window — pooled across firms × features ×
   time-within-window — and divides all of X by it. Two real consequences:
   (a) it does not equalize the *cross-feature* scale gap (log-modulus market
   cap ~12-25 vs. ratio features ~0.5), it only equalizes the *cross-window*
   energy; (b) `reg_W` in the CP search space already absorbs any global
   rescaling Optuna might want, so per-window RMS is largely cosmetic for the
   optimizer.
2. **The SURPRISE vs LEVELS distinction is thinner than its name suggests.**
   Operationally, SURPRISE differs from LEVELS only by that per-window divisor.
   Y is identical in both modes. CV/holdout numbers show the two modes
   within 0.5 R² points of each other in every cell.
3. **Coarser-than-baseline normalization is indefensible.** FE itself works at
   firm × feature granularity (one mean per `(i, j)`). Ridge fits per-feature
   targets with per-feature alpha. CP's only X-side scale layer is a single
   per-window scalar — strictly coarser than both baselines.
4. **The pooled residual objective is fighting Ridge directly.** v2's score is
   `R²(FE + γ·CP) − R²(FE)`, which rewards CP for rediscovering *any* signal
   Ridge already finds. The most natural way to push CP into Ridge-orthogonal
   territory is to make CP predict Ridge's residuals (booster architecture).
   Subtracting `R²(Ridge)` from the v2 score is a constant shift and does not
   change Optuna's search; only changing the *prediction architecture* does.

### Decisions for v3 (locked in by chat)
- **Drop SURPRISE mode** from the search universe. Only LEVELS L=2 and
  LEVELS L=4 are searched in v3. v2 SURPRISE caches remain on disk for any
  future robustness check but are not used.
- **Add a per-feature X scaling toggle (`FEATURE_X_SCALE`)** in the worker,
  computed per outer CV fold from training X only, applied symmetrically to
  X_tr and X_va. This mirrors what v2's `FEATURE_TARGET_SCALE` does to Y.
- **Run two parallel clusters for 72h:**
  1. `residual_delta_v3` — v2 architecture (FE-residual CP), with the new
     `FEATURE_X_SCALE` toggle. Cleanly isolates the X-side normalization
     question.
  2. `ridge_delta_v3` — Ridge-booster architecture. CP fits on the residual
     `Y − Ridge_OOF(X)` using nested `TimeSeriesSplit(3)` on outer-training to
     obtain out-of-fold Ridge predictions. Validation uses a single Ridge fit
     on full outer-training. Early training windows without OOF Ridge fall
     back to FE residual targets. Score = `R²(Ridge_va + γ·CP_va) − R²(Ridge_va)`.
     The Ridge variant used is the patched `ridge_structured_cp_matched_zero_filled_ts_cv`
     (inner-CV leakage fix from 2026-05-25).
- **Cluster layout:** 28 hosts × 8 workers each = 224 workers total. Per host:
  4 × `residual_delta_v3` (2 L=2 + 2 L=4) and 4 × `ridge_delta_v3`
  (2 L=2 + 2 L=4). One watchdog per host covering both studies via
  `--match _v3` substring.
- **Budget:** 72h per worker. With 56 workers/cell/study in parallel that's
  roughly 4,500 trials per L=2 cell and 1,100 per L=4 cell per study.

### What this isolates
- The v3-FE cluster vs. v2 isolates the value of `FEATURE_X_SCALE` alone. If
  it doesn't improve over v2, per-feature X scaling is not the missing piece.
- The v3-Ridge cluster vs. v3-FE isolates the value of the booster
  architecture (does CP have Ridge-orthogonal signal at all?).
- The combined picture (compared to v2) tells us whether (a) the search
  ceiling moves at all, and (b) which ingredient — X normalization or
  prediction architecture — drives any improvement.

### Risks / things to watch
- **Booster signal-to-noise.** Ridge already captures most of FE→Ridge.
  Booster CV deltas may sit in the 0.001-0.005 range instead of v2's
  0.01-0.015, which means TPE needs more trials before its sample-efficient
  exploitation kicks in. The 72h budget is sized for this.
- **Ridge OOF precompute time.** Nested 3×3 `TimeSeriesSplit` × 40 features ×
  7 candidate alphas takes ~5-10 min per booster worker at startup. 112
  booster workers all paying that once is acceptable.
- **Mixed CP target regime in booster.** Early training windows without OOF
  Ridge fall back to FE residuals. This is consistent ("subtract the best
  baseline available at that point") but means CP sees slightly different
  target distributions on early vs. late training windows. We will check this
  is not visibly hurting validation behavior in the first few completed trials
  before relying on the run.

### Implementation plan (in order)
1. Patch `prediction_new/worker.py` to add both new objectives, the
   `FEATURE_X_SCALE` toggle, and the booster pipeline (Ridge OOF precompute +
   ensemble scoring).
2. Patch `prediction_new/monitor.py` to accept `residual_delta_v3` and
   `ridge_delta_v3`.
3. Add `prediction_new/launch_rdv3_on_host.sh` (one watchdog + 4 + 4 workers).
4. Smoke test one trial per objective locally before the distributed launch.
5. Fan out across 28 hosts with the existing parallel SSH helper. Update this
   entry with the launch timestamp and a 12h checkpoint.

### Launch confirmed (2026-05-28 ~16:55 EDT)
- All four worker.py / monitor.py patches landed; lint clean.
- Local smoke validated:
  - `residual_delta_v3` worker on `dh2010pc08` ran trials at ~135% CPU for 20+ min
    with no errors (CP fits use 4 BLAS threads each so a single worker pulls
    1.0-2.5 cores when CP is the bottleneck).
  - `ridge_delta_v3` worker completed Ridge OOF precompute in **5.0s across
    3 folds** at startup and entered the trial loop cleanly. The booster
    code path imports and runs without errors.
- v3 journals wiped clean before distributed launch.
- Parallel fan-out via `launch_rdv3_on_host.sh` SSHed to all 30 prime hosts.
  - **28 hosts running** the full 8-worker + 1-watchdog layout (4 ×
    `residual_delta_v3` + 4 × `ridge_delta_v3`, all LEVELS, L=2 + L=4).
  - 2 hosts unreachable: `utmlab10-17` (No route to host) and `utmlab26-05`
    (publickey rejected, separate trust issue). Acceptable shortfall — the
    plan called for 28 hosts and we hit exactly 28.
- Spot check of 3 hosts (`utmlab10-02`, `utmlab20-03`, `utmlab26-29`): every
  worker is at 100-260% CPU, memory under 2% per worker, load avg ~11. No
  oversubscription or thrash.
- Per-study trial counts immediately after launch: 56 RUNNING on each of the
  four studies (= one open trial per active worker). First completed trials
  expected within ~15 min (L=2) to ~45 min (L=4, booster); first batch of
  meaningful TPE-driven progress around 4-8 h in.
- Next check-in: ~12 h after launch, monitor.py snapshot of all four v3
  studies + a sanity check that the fleet is still alive.

---

### Paper-narrative consequence
- If v3 lands cleanly, the paper's prediction section drops the
  SURPRISE/LEVELS dichotomy and instead frames CP either as
  (a) a fully per-feature-normalized CP that competes head-to-head with
  Ridge, or
  (b) a Ridge-booster ensemble — whichever variant wins on holdout.
- Either way the new framing matches the symmetry of FE and Ridge baselines,
  which is the right hygiene for a methods paper.

---

## 2026-05-26 (midday) — v1 holdout relaunched durably + early v2 results

### Why this entry
- The previous v1 top-5 holdout evaluator died silently overnight; relaunched
  today with crash-resilient tooling. Also: `residual_delta_v2` has been live
  on 28 hosts since 2026-05-25 ~17:04 and now has positive holdout-CV deltas in
  every cell, which is the first time CP has clearly outperformed FE-only on
  the residual objective.

### v1 top-5 holdout — silent failure root cause
- Original run (launched 2026-05-25 14:26) consumed ~7h of compute across
  two Loky workers but **never wrote its CSV**. No OOM event in syslog.
- Cause: the evaluator was a child of a Cursor terminal session (terminal 2987)
  and was not `nohup`'d. When the Cursor session was torn down, the parent
  bash exited and SIGHUP propagated to the still-attached python child.
- Stdout was being piped through `tail -200`, which also defers all output until
  process exit, so we had no incremental visibility either.

### Hardening applied to `evaluate_top_trials_test.py`
- Switched the joblib call to
  `Parallel(n_jobs=..., return_as="generator_unordered")` and now write
  `results/<output>.csv.partial` after **every** completed CP trial. The final
  CSV is written and the `.partial` removed only on full success.
- Each completed trial also logs a per-row summary line so the log file can be
  `tail`'d for live progress.

### Durable relaunch (2026-05-26 11:33)
- Launch shape:
  - `setsid nohup .../python -u evaluate_top_trials_test.py
     --objective residual_delta --top-k 5 --min-completed 40
     --n-jobs-cp 2 --modes LEVELS,SURPRISE --lookbacks 2,4
     --output results/residual_delta_top5_holdout_patched.csv
     > logs/evaluate_v1_top5_holdout_<ts>.log 2>&1 < /dev/null &`
- Verified detachment: `setsid` gave it its own session/process group
  (`SID == PGID == PID`), and after killing its parent bash wrapper it kept
  running with `PPID=1`. So this evaluator now survives terminal/Cursor/SSH loss.
- Patched CP-matched-Ridge inner-CV leakage (see next entry) is already in this
  run, so the Ridge baselines are honest.

### v1 holdout baselines (already computed; CP fits in progress)
| Cell          | FE      | Ridge per-feature | Ridge global | Ridge CP-matched |
|---------------|---------|-------------------|--------------|------------------|
| LEVELS L=2    | 0.72532 | 0.76694           | 0.76740      | 0.76647          |
| LEVELS L=4    | 0.72670 | 0.76893           | 0.76708      | 0.76836          |
| SURPRISE L=2  | 0.72532 | 0.76407           | 0.76379      | 0.76306          |
| SURPRISE L=4  | 0.72670 | 0.76645           | 0.76711      | 0.76543          |
- All three Ridge variants land within ~0.001 of each other in every cell, so
  the per-feature-alpha vs global-alpha distinction is not the source of any
  CP-vs-Ridge gap. The fairness fixes are stable.
- Ridge adds ~4 R² points over FE-only across the board.

### residual_delta_v2 — first holdout-CV snapshot after ~18h
Output of `prediction_new/monitor.py --objective residual_delta_v2`:
```
study                                             n_trials    best_delta  rank      reg_w   gamma    rms   feat_sc
cp_pred_levels_L2_residual_delta_v2                   1349       0.01400    10       20.1   1.351  False    True
cp_pred_levels_L4_residual_delta_v2                    280       0.01005    21       25.5   1.363  False    True
cp_pred_surprise_L2_residual_delta_v2                   994       0.01367     8       9.31   1.140  False    True
cp_pred_surprise_L4_residual_delta_v2                   241       0.00825     5       9.57   1.087  False    True
```
- For comparison, v1's best holdout-CV deltas after ~7 days were:
  - LEVELS L=2 ≈ 0.0023, LEVELS L=4 ≈ 0, SURPRISE L=2 ≈ 0.0048, SURPRISE L=4 ≈ 0.0048
- v2 deltas are **3–10× higher across all four cells**, and the optimizer is
  picking a convergent recipe in every cell:
  - `FEATURE_TARGET_SCALE = True`
  - `GAMMA ≈ 1.1–1.4` (CP residual at slightly amplified strength,
    **not** shrunk toward FE)
  - `USE_RMS_SCALING = False`
  - moderate ranks (5–21)
- Interpretation: per-feature target standardization is the new ingredient that
  was missing in v1; once high-variance features stop dominating the pooled
  loss, CP starts contributing signal that FE alone cannot reproduce.

### Open items
- Wait out the remaining ~30h of the 48h v2 budget before evaluating v2 trials
  on the test set. Do **not** run holdout on v2 trials early; that would turn
  the holdout into part of model selection.
- After 48h, freeze the per-cell top-k and run the (now durable) evaluator on
  the v2 trials. Same output format as v1 so the comparison is one CSV diff.

---

## 2026-05-25 (afternoon) — CP-matched Ridge inner-CV leakage patched

### What was wrong
- `ridge_structured_cp_matched_zero_filled_ts_cv` in
  `Code for paper/CP_struct_test_new.py` previously precomputed
  `y_firm_mean_full` once on the entire outer-training block, then reused the
  same vector when forming the inner-CV residual targets used to **select
  alpha**. So inner-fold alpha selection saw firm means computed with the help
  of future inner-validation rows from the same outer-training block.
- Net effect: a small but real leak in the CP-matched Ridge baseline. Magnitude
  was probably negligible (Ridge variants all sit within 0.001 of each other on
  the test set), but it was a real correctness issue and would have been hard
  to defend in review.

### Fix applied to `Code for paper/CP_struct_test_new.py`
- Inner-CV alpha search now computes `y_firm_mean_inner` from the inner-training
  rows only (`_within_firm_means_y(y_tr=yj[tr_rows], m_tr=mj[tr_rows], ...)`)
  and builds the zero-filled residual target from those inner means.
- The outer-training final fit still uses `y_firm_mean_full`, which is correct.
- This is the version used by `evaluate_top_trials_test.py` going forward
  (including the durable v1 holdout relaunched today).

### Impact on already-running v2 search
- The leakage was only in the Ridge baseline's alpha-tuning loop. CP target
  preparation in `prediction_new/worker.py` was never affected, so the patch
  does **not** invalidate any existing Optuna trials — neither v1 nor v2.
- Only the *evaluators* needed to be re-run with the patched function. The
  durable v1 holdout launched today already imports the patched version.

---

## 2026-05-25 (late afternoon) — residual_delta_v2 designed and launched

### Motivation
- v1 `residual_delta` finally produced positive but tiny CV improvements
  (~0.002–0.005) over FE-only after ~7 days of search across 4 cells. The
  optimizer kept gravitating to high ranks with high regularization, which
  effectively shrinks the CP factor matrices toward 0 — i.e. CP was learning to
  imitate FE rather than add new structure.
- Two specific hypotheses to test in v2:
  1. The pooled SSE objective is dominated by a handful of high-variance
     features, so CP wastes capacity reproducing variance that FE already
     captures. **Fix:** per-feature target standardization toggle.
  2. CP should be free to shrink its own contribution. **Fix:** explicit
     `GAMMA` scaling on the CP residual, with `GAMMA=0` corresponding to
     "fall back to FE-only" (a guaranteed `delta=0` safety floor).

### Search space (`prediction_new/prediction_config.py`)
- `RDV2_RANK_RANGE = (1, 25)` (tighter and smaller than v1's `(5, 80)`;
  residual structure plausibly low-rank)
- `RDV2_REG_W_RANGE = (1e-2, 1e5)` (wider upper bound so CP can be strongly
  shrunk if useful)
- `RDV2_GAMMA_RANGE = (0.0, 2.0)`
- `USE_RMS_SCALING ∈ {True, False}` (kept from v1)
- `FEATURE_TARGET_SCALE ∈ {True, False}` (new in v2)

### Code wiring
- `prediction_new/worker.py`:
  - `OBJECTIVES = ("pooled_r2", "residual_delta", "residual_delta_v2")` and a
    helper `RESIDUAL_OBJECTIVES` constant.
  - `make_objective` branches on `is_v2` to:
    - draw GAMMA + FEATURE_TARGET_SCALE,
    - compute per-feature scale `sqrt(SSE/n)` from inner-train mask,
    - apply scale to centered Y before RMS scaling,
    - unwind feature scale + RMS at predict time,
    - return `R²(FE + gamma * CP_residual) − R²(FE)`.
  - Print includes `gamma` and `feat_scale` for v2 progress lines.
- `prediction_new/monitor.py`:
  - Adds `residual_delta_v2` to `--objective` choices.
  - Adds `gamma` and `feat_sc` columns when v2 is selected.
- v2 studies use a fully separate journal namespace
  (`study_*_L*_residual_delta_v2.log`), so v1 journals are untouched.

### Distributed launch tooling
- `distributed_launcher.py` worked correctly for v1 but is **serial in SSH**.
  With 28 hosts × ~3 SSH calls/host for watchdog setup + ~2/host per worker,
  the v2 launch was projected at 5+ hours — unacceptable. Killed the
  serial launcher after 30 min (only 2 watchdogs started).
- Replaced the per-host driver with `prediction_new/launch_rdv2_on_host.sh`:
  - One SSH call per host runs this script; it backgrounds 1 watchdog +
    8 workers (2 per cell), all `nohup … &`, then exits.
  - SSH fan-out is parallelized in the launching shell (`for h in hosts; do
    ssh "$h" "bash <script>" & done; wait`).
- Net result: all 28 hosts fully launched in **4 seconds**.

### Cluster topology (running since 2026-05-25 ~17:04)
- 28 prime-numbered lab hosts (utmlab10/20/26-* primes minus two unreachable
  ones: `utmlab10-17`, `utmlab26-05`).
- Per host: 8 worker processes + 1 RAM watchdog (`cp_sweep_watchdog.py
  --match residual_delta_v2 --process-pct 85 --system-limit-gb ~50 …`).
- Per host worker mix: 2 × LEVELS L2, 2 × LEVELS L4, 2 × SURPRISE L2, 2 × SURPRISE L4.
- Time budget per worker: 48h (172800s). N-trials cap: 5000 (effectively a
  no-op next to the time budget).
- Thread caps: `OMP_NUM_THREADS=4`, etc., per worker. Watchdog cascades on the
  same `--match residual_delta_v2` string so it only ever kills v2 processes.

### Sanity checks
- Pre-launch smoke test on `utmlab10-02` confirmed v2 journals are created
  with the correct `_residual_delta_v2` suffix and that trials advance.
- After 4h, every cell had **positive** best deltas; after 18h, all four cells
  show 3–10× v1's best deltas (see entry above for the snapshot).

### Files added / modified
- Added: `Code for paper/prediction_new/launch_rdv2_on_host.sh`
- Added: `Code for paper/jobs/rdv2_48h_jobs.txt` (still useful for later
  re-launches via `distributed_launcher.py --job-file`; not used in the
  parallel SSH fan-out).
- Modified: `Code for paper/prediction_new/prediction_config.py`
- Modified: `Code for paper/prediction_new/worker.py`
- Modified: `Code for paper/prediction_new/monitor.py`

### Open items / discipline reminders
- Do not run `evaluate_top_trials_test.py` on v2 trials before the 48h budget
  ends; the holdout must remain a single-shot evaluation step.
- After v2 results stabilize, decide whether per-feature standardization
  should also become a regular CP-baseline ingredient in the paper narrative,
  or whether to present it explicitly as a v1→v2 improvement.

---

## 2026-05-02 (early afternoon) — Transformation integrity check + run-priority note

### Context
- Question raised: whether residual-focused optimization plus imputation plus
  `SURPRISE` mode could be accidentally applying duplicate/incorrect transforms.

### Audit result (current `prediction_new` code path)
- No duplicate-transform defect was found.
- `build_prediction_caches.py`:
  - Tucker imputes `X` windows only.
  - observed `X` cells are preserved exactly via `np.where(...)`.
  - `SURPRISE` exports RMS-normalized `X`; `LEVELS` exports unscaled-unit `X`
    with only missing entries imputed.
- `worker.py`:
  - FE-centering is applied once to `Y_tr` per outer fold.
  - optional `USE_RMS_SCALING` scales centered `Y` residual targets once.
  - `residual_delta` changes the objective score only
    (`R²(FE+CP) - R²(FE)`); it does not add a second transform stage.

### Interpretation
- The current concern is not a discovered preprocessing bug; it is a modeling
  allocation decision (for example, whether to keep heavy `SURPRISE` weighting
  or shift more budget toward `LEVELS`).
- Any move to de-emphasize `SURPRISE` should be recorded as an explicit search
  strategy choice, not a data-cleaning fix.

### Documentation sync
- Added a matching status addendum to `CP_RIDGE_HANDOFF.md` so handoff context
  and research chronology stay aligned.

---

## 2026-04-30 (evening) — Added unified top-trial holdout evaluator

### What was implemented
- Added:
  - `Code for paper/prediction_new/evaluate_top_trials_test.py`
- Purpose: one consolidated holdout evaluator for top Optuna trials across:
  - both objectives (`pooled_r2`, `residual_delta`)
  - both lookbacks (`L=2`, `L=4`)
  - both modes (`LEVELS`, `SURPRISE`)
- This replaces the narrow `L=2`-only diagnostic pattern with a reusable
  evaluator that can be run immediately once residual-delta studies reach a
  minimum completed-trial threshold.

### Output fields
- For each selected trial, outputs in one row:
  - objective/model metadata (`objective`, `mode`, `L`, trial rank/order, CP params)
  - holdout baselines:
    - `fe_test_r2`
    - `ridge_per_feature_alpha_test_r2`
    - `ridge_global_alpha_test_r2`
    - `ridge_cp_matched_zero_filled_test_r2`
  - CP metrics:
    - `cp_test_r2`
    - `cp_residual_delta_test` (`cp_test_r2 - fe_test_r2`)
    - deltas versus each Ridge baseline

### Selection gate
- Script enforces a minimum number of completed trials per study via
  `--min-completed` to avoid premature interpretation from near-empty studies.

### Example command
```bash
cd "/student/mcnama53/Projects/Tensor Research"

"/student/mcnama53/.local/share/mamba/envs/research/bin/python" \
  "Code for paper/prediction_new/evaluate_top_trials_test.py" \
  --objective residual_delta \
  --top-k 3 \
  --min-completed 20 \
  --lookbacks 2,4 \
  --modes LEVELS,SURPRISE \
  --output "Code for paper/prediction_new/results/residual_delta_top3_holdout.csv"
```

### Validation
- Lint check: clean.
- Python compile check: clean.

---

## 2026-04-30 (late afternoon) — Residual-delta run live status snapshot

### Runtime status check
- Verified post-relaunch distributed compute and watchdog coverage after the
  `residual_delta` pivot.
- Active host pool in the live snapshot: 21 hosts.
- Residual-delta workers in the live snapshot: 168 total (8 per host).
- Watchdogs: 1 active watchdog per host across the full 21-host pool.

### Resource utilization snapshot
- Host load levels were generally in the expected range for 8 workers on 24-core
  machines (roughly high single digits to low teens, with occasional higher nodes).
- RAM usage remained comfortably below watchdog limits:
  - Typical observed used RAM was approximately 8-17 GiB per 62 GiB host.
  - Watchdog threshold remains 53 GiB with 85% process policy and cascade cleanup.

### Study-state snapshot (residual-delta objective)
- Initial monitor showed all four residual-delta studies active with running
  trials and no completed trials yet at startup check:
  - `cp_pred_levels_L2_residual_delta`: 21 running
  - `cp_pred_levels_L4_residual_delta`: 63 running
  - `cp_pred_surprise_L2_residual_delta`: 21 running
  - `cp_pred_surprise_L4_residual_delta`: 63 running
- No immediate tracebacks/errors were detected in `residual_delta_*.log` files
  at startup verification.

### Interpretation
- Operationally, the relaunch is healthy: compute utilization is high, safety
  controls are active, and the corrected objective is now what the cluster is
  optimizing.
- Next meaningful checkpoint is first completed residual-delta trials, then
  top-k holdout evaluation using residual-delta-selected configurations.

---

## 2026-04-30 (afternoon) — CP objective corrected to residual improvement

### Root-cause diagnosis
- A major red flag appeared: many CP Optuna trials had identical CV/test R², and
  top-3 `L=2` test-set runs returned the same CP test score across different CP
  hyperparameters.
- Confirmed numerically that the old Optuna objective was effectively selecting
  FE-only behavior:
  - FE-only CV means exactly matched prior Optuna best scores:
    - `L=2`: `0.7318296596109586`
    - `L=4`: `0.7336781355701287`
  - FE-only test R² exactly matched the CP `L=2` top-3 test result:
    - `0.7253239306500501` (reported as `0.725324`)
- Interpretation: under the old pooled objective, CP could "win" by shrinking the
  residual component toward zero and reproducing only firm-feature fixed effects.

### Method change (critical)
- Changed distributed Optuna worker objective from pooled R² to **incremental
  residual value over FE baseline**:
  - new objective name: `residual_delta`
  - per-fold score:
    - `delta = R²(FE + CP residual) - R²(FE-only)`
  - study direction remains maximize.
- This forces CP to earn positive score only when it improves beyond FE means.

### Anti-duplication improvements
- Added objective-specific study/journal namespace in `worker.py`:
  - journal suffix: `_residual_delta`
  - study suffix: `_residual_delta`
- Added worker-specific sampler seeds (instead of global fixed seed per worker
  process) and enabled `TPESampler(..., constant_liar=True)` to reduce duplicate
  suggestions under heavy distributed concurrency.

### Ridge fairness changes captured in code
- Added CP-matched zero-filled Ridge baseline (same target handling style as
  current TensorLy CP residual training):
  - `Code for paper/CP_struct_test_new.py`
  - propagated into:
    - `Code for paper/prediction_new/evaluate_l2_top3_test.py`
    - `Code for paper/prediction_new/imputer_sensitivity.py`
- Baselines for `L=2`:
  - `LEVELS`: `0.764757`
  - `SURPRISE`: `0.755351`

### Relaunch execution
- Stopped old pooled-R² `worker.py` processes across active hosts.
- Relaunched distributed search with `--objective residual_delta`,
  24-hour budget (`--time-budget-s 86400`), and watchdog coverage.
- Launch layout per host: 8 workers
  - `LEVELS L=4`: 3
  - `SURPRISE L=4`: 3
  - `LEVELS L=2`: 1
  - `SURPRISE L=2`: 1
- Active hosts in this relaunch: 21.
- Immediate post-launch monitor:
  - `cp_pred_levels_L2_residual_delta`: 21 running
  - `cp_pred_levels_L4_residual_delta`: 63 running
  - `cp_pred_surprise_L2_residual_delta`: 21 running
  - `cp_pred_surprise_L4_residual_delta`: 63 running
- No immediate tracebacks in residual-delta worker logs at startup.

### Ongoing note
- The previous top-3 `L=2` test-set diagnostic finished and was not favorable to
  CP on pooled R²; this is now treated as evidence that the old objective was
  misaligned for model selection rather than the final word on CP signal.
- Going forward, model selection should use residual-delta studies first, then
  evaluate pooled/per-feature test behavior of those selected CP models.

---

## 2026-04-30 — Prediction search rebalance and Ridge fairness audit

### Distributed Optuna status
- The overnight CPRegressor Optuna search made substantial progress but confirmed
  that `L=4` trials are much slower than `L=2` trials.
- Morning snapshot before rebalancing:
  - `LEVELS L=2`: 88 complete, 36 running
  - `LEVELS L=4`: 9 complete, 39 running
  - `SURPRISE L=2`: 51 complete, 38 running
  - `SURPRISE L=4`: 9 complete, 39 running
- Rebalanced new capacity toward `L=4` only. Added/converted:
  - `utmlab20-05`
  - `utmlab20-23`
  - `utmlab20-29`
  - `utmlab26-02`
- Each of those hosts runs 12 workers: 6 for `LEVELS L=4` and 6 for
  `SURPRISE L=4`, with watchdog coverage and the new 24-hour budget
  (`--time-budget-s 86400`).
- Later snapshot showed `L=4` catching up:
  - `LEVELS L=2`: 98 complete, 26 running
  - `LEVELS L=4`: 29 complete, 43 running
  - `SURPRISE L=2`: 51 complete, 38 running
  - `SURPRISE L=4`: 29 complete, 42 running

### Interim L=2 test-set diagnostic
- Added `Code for paper/prediction_new/evaluate_l2_top3_test.py` to evaluate the
  top completed `L=2` Optuna CP trials on the held-out test set.
- The diagnostic selects the top 3 completed trials for each `L=2` study, fits
  each CP model on the 80% development block, and scores on the 20% hold-out
  test block.
- First completed CP test-set fit:
  - `LEVELS L=2`, trial `0`
  - CV R²: `0.7318296596109586`
  - test CP R²: `0.725324`
  - original Ridge test R²: `0.766939`
  - delta: `-0.041615`
- The remaining five top-3 `L=2` CP test fits were still running at the last
  check. This is an interim signal only; do not over-interpret one `L=2` result.

### Ridge comparison audit
- The original Ridge baseline was a strong observed-label, per-feature Ridge:
  each target feature selected its own `alpha` and Ridge trained only on observed
  target cells.
- A global-alpha Ridge check barely changed test R²:
  - `LEVELS L=2`: per-feature Ridge `0.766939`; global-alpha Ridge `0.767397`
  - `SURPRISE L=2`: per-feature Ridge `0.764074`; global-alpha Ridge `0.763794`
- This suggests Ridge's headline performance is mostly coming from the
  firm-feature fixed effects / persistence structure rather than per-feature
  alpha tuning.
- More importantly, TensorLy `CPRegressor` is not mask-aware for targets. CP is
  trained on firm-feature demeaned residual tensors where missing target
  residuals are zero-filled. Observed-label Ridge therefore had an advantage by
  ignoring missing target cells during fitting.
- For apples-to-apples comparison with current CP, changed Ridge to a
  CP-matched zero-filled residual baseline:
  - compute firm-feature fixed effects,
  - center targets,
  - set missing residuals to zero,
  - fit Ridge on all rows,
  - evaluate only on observed test cells.
- Updated scripts:
  - `Code for paper/CP_struct_test_new.py`
  - `Code for paper/prediction_new/evaluate_l2_top3_test.py`
  - `Code for paper/prediction_new/imputer_sensitivity.py`
- CP-matched Ridge `L=2` baselines:
  - `LEVELS L=2`: `0.764757`
  - `SURPRISE L=2`: `0.755351`
- These are only slightly lower than observed-label Ridge, reinforcing that the
  major baseline strength is persistence/FE rather than mask handling alone.

### Interpretation
- Aggregate pooled R² can hide the part of the story where CP is useful.
  Earlier diagnostics showed CP-minus-Ridge per-feature delta R² was positively
  related to lower persistence features. That remains the key hypothesis:
  Ridge/FE dominates highly persistent features, while CP may add value for
  less persistent features where historical firm-feature averages are weaker.
- A future methodological extension would be a truly target-mask-aware CP
  regression objective. That is likely a separate project; for the current paper
  the immediate goal is an apples-to-apples comparison using CP-matched Ridge,
  plus per-feature persistence/delta analysis and `L=4`/ensemble results.

---

## 2026-04-29 (night) — Distributed Optuna CP search scaled to 16 lab machines

### Context
- After correcting the prediction-window off-by-one bug and rebuilding the
  prediction caches, the flawed Optuna journals were cleared and the CPRegressor
  hyperparameter search was relaunched from clean shared journal files.
- The four active studies are:
  - `cp_pred_levels_L2`
  - `cp_pred_levels_L4`
  - `cp_pred_surprise_L2`
  - `cp_pred_surprise_L4`
- All studies use shared Optuna `JournalStorage` files under
  `Code for paper/prediction_new/optuna_journal`, so workers on different lab
  computers coordinate trial reservations through NFS.

### Actions
- Scaled the run to 16 computers:
  - local/controller: `dh2010pc08`
  - `utmlab10-02`, `utmlab10-03`, `utmlab10-05`, `utmlab10-07`,
    `utmlab10-11`, `utmlab10-19`, `utmlab10-23`, `utmlab10-29`
  - `utmlab20-03`, `utmlab20-05`, `utmlab20-07`, `utmlab20-11`,
    `utmlab20-13`, `utmlab20-17`, `utmlab20-19`
- Targeted roughly 12 Optuna workers per host, split evenly across the four
  studies. The latest journal check showed 47 running trials per study
  (188 active running trials total).
- Confirmed every active host has a `cp_sweep_watchdog.py` supervising
  `worker.py` with:
  - `--system-limit-gb 53`
  - `--process-pct 85`
  - `--cascade --cascade-match worker.py`
  - `--cleanup-margin-gb 5`

### First result
- The first completed trial landed in `LEVELS`, `L=2`:
  - trial `0`
  - CV R²: `0.7318296596109586`
  - runtime: approximately `74.9` minutes
  - params: `RANK_REGRESS=8`, `REG_W=98.1655`, `USE_RMS_SCALING=True`
- Other studies had not completed a trial yet at the last check, but workers
  were actively consuming CPU and no tracebacks/errors were found in the logs.

### Operational note
- Tonight's workers were launched with `--time-budget-s 36000` (10 hours), so
  they should stop automatically tomorrow morning in staggered waves.
- For future overnight/full-day runs, use a 24-hour worker budget instead:

```bash
--time-budget-s 86400
```

- The high `--n-trials 10000` cap is effectively just a safety ceiling; the wall
  clock budget is the real stop condition.

---

## 2026-04-29 (afternoon) — SSH agent unlocked cross-lab automation

### Context
- Earlier distributed runs depended on manually logging into each lab host once
  and reusing that host-specific ControlMaster socket. That worked, but it did
  not scale well because every new machine required a password login.
- The existing `~/.ssh/id_ed25519` key had an unknown passphrase, so it could not
  be loaded into an SSH agent for unattended fan-out.

### Actions
- Created a new lab automation key: `~/.ssh/id_ed25519_lab_auto`.
- Added its public key to `~/.ssh/authorized_keys` on the shared lab home.
- Started an SSH agent on a stable socket and loaded the new key:

```bash
ssh-agent -a ~/.ssh/lab-agent.sock > ~/.ssh/lab-agent.env
source ~/.ssh/lab-agent.env
ssh-add ~/.ssh/id_ed25519_lab_auto
```

- Updated `/student/mcnama53/.ssh/config` for `Host utmlab*` to use:
  - `IdentityFile ~/.ssh/id_ed25519_lab_auto`
  - `IdentityAgent ~/.ssh/lab-agent.sock`
  - `IdentitiesOnly yes`
- Extended `ControlPersist` from `10m` to `8h` so successful connections remain
  reusable during a work session.

### Validation
- Confirmed key/agent auth works with ControlMaster disabled:
  `ssh -F ~/.ssh/config -o BatchMode=yes -o ControlMaster=no -o ControlPath=none utmlab10-02 hostname`
  returned `dh2010pc02`.
- Confirmed the same for a host that had not been manually unlocked:
  `utmlab20-03` returned `dh2020pc03`.

### Operational note
- To re-enable automated lab fan-out after a reboot/logout or dead agent, run:

```bash
lab-agent
```

- This helper lives at `~/bin/lab-agent` and wraps the longer setup:

```bash
ssh-agent -a ~/.ssh/lab-agent.sock > ~/.ssh/lab-agent.env
source ~/.ssh/lab-agent.env
ssh-add ~/.ssh/id_ed25519_lab_auto
```

- After this one unlock on the controller lab machine, distributed launchers can
  SSH to other lab hosts using key/agent auth. The agent does not expose the key
  passphrase to scripts; it only signs SSH authentication challenges.

---

## 2026-04-29 (afternoon) — Generic distributed launcher added

### Context
- Prediction sensitivity tests and later Optuna/CP runs will need to fan out
  across lab computers. SSH from the current lab host (`dh2010pc08`) to other
  lab hosts requires either key auth or one manual password login with
  ControlMaster reuse.
- The user manually opened `ssh utmlab20-02`; after password authentication,
  noninteractive `ssh -o BatchMode=yes utmlab20-02 ...` works through the
  persisted ControlMaster connection.

### Actions
- Added `Code for paper/distributed_launcher.py`, a reusable SSH/NFS launcher
  for arbitrary commands.
- The launcher supports:
  - comma-separated host lists,
  - one command per host or a command list from `--job-file`,
  - per-job log files under `Code for paper/distributed_logs`,
  - `--dry-run` inspection,
  - host-local watchdog startup before worker launch.
- Installed matching host aliases in `/student/mcnama53/.ssh/config` so the lab
  account can use names like `utmlab20-02`.

### Watchdog policy
- Default RAM kill switch is **85% of host MemTotal**. On a 62.4 GB lab host,
  this becomes approximately 53 GB.
- Watchdogs use cascade mode but are scoped by `--cascade-match` to our worker
  substring (for example `imputer_sensitivity.py`) so they kill our runaway
  workers rather than other users' jobs.
- The cleanup margin is 5 GB by default, so a host over the limit is cleaned
  down to roughly 48 GB used on a 62.4 GB machine.

### Status
- Dry-run against `utmlab20-02` succeeded and produced the expected watchdog and
  job launch commands.
- A low-stakes distributed sensitivity smoke test is running:
  - local `dh2010pc08`: `validated_cv`, `LEVELS`, `L=2`
  - remote `dh2020pc02`: `hidden_best`, `LEVELS`, `L=2`

---

## 2026-04-29 (afternoon) — Sweep reliability audit and distributed re-run plan

### Context
- We are no longer working with the old fundamentals tensor. The live pipeline now
  uses `Code for paper/90-25_Q_Fundamentals_v2.csv`, 40 distinct feature specs,
  YTD cash-flow items differenced to quarterly flows, `aqcy` for acquisitions,
  and `ibadj12` as the non-duplicate annual/TTM income feature.
- A long Tucker grid sweep was started on the rebuilt MFI tensor with
  `Sweep_Tucker_Ranks.py --max-iter 50 --tol 1e-4 --init random` behavior. It
  was stopped after 386/576 rank combinations because high-rank errors increased
  with rank, which violates the monotonicity expected of the optimum.

### Findings
- The error formula itself matches the historical scripts' relative Frobenius
  error, adapted to observed entries:
  `||M * (X_hat - X)||_F / ||M * X||_F`.
- Historical dense scripts:
  `Code for paper/Decomposition_firms.py` and
  `Code for paper/Decompositions_FF.py` used
  `||X_hat - X||_F / ||X||_F`.
- Current sparse scripts:
  `Code for paper/Sweep_Tucker_Ranks.py` and
  `Code for paper/Build_PrePrediction_Exhibits.py` use the masked analogue.
- The failed Tucker grid is therefore most likely a solver-protocol problem, not
  a metric-definition problem: single random initialization, only 50 iterations,
  and no restarts.
- Example bad row from the stopped sweep: `(r1, r2, r3) = (67, 20, 20)` produced
  observed relative error `6.561361`, while larger ranks became even worse. That
  is not credible as an optimized reconstruction result.
- The CP rank sweep is also only preliminary. Its curve is broadly decreasing
  (rank 1: `0.537`, rank 99: `0.0488`) but uses one random initialization and no
  restart audit, so exact values and exact best rank should not be treated as
  paper-ready.

### Plan
- Preserve the stopped Tucker output for debugging:
  `Code for paper/pre_prediction_cache/tucker_rank_grid_sweep.csv` should be
  copied to `tucker_rank_grid_sweep_random50_bad.csv` before any new Tucker
  sweep overwrites it.
- Run a focused sanity check on `(67, 20, 20)`:
  `init="svd"`, `n_iter_max=500`, `tol=1e-6`, `verbose=True`.
- Compare TensorLy's printed reconstruction error to our observed-entry metric.
  If they agree and the error drops materially, the metric is vindicated and the
  earlier sweep settings were the bug.
- Re-run robust Tucker and CP sweeps using the lab machines. RAM is not the
  constraint; CPU time and restart coverage are. Use more iterations
  (`500` or `1000`), stricter tolerance (`1e-6` or `1e-7`), and multiple seeds.
- A single SVD-initialized sanity check can saturate one machine (observed
  ~2300 % CPU on the 24-core lab host), but the production sweep should still
  parallelize at the independent-job level. Cap each worker process to a modest
  BLAS thread count (for example 4 threads), run several worker processes per
  host, and distribute rank/seed jobs across lab machines.
- Always launch long CPU sweeps under a watchdog, even when memory appears low
  in pilot runs. The current tensor jobs are light on RAM (~2--3 GB used during
  the sanity audit on a 62 GB host), but distributed workers can pile up if a
  job hangs, BLAS thread counts are mis-set, or multiple users share the same
  machine. Use `Code for paper/cp_sweep_watchdog.py` or a small variant around
  each host-level launcher to enforce memory caps and cleanly kill runaway
  worker cascades.
- Distribute work by independent rank/seed jobs over the shared NFS home
  directory. Each worker should claim one job atomically, write one result file,
  and avoid concurrent writes to a single CSV. Aggregate after all workers
  finish.
- For each rank, report the best error across restarts plus mean/std across
  restarts. Plot both raw best error and cumulative-best error, since the
  optimizer's returned error can be non-monotone even though the true optimum is
  non-increasing with rank.

### Next
- Build a small `tucker_sanity_check.py` first, rather than relaunching the full
  grid blindly.
- If the sanity check passes, create distributed worker scripts for robust CP and
  Tucker sweeps, reusing the same lab-computer/NFS pattern planned for the
  prediction Optuna search.

---

## 2026-04-29 (afternoon) — Prediction imputer fixed and rank-CV'd

### Context
- The revamped prediction pipeline uses the v2 fundamentals file, 40 feature
  specs, and a dynamic top-50 market-cap universe at 2024Q4.
- The initial cache builder was replacing the entire input window with the
  Tucker reconstruction. That is not pure imputation: it also smooths observed
  reported accounting values.

### Decisions
- **Observed cells are preserved exactly.** `prediction_new/build_prediction_caches.py`
  now uses Tucker only to fill NaN cells. Observed cells pass through unchanged
  in both LEVELS and SURPRISE modes.
- **Tucker imputer uses SVD init.** The MFI Tucker audit showed random init can
  be badly under-optimized, while SVD init is stable and memory-safe on these
  tensors. The prediction-window tensors are tiny (`50 x 40 x L`), so SVD init
  is safe here too.
- **Rank selection is based on hidden observed cells.**
  `prediction_new/sweep_imputer_ranks_cv.py` hides 10% of observed entries,
  stratified by feature within each rolling window, fits mask-aware Tucker, and
  scores only the hidden cells. This directly validates the imputer's job.
- **Selected imputation ranks:**
  - `L=2`: `[2, 2, 2]`. Best mean holdout error was `[3, 3, 2]` at `0.4349`,
    but `[2, 2, 2]` was essentially tied at `0.4354` and is the parsimonious
    one-standard-error choice.
  - `L=4`: `[4, 4, 4]`. Best mean holdout error and one-standard-error choice:
    `0.4025`.

### Actions
- Added `prediction_new/sweep_imputer_ranks_cv.py`.
- Updated `prediction_new/prediction_config.py`:
  `IMPUTATION_RANKS = {2: [2, 2, 2], 4: [4, 4, 4]}`.
- Updated `prediction_new/build_prediction_caches.py` to use
  `init="svd"`, `n_iter_max=100`, `tol=1e-5`, and observed-cell preservation.
- Rebuilt prediction caches:
  - `tensor_levels_L2.pkl`: avg observed reconstruction error `0.4050`
  - `tensor_levels_L4.pkl`: avg observed reconstruction error `0.3630`
  - `tensor_surprise_L2.pkl`: avg observed reconstruction error `0.4050`
  - `tensor_surprise_L4.pkl`: avg observed reconstruction error `0.3630`
  - all four had `0` Tucker failures.

### Result files
- `prediction_new/sweep_results/imputer_rank_cv_stratified.csv`
- `prediction_new/sweep_results/imputer_rank_cv_stratified_summary.csv`
- `prediction_new/logs/imputer_rank_cv_stratified.log`
- `prediction_new/logs/build_caches.log`

---

## 2026-04-29 (late afternoon) — 40-feature spec finalized; pre_prediction_config aligned to paper

### Context
Live `FEATURE_SPECS` in `Code for paper/pre_prediction_config.py` had 39 entries
vs the paper's 40 (Table "Selected fundamental features used for constructing
the design matrix", `Paper_Draft/main.tex` lines 357–388 — 20 rows × 2 cols).
Working from a stale `pre_prediction_cache/audit/audit_summary.json` was
sending us down dead ends (it suggested multi-column priority lists with
collision risk that don't exist in the live spec). Deleted that stale audit
and re-derived everything directly from `Paper_Draft/main.tex` and the v2
panel.

### Decisions
- **Source of truth = paper table + v2 CSV; no audit summary.** Removed
  `pre_prediction_cache/audit/{audit_summary.json,audit_summary.md,
  feature_periodicity_audit_summary.json}` so future work doesn't anchor on
  outdated priority lists.
- **Empirical reconciliation of "Quarterly" vs "Annual" Income Before
  Extra Items.** Within-fiscal-year YTD differencing of `iby` recovers `ibq`
  in **99.91 %** of 53,781 firm-quarters (exact match, $\rho = 1.000$).
  Year-end `iby` equals $\sum \mathit{ibq}_{Q1..Q4}$ in **99.84 %** of 13,003
  firm-fiscal-years. Same algebra for `ibadj12 ↔ ibadjq`. Conclusion:
  **`iby` is mechanically just YTD-cumulated `ibq`** (same income-statement
  line item, two storage conventions). Therefore using `iby` with a
  `ytd_to_quarterly` transform for "Annual IB Ex Items" would collapse the
  feature into a literal duplicate of "Quarterly IB Ex Items".
- **40th feature: `ibadj12` (TTM, CSE-adjusted).** Correlation 0.78 with
  `ibq`, 0 % exact-match — genuinely distinct signal (per-quarter level vs
  trailing-12-month aggregate). `ibcy` and `niq` were also considered;
  `niq` (correlation 0.99, 82 % exact match with `ibq`) is too close, and
  `ibcy` (cash-flow-statement reconciliation) doesn't fit the "Annual" label.
- **Acquisitions source-column fix.** Live spec had `aqaq` (Acquisitions —
  Income Contribution, P&L line). v2 panel density: 17.5 %, and
  $\rho(\mathit{aqaq}, \mathit{aqcy}) = -0.16$ — completely different concept
  from acquisitions cash outflow. Switched to `aqcy` with
  `transform="ytd_to_quarterly"` (87.3 % dense, the cash-flow-statement
  line that "Acquisitions" naturally references).
- **No exact-line-item duplicates remain.** All 40 features now map to 40
  distinct Compustat columns. No collisions, no shared fallbacks. `ibq` and
  `ibadj12` are the closest pair (same accounting concept, different
  aggregation windows + CSE adjustment) and meet the user's
  "not the same exact line item" bar.

### Actions
- `Code for paper/pre_prediction_config.py`:
  - `LOCAL_FUNDAMENTALS_FILE` → `90-25_Q_Fundamentals_v2.csv`
  - `Acquisitions`: `("aqaq",)` → `("aqcy",)`, transform `as_reported` →
    `ytd_to_quarterly`
  - Inserted `FeatureSpec("Annual Income Before Extraordinary Items",
    ("ibadj12",))` between `Quarterly IB Ex Items` and `Income Taxes`
    (matches paper table row order)
- Deleted stale audit summaries (see Decisions).
- Verified post-edit: `len(FEATURE_SPECS) == 40`, set-equal to the paper
  labels, `LOCAL_FUNDAMENTALS_FILE` resolves to an existing 103 MB CSV.

### Density profile of the new 40-feature spec (against v2 panel, 59,037 rows)
- **≥ 90 %** (28 features): aoq, capxy, cheq, dvy, ceqq, cogsq, dlcq, epspxq,
  epsfxq, xidoq, fincfy, fopoy, ibq, txtq, invtq, ivncfy, ivacoy, dlttq,
  dltisy, nopiq, oancfy, pstkq, piq, rectq, sstky, saleq, spiq, seqq.
- **70 – 90 %** (5): aqcy, oibdpq, sivy, sppivy, intanq, mibtq.
- **60 – 70 %** (2): txbcofy, ibadj12.
- **45 – 60 %** (4): anoq, ciq, lnoq, ivstq.
- The bottom four are genuinely sparse Compustat fields; Tucker imputation
  is expected to absorb the missingness.

### Next
- Invalidate downstream caches before rebuilding:
  `pre_prediction_cache/fundamentals_panel_40_features.csv` (was built from
  the 39-feature pre-v2 spec) and `pre_prediction_cache/cp_relative_error.csv`
  (CP curve from the contaminated tensor). Keep raw v2 CSV.
- Rebuild panel + tensor; expected shape `(499, 40, 140)`.
- Tucker rank sweep under the watchdog, target observed relative error
  < 15 %.
- Then revisit `Sweep_Tucker_Ranks.py` / `Optimize_Tucker_Ranks.py`
  imports (the long-standing bug where they reference functions no longer
  exposed by `Build_PrePrediction_Exhibits.py`).

### Open questions
- Paper omits `atq` (Total Assets, 97.5 % dense) and `ltq` (Total
  Liabilities, 97.5 % dense) despite including their components. Worth
  flagging to Masoud as a possible future addition — but stays out of the
  current 40-feature spec.

### Watch-outs
- The paper table itself wasn't edited; it already had all 40 features in
  the correct order. Edits were only to our spec.
- Don't trust `pre_prediction_cache/audit/*` derived artifacts that survived
  this session (`feature_density.csv`, `feature_periodicity_audit.csv`,
  `feature_rule_table.csv`, `gics_density.csv`, etc.) — they were generated
  from the pre-v2, 39-feature spec and will need a full rebuild after the
  next tensor build.

---

## 2026-04-29 (afternoon) — Full comp.fundq pulled; v2 fundamentals CSV ready

### Context
Continued from morning entry below. After confirming via WRDS schema discovery
that the 15 missing YTD cash-flow columns *do* exist in `comp.fundq`, pulled
a full-history snapshot for our existing 499-gvkey S&P 500 universe.

### Decisions
- **Universe**: stick with the existing 499 gvkeys in
  `Code for paper/gvkeys_to_gics.csv` (a snapshot from Compustat Daily Updates
  - Index Constituents - S&P Current). No re-fetch of constituents at this
  point.
- **Filter**: standard Compustat dedupe filter
  `consol='C' AND popsrc='D' AND datafmt='STD' AND indfmt='INDL'`. All 499
  gvkeys come back as `INDL` rows — financial-firm `FS` filter loss is not
  an issue for this universe.
- **Transport**: `psycopg2` direct, not the `wrds` Python wrapper (which
  prompts interactively even when given `wrds_username=...`). One Duo push
  per fresh `psycopg2.connect`.

### Actions
- `Code for paper/wrds_q4_snapshot.py` — quick Q4-2024 sanity probe
  (499 rows × 648 cols, all 15 YTD columns 82–100 % populated). Read-only,
  no file written.
- `Code for paper/fetch_fundamentals_wrds.py` — full pull script with
  `--no-confirm`, `--start-date`, `--end-date`, `--output`, `--gvkey-source`
  flags and required-column density assertions before any file write.
- Pulled 1990-01-01 → 2024-12-31, all 648 columns, into
  `Code for paper/90-25_Q_Fundamentals_v2.csv` (103 MB, 59,037 rows,
  499 gvkeys, ~12 s of wire time at ~5 k rows/s). All 15 previously
  empty YTD cash-flow columns populated at 84–94 % density across the
  full panel.
- Verified file integrity by re-reading with both pandas C and python
  engines — 59,037 × 648 in both; the original-CSV (`90-25_Q_Fundamentals.csv`,
  87 MB, 378 columns) is left untouched.

### Next
- Update `pre_prediction_config.py` `LOCAL_FUNDAMENTALS_FILE` to point at
  the v2 CSV.
- Drop the now-unnecessary `transform="ytd_to_quarterly"` step from
  `FEATURE_SPECS` for the 15 cash-flow features (per the "no data
  engineering" preference: use raw YTD values; trust Tucker to absorb the
  cumulation structure).
- Rebuild the firm-level panel + tensor; expect observed density to jump
  from ~50 % → ~70-75 % since the previously-empty slabs become real data.
- Run a small Tucker rank sweep targeting < 15 % observed relative error
  under the (now patched) watchdog.

### Open questions
- After the Tucker sweep settles on a rank: re-run the prediction code
  paths? (Deferred per user.)
- Re-add the "Annual Income Before Extraordinary Items" feature
  (`iby` / `ibadj12`) to bring the feature count from 39 back to the audit's
  40? Decide before the Tucker sweep so the tensor shape is final.

### Watch-outs
- Don't read the v2 CSV while a write is still in progress — pandas C
  engine may silently return a partial frame without error. Prefer
  reading after the python process has confirmed exit.
- The SQL warning about `pd.read_sql_query` wanting an SQLAlchemy
  connectable instead of a raw psycopg2 connection is benign for now;
  ignore unless we hit a future-pandas deprecation.

---

## 2026-04-29 (morning) — Pre-Prediction CP sweep finished; root-caused MFI data gap

### Context
- Picking up an interrupted run of `Code for paper/Build_PrePrediction_Exhibits.py`
  (CP rank sweep) under a memory watchdog.
- Pivoted mid-session to scoping the MFI build (target: Tucker observed
  relative error < 15 %).

### Decisions
- **Watchdog rewrite.** `Code for paper/cp_sweep_watchdog.py` is now a generic
  Python-job RAM watchdog. CLI-driven (`--match`, `--user`, `--interval`,
  `--system-limit-gb`, `--process-pct`, `--cascade`, `--cascade-match`,
  `--cleanup-margin-gb`, `--grace-seconds`, `--start-timeout`,
  `--tmux-session`). Polls `/proc` directly so 5 s polling is essentially free
  (~20 ms / cycle, ~20 MB resident). With `--cascade`, on a system-RAM breach
  it kills the user's python processes by RSS desc until system used RAM is
  back below `system_limit_gb - cleanup_margin_gb`. Bug fix during session:
  the matched target is now always a candidate regardless of
  `--cascade-match` (previously a too-narrow `--cascade-match` could exclude
  the target itself).
- **CP rank-40 OOM root cause.** tensorly's default `init='svd'` for `parafac`
  with `mask=...` runs `svd_mask_repeats=5` rounds of SVD-then-impute, each
  allocating a full float64 working tensor. For our 500 × 39 × 140 tensor
  this was fine through rank 39 (= the features mode size) but blew past
  57 GB RSS at rank 40 because `partial_svd` on the 39-row mode-1 unfolding
  fell into a degraded path once it was asked for more components than rows.
- **CP fix: `init='random', random_state=SEED`.** Single-line change to
  `cp_error()` in `Build_PrePrediction_Exhibits.py`. Memory dropped from
  ~37 GB stable / 57 GB at rank 40 → **~0.4 GB stable through rank 100**.
  Time per rank also halved. Rank-40 error with random-init (0.0668) is
  marginally *better* than rank-39 with svd-init (0.0672) — no visible
  discontinuity at the methodology boundary, so the cached ranks 1–39 from
  the prior svd run can stay.
- **MFI data root cause: 12 of 39 features are 0.00 % dense.** Every YTD
  cash-flow source column referenced by `FEATURE_SPECS` is missing from the
  local `Code for paper/90-25_Q_Fundamentals.csv` extract. Confirmed
  programmatically:
  `capxy, dvy, oancfy, fincfy, ivncfy, ivacoy, dltisy, txbcofy, fopoy, sstky,
  sivy, sppivy, aolochy, aqcy, ibcy`. The audit's 22.20 % Tucker error and
  the just-completed CP curve are both polluted by ~30 % structurally empty
  tensor slabs.
- **WRDS confirmation (read-only).** `Code for paper/wrds_inspect_fundq.py`
  hits `comp.fundq` directly via `psycopg2` (the `wrds` Python wrapper
  prompts interactively even with `wrds_username=...`, hence the bypass).
  `comp.fundq` has 648 columns vs our local 378. All 15 missing YTD columns
  exist; **no quarterly siblings exist for any cash-flow item** (Compustat
  reports them YTD-only by design). The 3 already-q-substituted features
  (`iby→ibq`, `epsfxy→epsfxq`, `epspxy→epspxq`) are correct.

### Actions
- `Code for paper/cp_sweep_watchdog.py` — generalized watchdog (rewritten).
- `Code for paper/Build_PrePrediction_Exhibits.py` — added
  `init="random", random_state=SEED` to the `parafac` call; added `SEED` to
  the import list.
- Re-ran the CP sweep under watchdog. All 100 ranks completed. Final cache:
  `Code for paper/pre_prediction_cache/cp_relative_error.csv` (rank 100
  error = 0.0362). Plot: `Paper_Draft/Figures/Fundamentals/Relative_Error.eps`
  + `.pdf`.
- `Code for paper/wrds_inspect_fundq.py` — created.

### Next
- Pull a complete `comp.fundq` snapshot (all 648 columns) for the existing
  gvkey universe, 1990-01-01 → 2024-12-31, with the standard
  `consol='C', popsrc='D', datafmt='STD', indfmt='INDL'` filter. Save to
  `Code for paper/90-25_Q_Fundamentals_v2.csv` so the original isn't
  clobbered until verified.
- Rebuild the firm-level fundamentals panel and tensor with raw values
  ("no feature engineering" per current preference): the 15 cash-flow
  features become raw YTD readings, density jumps from 50 % → ~70 %.
- Run a small Tucker rank sweep targeting < 15 % observed relative error
  under the watchdog.
- **Bug**: `Sweep_Tucker_Ranks.py` and `Optimize_Tucker_Ranks.py` import
  `build_fundamentals_panel` / `build_fundamentals_tensor` from
  `Build_PrePrediction_Exhibits.py` — those names no longer exist (the
  current build script exposes only `load_raw_fundamentals` /
  `build_tensor` and returns `(tensor, mask)` rather than
  `(tensor, mask, firms, quarters, _)`). Fix when the new tensor pipeline
  lands.

### Open questions
- Should the missing 40th feature (audit's "Annual Income Before
  Extraordinary Items" using `iby` / `ibadj12`) be re-added now that we
  have `comp.fundq`? It would coexist with the existing
  "Quarterly Income Before Extraordinary Items" (using `ibq`).
- After the v2 tensor is built and Tucker rank chosen: re-run the prediction
  pipeline? (Flagged as a downstream concern; deferred for now.)

### Watch-outs
- WRDS requires a Duo push approval per fresh `psycopg2.connect`. Bulk pulls
  should keep one connection open and run all queries in it.
- `Code for paper/pre_prediction_cache/Build_PrePrediction_Exhibits.lock`
  is a stale Apr 26 file; nothing currently reads it. Safe to ignore.
- Cursor shell sandbox blocks DNS for non-allowlisted hosts. WRDS calls need
  `required_permissions: ["all"]` (or `["full_network"]` plus a valid
  resolved address) to reach `wrds-pgdata.wharton.upenn.edu`.
