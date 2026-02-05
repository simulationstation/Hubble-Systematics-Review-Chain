# Layman summary (current status)

This repo is a **late‑time systematics audit chain** for the Hubble tension.
It treats an **early‑time ΛCDM background as a fixed “anchor”** (e.g. `H0=67.4`) and asks:
“What *systematic distortions* would late‑time pipelines need to be consistent with that anchor?”

## Smoke tests vs real tests

- **Smoke / code-correctness tests:** `pytest` unit tests. These only check that our shuffles/nulls and core math behave as intended.
- **Real-data fits:** baseline fits and cut scans on the public **Pantheon+** and **Pantheon+SH0ES** (calibrators + Hubble‑flow) tables in `data/raw/…`.
- **Internal stress tests (not new data):** permutation nulls, correlated-cut Monte Carlo drift nulls, injection/recovery, and SBC. These are “is the pipeline fooling itself?” tests.

## The headline “tension reproduction” (real data)

On the **Pantheon+SH0ES ladder subset** (calibrators + Hubble‑flow SNe, `zHD`, `0.023≤z≤0.15`):

- Baseline reproduces an effective ladder scale `H0_eff ≈ 73.55 ± 1.08`.
- That is ~**9% higher** than the early‑time anchor `67.4`.
- In distance-modulus terms, the required shift is about **0.19 mag**.

See: `outputs/pantheon_plus_shoes_ladder_demo_v3/report.md`.

## Other late-time probes (sanity context; real data)

As a cross-check, we also ran a simple **stack** of late-time probes that do *not* include the SH0ES
calibrator step: Pantheon+ (SN only) + DESI 2024 BAO + chronometers. That stack prefers
`H0 ≈ 68.21 ± 0.31`, i.e. close to the early-time anchor and far from the SH0ES ladder value.

See:
- `outputs/stack_sn_bao_cc_stress_v2/report.md`

## Independent siren probe (real data; with metadata cuts)

We added a **selection-corrected Gate‑2 dark-siren** adapter that ingests per-event `logL(H0)` curves
on a common grid (independent of SN/BAO/CC) and enables the same cut/time drift audits using event
QC metrics (`ess_min`, `n_good_min`) and event time parsed from the event name.

**Important:** you noted the upstream Gate‑2 product is **not complete yet**, so treat these results
as *experimental plumbing checks*, not as a mature physics constraint.

- Baseline (moment summary of the selection-corrected posterior): `H0_eff ≈ 60.84 ± 12.74` (broad).  
  Cut scan over `ess_min` shows large variation once `N` becomes small. Under a siren-specific
  correlated-cut null (event-level Gaussian surrogate), the observed drift across the ESS scan is
  **unlikely** (max-pair p≈`2e-5`; path-length p≈`0` in 50k sims). A time-bin shuffle null shows no
  significant time drift (p≈0.58–0.62).

See:
- `outputs/siren_gate2_grid_audit_v1/report.md` (legacy linear-Gaussian null)
- `outputs/siren_gate2_grid_audit_v2/report.md` (siren-specific null)

## What we checked (real‑data stability tests)

These are “does the inferred ladder scale drift when we change cuts/epochs?” tests.

- **Time/epoch stability (shuffle null; fixed-support calibrators):** no strong evidence of epoch‑dependent `H0` in this dataset at current sensitivity.  
See: `outputs/pantheon_plus_shoes_ladder_time_invariance_null_fixedcal_v3/report.md`.
- **Per‑survey time drift (grouped null):** no individual high‑N survey shows a small p‑value for drift.  
  See: `outputs/pantheon_plus_shoes_ladder_group_time_drift_v1/report.md`.
- **Quality cuts (Hubble‑flow only, calibrators always included):**
  - by measurement uncertainty (`m_b_corr_err_DIAG`): no unusual drift  
    `outputs/pantheon_plus_shoes_ladder_quality_cut_scan_v1/report.md`
  - by light-curve fit probability (`FITPROB`): no unusual drift  
    `outputs/pantheon_plus_shoes_ladder_fitprob_cut_scan_v2/report.md` (note: `..._v1` is an obsolete partial run)
  - by reduced chi2 (`FITCHI2/NDOF`): mild drift hints but not significant (p≈0.16–0.34)  
    `outputs/pantheon_plus_shoes_ladder_redchi2_cut_scan_v2/report.md`

## “How big would a systematic need to be?” (injection/recovery)

We injected simple distortion patterns and measured how strongly they bias the inferred `delta_lnH0`.
The key output is the **amplitude required to fake the full ladder offset**.

Examples (all in magnitudes, in our parameterization):

- Calibrator-vs-Hubble‑flow offset needs **~0.19 mag** (this is essentially the full H0 gap).
- A quality/selection proxy linear in `m_b_corr_err_DIAG` needs **~0.45 mag per 1σ(error)** (HF only).
- A calibrator-only proxy linear in `PKMJDERR` needs **~0.31 mag per 1σ(PKMJDERR)**.
- A calibrator-only proxy linear in `m_b_corr_err_VPEC` needs **~0.15 mag per 1σ(VPEC error)**.
- A proxy linear in `m_b_corr_err_RAW` would need **~5.0 mag per 1σ(RAW error)** (i.e. effectively impossible).
- A pure sky dipole needs **~0.40 mag**.
- A pure time trend needs **~0.68 mag per 1σ(time)**.
- A host-mass step needs **~2.35 mag**.

See: `outputs/pantheon_plus_shoes_ladder_injection_misspec_v4/report.md`.

## Joint “anchor-consistency” inference (real data)

If we **force** agreement with the early-time anchor using the *other* late-time probes
SN-only+BAO+chronometers, then the SH0ES-style ladder subset can only be made
compatible by introducing a large calibrator↔Hubble-flow offset:

- In a joint stack, the inferred `calibrator_offset_mag ≈ +0.162 ± 0.032` mag.
- That is roughly **~85% ± 17%** of the full **~0.19 mag** ladder gap (simple ratio; not a calibrated physical prior).

See:
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_v1/report.md`
- External-prior stress test: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_tight_v1/report.md`

In the tight-prior stress test (`sigma_calibrator_offset_mag=0.02`), the fit is forced toward a
much smaller offset (`calibrator_offset_mag ≈ +0.046 ± 0.017`), the combined `H0_eff` shifts slightly
upward (`≈ 68.57`), and the model evidence drops notably. This is a concrete “external prior gate”:
if external calibration work can truly bound the calibrator↔HF offset at the ~0.02 mag level, this
joint stack would no longer be able to reconcile the ladder subset without paying a strong
likelihood/evidence penalty.

A small prior-width sweep confirms this trend: once `sigma_calibrator_offset_mag` is pushed below
~0.05 mag, the log evidence falls quickly (ΔlogZ ≈ -2.3 at 0.05, ≈ -7.5 at 0.02, ≈ -9.9 at 0.01).
See: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_prior_sigma_sweep_v1/report.md`.

We repeated the same “external prior gate” idea for **per-survey** calibrator offsets and for
**per-epoch time-bin** offsets (both are more “instrument-like” than a single global step). Under a
tight `σ=0.02` mag bound:

- the largest per-survey calibrator offset is forced down to ~0.008 mag and the evidence drops
  by ΔlogZ ≈ -11.3,
- the largest calibrator time-bin offset is forced down to ~0.021 mag and the evidence drops by
  ΔlogZ ≈ -10.1,

while `delta_lnH0` shifts upward (the model “gives up” some anchor-consistency once the calibrator
step is externally constrained).

See: `outputs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1/report.md`.

### “Actual” per-survey / per-epoch bounds (covariance-implied)

As a more data-grounded (but still not fully independent) alternative to “assume σ=0.02 mag”, we
derived per-survey and per-epoch prior widths directly from the *published* Pantheon+SH0ES
STAT+SYS covariance by measuring typical correlated variance within each group.

This produces a sigma-override file with values like:

- `calibrator_offset_mag ≈ 0.023` mag,
- per-survey calibrator offsets typically σ≈0.02–0.05 mag,
- per-epoch time-bin offsets σ≈0.02–0.03 mag,

written to `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_cov_v1.json`.
See the derivation script: `scripts/derive_pantheon_shoes_cov_priors.py`.

When we re-run the joint anchor-consistency fit under these bounds:

- the global calibrator offset is forced down from ~0.16 mag to ~0.057 mag (ΔlogZ ≈ -6.7),
- per-survey calibrator offsets top out around ~0.034 mag (ΔlogZ ≈ -11.3),
- calibrator time-bin offsets top out around ~0.049 mag (ΔlogZ ≈ -9.1).

See: `outputs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1/report.md`.

As a cross-validated “does this help on held-out calibrators?” check, the calibrator-holdout
predictive score still improves when you allow a calibrator offset *even under* the covariance-implied
prior, but the gain is smaller than when the offset is essentially unconstrained:

- Δlogp ≈ +5.74 for a free (wide-prior) calibrator offset,
- Δlogp ≈ +4.20 for the covariance-implied calibrator offset.

See: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_cov_implied_v1/report.md`.

### Calibration-only covariance grouping (Pantheon+SH0ES `CALIB.cov`)

Pantheon+SH0ES ships *systematic-group* covariance blocks under `sytematic_groupings/`.
Using the **CALIB-only** covariance (`Pantheon+SH0ES_122221_CALIB.cov`) as the summary source yields
tighter “instrument-like” bounds, including σ(`calibrator_offset_mag`)≈0.019 mag.

When we rerun the same joint anchor-consistency stack under these CALIB-only bounds:

- the inferred global calibrator offset shrinks to `calibrator_offset_mag ≈ 0.045 ± 0.016`,
- the “tension reduction fraction” drops to ≈0.09 (vs ≈0.37 when the step is effectively unconstrained),
- calibrator-holdout predictive scoring still prefers the offset, but with a smaller gain (Δlogp ≈ +4.1).

See:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1/report.md`

### Systematic-group covariance blocks (full set in `sytematic_groupings/`)

Pantheon+SH0ES ships **many** systematic-group covariance blocks (e.g. `BS21`, `C11`, `SPECEFF`,
`SCATTERMOD`, …). If we repeat the *same* “derive a coherent-scale sigma from the covariance”
construction on each block independently, the implied coherent-step scale is consistently small:
σ(`calibrator_offset_mag`) is typically about **0.02 mag** across all groupings.

When we rerun the joint anchor-consistency stack with each grouping’s implied priors:

- the fit supports only `calibrator_offset_mag ≈ 0.04–0.05` (not ~0.16),
- the corresponding “tension reduction fraction” stays small (≈0.07–0.09),
- calibrator-holdout predictive scoring still prefers a calibrator offset, but with reduced gains
  (Δlogp ≈ +3.6–+4.1 vs +5.2 when the offset is effectively unconstrained).

See:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_groupings_gates_extgrid_all_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_groupings_extgrid_all_v1/report.md`

Interpretation (plain English): if you take *any* of the shipped systematic-group covariance blocks
as a “budget” for coherent calibration-like shifts, none of them are remotely large enough to
permit the ~0.16–0.19 mag step that would be needed to fully explain the ladder-vs-anchor gap.

### Per-survey/epoch (“survey×time”) calibration constraints

To test a more granular “per survey, per observing epoch” failure mode, we also derived a
survey×time-bin sigma file from `CALIB.cov` and enabled the corresponding mechanism
`survey_pkmjd_bins` (per-survey time-bin offsets on calibrators).

Result:

- In the joint anchor-consistency stack, `survey_pkmjd_bins` does **not** meaningfully reduce the
  tension under CALIB.cov-derived bounds.
- In calibrator-holdout predictive scoring, `survey_pkmjd_bins` gives only a small improvement when
  unconstrained (Δlogp ≈ +1.1) and an even smaller one when constrained (Δlogp ≈ +0.5).

See:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_surveytime_gates_extgrid_all_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_calibcov_extgrid_all_v1/report.md`

We also ran an injection map: inject a magnitude shift into a **single** survey×time bin and ask
how much it leaks into `delta_lnH0` when the model does not include that mechanism. The required
amplitudes to fake the full ladder-vs-anchor `delta_lnH0` are **multiple magnitudes** (≫0.1 mag),
which is not physically plausible as a calibration error.

See:
- Injection mapping: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_survey_pkmjd_bins_misspec_v1/report.md`

### Global time-bin offsets on calibrators (`pkmjd_bins`)

The same CALIB.cov-derived priors also bound **global** (survey-agnostic) calibrator time-bin offsets
(`pkmjd_bins` on calibrators) at the ~0.02 mag level per bin.

Result:

- In the joint anchor-consistency stack, constrained calibrator time-bin offsets reduce the tension
  only slightly (tension-reduction fraction ≈0.02) and improve held-out calibrators modestly
  (Δlogp ≈ +1.4).
- Injection mapping shows that faking the full ladder-vs-anchor offset via a **single** calibrator
  time bin would require an offset of order **~1 mag** (≈0.9–1.5 mag depending on the bin), which is
  far beyond any reasonable calibration drift.

See:
- Stack mechanism scan (bounded): `outputs/stack_sn_bao_cc_plus_ladder_mechanism_attribution_calibcov_bounded_extgrid_more_v1/report.md`
- Stack calibrator holdout (bounded): `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_v1/report.md`
- Injection mapping: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_pkmjd_bins_misspec_v1/report.md`

### External calibration covariance (Brout+21 “FRAGILISTIC”; survey-level)

Pantheon+ ships an external calibration product in its DataRelease calibration folder:
`FRAGILISTIC_COVARIANCE.npz`, described as the covariance of photometric zeropoint offsets from
Brout et al. (2021). This provides a more “instrument-like” calibration constraint than an
arbitrary σ=0.02 mag gate.

We download that file into:
`data/raw/pantheon_plus_calibration/FRAGILISTIC_COVARIANCE.npz`
and compress its per-(survey,band) zeropoint covariance into **one effective per-survey magnitude
offset prior** (mean across that survey’s bands). This produces survey-level sigmas typically at
the **few-mmag** level (σ≈0.002–0.008 mag) and writes them to:
`data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_fragilistic_v1.json`
via `scripts/derive_pantheon_shoes_fragilistic_priors.py`.

When we rerun the joint anchor-consistency stack with these tight **survey** calibration priors:

- the model still wants essentially the same large calibrator↔HF step when allowed:
  `calibrator_offset_mag ≈ 0.164 ± 0.031`,
- and calibrator-holdout predictive scoring still strongly prefers the calibrator offset even under
  the tight survey-calibration priors (Δlogp ≈ +5.90).

See:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1/report.md`

### SH0ES calibrator-chain prior scale (linear-system products)

The Pantheon+/SH0ES DataRelease includes a compact **linear system** for the full ladder
(`SH0ES_Data/ally_...fits`, `alll_...fits`, `allc_...fits`) plus `lstsq_results.txt`.
The last entry (index 46, “`fivelogH0`” per the SH0ES example script) reports σ≈0.028 mag.

Treating that σ as a *calibrator-chain-inspired* prior width on an **additional** `calibrator_offset_mag`
(i.e., “how big could an unmodeled extra calibrator-step effect plausibly be?”), the joint stack under
FRAGILISTIC survey priors supports only `calibrator_offset_mag ≈ 0.074 ± 0.021` (tension-reduction frac ≈ 0.16).

See:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1/report.md`

## Calibrator holdout (cross-validated; real data)

A key “is this just overfitting a few points?” check is to **hold out calibrators** while keeping
the Hubble‑flow sample fixed in TRAIN, then score held‑out calibrators under the posterior
predictive distribution.

Result: calibrator-only mechanisms improve held‑out predictive performance substantially:

- Random calibrator holdout: Δlogp ≈ **+10.2** for `calibrator_offset_mag` and Δlogp ≈ **+8.3** for
  calibrator time-bin offsets (`pkmjd_bins` applied to calibrators).
- Survey-by-survey calibrator holdout: improvements persist but are smaller (Δlogp ≈ +3.8 and +3.5).

See:
- `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1/report.md`
- `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1/report.md`

A full-covariance robustness check (fewer reps due to cost) shows similar behavior:
Δlogp ≈ **+8.5** (random) and ≈ **+3.2** (survey holdout) for `calibrator_offset_mag`.

See:
- `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`
- `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1/report.md`

### Joint-stack version (anchor-consistency context)

We repeated the same “hold out calibrators, keep hubble-flow fixed” idea inside the **joint**
anchor-consistency stack (SN-only + BAO + chronometers + ladder). Here we also keep the non-ladder
probes fixed in TRAIN (splits apply only within the ladder part).

Result: the calibrator-only mechanisms still help held-out calibrators:

- Random calibrator holdout (diag errors): Δlogp ≈ **+5.7** for `calibrator_offset_mag`
- Random calibrator holdout (full cov): Δlogp ≈ **+6.2** for `calibrator_offset_mag`

See:
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`

#### Mechanism scan (joint stack)

We tested a few “explicit proxy” replacements inside the same joint-stack holdout:

- `PKMJDERR` linear term on calibrators (`pkmjd_err_linear`) gives Δlogp ≈ **+4.6** (close to the
  time-bin offsets Δlogp ≈ +5.0, but below `calibrator_offset_mag` Δlogp ≈ +5.7).
- `m_b_corr_err_VPEC` linear term on calibrators (`m_b_corr_err_vpec_linear`) gives essentially **no**
  improvement (Δlogp ≈ +0.1).

See:
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1/report.md`

We also ran a **survey holdout** (hold out calibrators from one survey at a time). Effects persist
but are smaller, with calibrator time-bin offsets still best among these candidates (Δlogp ≈ +1.9,
vs +1.7 for `calibrator_offset_mag`, vs +1.5 for `pkmjd_err_linear`):

- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1/report.md`

A full-covariance variant shows the same ordering with slightly larger improvements
(Δlogp ≈ +2.34 for calibrator time-bin offsets; +2.29 for `calibrator_offset_mag`; +1.45 for
`pkmjd_err_linear`):

- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1/report.md`

We also repeated the joint-stack holdout with **external late-time H0 probes** added as `h0_grid`
constraints (TRGB / strong lenses / masers). The same calibrator-only effects persist:

- Random holdout: Δlogp ≈ **+5.23** (`calibrator_offset_mag`), ≈ **+4.78** (time bins), ≈ **+4.25**
  (`PKMJDERR` linear).  
  See: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1/report.md`.
- Survey holdout: Δlogp ≈ **+1.86** (time bins) and **+1.57** (`calibrator_offset_mag`).  
  See: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1/report.md`.

This does **not** prove a physical mechanism (we are not modeling the full photometry pipeline), but
it does say the required calibrator↔HF offset behaves like a **coherent effect**, not a tiny-N
artifact.

## External-probe bracketing (TRGB / strong lenses / megamasers)

As a **stress test** (not an independence claim), we added a small set of published, approximately-Gaussian
late-time `H0` constraints in two equivalent ways:

- as `gaussian_measurement` constraints (single-number summary), and
- as `h0_grid` posteriors (grid PDFs in `data/processed/h0_grids/`, generated by
  `scripts/build_h0_grids_from_external_constraints.py`).

- Low-H0 leaning: TRGB (Freedman 2021 review summary) + TDCOSMO+SLACS (Birrer 2020)
- High-H0 leaning: H0LiCOW (Wong 2020) + Megamaser Cosmology Project (Pesce 2020)

Result: the joint fit still requires essentially the same **calibrator↔Hubble-flow offset** (`~0.15–0.16 mag`).
The low-H0 constraints are consistent with the stack; the high-H0 constraints show visible tension (large per-part chi2),
but they do not pull the combined `H0_eff` up to ~73.

See:
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1/report.md`

## “Mechanism replacements” for the calibrator offset (so far: no)

We tested a few simple, metadata-shaped alternatives in the joint stack (e.g. calibrator-only time trends,
calibrator-by-survey offsets). None reduced the required ~0.16 mag offset, and evidence typically worsened
(consistent with “extra flexibility, no real explanatory power”).

See:
- `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_plus_calTimeLinear_v1/report.md`
- `outputs/stack_sn_bao_cc_siren_plus_ladder_calSurvey_only_v1/report.md`
- `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_plus_calSurvey_v1/report.md`

We also tested calibrator-only **epoch bin offsets** as a replacement mechanism. They can create
large per-epoch shifts (up to ~0.24 mag in one bin), but they are still penalized by evidence
relative to a single `calibrator_offset_mag`:
- `outputs/stack_sn_bao_cc_plus_ladder_calTimeBins_only_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_plus_calTimeBins_v1/report.md`

Finally, we tested a very simple calibrator-only proxy: a linear term in `PKMJDERR` (the time of
maximum uncertainty). This improves the joint fit slightly, but it does **not** replace the full
calibrator offset:

- With `pkmjd_err_linear` on calibrators: log evidence ≈ **861.79**, ladder chi2 ≈ **297.31**
  (`outputs/stack_sn_bao_cc_plus_ladder_pkmjderr_linear_cal_v1/report.md`).
- With `calibrator_offset_mag`: log evidence ≈ **871.22**, ladder chi2 ≈ **280.14**
  (`outputs/stack_sn_bao_cc_plus_ladder_cal_offset_v1/report.md`).

So `PKMJDERR` is a plausible *marker* for whatever drives the calibrator↔HF mismatch, but it does not
fully account for it in this model family.

## “Are any of these probable yet?”

We have a **probable “shape”** but not a probable **physical cause** yet.

- The existence of a coherent **calibrator↔HF offset** is supported in this simplified framework:
  it is required by the joint anchor-consistency stack and it improves held‑out calibrator
  predictive scoring.
- However, we still do **not** have a specific, independently validated metadata mechanism
  (time/sky/quality/host) that can explain most of the ~0.19 mag ladder shift at an amplitude that
  is clearly physically plausible.

One concrete sign of this: cross-validated predictive scoring and exact Gaussian-model evidence both
**penalize** adding flexible HF redshift splines and low‑ℓ sky modes on the ladder subset (they make
held-out predictions worse and reduce evidence despite slightly improving in-sample chi2).

See:
- `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`
- `outputs/pantheon_plus_shoes_ladder_level_sweep_v2/report.md`

## What to do next

High-value next steps that keep the systematics-first framing:

1. **External priors:** encode realistic calibration bounds (per survey, per epoch) and quantify how much of the gap can be absorbed *without* implausible amplitudes.
2. **Add more probes via `h0_grid`:** strong lenses, TRGB, additional siren posteriors (as grid files) and run the same “anchor consistency” stress test.
3. **Richer selection stress tests:** cuts on additional metadata (e.g. `PKMJDERR`, `HOST_LOGMASS_ERR`) and joint “survey×epoch” distortion models, validated by SBC.
