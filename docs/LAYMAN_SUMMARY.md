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
- A pure sky dipole needs **~0.40 mag**.
- A pure time trend needs **~0.68 mag per 1σ(time)**.
- A host-mass step needs **~2.35 mag**.

See: `outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/report.md`.

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

## External-probe bracketing (TRGB / strong lenses / megamasers)

As a **stress test** (not an independence claim), we added a small set of published, approximately-Gaussian
late-time `H0` constraints via the `gaussian_measurement` adapter:

- Low-H0 leaning: TRGB (Freedman 2021 review summary) + TDCOSMO+SLACS (Birrer 2020)
- High-H0 leaning: H0LiCOW (Wong 2020) + Megamaser Cosmology Project (Pesce 2020)

Result: the joint fit still requires essentially the same **calibrator↔Hubble-flow offset** (`~0.15–0.16 mag`).
The low-H0 constraints are consistent with the stack; the high-H0 constraints show visible tension (large per-part chi2),
but they do not pull the combined `H0_eff` up to ~73.

See:
- `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_extH0_low_v1/report.md`
- `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_extH0_high_v1/report.md`
- `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_extH0_all_v1/report.md`

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

## “Are any of these probable yet?”

Not yet. The tests so far *don’t* pick out a specific small/metadata‑tracked distortion that can explain most of the ~0.19 mag ladder shift.
The only mechanisms that fully close the gap require **very large** corrections.

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
