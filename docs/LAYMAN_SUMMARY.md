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
calibrator step: Pantheon+ (SN only) + DESI 2024 BAO + chronometers, optionally plus a GR-control
siren H0 posterior grid. That stack prefers `H0 ≈ 68.21 ± 0.31`, i.e. close to the early-time
anchor and far from the SH0ES ladder value.

See:
- `outputs/stack_sn_bao_cc_stress_v2/report.md`
- `outputs/stack_sn_bao_cc_siren_stress_v2/report.md`

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
- A pure sky dipole needs **~0.40 mag**.
- A pure time trend needs **~0.68 mag per 1σ(time)**.
- A host-mass step needs **~2.35 mag**.

See: `outputs/pantheon_plus_shoes_ladder_injection_misspec_v2/report.md`.

## Joint “anchor-consistency” inference (real data)

If we **force** agreement with the early-time anchor using the *other* late-time probes
SN-only+BAO+chronometers(+siren GR control), then the SH0ES-style ladder subset can only be made
compatible by introducing a large calibrator↔Hubble-flow offset:

- In a joint stack, the inferred `calibrator_offset_mag ≈ +0.162 ± 0.032` mag.
- That is roughly **~85% ± 17%** of the full **~0.19 mag** ladder gap (simple ratio; not a calibrated physical prior).

See: `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_v2/report.md`.

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
