# Spec status (systematics-first Hubble-tension chain)

This file maps the “systematics-first” spec you described (CatWISE-style stability scans + inverse-reconstruction discipline) to what exists **in this repo** today.

## What this chain is built to test

- **Primary target:** late-time distance-scale pipelines (SN ladder, SN-only+BAO+CC, siren H0 posteriors) under a **fixed early-time anchor**.
- **Not a new-cosmology fitter:** by default, the anchor (`anchor.H0`, `anchor.rd_Mpc`, etc.) is treated as fixed; late-time data are used to infer **distortion/systematics terms**.

## Implemented building blocks (code)

- **Config-driven runner + reporting:** `src/hubble_systematics/audit/runner.py`, `src/hubble_systematics/audit/reporting.py`
- **Mechanism ladder (L0–L4):** implemented per-probe (e.g. `src/hubble_systematics/probes/pantheon_plus_shoes_ladder.py`)
- **Stability scans (cut sweeps):** `src/hubble_systematics/audit/tasks.py#run_cut_scan`
- **Correlated-cut drift null MC:** `src/hubble_systematics/audit/correlated_cut_null.py`
- **Time/epoch invariance tests:**
  - split-fit: `src/hubble_systematics/audit/split_fit.py`
  - split-null: `src/hubble_systematics/audit/split_null.py`
  - grouped split-null (per-survey): `src/hubble_systematics/audit/group_split_null.py`
- **Injection / recovery:** `src/hubble_systematics/audit/injection.py`
- **SBC / coverage:** `src/hubble_systematics/audit/sbc.py`
- **Cross-validated hemisphere scan:** `src/hubble_systematics/audit/hemisphere_scan.py`
- **Predictive scoring (CV log predictive density):** `src/hubble_systematics/audit/predictive_score.py`
- **Exact Gaussian log evidence:** `src/hubble_systematics/gaussian_linear_model.py#log_marginal_likelihood`
- **Joint / stacked inference:** `src/hubble_systematics/joint.py` (+ stack handling in `src/hubble_systematics/audit/tasks.py`)

## Implemented probes (real-data adapters)

- `pantheon_plus` (SN-only, with stable one-hot design support): `src/hubble_systematics/probes/pantheon_plus.py`
- `pantheon_plus_shoes_ladder` (calibrators + Hubble-flow SN subset): `src/hubble_systematics/probes/pantheon_plus_shoes_ladder.py`
- `bao` (preprocessed BAO summaries): `src/hubble_systematics/probes/bao.py`
- `chronometers` (cosmic chronometers): `src/hubble_systematics/probes/chronometers.py`
- `h0_grid` (a 1D posterior grid / likelihood for H0, linear or ln-space): `src/hubble_systematics/probes/h0_grid.py`
- `gaussian_measurement` (single-number Gaussian constraint on `H0` or `rd_Mpc`): `src/hubble_systematics/probes/gaussian_measurement.py`
- `siren_gate2_grid` (Gate-2 dark-siren grid + event metadata; selection-corrected): `src/hubble_systematics/probes/siren_gate2_grid.py`
  - Note: upstream Gate-2 products may be incomplete; treat as experimental plumbing until validated.
- `stack` (combine multiple parts with shared parameters by name): handled in `src/hubble_systematics/audit/tasks.py`

## “Audit packet” configs (how to run each spec component)

- **Ladder reproduction + cut scan + correlated-cut null + injection + SBC:** `configs/pantheon_plus_shoes_ladder_audit.yaml`
- **Predictive scoring (CV):** `configs/pantheon_plus_shoes_ladder_predictive_score_v2.yaml`
- **Predictive scoring (hold out calibrators; HF fixed in TRAIN):**
  `configs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1.yaml`
  and the survey-holdout variant
  `configs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1.yaml`
  (plus full-cov robustness variants
  `configs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1.yaml` and
  `configs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1.yaml`)
- **Predictive scoring (stack; hold out calibrators inside ladder part while keeping other probes in TRAIN):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1.yaml` and the full-cov variant
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1.yaml`
- **Predictive scoring (stack; mechanism scan on calibrator-only proxies):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1.yaml`
- **Predictive scoring (stack; survey holdout + mechanism scan on calibrator-only proxies):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1.yaml`
  and the full-cov variant
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1.yaml`
- **Model-evidence ladder sweep (L2→L4):** `configs/pantheon_plus_shoes_ladder_level_sweep_v2.yaml`
- **Time invariance null (fixed calibrators):** `configs/pantheon_plus_shoes_ladder_time_invariance_null_fixedcal_v3.yaml`
- **Quality-cut scans (FITPROB / redchi2 / etc.):** `configs/pantheon_plus_shoes_ladder_fitprob_cut_scan.yaml`, `configs/pantheon_plus_shoes_ladder_redchi2_cut_scan.yaml`, …
- **Mechanism sweeps (including quality/selection proxy):** `configs/pantheon_plus_shoes_ladder_sweep_v2.yaml` (adds `m_b_corr_err_linear`)
- **External-prior sensitivity sweep (tight vs loose priors):** `configs/pantheon_plus_shoes_ladder_external_prior_sweep.yaml`
- **Joint “anchor consistency” stress test (SN-only+BAO+CC+siren + ladder with calibrator offset):**
  `configs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_v2.yaml`
- **Joint “anchor consistency” stress test (no sirens):**
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_v1.yaml`
  and the external-prior stress variant
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_tight_v1.yaml`
- **Calibrator-step mechanism alternatives (no sirens):**
  `configs/stack_sn_bao_cc_plus_ladder_calTimeBins_only_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_plus_calTimeBins_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_calSurvey_only_min4_v1.yaml`,
  and `configs/stack_sn_bao_cc_plus_ladder_pkmjderr_linear_cal_v1.yaml`
- **External-prior gates (scan):**
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_prior_sigma_sweep_v1.yaml`
- **External-prior gates (per-survey / per-epoch mechanisms):**
  `configs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1.yaml`
- **Injection suite (adds new metadata proxies):**
  `configs/pantheon_plus_shoes_ladder_injection_misspec_v4.yaml`
- **SBC (new metadata proxies):**
  `configs/pantheon_plus_shoes_ladder_sbc_new_proxies_v1.yaml`
- **External H0 probes as `h0_grid` posteriors (TRGB / lenses / masers):**
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1.yaml`
- **Holdout battery with external H0 grids included:**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1.yaml` and
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1.yaml`

## What’s still incomplete vs the full ambition

- **More probes:** strong lenses, TRGB-only ladders, additional siren products, etc. (requires adding/copying datasets into `data/processed/...` with source metadata).
- **Metadata-tied priors:** current “external priors” are *synthetic knobs* (tight vs loose). A full version should tie priors to survey calibration logs, overlap residuals, or external calibrators (Gaia / standard-star networks) and validate them with holdouts.
- **Instrument-level forward sims:** current injection/SBC operates at the *compressed data-vector* level (linearized magnitude / distance residuals). A full end-to-end simulator for Cepheids/TRGB/lens modeling is not yet implemented here.

## Live status snapshot (where to read results fast)

Start at `docs/LAYMAN_SUMMARY.md` and the latest “real run” reports:

- `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1/report.md`
