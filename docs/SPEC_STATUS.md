# Spec status (systematics-first Hubble-tension chain)

This file maps the “systematics-first” spec you described (CatWISE-style stability scans + inverse-reconstruction discipline) to what exists **in this repo** today.

## What this chain is built to test

- **Primary target:** late-time distance-scale pipelines (SN ladder, SN-only+BAO+CC, siren H0 posteriors) under a **fixed early-time anchor**.
- **Not a new-cosmology fitter:** by default, the anchor (`anchor.H0`, `anchor.rd_Mpc`, etc.) is treated as fixed; late-time data are used to infer **distortion/systematics terms**.

## Implemented building blocks (code)

- **Config-driven runner + reporting:** `src/hubble_systematics/audit/runner.py`, `src/hubble_systematics/audit/reporting.py`
- **Predictive-score driver attribution (who drives Δlogp?):**
  - `scripts/rank_predictive_score_drivers.py`
  - `scripts/report_hero_calibrators.py` (metadata/photometry provenance for “hero” CIDs)
  - `scripts/report_cid_discordance.py` (quantifies duplicate-CID reductions; supports CID-holdout audit interpretation)
- **Mechanism ladder (L0–L4):** implemented per-probe (e.g. `src/hubble_systematics/probes/pantheon_plus_shoes_ladder.py`)
- **Stability scans (cut sweeps):** `src/hubble_systematics/audit/tasks.py#run_cut_scan`
- **Correlated-cut drift null MC:** `src/hubble_systematics/audit/correlated_cut_null.py`
- **Time/epoch invariance tests:**
  - split-fit: `src/hubble_systematics/audit/split_fit.py`
  - split-null: `src/hubble_systematics/audit/split_null.py`
  - grouped split-null (per-survey): `src/hubble_systematics/audit/group_split_null.py`
- **Injection / recovery:** `src/hubble_systematics/audit/injection.py`
- **SBC / coverage:** `src/hubble_systematics/audit/sbc.py`
- **Prior-MC forward simulator (prior → bias bound):** `src/hubble_systematics/audit/prior_mc.py`
- **Cross-validated hemisphere scan:** `src/hubble_systematics/audit/hemisphere_scan.py`
- **Predictive scoring (CV log predictive density):** `src/hubble_systematics/audit/predictive_score.py`
- **Permutation-null tests (metadata proxy “realness”):** `scripts/proxy_permutation_null.py` (permute within-survey / global; supports multiprocessing)
- **Exact Gaussian log evidence:** `src/hubble_systematics/gaussian_linear_model.py#log_marginal_likelihood`
- **Joint / stacked inference:** `src/hubble_systematics/joint.py` (+ stack handling in `src/hubble_systematics/audit/tasks.py`)
- **External calibration adapters:**
  - FRAGILISTIC (Brout+21) filter-level correlated prior: `src/hubble_systematics/external_calibration/fragilistic.py`

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
- **Predictive scoring (stack; covproj term ablations + survey holdout):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1.yaml`
- **Predictive scoring (stack; host-mass scope test under covproj bounds):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_hostmass_scope_v1.yaml` and
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_covproj_hostmass_scope_v1.yaml`
- **Predictive scoring (stack; mechanism scan on calibrator-only proxies):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1.yaml`
- **Mechanism attribution scans (stack; compare cal_offset vs time-bin proxies):**
  `configs/stack_sn_bao_cc_plus_ladder_mechanism_attribution_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_extgrid_more_v1.yaml`,
  and the CALIB.cov-bounded variants
  `configs/stack_sn_bao_cc_plus_ladder_mechanism_attribution_calibcov_bounded_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_v1.yaml`,
  plus full-cov variants:
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_fullcov_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_fullcov_v1.yaml`
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
- **Covariance-implied per-survey / per-epoch bounds (derived from published STAT+SYS cov):**
  `scripts/derive_pantheon_shoes_cov_priors.py` and
  `configs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1.yaml`
- **External calibration covariance (Brout+21 “FRAGILISTIC”; per-survey priors):**
  `scripts/derive_pantheon_shoes_fragilistic_priors.py`,
  `configs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1.yaml`
- **External calibration metadata proxy (SNANA kcor variants; per-survey(+time-bin) priors):**
  `data/raw/pantheon_plus_calibration/SNANA_kcor/`,
  `scripts/derive_pantheon_shoes_kcor_variant_priors.py`,
  extended cal/hf survey-offset variant:
  `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_kcor_variants_calhf_v1.json`,
  merged multi-source gate helper:
  `scripts/merge_sigma_overrides.py`,
  combined kcor+SH0ES-linear-system gate:
  `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_kcor_calhf_plus_shoeslin_v1.json`,
  `configs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_fullcov_v1.yaml`,
  plus kcor-gated mechanism-attribution scans:
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_fullcov_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_mechanism_attribution_kcor_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_mechanism_attribution_extgrid_more_v1.yaml`,
  and kcor-calibrated injection/SBC checks:
  `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_misspec_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_modeled_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_kcor_c_x1_biascor_misspec_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_kcor_c_x1_biascor_modeled_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_sbc_survey_pkmjd_bins_kcor_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_sbc_kcor_c_x1_biascor_v1.yaml`,
  plus full-cov variants:
  `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_misspec_fullcov_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_modeled_fullcov_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_sbc_survey_pkmjd_bins_kcor_fullcov_v1.yaml`
  plus constrained-decomposition sweeps/holdouts under the combined kcor+SH0ES-linear-system gate:
  `configs/stack_sn_bao_cc_plus_ladder_constrained_decomp_kcor_calhf_shoeslin_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_extgrid_more_v1.yaml`
- **Calibration-only covariance grouping (Pantheon+SH0ES `CALIB.cov`; per-survey + calibrator/time priors):**
  `data/raw/pantheon_plus_shoes/sytematic_groupings/Pantheon+SH0ES_122221_CALIB.cov`,
  `configs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1.yaml`
- **Systematic-group covariance gates (all `sytematic_groupings/*.cov` blocks):**
  `configs/stack_sn_bao_cc_plus_ladder_groupings_gates_extgrid_all_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_groupings_extgrid_all_v1.yaml`
- **SH0ES linear-system prior scale (fivelogH0 σ → calibrator_offset_mag σ):**
  `data/raw/shoes_linear_system/`,
  `scripts/derive_shoes_linear_system_fivelogh0_prior.py`,
  `configs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1.yaml`
- **FRAGILISTIC filter-level correlated prior (no per-survey σ compression):**
  `configs/stack_sn_bao_cc_plus_ladder_fragilistic_filter_shoes_gates_v1.yaml`
- **Injection suite (adds new metadata proxies):**
  `configs/pantheon_plus_shoes_ladder_injection_misspec_v4.yaml`
- **Injection suite (survey×time-bin proxy):**
  `configs/pantheon_plus_shoes_ladder_injection_calibcov_survey_pkmjd_bins_misspec_v1.yaml`
- **Injection suite (global calibrator time-bin proxy):**
  `configs/pantheon_plus_shoes_ladder_injection_calibcov_pkmjd_bins_misspec_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_calibcov_pkmjd_bins_modeled_v1.yaml`
- **SBC (new metadata proxies):**
  `configs/pantheon_plus_shoes_ladder_sbc_new_proxies_v1.yaml`
- **SBC (survey×time-bin proxy):**
  `configs/pantheon_plus_shoes_ladder_sbc_survey_pkmjd_bins_calibcov_v1.yaml`
- **SBC (global calibrator time-bin proxy):**
  `configs/pantheon_plus_shoes_ladder_sbc_pkmjd_bins_calibcov_v1.yaml`
- **Prior-MC bias bound (kcor time-bin priors):**
  `configs/pantheon_plus_shoes_ladder_prior_mc_kcor_timebins_v1.yaml`
- **Prior-MC bias bound (combined kcor cal/hf + SH0ES-linear-system):**
  `configs/pantheon_plus_shoes_ladder_prior_mc_constrained_kcor_calhf_shoeslin_v1.yaml`
- **Prior-MC bias bound (host mass only; covproj bounds):**
  `configs/pantheon_plus_shoes_ladder_prior_mc_host_mass_step_covproj_v1.yaml`
- **Prior-MC + injection + SBC under merged “realcal survey×epoch” constraints:**
  `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_realcal_surveytime_shoeslin_covproj_v1.json`,
  `configs/pantheon_plus_shoes_ladder_prior_mc_realcal_surveytime_shoeslin_covproj_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_realcal_surveytime_shoeslin_covproj_misspec_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_realcal_surveytime_shoeslin_covproj_modeled_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_sbc_realcal_surveytime_shoeslin_covproj_v1.yaml`
- **SBC under “realcal” constraints + FRAGILISTIC filter priors:**
  `configs/pantheon_plus_shoes_ladder_sbc_realcal_surveytime_shoeslin_covproj_fragfilter_v1.yaml`
- **Injection/SBC under CALIB-only priors (survey/time injections):**
  `configs/pantheon_plus_shoes_ladder_injection_calibcov_misspec_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_anchor_enforced_v1.yaml`,
  `configs/pantheon_plus_shoes_ladder_sbc_calibcov_v1.yaml`
- **External H0 probes as `h0_grid` posteriors (TRGB / lenses / masers):**
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1.yaml`
- **External H0 probes (expanded; includes SBF + STRIDES):**
  `data/processed/external_constraints/h0_constraints_2026-02-05.json`,
  `scripts/build_h0_grids_from_external_constraints.py`,
  `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_more_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_more_v1.yaml`
- **Holdout battery with external H0 grids included:**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1.yaml` and
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1.yaml`
  plus the CALIB-only external-prior variants:
  `configs/stack_sn_bao_cc_plus_ladder_calibcov_gates_extgrid_all_v1.yaml` and
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_extgrid_all_v1.yaml`
- **CID-holdout battery (calibrators held out by CID; ext grids; realcal survey×epoch pack):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1.yaml`
- **CID-holdout battery with FRAGILISTIC filter priors (same realcal pack + ext grids):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_v1.yaml`
- **CID-holdout battery with calibrator-CID duplicate handling (dedup / drop-discordant):**
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dedup_cal_bestfitprob_v1.yaml`,
  `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dropdiscord_cal_bestfitprob_v1.yaml`

## What’s still incomplete vs the full ambition

- **More probes:** strong lenses, TRGB-only ladders, additional siren products, etc. (requires adding/copying datasets into `data/processed/...` with source metadata).
- **Metadata-tied priors:** current “external priors” are *synthetic knobs* (tight vs loose). A full version should tie priors to survey calibration logs, overlap residuals, or external calibrators (Gaia / standard-star networks) and validate them with holdouts.
- **Note:** we now include one concrete metadata-tied prior source (the Brout+21 zeropoint covariance shipped with Pantheon+), but we still lack true per-epoch calibration logs and photometry-level forward modeling.
- **Instrument-level forward sims:** current injection/SBC operates at the *compressed data-vector* level (linearized magnitude / distance residuals). A full end-to-end simulator for Cepheids/TRGB/lens modeling is not yet implemented here.

## Live status snapshot (where to read results fast)

Start at `docs/LAYMAN_SUMMARY.md` and the latest “real run” reports:

- `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1/report.md`
