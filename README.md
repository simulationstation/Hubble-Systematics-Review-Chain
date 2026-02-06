# Hubble-Systematics-Review-Chain

Systematics-first, *audit-style* pipeline for late-time distance-scale probes, patterned on:

- stability scans (cuts/thresholds) + correlated-cut drift nulls
- mechanism ladders (low-dimensional knobs → structured residual closure)
- injection/recovery suites
- SBC/coverage gates
- time/epoch and sky (low-ℓ) invariance tests

## Current findings (real data; 2026-02-06)

These are **real-data** runs on the public Pantheon+ / Pantheon+SH0ES tables in `data/raw/…`,
using this repo’s **linear-Gaussian audit models** (not a full end-to-end SH0ES reanalysis).

- **Ladder reproduction:** `H0_eff = 73.552 ± 1.078` vs anchor `67.4` (equivalent `|Δμ| ≈ 0.19 mag`).  
  Report: `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_predictive_score_v2.yaml`
- **Late-time non-ladder stack:** Pantheon+ (SN-only) + DESI BAO + CC gives `H0_eff = 68.208 ± 0.314` (close to anchor).  
  Report: `outputs/stack_sn_bao_cc_stress_v2/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_stress.yaml`
- **Anchor-consistency (no sirens):** adding the SH0ES ladder subset forces a large calibrator↔HF offset:  
  `calibrator_offset_mag = +0.1616 ± 0.0318` mag.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_v1.yaml`
- **Ladder calibrator holdout (CV; real data):** holding out calibrators (keeping hubble-flow fixed in TRAIN) shows a large out-of-sample improvement from calibrator-only mechanisms:  
  Δlogp ≈ +10.2 / +8.5 (diag/full-cov) for `calibrator_offset_mag`, and Δlogp ≈ +8.3 / +6.1 for calibrator time-bin offsets (`pkmjd_bins` on calibrators).  
  Survey-by-survey holdout also improves (Δlogp ≈ +3.8 / +3.2 for `calibrator_offset_mag`).  
  In these random holdout splits, `calibrator_offset_mag` beats the baseline in **99.5–100%** of splits (depending on diag/fullcov).  
  Reports: `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1.yaml`, `configs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1.yaml`, `configs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1.yaml`, `configs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1.yaml`
- **Joint-stack calibrator holdout (real data):** in the *anchor-consistency* joint stack (SN-only+BAO+CC+ladder), holding out calibrators (keeping hubble-flow + other probes fixed in TRAIN) still prefers calibrator-only corrections:  
  Δlogp ≈ +5.7 / +6.2 (diag/full-cov) for `calibrator_offset_mag`.  
  In these random holdout splits, `calibrator_offset_mag` beats the baseline in **93.5–99%** of splits (depending on diag/fullcov).  
  A simple calibrator-only proxy `pkmjd_err_linear` (linear in `PKMJDERR`) also improves held-out calibrators (Δlogp ≈ +4.6), while a `m_b_corr_err_VPEC` linear proxy does not.  
  Survey-holdout inside the same joint stack is smaller but still improves (Δlogp ≈ +1.9 / +2.3 for calibrator time-bin offsets; +1.7 / +2.3 for `calibrator_offset_mag`).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1.yaml`
- **External-prior stress test:** forcing a tight prior on `calibrator_offset_mag` degrades evidence and shifts the fit (does *not* recover `H0≈73`).  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_tight_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_tight_v1.yaml`
- **External-prior “gate” sweep (real data):** if external calibration work can truly bound calibrator-step distortions at the ~0.02 mag level, the joint anchor-consistency fit pays a large evidence penalty and cannot maintain the large ~0.16 mag calibrator↔HF offset.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1.yaml`
- **Covariance-implied calibration bounds (real data):** we can also derive “budget-like” bounds for per-survey and per-epoch calibrator offsets from the published Pantheon+SH0ES STAT+SYS covariance. Under these bounds (typical σ≈0.02–0.05 mag), the joint fit cannot sustain a 0.16 mag calibrator↔HF step without a large evidence penalty.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1.yaml`  
  Derivation: `scripts/derive_pantheon_shoes_cov_priors.py`
- **External calibration covariance (Brout+21 “FRAGILISTIC”; real data):** using the public zeropoint-offset covariance shipped with the Pantheon+ DataRelease (typical per-survey σ≈0.002–0.008 mag), constraining per-survey offsets does **not** remove the need for a large calibrator↔HF step (the fit still wants `calibrator_offset_mag ≈ 0.16` when allowed). Calibrator holdout predictive scoring still prefers `calibrator_offset_mag` even under these tight survey-calibration priors.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1.yaml`  
  Derivation: `scripts/derive_pantheon_shoes_fragilistic_priors.py`
- **Calibration-only covariance gate (Pantheon+SH0ES `CALIB.cov`; real data):** using the Pantheon+SH0ES *calibration-only* covariance grouping (`sytematic_groupings/Pantheon+SH0ES_122221_CALIB.cov`) to derive prior widths gives σ(`calibrator_offset_mag`)≈0.019 mag. Under these bounds, the joint anchor-consistency fit can only support a much smaller step: `calibrator_offset_mag ≈ 0.045 ± 0.016` (tension-reduction frac ≈ 0.09 vs ≈ 0.37 when free). Calibrator-holdout predictive scoring still prefers the mechanism, but with a smaller gain (Δlogp ≈ +4.1 vs +5.7 when free).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1.yaml`  
  Derivation: `scripts/derive_pantheon_shoes_cov_priors.py --raw-cov-path data/raw/pantheon_plus_shoes/sytematic_groupings/Pantheon+SH0ES_122221_CALIB.cov`
- **All systematic-group covariance blocks (Pantheon+SH0ES `sytematic_groupings/*.cov`; real data):** repeating the same “covariance-implied prior” construction on *every* shipped systematic-group covariance block yields very similar coherent-scale bounds (σ(`calibrator_offset_mag`)≈0.019–0.022 mag). Under these bounds, the joint fit only supports `calibrator_offset_mag ≈ 0.04–0.05` and tension-reduction fractions ≈0.07–0.09. Calibrator-holdout predictive scoring still prefers a calibrator offset, but with reduced gains (Δlogp ≈ +3.6–+4.1 vs +5.2 when free).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_groupings_gates_extgrid_all_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_groupings_extgrid_all_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_groupings_gates_extgrid_all_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_groupings_extgrid_all_v1.yaml`
- **Per-survey/epoch calibration constraints (CALIB.cov “survey×time”; real data):** allowing per-survey time-bin offsets (`survey_pkmjd_bins`) does **not** meaningfully reduce the anchor-consistency tension under CALIB.cov-derived bounds; calibrator-holdout improvement is small (Δlogp ≈ +0.5 when constrained). An injection map shows that moving `delta_lnH0` by the full ladder-vs-anchor amount would require a single survey×epoch-bin offset of **multiple magnitudes** (≫0.1 mag).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_surveytime_gates_extgrid_all_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_calibcov_extgrid_all_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_calibcov_survey_pkmjd_bins_misspec_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_surveytime_gates_extgrid_all_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_calibcov_extgrid_all_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_calibcov_survey_pkmjd_bins_misspec_v1.yaml`
- **CALIB.cov time-bin offsets on calibrators (`pkmjd_bins`; real data + injection):** under CALIB.cov-derived bounds (σ≈0.02 mag per time-bin), calibrator time-bin offsets provide only small tension reduction in the joint stack (≈0.02) and modest held-out calibrator gains (Δlogp ≈ +1.1–+1.4 for time bins alone; fullcov/diag). Injection mapping shows that faking the full ladder-vs-anchor offset via a **single** calibrator time bin would require a **~1 mag** shift (≈0.9–1.5 mag depending on bin), which is not physically plausible as calibration drift.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_mechanism_attribution_calibcov_bounded_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_fullcov_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_calibcov_pkmjd_bins_misspec_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_mechanism_attribution_calibcov_bounded_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_fullcov_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_calibcov_pkmjd_bins_misspec_v1.yaml`
- **External calibration metadata (SNANA kcor variants; real data):** ingesting the Pantheon+ DataRelease `SNANA_kcor` calibration files and turning the **spread across calibration variants** into per-survey(+time-bin) prior widths yields σ≈0.01–0.015 mag for the affected surveys. Under these bounds, per-survey epoch-bin offsets (`survey_pkmjd_bins`) remain small (max |mean| ≈ 0.006 mag across 40 coefficients) and provide essentially no held-out-calibrator gain (Δlogp ≈ +0.07) once constrained; the joint stack still prefers a large global calibrator offset when allowed.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_fullcov_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_fullcov_v1.yaml`  
  Prior derivation: `scripts/derive_pantheon_shoes_kcor_variant_priors.py` (outputs `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_kcor_variants_v1.json`)
- **Forward “how much can this explain?” bound (prior MC; real data + simulator):** using the new `prior_mc` forward simulator to draw random time-bin calibration drifts from the kcor-variant priors and refit the ladder shows the induced `delta_lnH0` shift has p95(|.|) ≈ 0.0199, corresponding to **~21% of the full ladder-vs-anchor tension** (p99 ≈ 28%).  
  Report: `outputs/pantheon_plus_shoes_ladder_prior_mc_kcor_timebins_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_prior_mc_kcor_timebins_v1.yaml`
- **Injection mapping (kcor priors; survey×time → H0 shift):** injecting a single per-survey epoch-bin magnitude offset into calibrators maps to a tiny change in `delta_lnH0` (typical slopes ≈0.02–0.03 in `delta_lnH0` per mag). Matching the full ladder-vs-anchor `delta_lnH0` would require a single-bin offset of **several magnitudes**, which is far beyond plausible calibration drift; when `survey_pkmjd_bins` is explicitly modeled under the kcor priors, the fitted per-bin offsets remain tiny (max |mean| ≈ 0.005 mag).  
  Reports: `outputs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_misspec_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_modeled_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_misspec_fullcov_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_modeled_fullcov_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_misspec_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_modeled_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_misspec_fullcov_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_kcor_survey_pkmjd_bins_modeled_fullcov_v1.yaml`
- **Mechanism attribution (SALT2 metadata; real data):** adding calibrator-only linear terms in SALT2 stretch `x1` improves held-out calibrators modestly (Δlogp ≈ +1.1 in full-cov), but does **not** replace the need for a free `calibrator_offset_mag` (still Δlogp ≈ +5.4 under kcor gates). In a baseline-sweep (anchor-consistency), `x1`/`c`/`biasCor_m_b` linear terms reduce the inferred `delta_lnH0` by only **~1–2%** (vs **~24%** for `calibrator_offset_mag`).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_fullcov_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_mechanism_attribution_extgrid_more_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_fullcov_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_mechanism_attribution_extgrid_more_v1.yaml`
- **Injection bound (SALT2 metadata → H0 shift):** if you omit the `c_linear_mag` / `x1_linear_mag` / `biascor_m_b_linear_mag` terms, an injected calibrator-only dependence would need to be extremely large (≈0.6–0.9 mag per 1σ in `c`/`x1`/`biasCor_m_b`) to mimic the full ladder-vs-anchor `delta_lnH0`; when these terms are modeled, leakage into `delta_lnH0` is consistent with zero.  
  Reports: `outputs/pantheon_plus_shoes_ladder_injection_kcor_c_x1_biascor_misspec_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_kcor_c_x1_biascor_modeled_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_injection_kcor_c_x1_biascor_misspec_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_kcor_c_x1_biascor_modeled_v1.yaml`
- **CID-group holdout (joint stack; real data):** using a stricter holdout split that withholds *entire SNe* by name (`CID`) in the ladder part (while keeping hubble-flow + other probes fixed in TRAIN) still prefers a calibrator offset, though the gain is smaller on this split family (Δlogp ≈ +0.32 for `calibrator_offset_mag`, vs ≈ +0.10 for `x1_linear_cal`).  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_mechanism_attribution_kcor_extgrid_more_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_mechanism_attribution_kcor_extgrid_more_v1.yaml`
- **Combined “external gate” decomposition (kcor+SH0ES-linear-system; real data):** using kcor-variant calibration widths to bound per-survey/per-epoch offsets **and** a SH0ES-linear-system-inspired σ(`calibrator_offset_mag`)≈0.028, bounded survey/epoch fields alone explain ~0% of the joint-stack shift, while a bounded global `calibrator_offset_mag` reduces `delta_lnH0` by only ~11%. A metadata-rich constrained model can move `delta_lnH0` by ~16% but is disfavored by evidence.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_constrained_decomp_kcor_calhf_shoeslin_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_extgrid_more_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_constrained_decomp_kcor_calhf_shoeslin_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_extgrid_more_v1.yaml`  
  Priors: `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_kcor_calhf_plus_shoeslin_v1.json` (built via `scripts/merge_sigma_overrides.py`)
- **Prior-MC bound under combined gates (real data):** drawing unmodeled calibrator-step distortions from the combined kcor+SH0ES-linear-system priors yields p95(|δ`delta_lnH0`|)≈0.031, i.e. ≈39% of a reference “full tension” scale ln(73/67.4); in 20k draws, P(>100%)≈0.  
  Report: `outputs/pantheon_plus_shoes_ladder_prior_mc_constrained_kcor_calhf_shoeslin_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_prior_mc_constrained_kcor_calhf_shoeslin_v1.yaml`
- **Cov-projected “external” bounds for metadata proxies (JLA_SALT2; real data):** projecting the published JLA_SALT2 systematic covariance onto proxy vectors yields realistic prior widths for metadata terms (typical σ≈0.021–0.045 mag for `pkmjd_err_linear_mag`, `host_mass_step_mag`, `c/x1/biascor`, `mwebv`). Under these bounds, a CID-group holdout inside the joint stack prefers the **bounded metadata-rich** ladder model: Δlogp ≈ +0.40 (diag), exceeding the bounded `calibrator_offset_mag` model (Δlogp ≈ +0.28). Term ablations show the gain is dominated by **`host_mass_step` (Δlogp ≈ +0.29)** and **`pkmjd_err_linear` (Δlogp ≈ +0.26)**, while `mwebv`/`c_linear` contribute little.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/report.md`  
  Driver ranking: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/driver_ranking.md` (via `scripts/rank_predictive_score_drivers.py`)  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1.yaml`  
  Priors: `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_kcor_calhf_shoeslin_plus_covproj_JLA_SALT2_cal_v1.json` (built via `scripts/derive_proxy_priors_from_cov_projection.py` + `scripts/merge_sigma_overrides.py`)
- **Permutation-null “is this tied to the actual metadata?” (real data; covproj CID holdout):** for the same CID-holdout term ablations, a *within-survey* permutation-null test supports `host_mass_step` as a “real” (non-random) driver but does **not** support `pkmjd_err_linear`:
  - `+fields+host_mass_step`: observed mean Δlogp ≈ +0.2915; permuting `HOST_LOGMASS` among calibrators within each `IDSURVEY` gives p(Δlogp≥obs) ≈ **0.0138** (n=5000).
  - `+fields+pkmjd_err`: observed mean Δlogp ≈ +0.2647; permuting `PKMJDERR` among calibrators within each `IDSURVEY` gives p ≈ **0.127** (n=5000), consistent with a “generic regressor” rather than a specific metadata-linked effect.
  - Full `+bounded_fields_plus_metadata_bounded`: observed mean Δlogp ≈ +0.3985; permuting `HOST_LOGMASS` within-survey gives p ≈ **0.0778** (n=5000), i.e. marginal.
  Artifacts: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/permutation_null__fields_host_mass_step_host_logmass_within_survey_n5000.json`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/permutation_null__fields_pkmjd_err_pkmjd_err_within_survey_n5000.json`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/permutation_null__bounded_fields_plus_metadata_bounded_host_logmass_within_survey_n5000.json`
- **Harder generalization (calibrator survey holdout; covproj term ablations):** holding out calibrator SNe survey-by-survey (`group_var=idsurvey`, HF always in TRAIN) still yields large gains from the same bounded terms:
  - `+fields+host_mass_step`: Δlogp ≈ +1.35 (win-rate 100%)
  - `+fields+pkmjd_err`: Δlogp ≈ +1.28 (win-rate 90%)
  - Full `+bounded_fields_plus_metadata_bounded`: Δlogp ≈ +1.93 (win-rate 90%)
  Permutation-null on this split family (within-survey; n=5000): p ≈ **0.0104** for `host_mass_step` (permute `HOST_LOGMASS`) and p ≈ **0.1516** for `pkmjd_err` (permute `PKMJDERR`).
  Report: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1/report.md`  
  Artifacts: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1/permutation_null__fields_host_mass_step_host_logmass_within_survey_n5000.json`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1/permutation_null__fields_pkmjd_err_pkmjd_err_within_survey_n5000.json`
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1.yaml`
- **Host-mass effect is calibrator-specific (real data; covproj bounds):** if the host-mass step is allowed to apply to *all* ladder SNe instead of calibrators only, its held-out-calibrator benefit drops sharply, and an HF-only mass step gives ~no gain. On calibrator survey holdout:
  - calibrator-only: Δlogp ≈ +1.23 (vs bounded fields)
  - all-SNe: Δlogp ≈ +0.38
  - HF-only: Δlogp ≈ +0.00
  Report: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_hostmass_scope_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_hostmass_scope_v1.yaml`
- **Same conclusion under random calibrator holdout (real data; covproj bounds):** random splits (200 reps) show a much larger gap:
  - calibrator-only: Δlogp ≈ +3.50 (vs bounded fields)
  - all-SNe: Δlogp ≈ +1.07
  - HF-only: Δlogp ≈ +0.00
  Report: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_covproj_hostmass_scope_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_covproj_hostmass_scope_v1.yaml`
- **Calibration gates for these proxy terms (prior-MC + SBC + injections; real data + simulator):** under the combined kcor+SH0ES-linear+covproj bounds, a forward “prior-MC” draw of unmodeled distortions can explain p50/p95/p99 ≈ **20% / 60% / 79%** of a reference ln(73/67.4) tension scale (but P(>100%) remains ≈0). Repeated-noise SBC shows **no undercoverage** (it is conservative/over-covered), and injections confirm the dominant single-term H0-shift risks are `pkmjd_err_linear_mag` and `host_mass_step_mag` (each ≈15% of the reference scale at 1σ).  
  Reports: `outputs/pantheon_plus_shoes_ladder_prior_mc_constrained_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_sbc_constrained_covproj_JLA_SALT2_cal_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_covproj_metadata_misspec_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_covproj_metadata_modeled_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_prior_mc_constrained_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_v1.yaml`, `configs/pantheon_plus_shoes_ladder_sbc_constrained_covproj_JLA_SALT2_cal_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_covproj_metadata_misspec_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_covproj_metadata_modeled_v1.yaml`
- **Same suite with “realcal survey×epoch” constraints (real data):** using a merged external gate pack (kcor-variant priors + calibcov survey×time bins (min3) + SH0ES linear-system prior + covproj bounds), the prior-MC bound remains similar: frac(tension) p50/p95/p99 ≈ **0.211 / 0.621 / 0.811**, and SBC remains **conservative/over-covered** (e.g. 68% intervals cover the truth ≫68% for `delta_lnH0`).  
  Reports: `outputs/pantheon_plus_shoes_ladder_prior_mc_realcal_surveytime_shoeslin_covproj_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_sbc_realcal_surveytime_shoeslin_covproj_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_realcal_surveytime_shoeslin_covproj_misspec_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_injection_realcal_surveytime_shoeslin_covproj_modeled_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_prior_mc_realcal_surveytime_shoeslin_covproj_v1.yaml`, `configs/pantheon_plus_shoes_ladder_sbc_realcal_surveytime_shoeslin_covproj_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_realcal_surveytime_shoeslin_covproj_misspec_v1.yaml`, `configs/pantheon_plus_shoes_ladder_injection_realcal_surveytime_shoeslin_covproj_modeled_v1.yaml`
- **Host-mass alone “how much can it explain?” bound (prior-MC; covproj bounds):** drawing an unmodeled calibrator-only host-mass step with σ from the cov-projected external bounds yields frac(tension) p50/p95/p99 ≈ **0.124 / 0.360 / 0.473** of ln(73/67.4).  
  Report: `outputs/pantheon_plus_shoes_ladder_prior_mc_host_mass_step_covproj_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_prior_mc_host_mass_step_covproj_v1.yaml`
- **SH0ES calibrator-chain prior scale (linear-system; real data product):** the SH0ES DataRelease includes a compact linear system (`SH0ES_Data/all[LCY]_...fits`, `lstsq_results.txt`) with σ(`fivelogH0`)≈0.028 mag. Treating that as a *calibrator-chain-inspired* prior width for an additional `calibrator_offset_mag`, the joint stack under FRAGILISTIC survey priors supports only `calibrator_offset_mag ≈ 0.074 ± 0.021` (tension-reduction frac ≈ 0.16).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1.yaml`  
  Derivation: `scripts/derive_shoes_linear_system_fivelogh0_prior.py`
- **Injection mapping (calibrator step → H0 shift):** with an unmodeled calibrator-only magnitude shift injected into the ladder data, the induced bias in `delta_lnH0` follows the expected slope `d(delta_lnH0)/d(Δm)≈0.46`, so faking the full ladder-vs-anchor offset requires Δm≈0.19 mag.  
  Report: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_misspec_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_injection_calibcov_misspec_v1.yaml`
- **Anti-overfit gates:** cross-validated predictive scoring and exact Gaussian log-evidence both *penalize* adding flexible closure terms (HF redshift splines / sky low-ℓ modes) on the ladder subset.  
  Reports: `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`, `outputs/pantheon_plus_shoes_ladder_level_sweep_v2/report.md`
- **External H0 probes as `h0_grid` (TRGB / lenses / masers; stress-test):** adding these does not remove the need for a large calibrator↔HF offset in the joint stack, and calibrator holdout improvements persist.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1.yaml`
- **More external H0 grids (SBF + STRIDES; stress-test):** adding two additional late-time constraints (`sbf_blakeslee_2021`, `strides_shajib_2020`) still leaves the joint fit preferring a large calibrator↔HF step (`calibrator_offset_mag ≈ 0.15`) and leaves calibrator-holdout gains essentially unchanged (Δlogp ≈ +4.9 for `calibrator_offset_mag`).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_more_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_more_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_more_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_more_v1.yaml`  
  Build grids: `scripts/build_h0_grids_from_external_constraints.py` (inputs: `data/processed/external_constraints/h0_constraints_2026-02-05.json`)
- **CID holdout under the “realcal survey×epoch” pack (real data; stack + ext grids):** calibrator-CID holdout improvements persist under the same merged sigma pack: Δlogp ≈ +0.31 (`cal_offset_bounded`) and ≈ +0.38 (`+bounded_fields_plus_metadata_bounded`) vs baseline.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1/report.md`  
  Driver ranking: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1/driver_ranking.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1.yaml`
- **Independent probe adapter (sirens, Gate-2):** selection-corrected per-event `logL(H0)` grid + metadata cut/time drift audit (experimental; upstream Gate-2 product not complete yet).  
  Report: `outputs/siren_gate2_grid_audit_v2/report.md`  
  Reproduce: `configs/siren_gate2_grid_audit_v2.yaml`

For the full status + what’s still missing vs the full ambition, start at:
- `docs/LAYMAN_SUMMARY.md`
- `docs/SPEC_STATUS.md`

## Quickstart (uses existing venvs in this workspace)

Run with this repo’s venv:

```bash
.venv/bin/hubble-audit --help
.venv/bin/hubble-audit run configs/pantheon_plus_audit.yaml
```

Or run with the Project venv (also present in this workspace):

```bash
PYTHONPATH=src ../PROJECT/.venv/bin/python -m hubble_systematics.cli --help
```

Or install editable into that venv:

```bash
../PROJECT/.venv/bin/pip install -e .
hubble-audit --help
```

Example: run the Pantheon+ audit packet (baseline + cut scan + correlated-null drift MC):

```bash
PYTHONPATH=src ../PROJECT/.venv/bin/python -m hubble_systematics.cli run configs/pantheon_plus_audit.yaml
```

Example: ladder time-invariance null (fixed-support calibrators; shuffle-within-survey null):

```bash
PYTHONPATH=src ../PROJECT/.venv/bin/python -m hubble_systematics.cli run configs/pantheon_plus_shoes_ladder_time_invariance_null_fixedcal_v3.yaml
```

Outputs are written under `outputs/<run_id>/`.

## Tests

```bash
.venv/bin/pytest
```
