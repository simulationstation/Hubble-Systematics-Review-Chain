# Excitement log

This file is for “potentially exciting” intermediate indications. Treat entries as **hypotheses to
validate**, not conclusions.

## 2026-02-05 — Calibrator holdout generalization (real data; simplified model)

Holding out **calibrator SNe** while keeping the **Hubble-flow** sample fixed in TRAIN produces a
large out-of-sample predictive improvement from **calibrator-only** mechanisms:

- Random calibrator holdout: Δlogp ≈ +10.2 for `calibrator_offset_mag` (and Δlogp ≈ +8.3 for
  calibrator time-bin offsets via `pkmjd_bins` on calibrators).
- Survey-by-survey calibrator holdout: improvements persist but are smaller (Δlogp ≈ +3.8 and +3.5).

Artifacts:
- `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1/report.md`
- `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1/report.md`
- Full-cov robustness:
  - `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`
  - `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1/report.md`
- Joint-stack (SN-only+BAO+CC+ladder) version:
  - `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1/report.md`
  - `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`

Caveats:
- This is a **linear-Gaussian audit model** on the public Pantheon+SH0ES table, not an end-to-end
  SH0ES reanalysis.
- Predictive scoring currently uses **subset covariances** (no cross-cov between TRAIN and TEST) and
  in these runs we used **diagonalized errors** for speed/stability.
- This does **not** identify the *physical* origin (instrument/calibration/selection) of the
  calibrator↔HF mismatch; it only indicates the mismatch behaves like a **coherent** effect rather
  than a tiny-N artifact within this framework.

## 2026-02-05 — PKMJDERR proxy is a strong “replacement” candidate (joint stack; CV)

Inside the joint anchor-consistency stack (SN-only+BAO+CC+ladder), a simple calibrator-only linear
term in `PKMJDERR` (time-of-maximum uncertainty) gives a large held-out calibrator predictive
improvement:

- `pkmjd_err_linear` on calibrators: Δlogp ≈ +4.6 (vs baseline)
- compared to `calibrator_offset_mag`: Δlogp ≈ +5.7
- while `m_b_corr_err_VPEC` linear gives ~0 improvement (Δlogp ≈ +0.1)

Artifacts:
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1/report.md`

## 2026-02-05 — Joint-stack survey holdout persists (robustness; real data)

Holding out calibrator SNe **survey-by-survey** inside the joint anchor-consistency stack
(SN-only+BAO+CC+ladder) still prefers calibrator-only corrections, though the improvement is smaller
than random holdout (as expected for a harder generalization test):

- Diagonal errors: Δlogp ≈ +1.92 for `cal_time_bins`, +1.72 for `cal_offset`, +1.51 for
  `pkmjderr_linear_cal`.
- Full cov: Δlogp ≈ +2.34 for `cal_time_bins`, +2.29 for `cal_offset`, +1.45 for
  `pkmjderr_linear_cal`.

Artifacts:
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1/report.md`
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1/report.md`

## 2026-02-05 — External-prior “gate” is strong (real data; per-survey/per-epoch variants)

If external calibration work can really bound calibrator-step distortions at the ~0.02 mag level,
the joint anchor-consistency stack cannot maintain the large ~0.16 mag calibrator↔HF correction
without paying a big evidence penalty:

- Tight `sigma_calibrator_offset_mag=0.02`: ΔlogZ ≈ -7.5.
- Tight per-survey calibrator offsets (`sigma_cal_survey_offset_mag=0.02`): max |offset| ≈ 0.008 mag and ΔlogZ ≈ -11.3.
- Tight calibrator time-bin offsets (`sigma_pkmjd_bin_offset_mag=0.02`): max |offset| ≈ 0.021 mag and ΔlogZ ≈ -10.1.

Artifact:
- `outputs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1/report.md`

## 2026-02-05 — External H0 grids don’t remove the calibrator↔HF mismatch (stress-test; real data)

Adding TRGB / strong-lens / megamaser H0 constraints as `h0_grid` posteriors does not remove the need
for a large calibrator↔HF offset in the joint stack (still ~0.15–0.16 mag), and calibrator-holdout
predictive improvements persist.

Artifacts:
- Baseline fits: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1/report.md`,
  `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1/report.md`,
  `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1/report.md`,
  and with additional grids (SBF + STRIDES): `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_more_v1/report.md`
- Holdouts: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1/report.md`,
  `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1/report.md`,
  and the expanded-grid holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_more_v1/report.md`

## 2026-02-05 — Covariance-implied “realistic” per-survey/epoch bounds still too small (real data)

If we derive prior widths for calibrator-step distortions directly from the published Pantheon+SH0ES
STAT+SYS covariance (typical σ≈0.02–0.05 mag), the joint anchor-consistency stack cannot sustain the
full ~0.16 mag calibrator↔HF step:

- Global `calibrator_offset_mag` is forced down to ~0.057 mag (ΔlogZ ≈ -6.7).
- Per-survey calibrator offsets top out around ~0.034 mag (ΔlogZ ≈ -11.3).
- Calibrator time-bin offsets top out around ~0.049 mag (ΔlogZ ≈ -9.1).

Artifacts:
- Derivation: `scripts/derive_pantheon_shoes_cov_priors.py`
- Sigma file: `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_cov_v1.json`
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1/report.md`

## 2026-02-05 — Brout+21 “FRAGILISTIC” zeropoint covariance implies mmag survey priors (real data)

Pantheon+ ships an external calibration product (`FRAGILISTIC_COVARIANCE.npz`) described as the
zeropoint-offset covariance from Brout+21. When compressed into per-survey offset priors, typical
sigmas are only a few mmag (σ≈0.002–0.008 mag).

Applying these tight survey-calibration priors:

- does **not** reduce the needed calibrator↔HF step when that step is allowed:
  `calibrator_offset_mag ≈ 0.164 ± 0.031` (still ~0.16),
- and calibrator-holdout predictive scoring still strongly prefers the calibrator offset under
  these priors (Δlogp ≈ +5.90).

Artifacts:
- Data: `data/raw/pantheon_plus_calibration/FRAGILISTIC_COVARIANCE.npz`
- Derivation: `scripts/derive_pantheon_shoes_fragilistic_priors.py`

## 2026-02-06 — Under cov-projected proxy bounds, `host_mass_step` + `pkmjd_err` dominate CID-holdout gains

Using “budget-like” external bounds for SALT2 metadata terms derived by projecting the published
JLA_SALT2 systematic covariance onto proxy vectors (σ≈0.021–0.045 mag), the joint-stack CID-holdout
prefers a bounded metadata-rich model with a sizable out-of-sample gain (Δlogp ≈ +0.40).

In term ablations, nearly all of that improvement comes from:
- `host_mass_step` (Δlogp ≈ +0.29), and
- `pkmjd_err_linear` (Δlogp ≈ +0.26),
with smaller contributions from `x1_linear` and negligible impact from `mwebv`/`c_linear`.

This suggests the “most leverage” metadata terms (in this simplified linear-Gaussian audit) are
highly specific and therefore good targets for *real* external calibration constraints (Gaia cross-cal,
overlap residuals, zeropoint logs).

Artifacts:
- Covproj CID holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_extgrid_more_v1/report.md`
- Term ablations: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/report.md`
- Driver ranking: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/driver_ranking.md`
- Injection mapping (misspec/modeled): `outputs/pantheon_plus_shoes_ladder_injection_covproj_metadata_misspec_v1/report.md`,
  `outputs/pantheon_plus_shoes_ladder_injection_covproj_metadata_modeled_v1/report.md`
- Sigma file: `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_fragilistic_v1.json`
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1/report.md`

## 2026-02-06 — Permutation-null supports host-mass as “real”, not PKMJDERR (real data)

On the same covproj CID-holdout term-ablation run, a within-survey permutation-null suggests the
`host_mass_step` gain is tied to the *actual* `HOST_LOGMASS` values (unlikely under permutation),
while the `pkmjd_err_linear` gain is not:

- `+fields+host_mass_step` (permute `HOST_LOGMASS` within `IDSURVEY`): p(Δlogp≥obs) ≈ 0.0138 (n=5000)
- `+fields+pkmjd_err` (permute `PKMJDERR` within `IDSURVEY`): p ≈ 0.127 (n=5000)

Artifacts:
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/permutation_null__fields_host_mass_step_host_logmass_within_survey_n5000.json`
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_covproj_term_ablations_v1/permutation_null__fields_pkmjd_err_pkmjd_err_within_survey_n5000.json`

On the harder **calibrator survey holdout** split family, the same story holds:
- `+fields+host_mass_step`: p ≈ 0.0104 (n=5000)
- `+fields+pkmjd_err`: p ≈ 0.1516 (n=5000)

Artifacts:
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1/permutation_null__fields_host_mass_step_host_logmass_within_survey_n5000.json`
- `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1/permutation_null__fields_pkmjd_err_pkmjd_err_within_survey_n5000.json`

## 2026-02-05 — CALIB-only covariance gate sharply bounds the calibrator step (real data)

Pantheon+SH0ES also ships systematic-group covariance blocks. Using the **CALIB-only** grouping
(`Pantheon+SH0ES_122221_CALIB.cov`) to derive sigma overrides yields a tight bound
σ(`calibrator_offset_mag`)≈0.019 mag.

Under that bound, the joint anchor-consistency stack can only support a much smaller step:

- `calibrator_offset_mag ≈ 0.045 ± 0.016` (tension-reduction frac ≈ 0.09 vs ≈ 0.37 when free),
- calibrator-holdout predictive scoring still prefers the mechanism but with smaller gain
  (Δlogp ≈ +4.1 vs +5.7 when free).

Artifacts:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1/report.md`

## 2026-02-05 — All systematic-group covariance blocks imply ~0.02 mag coherent scale (real data)

Pantheon+SH0ES ships many systematic-group covariance blocks under `sytematic_groupings/`. Deriving
sigma overrides from each grouping separately yields very similar coherent-step scales:
σ(`calibrator_offset_mag`)≈0.019–0.022 mag across the full set.

Under those grouping-implied bounds, the joint stack only supports a small step
(`calibrator_offset_mag ≈ 0.04–0.05`) and tension reduction fractions stay at the ~0.07–0.09 level.

Artifacts:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_groupings_gates_extgrid_all_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_groupings_extgrid_all_v1/report.md`

## 2026-02-05 — Per-survey×epoch offsets are too weak to matter (real data + injection map)

Allowing per-survey time-bin offsets (`survey_pkmjd_bins`) provides only a small calibrator-holdout
gain (Δlogp ≈ +1.1 when unconstrained; ≈ +0.5 under CALIB.cov-derived bounds).

An injection map shows that shifting `delta_lnH0` by the full ladder-vs-anchor amount via a **single**
survey×time-bin would require an implausibly large offset (multiple magnitudes).

Artifacts:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_surveytime_gates_extgrid_all_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_calibcov_extgrid_all_v1/report.md`
- Injection: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_survey_pkmjd_bins_misspec_v1/report.md`

## 2026-02-05 — SH0ES linear-system σ(fivelogH0) used as a calibrator-step prior (real data product)

The SH0ES DataRelease includes `lstsq_results.txt` with σ(`fivelogH0`)≈0.028 mag. Treating that as a
calibrator-chain-inspired prior width for an *additional* `calibrator_offset_mag` reduces the step
supported by the joint stack under FRAGILISTIC survey priors:

- `calibrator_offset_mag ≈ 0.074 ± 0.021` (tension-reduction frac ≈ 0.16).

Artifacts:
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1/report.md`
- Derivation: `scripts/derive_shoes_linear_system_fivelogh0_prior.py`

## 2026-02-05 — SNANA kcor calibration variants bound survey×epoch offsets (real data + metadata)

Pantheon+ ships SNANA kcor calibration FITS files (`data/raw/pantheon_plus_calibration/SNANA_kcor/`)
that include a `ZPoff` table per file. Using the spread across available calibration variants as a
heuristic per-survey calibration scale yields σ≈0.01–0.015 mag for the relevant low-z surveys.

Under these bounds:
- per-survey epoch-bin offsets (`survey_pkmjd_bins`) remain tiny (max |mean| ≈ 0.006 mag),
- held-out calibrators show essentially no gain from `survey_pkmjd_bins` once constrained (Δlogp ≈ +0.07),
- and the joint stack still prefers a large global `calibrator_offset_mag` when that is allowed.

Artifacts:
- Priors: `data/processed/external_calibration/pantheon_plus_shoes_sigma_overrides_from_kcor_variants_v1.json` (derived by `scripts/derive_pantheon_shoes_kcor_variant_priors.py`)
- Gate sweep: `outputs/stack_sn_bao_cc_plus_ladder_surveytime_kcor_gates_extgrid_more_v1/report.md`
- Holdout: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_surveytime_kcor_extgrid_more_v1/report.md`

Follow-on:
- Use the new `prior_mc` task to translate these bounds into an “explainable fraction” of the ladder
  offset. (Example: `outputs/pantheon_plus_shoes_ladder_prior_mc_kcor_timebins_v1/report.md`.)
