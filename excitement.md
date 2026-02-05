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
  `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1/report.md`
- Holdouts: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1/report.md`,
  `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1/report.md`

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
