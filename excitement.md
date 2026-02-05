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
