# Audit report

Run dir: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_anchor_enforced_v1`

## Baseline fit


- N: 354

- Ladder level: L2

- chi2/dof: 304.90 / 333

- log evidence: 214.15

- H0_eff (from delta_lnH0): 67.400 ± 0.007

- Equivalent Δμ [mag]: -0.0000 ± 0.0002

- Calibrator-offset-equivalent H0 [km/s/Mpc]: 65.864 ± 0.501

- calibrator_offset_mag: +0.0501 ± 0.0165

- Time-bin offsets: max |mean|=0.0311 (n=4)

## Injection/recovery


- inject_calibrator_offset_recovery: calibrator_offset_mag → calibrator_offset_mag, slope=0.4503, amp_for_|delta_lnH0|=0.000 (`outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_anchor_enforced_v1/figures/injection_suite_inject_calibrator_offset_recovery.png`)

- inject_idsurvey_5_recovery: idsurvey_offset_mag → survey_offset_5, slope=0.5014, amp_for_|delta_lnH0|=0.000 (`outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_anchor_enforced_v1/figures/injection_suite_inject_idsurvey_5_recovery.png`)

- inject_pkmjd_bin1_recovery: pkmjd_bin_offset_mag → pkmjd_bin_offset_1, slope=0.1145, amp_for_|delta_lnH0|=0.000 (`outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_anchor_enforced_v1/figures/injection_suite_inject_pkmjd_bin1_recovery.png`)
