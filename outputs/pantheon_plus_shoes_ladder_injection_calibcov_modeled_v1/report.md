# Audit report

Run dir: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_v1`

## Baseline fit


- N: 354

- Ladder level: L2

- chi2/dof: 290.49 / 337

- log evidence: 223.04

- H0_eff (from delta_lnH0): 73.588 ± 1.239

- Equivalent Δμ [mag]: -0.1907 ± 0.0366

- Calibrator-offset-equivalent H0 [km/s/Mpc]: 67.400 ± 0.599

- calibrator_offset_mag: -0.0000 ± 0.0193

## Injection/recovery


- inject_calibrator_offset_recovery: calibrator_offset_mag → calibrator_offset_mag, slope=0.0003, amp_for_|delta_lnH0|=277.077 (`outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_v1/figures/injection_suite_inject_calibrator_offset_recovery.png`)

- inject_idsurvey_5_hf: idsurvey_offset_mag → delta_lnH0, slope=-0.0719, amp_for_|delta_lnH0|=1.222 (`outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_v1/figures/injection_suite_inject_idsurvey_5_hf.png`)

- inject_pkmjd_bin1_cal: pkmjd_bin_offset_mag → delta_lnH0, slope=0.0516, amp_for_|delta_lnH0|=1.704 (`outputs/pantheon_plus_shoes_ladder_injection_calibcov_modeled_v1/figures/injection_suite_inject_pkmjd_bin1_cal.png`)
