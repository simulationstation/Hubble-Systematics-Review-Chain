# Audit report

Run dir: `outputs/pantheon_plus_shoes_ladder_injection_misspec_v3`

## Baseline fit


- N: 354

- Ladder level: L2

- chi2/dof: 275.23 / 338

- log evidence: 207.85

- H0_eff (from delta_lnH0): 73.552 ± 1.078

- Equivalent Δμ [mag]: -0.1897 ± 0.0318

## Injection/recovery


- cal_offset: calibrator_offset_mag → delta_lnH0, slope=0.4599, amp_for_|delta_lnH0|=0.190 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_cal_offset.png`)

- dust_mwebv: mwebv_linear_mag → delta_lnH0, slope=-0.0034, amp_for_|delta_lnH0|=25.851 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_dust_mwebv.png`)

- hf_offset: hf_offset_mag → delta_lnH0, slope=-0.4607, amp_for_|delta_lnH0|=0.190 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_hf_offset.png`)

- host_mass_step: host_mass_step_mag → delta_lnH0, slope=0.0372, amp_for_|delta_lnH0|=2.348 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_host_mass_step.png`)

- quality_m_b_corr_err: m_b_corr_err_linear_mag → delta_lnH0, slope=0.1930, amp_for_|delta_lnH0|=0.452 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_quality_m_b_corr_err.png`)

- sky_dipole_cmb: sky_dipole_mag → delta_lnH0, slope=0.2206, amp_for_|delta_lnH0|=0.396 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_sky_dipole_cmb.png`)

- time_pkmjd: pkmjd_linear_mag → delta_lnH0, slope=0.1283, amp_for_|delta_lnH0|=0.681 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_time_pkmjd.png`)

- z_linear: z_linear_mag → delta_lnH0, slope=-0.4190, amp_for_|delta_lnH0|=0.208 (`outputs/pantheon_plus_shoes_ladder_injection_misspec_v3/figures/injection_suite_z_linear.png`)
