# Audit report

Run dir: `outputs/pantheon_plus_shoes_ladder_mechanisms_v1`

## Baseline fit


- N: 354

- Ladder level: L4

- chi2/dof: 251.85 / 307

- H0_eff (from delta_lnH0): 74.217 ± 7.165

- Equivalent Δμ [mag]: -0.2092 ± 0.2096

## Cut scan

- Figure: `outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/cut_scan_param.png`

## Correlated-cut null


- Drift param: delta_lnH0

- p-values (end_to_end / max_pair / path_length): {'end_to_end': 0.455, 'max_pair': 0.305, 'path_length': 0.5}

- Figure: `outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/correlated_cut_null_hist.png`

## Injection/recovery


- calibrator_offset: calibrator_offset_mag → delta_lnH0 (`outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/injection_suite_calibrator_offset.png`)

- dust_mwebv: mwebv_linear_mag → delta_lnH0 (`outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/injection_suite_dust_mwebv.png`)

- host_mass_step: host_mass_step_mag → delta_lnH0 (`outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/injection_suite_host_mass_step.png`)

- time_pkmjd: pkmjd_linear_mag → delta_lnH0 (`outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/injection_suite_time_pkmjd.png`)

## SBC

- Figure: `outputs/pantheon_plus_shoes_ladder_mechanisms_v1/figures/sbc_coverage.png`
