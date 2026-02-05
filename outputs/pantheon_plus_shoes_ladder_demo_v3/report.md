# Audit report

Run dir: `outputs/pantheon_plus_shoes_ladder_demo_v3`

## Baseline fit


- N: 354

- Ladder level: L2

- chi2/dof: 275.23 / 338

- H0_eff (from delta_lnH0): 73.552 ± 1.078

- Equivalent Δμ [mag]: -0.1897 ± 0.0318

## Cut scan

- Figure: `outputs/pantheon_plus_shoes_ladder_demo_v3/figures/cut_scan_param.png`

## Correlated-cut null


- Drift param: delta_lnH0

- p-values (end_to_end / max_pair / path_length): {'end_to_end': 0.495, 'max_pair': 0.515, 'path_length': 0.79}

- Figure: `outputs/pantheon_plus_shoes_ladder_demo_v3/figures/correlated_cut_null_hist.png`

## Injection/recovery


- Mechanism: calibrator_offset_mag

- Param of interest: delta_lnH0

- Approx bias slope d(param)/d(amp): 0.4599

- Amp to explain |delta_lnH0|: 0.190

- Figure: `outputs/pantheon_plus_shoes_ladder_demo_v3/figures/injection_suite.png`

## SBC

- Figure: `outputs/pantheon_plus_shoes_ladder_demo_v3/figures/sbc_coverage.png`
