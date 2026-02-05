# Audit report

Run dir: `outputs/siren_gate2_grid_audit_v1`

## Baseline fit


- N: 33

- Ladder level: L0

- chi2/dof: 0.01 / 32

- log evidence: -40.49

- H0_eff (from delta_lnH0): 60.836 ± 12.735

- Equivalent Δμ [mag]: 0.2225 ± 0.4546

## Cut scan


- Cut var: ess_min

- Cut mode: geq

- Figure: `outputs/siren_gate2_grid_audit_v1/figures/cut_scan_param.png`

## Correlated-cut null


- Drift param: delta_lnH0

- Cut mode: geq

- p-values (end_to_end / max_pair / path_length): {'end_to_end': 0.814, 'max_pair': 0.806, 'path_length': 0.884}

- Figure: `outputs/siren_gate2_grid_audit_v1/figures/correlated_cut_null_hist.png`

## Injection/recovery


- Mechanism: y_offset

- Param of interest: delta_lnH0

- Approx bias slope d(param)/d(amp): 0.7823

- Amp to explain |delta_lnH0|: 0.131

- Figure: `outputs/siren_gate2_grid_audit_v1/figures/injection_suite.png`

## SBC

- Figure: `outputs/siren_gate2_grid_audit_v1/figures/sbc_coverage.png`

## Split fit


## Split null


- Split var: event_mjd (bins)

- Param: delta_lnH0

- Shuffle within: pe_analysis_id

- p-values (span / chi2_const): {'chi2_const': 0.614, 'span': 0.578}

- Figure: `outputs/siren_gate2_grid_audit_v1/figures/split_null_hist.png`
