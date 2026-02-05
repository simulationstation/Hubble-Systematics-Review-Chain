# Audit report

Run dir: `outputs/siren_gate2_grid_audit_v2`

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

- Figure: `outputs/siren_gate2_grid_audit_v2/figures/cut_scan_param.png`

## Correlated-cut null


- Drift param: delta_lnH0

- Cut mode: geq

- p-values (end_to_end / max_pair / path_length): {'end_to_end': 0.08318, 'max_pair': 2e-05, 'path_length': 0.0}

- Figure: `outputs/siren_gate2_grid_audit_v2/figures/correlated_cut_null_hist.png`

## Split fit


## Split null


- Split var: event_mjd (bins)

- Param: delta_lnH0

- Shuffle within: pe_analysis_id

- p-values (span / chi2_const): {'chi2_const': 0.621, 'span': 0.5762}

- Figure: `outputs/siren_gate2_grid_audit_v2/figures/split_null_hist.png`
