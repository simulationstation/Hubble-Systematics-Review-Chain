# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_covproj_term_ablations_v1`

## Baseline fit


- N: 1713

- Ladder level: L1

- chi2/dof: 1555.90 / 1695

- log evidence: 878.31

- H0_eff (from delta_lnH0): 69.058 ± 0.286

- Equivalent Δμ [mag]: -0.0528 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1187.9052981967423, n=1322

  - pantheon_plus_shoes_ladder: chi2=317.4684070850479, n=354

  - bao: chi2=22.12991651763276, n=12

  - chronometers: chi2=12.427807668530884, n=19

  - h0_grid: chi2=0.1907456913470504, n=1

  - h0_grid: chi2=3.0555759140661314, n=1

  - h0_grid: chi2=0.20129604151859348, n=1

  - h0_grid: chi2=6.235869212382085, n=1

  - h0_grid: chi2=3.49652084063892, n=1

  - h0_grid: chi2=2.7869761158027972, n=1

## Predictive score


- Mode: group_holdout

- N splits: 10

- Train frac: None

- Always include calibrators: False

- Always include hubble-flow: True

- +bounded_calibration_fields: mean_logp=1.82, Δlogp_vs_base=0.12

- +bounded_fields_plus_metadata_bounded: mean_logp=3.63, Δlogp_vs_base=1.93

- +cal_offset_bounded: mean_logp=2.89, Δlogp_vs_base=1.19

- +fields+host_mass_step: mean_logp=3.05, Δlogp_vs_base=1.35

- +fields+pkmjd_err: mean_logp=2.98, Δlogp_vs_base=1.28

- baseline: mean_logp=1.70, Δlogp_vs_base=0.00
