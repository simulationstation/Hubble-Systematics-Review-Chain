# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_extgrid_more_v1`

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

- N splits: 43

- Train frac: None

- Always include calibrators: False

- Always include hubble-flow: True

- +bounded_calibration_fields: mean_logp=0.46, Δlogp_vs_base=0.03

- +bounded_fields_plus_metadata_bounded: mean_logp=0.83, Δlogp_vs_base=0.40

- +cal_offset_bounded: mean_logp=0.72, Δlogp_vs_base=0.28

- baseline: mean_logp=0.43, Δlogp_vs_base=0.00
