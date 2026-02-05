# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_calibcov_bounded_extgrid_more_v1`

## Baseline fit


- N: 1713

- Ladder level: L1

- chi2/dof: 1549.31 / 1695

- log evidence: 879.62

- H0_eff (from delta_lnH0): 69.065 ± 0.286

- Equivalent Δμ [mag]: -0.0530 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1186.4670132157412, n=1322

  - pantheon_plus_shoes_ladder: chi2=312.2497722974261, n=354

  - bao: chi2=22.258838743804475, n=12

  - chronometers: chi2=12.428163231720566, n=19

  - h0_grid: chi2=0.18684303485860362, n=1

  - h0_grid: chi2=3.044316456029821, n=1

  - h0_grid: chi2=0.20312192797502474, n=1

  - h0_grid: chi2=6.212890726593692, n=1

  - h0_grid: chi2=3.4858240822378757, n=1

  - h0_grid: chi2=2.7779407653707833, n=1

## Predictive score


- Mode: random

- N splits: 200

- Train frac: 0.7

- Always include calibrators: False

- Always include hubble-flow: True

- +cal_offset: mean_logp=8.45, Δlogp_vs_base=3.44

- +cal_offset_plus_cal_time_bins: mean_logp=9.16, Δlogp_vs_base=4.16

- +cal_offset_plus_pkmjderr_linear_cal: mean_logp=9.58, Δlogp_vs_base=4.57

- +cal_offset_plus_pkmjderr_plus_cal_time_bins: mean_logp=10.16, Δlogp_vs_base=5.15

- +cal_time_bins: mean_logp=6.42, Δlogp_vs_base=1.42

- +pkmjderr_linear_cal: mean_logp=9.37, Δlogp_vs_base=4.36

- +vpecerr_linear_cal: mean_logp=5.19, Δlogp_vs_base=0.18

- baseline: mean_logp=5.01, Δlogp_vs_base=0.00
