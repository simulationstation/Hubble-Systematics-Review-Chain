# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_attribution_kcor_extgrid_more_fullcov_v1`

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


- Mode: random

- N splits: 200

- Train frac: 0.7

- Always include calibrators: False

- Always include hubble-flow: True

- +biascor_m_b_linear_cal: mean_logp=6.47, Δlogp_vs_base=-0.20

- +c_linear_cal: mean_logp=6.63, Δlogp_vs_base=-0.05

- +c_x1_biascor_cal: mean_logp=7.57, Δlogp_vs_base=0.89

- +cal_offset: mean_logp=12.11, Δlogp_vs_base=5.44

- +cal_offset_plus_c_x1_biascor_cal: mean_logp=12.11, Δlogp_vs_base=5.44

- +cal_offset_plus_survey_pkmjd_bins_plus_c_x1_biascor_cal: mean_logp=12.13, Δlogp_vs_base=5.45

- +survey_pkmjd_bins: mean_logp=6.73, Δlogp_vs_base=0.06

- +survey_pkmjd_bins_plus_c_x1_biascor_cal: mean_logp=7.61, Δlogp_vs_base=0.94

- +x1_linear_cal: mean_logp=7.72, Δlogp_vs_base=1.05

- baseline: mean_logp=6.67, Δlogp_vs_base=0.00
