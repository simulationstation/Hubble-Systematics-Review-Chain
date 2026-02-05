# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_more_v1`

## Baseline fit


- N: 1713

- Ladder level: L1

- chi2/dof: 1531.63 / 1695

- log evidence: 866.17

- H0_eff (from delta_lnH0): 69.048 ± 0.286

- Equivalent Δμ [mag]: -0.0524 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1181.1403400392865, n=1322

  - pantheon_plus_shoes_ladder: chi2=300.06405759462234, n=354

  - bao: chi2=21.960204668640216, n=12

  - chronometers: chi2=12.427389988268006, n=19

  - h0_grid: chi2=0.1959974710676067, n=1

  - h0_grid: chi2=3.0705796303884525, n=1

  - h0_grid: chi2=0.19888095713122134, n=1

  - h0_grid: chi2=6.266489041611316, n=1

  - h0_grid: chi2=3.5107695350498482, n=1

  - h0_grid: chi2=2.7990130023874356, n=1

## Predictive score


- Mode: random

- N splits: 200

- Train frac: 0.7

- Always include calibrators: False

- Always include hubble-flow: True

- +cal_offset: mean_logp=9.70, Δlogp_vs_base=4.89

- +cal_time_bins: mean_logp=9.43, Δlogp_vs_base=4.62

- +pkmjderr_linear_cal: mean_logp=8.83, Δlogp_vs_base=4.01

- baseline: mean_logp=4.82, Δlogp_vs_base=0.00
