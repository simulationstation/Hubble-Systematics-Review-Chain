# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1`

## Baseline fit


- N: 1707

- Ladder level: L1

- chi2/dof: 1514.33 / 1689

- log evidence: 860.18

- H0_eff (from delta_lnH0): 68.710 ± 0.301

- Equivalent Δμ [mag]: -0.0418 ± 0.0095

- Per-part chi2:

  - pantheon_plus: chi2=1181.2420066235086, n=1322

  - pantheon_plus_shoes_ladder: chi2=303.2302999244981, n=354

  - bao: chi2=17.40403647898752, n=12

  - chronometers: chi2=12.449773628074224, n=19

## Predictive score


- Mode: random

- N splits: 200

- Train frac: 0.7

- Always include calibrators: False

- Always include hubble-flow: True

- +cal_offset_calibcov_survey_priors: mean_logp=8.07, Δlogp_vs_base=4.12

- +cal_offset_free: mean_logp=9.70, Δlogp_vs_base=5.74

- baseline: mean_logp=3.96, Δlogp_vs_base=0.00

- baseline_calibcov_survey_priors: mean_logp=4.17, Δlogp_vs_base=0.21
