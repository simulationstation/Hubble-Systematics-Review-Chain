# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_extgrid_all_v1`

## Baseline fit


- N: 1711

- Ladder level: L1

- chi2/dof: 1524.87 / 1693

- log evidence: 864.76

- H0_eff (from delta_lnH0): 68.926 ± 0.290

- Equivalent Δμ [mag]: -0.0486 ± 0.0091

- Per-part chi2:

  - pantheon_plus: chi2=1181.1762329554044, n=1322

  - pantheon_plus_shoes_ladder: chi2=301.1800266396357, n=354

  - bao: chi2=20.062183861237326, n=12

  - chronometers: chi2=12.427320494526015, n=19

  - h0_grid: chi2=0.2651785039832503, n=1

  - h0_grid: chi2=0.1708201134257844, n=1

  - h0_grid: chi2=6.642840303351365, n=1

  - h0_grid: chi2=2.946678913767049, n=1

## Predictive score


- Mode: random

- N splits: 200

- Train frac: 0.7

- Always include calibrators: False

- Always include hubble-flow: True

- +cal_offset_calibcov_survey_priors: mean_logp=8.33, Δlogp_vs_base=3.85

- +cal_offset_free: mean_logp=9.70, Δlogp_vs_base=5.23

- baseline: mean_logp=4.48, Δlogp_vs_base=0.00

- baseline_calibcov_survey_priors: mean_logp=4.70, Δlogp_vs_base=0.22
