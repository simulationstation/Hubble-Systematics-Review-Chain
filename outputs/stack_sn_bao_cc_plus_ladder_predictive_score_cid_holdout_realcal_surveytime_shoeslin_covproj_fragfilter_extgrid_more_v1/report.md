# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_v1`

## Baseline fit


- N: 1713

- Ladder level: L1

- chi2/dof: 1566.74 / 1606

- log evidence: 879.88

- H0_eff (from delta_lnH0): 69.068 ± 0.286

- Equivalent Δμ [mag]: -0.0531 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1194.820101374756, n=1322

  - pantheon_plus_shoes_ladder: chi2=321.2944075870158, n=354

  - bao: chi2=22.31093012063113, n=12

  - chronometers: chi2=12.428316053359971, n=19

  - h0_grid: chi2=0.1852869391019719, n=1

  - h0_grid: chi2=3.0398000950860684, n=1

  - h0_grid: chi2=0.20385759125036368, n=1

  - h0_grid: chi2=6.203673663443174, n=1

  - h0_grid: chi2=3.481532484698433, n=1

  - h0_grid: chi2=2.7743159668268715, n=1

## Predictive score


- Mode: group_holdout

- N splits: 43

- Train frac: None

- Always include calibrators: False

- Always include hubble-flow: True

- +bounded_calibration_fields: mean_logp=0.49, Δlogp_vs_base=0.09

- +bounded_fields_plus_metadata_bounded: mean_logp=0.80, Δlogp_vs_base=0.40

- +cal_offset_bounded: mean_logp=0.71, Δlogp_vs_base=0.31

- baseline: mean_logp=0.40, Δlogp_vs_base=0.00
