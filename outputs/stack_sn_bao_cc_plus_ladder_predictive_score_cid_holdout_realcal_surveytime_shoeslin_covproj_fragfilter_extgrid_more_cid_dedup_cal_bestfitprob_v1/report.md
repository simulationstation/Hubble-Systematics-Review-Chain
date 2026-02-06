# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dedup_cal_bestfitprob_v1`

## Baseline fit


- N: 1679

- Ladder level: L1

- chi2/dof: 1530.45 / 1572

- log evidence: 853.61

- H0_eff (from delta_lnH0): 69.053 ± 0.287

- Equivalent Δμ [mag]: -0.0526 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1194.732142796547, n=1322

  - pantheon_plus_shoes_ladder: chi2=285.2351849871784, n=320

  - bao: chi2=22.046626972733776, n=12

  - chronometers: chi2=12.427595425548896, n=19

  - h0_grid: chi2=0.19330663330902595, n=1

  - h0_grid: chi2=3.0629130900187005, n=1

  - h0_grid: chi2=0.20011244742140738, n=1

  - h0_grid: chi2=6.250843040856718, n=1

  - h0_grid: chi2=3.503489532916471, n=1

  - h0_grid: chi2=2.7928628836806615, n=1

## Predictive score


- Mode: group_holdout

- N splits: 43

- Train frac: None

- Always include calibrators: False

- Always include hubble-flow: True

- +bounded_calibration_fields: mean_logp=0.30, Δlogp_vs_base=0.04

- +bounded_fields_plus_metadata_bounded: mean_logp=0.48, Δlogp_vs_base=0.22

- +cal_offset_bounded: mean_logp=0.47, Δlogp_vs_base=0.21

- baseline: mean_logp=0.26, Δlogp_vs_base=0.00
