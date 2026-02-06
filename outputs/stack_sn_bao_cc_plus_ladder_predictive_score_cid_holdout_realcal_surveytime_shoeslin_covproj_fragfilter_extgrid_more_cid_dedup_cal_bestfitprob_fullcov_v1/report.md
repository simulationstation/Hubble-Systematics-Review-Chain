# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dedup_cal_bestfitprob_fullcov_v1`

## Baseline fit


- N: 1679

- Ladder level: L1

- chi2/dof: 1530.45 / 1572

- log evidence: 853.61

- H0_eff (from delta_lnH0): 69.053 ± 0.287

- Equivalent Δμ [mag]: -0.0526 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1194.732142796547, n=1322

  - pantheon_plus_shoes_ladder: chi2=285.23518498719545, n=320

  - bao: chi2=22.046626972703898, n=12

  - chronometers: chi2=12.42759542554882, n=19

  - h0_grid:trgb_freedman_2021_review: chi2=0.19330663330995113, n=1

  - h0_grid:sbf_blakeslee_2021: chi2=3.062913090021344, n=1

  - h0_grid:tdcosmo_slacs_birrer_2020: chi2=0.20011244742098186, n=1

  - h0_grid:h0licow_wong_2020: chi2=6.250843040862113, n=1

  - h0_grid:strides_shajib_2020: chi2=3.5034895329189815, n=1

  - h0_grid:megamaser_pesce_2020: chi2=2.792862883682782, n=1

## Predictive score


- Mode: group_holdout

- N splits: 43

- Train frac: None

- Always include calibrators: False

- Always include hubble-flow: True

- +bounded_calibration_fields: mean_logp=0.19, Δlogp_vs_base=0.06

- +bounded_fields_plus_metadata_bounded: mean_logp=0.41, Δlogp_vs_base=0.28

- +cal_offset_bounded: mean_logp=0.40, Δlogp_vs_base=0.27

- baseline: mean_logp=0.13, Δlogp_vs_base=0.00
