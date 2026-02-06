# Audit report

Run dir: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1`

## Baseline fit


- N: 1713

- Ladder level: L1

- chi2/dof: 1551.99 / 1695

- log evidence: 879.43

- H0_eff (from delta_lnH0): 69.066 ± 0.286

- Equivalent Δμ [mag]: -0.0530 ± 0.0090

- Per-part chi2:

  - pantheon_plus: chi2=1187.2314834349042, n=1322

  - pantheon_plus_shoes_ladder: chi2=314.1561800480818, n=354

  - bao: chi2=22.26417508516603, n=12

  - chronometers: chi2=12.428178646959873, n=19

  - h0_grid: chi2=0.18668308034777256, n=1

  - h0_grid: chi2=3.043852924427548, n=1

  - h0_grid: chi2=0.2031973455843335, n=1

  - h0_grid: chi2=6.21194474373191, n=1

  - h0_grid: chi2=3.4853836439390284, n=1

  - h0_grid: chi2=2.7775687531979023, n=1

## Predictive score


- Mode: group_holdout

- N splits: 43

- Train frac: None

- Always include calibrators: False

- Always include hubble-flow: True

- +bounded_calibration_fields: mean_logp=0.51, Δlogp_vs_base=0.09

- +bounded_fields_plus_metadata_bounded: mean_logp=0.81, Δlogp_vs_base=0.38

- +cal_offset_bounded: mean_logp=0.73, Δlogp_vs_base=0.31

- baseline: mean_logp=0.43, Δlogp_vs_base=0.00
