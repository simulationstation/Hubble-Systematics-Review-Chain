# Predictive-score driver ranking
- Run dir: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_h0_holdout_realcal_fullcov_v1`
- Mode: `group_holdout` (group_var=`stack_part`, n_groups=6, min_group_n=1)
- Baseline label: `baseline`

## Model ranking

| model | mean Δlogp vs base | win-rate | sd(Δlogp) |
|---|---:|---:|---:|
| `baseline` | +0.000 | 0.0% | 0.000 |
| `+bounded_calibration_fields` | -0.027 | 16.7% | 0.022 |
| `+cal_offset_bounded` | -0.117 | 16.7% | 0.094 |
| `+bounded_fields_plus_metadata_bounded` | -0.132 | 16.7% | 0.106 |

## Top group drivers (per model)

Groups are ordered as in the underlying split generator (sorted unique `stack_part` values after filters).

### `baseline`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | +0.000 |
| `h0_grid:megamaser_pesce_2020` | 1 | +0.000 |
| `h0_grid:sbf_blakeslee_2021` | 1 | +0.000 |
| `h0_grid:strides_shajib_2020` | 1 | +0.000 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.000 |
| `h0_grid:trgb_freedman_2021_review` | 1 | +0.000 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | +0.000 |
| `h0_grid:megamaser_pesce_2020` | 1 | +0.000 |
| `h0_grid:sbf_blakeslee_2021` | 1 | +0.000 |
| `h0_grid:strides_shajib_2020` | 1 | +0.000 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.000 |
| `h0_grid:trgb_freedman_2021_review` | 1 | +0.000 |

### `+bounded_calibration_fields`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.005 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.012 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.026 |
| `h0_grid:strides_shajib_2020` | 1 | -0.031 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.033 |
| `h0_grid:h0licow_wong_2020` | 1 | -0.068 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | -0.068 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.033 |
| `h0_grid:strides_shajib_2020` | 1 | -0.031 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.026 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.012 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.005 |

### `+cal_offset_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.020 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.054 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.110 |
| `h0_grid:strides_shajib_2020` | 1 | -0.131 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.139 |
| `h0_grid:h0licow_wong_2020` | 1 | -0.290 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | -0.290 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.139 |
| `h0_grid:strides_shajib_2020` | 1 | -0.131 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.110 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.054 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.020 |

### `+bounded_fields_plus_metadata_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.023 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.062 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.124 |
| `h0_grid:strides_shajib_2020` | 1 | -0.147 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.156 |
| `h0_grid:h0licow_wong_2020` | 1 | -0.327 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | -0.327 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.156 |
| `h0_grid:strides_shajib_2020` | 1 | -0.147 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.124 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.062 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.023 |

