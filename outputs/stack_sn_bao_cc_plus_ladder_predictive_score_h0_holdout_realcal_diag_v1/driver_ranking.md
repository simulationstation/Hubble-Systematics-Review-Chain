# Predictive-score driver ranking
- Run dir: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_h0_holdout_realcal_diag_v1`
- Mode: `group_holdout` (group_var=`stack_part`, n_groups=6, min_group_n=1)
- Baseline label: `baseline`

## Model ranking

| model | mean Δlogp vs base | win-rate | sd(Δlogp) |
|---|---:|---:|---:|
| `baseline` | +0.000 | 0.0% | 0.000 |
| `+bounded_calibration_fields` | -0.054 | 16.7% | 0.052 |
| `+cal_offset_bounded` | -0.251 | 16.7% | 0.229 |
| `+bounded_fields_plus_metadata_bounded` | -0.319 | 16.7% | 0.286 |

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
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.017 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.006 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.056 |
| `h0_grid:strides_shajib_2020` | 1 | -0.067 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.069 |
| `h0_grid:h0licow_wong_2020` | 1 | -0.144 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | -0.144 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.069 |
| `h0_grid:strides_shajib_2020` | 1 | -0.067 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.056 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.006 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.017 |

### `+cal_offset_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.069 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.053 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.252 |
| `h0_grid:strides_shajib_2020` | 1 | -0.301 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.312 |
| `h0_grid:h0licow_wong_2020` | 1 | -0.659 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | -0.659 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.312 |
| `h0_grid:strides_shajib_2020` | 1 | -0.301 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.252 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.053 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.069 |

### `+bounded_fields_plus_metadata_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.083 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.077 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.317 |
| `h0_grid:strides_shajib_2020` | 1 | -0.378 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.394 |
| `h0_grid:h0licow_wong_2020` | 1 | -0.830 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `h0_grid:h0licow_wong_2020` | 1 | -0.830 |
| `h0_grid:sbf_blakeslee_2021` | 1 | -0.394 |
| `h0_grid:strides_shajib_2020` | 1 | -0.378 |
| `h0_grid:megamaser_pesce_2020` | 1 | -0.317 |
| `h0_grid:trgb_freedman_2021_review` | 1 | -0.077 |
| `h0_grid:tdcosmo_slacs_birrer_2020` | 1 | +0.083 |

