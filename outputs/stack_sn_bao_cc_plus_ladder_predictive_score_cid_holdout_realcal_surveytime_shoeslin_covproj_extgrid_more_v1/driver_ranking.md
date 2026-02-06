# Predictive-score driver ranking
- Run dir: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1`
- Mode: `group_holdout` (group_var=`cid`, n_groups=43, min_group_n=1)
- Baseline label: `baseline`

## Model ranking

| model | mean Δlogp vs base | win-rate | sd(Δlogp) |
|---|---:|---:|---:|
| `+bounded_fields_plus_metadata_bounded` | +0.381 | 48.8% | 1.783 |
| `+cal_offset_bounded` | +0.307 | 60.5% | 0.955 |
| `+bounded_calibration_fields` | +0.089 | 46.5% | 0.484 |
| `baseline` | +0.000 | 0.0% | 0.000 |

## Top group drivers (per model)

Groups are ordered as in the underlying split generator (sorted unique `cid` values after filters).

### `+bounded_fields_plus_metadata_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +10.964 |
| `2019np` | 2 | +2.077 |
| `2002cr` | 2 | +1.996 |
| `2011by` | 1 | +1.655 |
| `1998dh` | 2 | +1.292 |
| `2001el` | 1 | +1.069 |
| `1994ae` | 1 | +1.029 |
| `2002dp` | 2 | +0.828 |
| `2012ht` | 2 | +0.666 |
| `2007sr` | 2 | +0.590 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `1998aq` | 1 | -1.307 |
| `2009Y` | 3 | -1.170 |
| `1999dq` | 2 | -0.889 |
| `2011fe` | 2 | -0.865 |
| `2018gv` | 3 | -0.687 |
| `2017cbv` | 2 | -0.507 |
| `2009ig` | 3 | -0.434 |
| `2005df` | 1 | -0.373 |
| `2012fr` | 2 | -0.277 |
| `1981B` | 1 | -0.182 |

### `+cal_offset_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +5.667 |
| `2019np` | 2 | +1.486 |
| `2002cr` | 2 | +1.261 |
| `2012ht` | 2 | +0.969 |
| `2011by` | 1 | +0.953 |
| `1998dh` | 2 | +0.890 |
| `1994ae` | 1 | +0.752 |
| `2002dp` | 2 | +0.686 |
| `2015F` | 2 | +0.678 |
| `2009ig` | 3 | +0.633 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009Y` | 3 | -0.618 |
| `1999dq` | 2 | -0.559 |
| `1998aq` | 1 | -0.524 |
| `2017cbv` | 2 | -0.425 |
| `2018gv` | 3 | -0.312 |
| `2002fk` | 2 | -0.273 |
| `2011fe` | 2 | -0.265 |
| `2003du` | 3 | -0.255 |
| `2005df_ANU` | 1 | -0.205 |
| `2021pit` | 1 | -0.180 |

### `+bounded_calibration_fields`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +3.066 |
| `2007sr` | 2 | +0.472 |
| `2012ht` | 2 | +0.404 |
| `2015F` | 2 | +0.299 |
| `2011by` | 1 | +0.293 |
| `2008fv_comb` | 1 | +0.166 |
| `2019np` | 2 | +0.130 |
| `2013aa` | 2 | +0.102 |
| `2002cr` | 2 | +0.060 |
| `2021hpr` | 1 | +0.056 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009Y` | 3 | -0.565 |
| `1994ae` | 1 | -0.137 |
| `2011fe` | 2 | -0.135 |
| `2021pit` | 1 | -0.094 |
| `2018gv` | 3 | -0.092 |
| `1998aq` | 1 | -0.061 |
| `2012fr` | 2 | -0.056 |
| `2005df_ANU` | 1 | -0.047 |
| `2005df` | 1 | -0.037 |
| `2012cg` | 2 | -0.028 |

### `baseline`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `1981B` | 1 | +0.000 |
| `1990N` | 1 | +0.000 |
| `1994ae` | 1 | +0.000 |
| `1995al` | 1 | +0.000 |
| `1997bp` | 1 | +0.000 |
| `1997bq` | 1 | +0.000 |
| `1998aq` | 1 | +0.000 |
| `1998dh` | 2 | +0.000 |
| `1999cp` | 2 | +0.000 |
| `1999dq` | 2 | +0.000 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `1981B` | 1 | +0.000 |
| `1990N` | 1 | +0.000 |
| `1994ae` | 1 | +0.000 |
| `1995al` | 1 | +0.000 |
| `1997bp` | 1 | +0.000 |
| `1997bq` | 1 | +0.000 |
| `1998aq` | 1 | +0.000 |
| `1998dh` | 2 | +0.000 |
| `1999cp` | 2 | +0.000 |
| `1999dq` | 2 | +0.000 |

