# Predictive-score driver ranking
- Run dir: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_v1`
- Mode: `group_holdout` (group_var=`cid`, n_groups=43, min_group_n=1)
- Baseline label: `baseline`

## Model ranking

| model | mean Δlogp vs base | win-rate | sd(Δlogp) |
|---|---:|---:|---:|
| `+bounded_fields_plus_metadata_bounded` | +0.397 | 48.8% | 1.811 |
| `+cal_offset_bounded` | +0.307 | 60.5% | 0.968 |
| `+bounded_calibration_fields` | +0.089 | 53.5% | 0.474 |
| `baseline` | +0.000 | 0.0% | 0.000 |

## Top group drivers (per model)

Groups are ordered as in the underlying split generator (sorted unique `cid` values after filters).

### `+bounded_fields_plus_metadata_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +11.129 |
| `2019np` | 2 | +2.155 |
| `2002cr` | 2 | +2.041 |
| `2011by` | 1 | +1.679 |
| `1998dh` | 2 | +1.332 |
| `2001el` | 1 | +1.087 |
| `1994ae` | 1 | +1.053 |
| `2002dp` | 2 | +0.852 |
| `2012ht` | 2 | +0.638 |
| `2007sr` | 2 | +0.609 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `1998aq` | 1 | -1.325 |
| `2009Y` | 3 | -1.177 |
| `2011fe` | 2 | -0.953 |
| `1999dq` | 2 | -0.918 |
| `2018gv` | 3 | -0.685 |
| `2017cbv` | 2 | -0.451 |
| `2009ig` | 3 | -0.402 |
| `2005df` | 1 | -0.380 |
| `2012fr` | 2 | -0.262 |
| `1981B` | 1 | -0.186 |

### `+cal_offset_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +5.770 |
| `2019np` | 2 | +1.500 |
| `2002cr` | 2 | +1.259 |
| `2012ht` | 2 | +0.981 |
| `2011by` | 1 | +0.950 |
| `1998dh` | 2 | +0.889 |
| `1994ae` | 1 | +0.754 |
| `2015F` | 2 | +0.704 |
| `2002dp` | 2 | +0.683 |
| `2001el` | 1 | +0.620 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009Y` | 3 | -0.674 |
| `1999dq` | 2 | -0.544 |
| `1998aq` | 1 | -0.511 |
| `2011fe` | 2 | -0.356 |
| `2017cbv` | 2 | -0.302 |
| `2021pit` | 1 | -0.278 |
| `2018gv` | 3 | -0.269 |
| `2002fk` | 2 | -0.250 |
| `2003du` | 3 | -0.227 |
| `2005df_ANU` | 1 | -0.196 |

### `+bounded_calibration_fields`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +3.010 |
| `2007sr` | 2 | +0.487 |
| `2012ht` | 2 | +0.318 |
| `2011by` | 1 | +0.299 |
| `2015F` | 2 | +0.257 |
| `2019np` | 2 | +0.170 |
| `2008fv_comb` | 1 | +0.169 |
| `2013aa` | 2 | +0.088 |
| `2002cr` | 2 | +0.074 |
| `2021hpr` | 1 | +0.061 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009Y` | 3 | -0.541 |
| `2011fe` | 2 | -0.154 |
| `1994ae` | 1 | -0.132 |
| `2018gv` | 3 | -0.089 |
| `2021pit` | 1 | -0.072 |
| `1998aq` | 1 | -0.064 |
| `2013dy` | 2 | -0.050 |
| `2005df_ANU` | 1 | -0.049 |
| `2012cg` | 2 | -0.043 |
| `2012fr` | 2 | -0.039 |

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

