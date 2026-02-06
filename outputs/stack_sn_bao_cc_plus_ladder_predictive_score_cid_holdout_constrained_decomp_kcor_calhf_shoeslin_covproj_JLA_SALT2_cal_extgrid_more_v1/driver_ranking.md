# Predictive-score driver ranking
- Run dir: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_constrained_decomp_kcor_calhf_shoeslin_covproj_JLA_SALT2_cal_extgrid_more_v1`
- Mode: `group_holdout` (group_var=`cid`, n_groups=43, min_group_n=1)
- Baseline label: `baseline`

## Model ranking

| model | mean Δlogp vs base | win-rate | sd(Δlogp) |
|---|---:|---:|---:|
| `+bounded_fields_plus_metadata_bounded` | +0.399 | 48.8% | 1.869 |
| `+cal_offset_bounded` | +0.284 | 60.5% | 0.917 |
| `+bounded_calibration_fields` | +0.029 | 46.5% | 0.134 |
| `baseline` | +0.000 | 0.0% | 0.000 |

## Top group drivers (per model)

Groups are ordered as in the underlying split generator (sorted unique `cid` values after filters).

### `+bounded_fields_plus_metadata_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +11.357 |
| `2019np` | 2 | +2.260 |
| `2002cr` | 2 | +2.127 |
| `2011by` | 1 | +1.769 |
| `1998dh` | 2 | +1.586 |
| `2001el` | 1 | +1.248 |
| `2002dp` | 2 | +0.980 |
| `1994ae` | 1 | +0.939 |
| `2012ht` | 2 | +0.802 |
| `2015F` | 2 | +0.721 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009ig` | 3 | -1.515 |
| `1998aq` | 1 | -1.403 |
| `2009Y` | 3 | -1.153 |
| `2011fe` | 2 | -1.017 |
| `1999dq` | 2 | -0.672 |
| `2018gv` | 3 | -0.522 |
| `2005df` | 1 | -0.403 |
| `2017cbv` | 2 | -0.258 |
| `2013aa` | 2 | -0.181 |
| `2008fv_comb` | 1 | -0.147 |

### `+cal_offset_bounded`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +5.435 |
| `2019np` | 2 | +1.451 |
| `2002cr` | 2 | +1.181 |
| `2011by` | 1 | +0.908 |
| `2012ht` | 2 | +0.890 |
| `1998dh` | 2 | +0.839 |
| `1994ae` | 1 | +0.745 |
| `2002dp` | 2 | +0.644 |
| `2015F` | 2 | +0.634 |
| `2001el` | 1 | +0.629 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009Y` | 3 | -0.650 |
| `1999dq` | 2 | -0.528 |
| `1998aq` | 1 | -0.519 |
| `2017cbv` | 2 | -0.339 |
| `2011fe` | 2 | -0.336 |
| `2002fk` | 2 | -0.268 |
| `2018gv` | 3 | -0.264 |
| `2021pit` | 1 | -0.263 |
| `2003du` | 3 | -0.250 |
| `2005df_ANU` | 1 | -0.199 |

### `+bounded_calibration_fields`

Top gains:

| group | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 4 | +0.855 |
| `2007sr` | 2 | +0.165 |
| `2011by` | 1 | +0.094 |
| `2012ht` | 2 | +0.085 |
| `2015F` | 2 | +0.069 |
| `2002cr` | 2 | +0.049 |
| `1998dh` | 2 | +0.042 |
| `2009ig` | 3 | +0.037 |
| `2019np` | 2 | +0.032 |
| `2008fv_comb` | 1 | +0.032 |

Top losses:

| group | n_test | Δlogp |
|---|---:|---:|
| `2009Y` | 3 | -0.116 |
| `2011fe` | 2 | -0.041 |
| `1994ae` | 1 | -0.038 |
| `2018gv` | 3 | -0.017 |
| `1999dq` | 2 | -0.016 |
| `2001el` | 1 | -0.014 |
| `1999cp` | 2 | -0.014 |
| `2013dy` | 2 | -0.012 |
| `2005df_ANU` | 1 | -0.010 |
| `2012cg` | 2 | -0.010 |

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

