# CID duplicate discordance report

- Run dir: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dropdiscord_cal_bestfitprob_v1`
- Scope: `cal`
- Duplicate CIDs (n>1): 11
- Flag thresholds: range ≥ 0.050 mag OR max |Δm|/σ ≥ 2.00
- Flagged: 2

## Top duplicate CIDs

| CID | n_rows | surveys | m_b_corr range | max |Δm|/σ | σ_diag med | FITPROB min..max | chi2/ndof med |
|---|---:|---|---:|---:|---:|---|---:|
| `2007af` | 3 | `CFA3K`, `LOSS2`, `SOUSA` | 0.06440 | 0.133 | 0.33809 | 0.0000..0.2846 | 1.231 |
| `1999dq` | 2 | `CFA2`, `LOSS2` | 0.04830 | 0.128 | 0.26620 | 0.0000..0.0055 | 2.096 |
| `2003du` | 3 | `CFA3S`, `LOSS2`, `LOWZ/JRK07` | 0.05200 | 0.126 | 0.29116 | 0.0000..0.9989 | 1.348 |
| `2017erp` | 3 | `CNIa0.02`, `FOUND`, `LOSS1` | 0.03190 | 0.063 | 0.34690 | 0.0000..1.0000 | 0.852 |
| `2018gv` | 2 | `CNIa0.02`, `FOUND` | 0.03020 | 0.062 | 0.34624 | 0.0000..0.9996 | 2.343 |
| `2006D` | 3 | `CFA3K`, `CSP`, `LOSS2` | 0.02450 | 0.061 | 0.28423 | 0.0000..0.0084 | 1.839 |
| `2019np` | 2 | `CNIa0.02`, `FOUND` | 0.02700 | 0.060 | 0.31924 | 0.4740..0.9985 | 0.818 |
| `2002fk` | 2 | `CFA3S`, `LOSS2` | 0.02190 | 0.052 | 0.29907 | 0.0000..0.0012 | 1.796 |
| `2002cr` | 2 | `CFA3S`, `LOSS2` | 0.01600 | 0.045 | 0.25195 | 0.0000..0.9724 | 1.398 |
| `2005cf` | 3 | `CFA3K`, `LOSS2`, `SOUSA` | 0.01340 | 0.028 | 0.32447 | 0.0010..0.9252 | 0.808 |
| `2012cg` | 2 | `LOSS1`, `SOUSA` | 0.02160 | 0.019 | 0.79026 | 0.0002..0.0004 | 1.618 |

## Flagged per-row details

### `2007af`

- n_rows=3, m_b_corr range=0.06440 mag, max |Δm|/σ=0.133

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00625 | 12.81370 | 0.32481 | 0.0000 | 2.191 | 54174.50 | 0.0490 | 0.03370 | 10.331 |
| 56 | `SOUSA` | Y | N | 0.00625 | 12.74930 | 0.36055 | 0.0268 | 1.231 | 54174.50 | 0.0464 | 0.03370 | 10.331 |
| 64 | `CFA3K` | Y | N | 0.00625 | 12.78070 | 0.33809 | 0.2846 | 1.076 | 54174.50 | 0.0320 | 0.03370 | 10.331 |


### `2003du`

- n_rows=3, m_b_corr range=0.05200 mag, max |Δm|/σ=0.126

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00747 | 13.54130 | 0.28787 | 0.0247 | 1.348 | 52766.40 | 0.0980 | 0.00862 | 9.345 |
| 63 | `CFA3S` | Y | N | 0.00747 | 13.48930 | 0.29352 | 0.0000 | 2.089 | 52766.20 | 0.0547 | 0.00862 | 9.345 |
| 50 | `LOWZ/JRK07` | Y | N | 0.00747 | 13.51550 | 0.29116 | 0.9989 | 0.573 | 52767.30 | 0.1593 | 0.00862 | 9.345 |


## Hero CIDs (from driver ranking)

- Model: `+bounded_fields_plus_metadata_bounded`

| CID | n_test | Δlogp | n_rows | m_b_corr range | max |Δm|/σ |
|---|---:|---:|---:|---:|---:|
| `2007af` | 3 | 7.060 | 3 | 0.0644 | 0.133 |
| `2019np` | 2 | 2.242 | 2 | 0.0270 | 0.060 |
| `2002cr` | 2 | 2.095 | 2 | 0.0160 | 0.045 |
| `2011by` | 1 | 1.687 | 1 | 0.0000 | nan |
| `1994ae` | 1 | 1.113 | 1 | 0.0000 | nan |
| `2001el` | 1 | 1.095 | 1 | 0.0000 | nan |
| `1999cp` | 1 | 0.749 | 1 | 0.0000 | nan |
| `2012ht` | 1 | 0.747 | 1 | 0.0000 | nan |
| `2015F` | 1 | 0.686 | 1 | 0.0000 | nan |
| `2002dp` | 1 | 0.524 | 1 | 0.0000 | nan |

### Hero per-row details

#### `2007af`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00625 | 12.81370 | 0.32481 | 0.0000 | 2.191 | 54174.50 | 0.0490 | 0.03370 | 10.331 |
| 56 | `SOUSA` | Y | N | 0.00625 | 12.74930 | 0.36055 | 0.0268 | 1.231 | 54174.50 | 0.0464 | 0.03370 | 10.331 |
| 64 | `CFA3K` | Y | N | 0.00625 | 12.78070 | 0.33809 | 0.2846 | 1.076 | 54174.50 | 0.0320 | 0.03370 | 10.331 |

#### `2019np`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | `CNIa0.02` | Y | N | 0.00648 | 13.20780 | 0.31733 | 0.9985 | 0.635 | 58510.50 | 0.0548 | 0.01727 | 10.630 |
| 150 | `FOUND` | Y | N | 0.00648 | 13.18080 | 0.32116 | 0.4740 | 1.001 | 58510.40 | 0.0507 | 0.01727 | 10.630 |

#### `2002cr`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00954 | 13.97790 | 0.25045 | 0.0000 | 2.280 | 52408.90 | 0.0301 | 0.02117 | 10.426 |
| 63 | `CFA3S` | Y | N | 0.00954 | 13.99390 | 0.25345 | 0.9724 | 0.516 | 52408.90 | 0.0457 | 0.02117 | 10.426 |

#### `2011by`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 51 | `LOSS1` | Y | N | 0.00349 | 12.54030 | 0.55206 | 0.2988 | 1.078 | 55690.90 | 0.0344 | 0.01200 | 10.417 |

#### `1994ae`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 61 | `CFA1` | Y | N | 0.00588 | 12.92920 | 0.33857 | 0.0278 | 1.351 | 49686.00 | 0.0596 | 0.02641 | 10.196 |

#### `2001el`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | `LOWZ/JRK07` | Y | N | 0.00333 | 12.24810 | 0.59039 | 0.0000 | 5.091 | 52183.00 | 0.0329 | 0.01222 | 11.280 |

#### `1999cp`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | `LOWZ/JRK07` | Y | N | 0.00954 | 13.96050 | 0.25219 | 0.0002 | 3.248 | 51364.10 | 0.0908 | 0.02112 | 10.441 |

#### `2012ht`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `CSP` | Y | N | 0.00465 | 12.78000 | 0.41953 | 1.0000 | 0.406 | 56296.20 | 0.0483 | 0.02516 | 9.530 |

#### `2015F`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `CSP` | Y | N | 0.00488 | 12.28600 | 0.40613 | 0.9999 | 0.638 | 57107.30 | 0.0301 | 0.17476 | 12.198 |

#### `2002dp`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | FITPROB | chi2/ndof | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 63 | `CFA3S` | Y | N | 0.01061 | 14.12760 | 0.30783 | 0.7992 | 0.742 | 52451.20 | 0.1552 | 0.04206 | 10.530 |

