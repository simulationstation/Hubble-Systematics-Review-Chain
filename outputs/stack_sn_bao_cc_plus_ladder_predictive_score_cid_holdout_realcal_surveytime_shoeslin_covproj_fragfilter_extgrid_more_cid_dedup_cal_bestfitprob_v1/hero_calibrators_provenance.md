# Hero calibrator provenance

- Source run: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dedup_cal_bestfitprob_v1`
- Driver model: `+bounded_fields_plus_metadata_bounded`
- Top-k: 10

This report lists the highest-leverage **calibrator** SNe (by Δlogp in a CID holdout) and shows their key table metadata and which surveys/photometry reductions they appear under.

## Top drivers

| CID | n_test | Δlogp |
|---|---:|---:|
| `2007af` | 1 | +2.018 |
| `2011by` | 1 | +1.709 |
| `2019np` | 1 | +1.220 |
| `1994ae` | 1 | +1.153 |
| `2002cr` | 1 | +1.069 |
| `2001el` | 1 | +1.063 |
| `2012ht` | 1 | +0.834 |
| `2015F` | 1 | +0.799 |
| `1999cp` | 1 | +0.759 |
| `2002dp` | 1 | +0.517 |

## Per-CID details

### `2007af`

- Holdout Δlogp: +2.018 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CFA3K`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | `CFA3K` | Y | N | 0.00625 | 12.78070 | 0.33809 | 32.03370 | 0.33917 | 31.77080 | 54174.50 | 0.0320 | 0.03370 | 10.331 | 0.2846 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3K` | `CFA3K` | `CFA3K B`, `CFA3K V`, `CFA3K i`, `CFA3K r` | `Land2` |

### `2011by`

- Holdout Δlogp: +1.709 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `LOSS1`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 51 | `LOSS1` | Y | N | 0.00349 | 12.54030 | 0.55206 | 31.79330 | 0.55272 | 31.63360 | 55690.90 | 0.0344 | 0.01200 | 10.417 | 0.2988 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `LOSS1` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |

### `2019np`

- Holdout Δlogp: +1.220 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CNIa0.02`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | `CNIa0.02` | Y | N | 0.00648 | 13.20780 | 0.31733 | 32.46080 | 0.31847 | 32.33020 | 58510.50 | 0.0548 | 0.01727 | 10.630 | 0.9985 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CNIa0.02` | `CNIa0.2` | `CNIa0.2 LCO B`, `CNIa0.2 LCO V`, `CNIa0.2 LCO i`, `CNIa0.2 LCO r`, `CNIa0.2 RP B`, `CNIa0.2 RP V`, `CNIa0.2 RP i`, `CNIa0.2 RP r` | `ASASSN` |

### `1994ae`

- Holdout Δlogp: +1.153 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CFA1`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 61 | `CFA1` | Y | N | 0.00588 | 12.92920 | 0.33857 | 32.18220 | 0.33964 | 32.11940 | 49686.00 | 0.0596 | 0.02641 | 10.196 | 0.0278 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA1` |  |  | `Land2` |

### `2002cr`

- Holdout Δlogp: +1.069 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CFA3S`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 63 | `CFA3S` | Y | N | 0.00954 | 13.99390 | 0.25345 | 33.24690 | 0.25488 | 33.11560 | 52408.90 | 0.0457 | 0.02117 | 10.426 | 0.9724 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3S` | `CFA3S` | `CFA3S B`, `CFA3S I`, `CFA3S R`, `CFA3S V` | `Land2` |

### `2001el`

- Holdout Δlogp: +1.063 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `LOWZ/JRK07`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | `LOWZ/JRK07` | Y | N | 0.00333 | 12.24810 | 0.59039 | 31.50110 | 0.59101 | 31.28590 | 52183.00 | 0.0329 | 0.01222 | 11.280 | 0.0000 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `LOWZ/JRK07` |  |  | `Land2` |

### `2012ht`

- Holdout Δlogp: +0.834 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CSP`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `CSP` | Y | N | 0.00465 | 12.78000 | 0.41953 | 32.03300 | 0.42040 | 31.93480 | 56296.20 | 0.0483 | 0.02516 | 9.530 | 1.0000 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CSP` | `CSPDR3` | `CSPDR3 B`, `CSPDR3 V`, `CSPDR3 V1`, `CSPDR3 V2`, `CSPDR3 V3`, `CSPDR3 g`, `CSPDR3 i`, `CSPDR3 r` | `CSPDR3`, `CSPDR3_supercal` |

### `2015F`

- Holdout Δlogp: +0.799 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CSP`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `CSP` | Y | N | 0.00488 | 12.28600 | 0.40613 | 31.53900 | 0.40703 | 31.44890 | 57107.30 | 0.0301 | 0.17476 | 12.198 | 0.9999 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CSP` | `CSPDR3` | `CSPDR3 B`, `CSPDR3 V`, `CSPDR3 V1`, `CSPDR3 V2`, `CSPDR3 V3`, `CSPDR3 g`, `CSPDR3 i`, `CSPDR3 r` | `CSPDR3`, `CSPDR3_supercal` |

### `1999cp`

- Holdout Δlogp: +0.759 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `LOWZ/JRK07`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | `LOWZ/JRK07` | Y | N | 0.00954 | 13.96050 | 0.25219 | 33.21350 | 0.25363 | 33.11560 | 51364.10 | 0.0908 | 0.02112 | 10.441 | 0.0002 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `LOWZ/JRK07` |  |  | `Land2` |

### `2002dp`

- Holdout Δlogp: +0.517 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CFA3S`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 63 | `CFA3S` | Y | N | 0.01061 | 14.12760 | 0.30783 | 33.38060 | 0.30901 | 33.18590 | 52451.20 | 0.1552 | 0.04206 | 10.530 | 0.7992 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3S` | `CFA3S` | `CFA3S B`, `CFA3S I`, `CFA3S R`, `CFA3S V` | `Land2` |

