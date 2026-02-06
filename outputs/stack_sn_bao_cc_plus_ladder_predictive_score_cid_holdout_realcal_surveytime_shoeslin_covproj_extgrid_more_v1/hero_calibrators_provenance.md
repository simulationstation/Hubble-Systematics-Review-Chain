# Hero calibrator provenance

- Source run: `/home/primary/Hubble-Systematics-Review-Chain/outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_extgrid_more_v1`
- Driver model: `+bounded_fields_plus_metadata_bounded`
- Top-k: 10

This report lists the highest-leverage **calibrator** SNe (by Δlogp in a CID holdout) and shows their key table metadata and which surveys/photometry reductions they appear under.

## Top drivers

| CID | n_test | Δlogp |
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

## Per-CID details

### `2007af`

- Holdout Δlogp: +10.964 (n_test=4)
- Rows in ladder table: 4
- Surveys present: `CFA3K`, `CSP`, `LOSS2`, `SOUSA`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `CSP` | Y | N | 0.00625 | 12.84690 | 0.32168 | 32.09990 | 0.32281 | 31.77080 | 54175.00 | 0.0305 | 0.03370 | 10.331 | 0.0005 |
| 56 | `SOUSA` | Y | N | 0.00625 | 12.74930 | 0.36055 | 32.00230 | 0.36156 | 31.77080 | 54174.50 | 0.0464 | 0.03370 | 10.331 | 0.0268 |
| 57 | `LOSS2` | Y | N | 0.00625 | 12.81370 | 0.32481 | 32.06670 | 0.32593 | 31.77080 | 54174.50 | 0.0490 | 0.03370 | 10.331 | 0.0000 |
| 64 | `CFA3K` | Y | N | 0.00625 | 12.78070 | 0.33809 | 32.03370 | 0.33917 | 31.77080 | 54174.50 | 0.0320 | 0.03370 | 10.331 | 0.2846 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3K` | `CFA3K` | `CFA3K B`, `CFA3K V`, `CFA3K i`, `CFA3K r` | `Land2` |
| `CSP` | `CSPDR3` | `CSPDR3 B`, `CSPDR3 V`, `CSPDR3 V1`, `CSPDR3 V2`, `CSPDR3 V3`, `CSPDR3 g`, `CSPDR3 i`, `CSPDR3 r` | `CSPDR3`, `CSPDR3_supercal` |
| `LOSS2` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |
| `SOUSA` | `SWIFT` | `SWIFT B`, `SWIFT V` | `SWIFT` |

### `2019np`

- Holdout Δlogp: +2.077 (n_test=2)
- Rows in ladder table: 2
- Surveys present: `CNIa0.02`, `FOUND`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | `CNIa0.02` | Y | N | 0.00648 | 13.20780 | 0.31733 | 32.46080 | 0.31847 | 32.33020 | 58510.50 | 0.0548 | 0.01727 | 10.630 | 0.9985 |
| 150 | `FOUND` | Y | N | 0.00648 | 13.18080 | 0.32116 | 32.43380 | 0.32229 | 32.33020 | 58510.40 | 0.0507 | 0.01727 | 10.630 | 0.4740 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CNIa0.02` | `CNIa0.2` | `CNIa0.2 LCO B`, `CNIa0.2 LCO V`, `CNIa0.2 LCO i`, `CNIa0.2 LCO r`, `CNIa0.2 RP B`, `CNIa0.2 RP V`, `CNIa0.2 RP i`, `CNIa0.2 RP r` | `ASASSN` |
| `FOUND` | `PS1`, `oldPS1`, `newPS1` | `PS1 g`, `PS1 i`, `PS1 r`, `PS1 z`, `newPS1 g`, `newPS1 i`, `newPS1 r`, `newPS1 z`, `oldPS1 g`, `oldPS1 i`, `oldPS1 r`, `oldPS1 z` | `Foundation` |

### `2002cr`

- Holdout Δlogp: +1.996 (n_test=2)
- Rows in ladder table: 2
- Surveys present: `CFA3S`, `LOSS2`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00954 | 13.97790 | 0.25045 | 33.23090 | 0.25191 | 33.11560 | 52408.90 | 0.0301 | 0.02117 | 10.426 | 0.0000 |
| 63 | `CFA3S` | Y | N | 0.00954 | 13.99390 | 0.25345 | 33.24690 | 0.25488 | 33.11560 | 52408.90 | 0.0457 | 0.02117 | 10.426 | 0.9724 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3S` | `CFA3S` | `CFA3S B`, `CFA3S I`, `CFA3S R`, `CFA3S V` | `Land2` |
| `LOSS2` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |

### `2011by`

- Holdout Δlogp: +1.655 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `LOSS1`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 51 | `LOSS1` | Y | N | 0.00349 | 12.54030 | 0.55206 | 31.79330 | 0.55272 | 31.63360 | 55690.90 | 0.0344 | 0.01200 | 10.417 | 0.2988 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `LOSS1` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |

### `1998dh`

- Holdout Δlogp: +1.292 (n_test=2)
- Rows in ladder table: 2
- Surveys present: `CFA2`, `LOSS2`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00814 | 13.46340 | 0.31588 | 32.71640 | 0.31703 | 32.49960 | 51029.60 | 0.0322 | 0.05788 | 10.935 | 0.0002 |
| 62 | `CFA2` | Y | N | 0.00814 | 13.36210 | 0.31631 | 32.61510 | 0.31746 | 32.49960 | 51030.10 | 0.0722 | 0.05788 | 10.935 | 0.0021 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA2` |  |  | `Land2` |
| `LOSS2` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |

### `2001el`

- Holdout Δlogp: +1.069 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `LOWZ/JRK07`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | `LOWZ/JRK07` | Y | N | 0.00333 | 12.24810 | 0.59039 | 31.50110 | 0.59101 | 31.28590 | 52183.00 | 0.0329 | 0.01222 | 11.280 | 0.0000 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `LOWZ/JRK07` |  |  | `Land2` |

### `1994ae`

- Holdout Δlogp: +1.029 (n_test=1)
- Rows in ladder table: 1
- Surveys present: `CFA1`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 61 | `CFA1` | Y | N | 0.00588 | 12.92920 | 0.33857 | 32.18220 | 0.33964 | 32.11940 | 49686.00 | 0.0596 | 0.02641 | 10.196 | 0.0278 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA1` |  |  | `Land2` |

### `2002dp`

- Holdout Δlogp: +0.828 (n_test=2)
- Rows in ladder table: 2
- Surveys present: `CFA3S`, `LOSS2`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.01061 | 14.04010 | 0.27324 | 33.29310 | 0.27457 | 33.18590 | 52451.30 | 0.0446 | 0.04206 | 10.530 | 0.0000 |
| 63 | `CFA3S` | Y | N | 0.01061 | 14.12760 | 0.30783 | 33.38060 | 0.30901 | 33.18590 | 52451.20 | 0.1552 | 0.04206 | 10.530 | 0.7992 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3S` | `CFA3S` | `CFA3S B`, `CFA3S I`, `CFA3S R`, `CFA3S V` | `Land2` |
| `LOSS2` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |

### `2012ht`

- Holdout Δlogp: +0.666 (n_test=2)
- Rows in ladder table: 2
- Surveys present: `CSP`, `SOUSA`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `CSP` | Y | N | 0.00465 | 12.78000 | 0.41953 | 32.03300 | 0.42040 | 31.93480 | 56296.20 | 0.0483 | 0.02516 | 9.530 | 1.0000 |
| 56 | `SOUSA` | Y | N | 0.00465 | 12.67790 | 0.44119 | 31.93090 | 0.44202 | 31.93480 | 56295.80 | 0.0986 | 0.02516 | 9.530 | 0.8176 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CSP` | `CSPDR3` | `CSPDR3 B`, `CSPDR3 V`, `CSPDR3 V1`, `CSPDR3 V2`, `CSPDR3 V3`, `CSPDR3 g`, `CSPDR3 i`, `CSPDR3 r` | `CSPDR3`, `CSPDR3_supercal` |
| `SOUSA` | `SWIFT` | `SWIFT B`, `SWIFT V` | `SWIFT` |

### `2007sr`

- Holdout Δlogp: +0.590 (n_test=2)
- Rows in ladder table: 2
- Surveys present: `CFA3K`, `LOSS2`

| IDSURVEY | survey | cal? | HF? | zHD | m_b_corr | σ_diag | MU_SH0ES | σ(MU) | CEPH_DIST | PKMJD | PKMJDERR | MWEBV | HOST_LOGMASS | FITPROB |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 57 | `LOSS2` | Y | N | 0.00571 | 12.42690 | 0.37494 | 31.67990 | 0.37591 | 31.60250 | 54448.90 | 0.2466 | 0.04048 | 10.682 | 0.0000 |
| 64 | `CFA3K` | Y | N | 0.00571 | 12.37620 | 0.38481 | 31.62920 | 0.38576 | 31.60250 | 54449.20 | 0.1643 | 0.04048 | 10.682 | 0.2686 |

Photometry provenance (external products used for calibration priors):

| survey | FRAGILISTIC groups | FRAGILISTIC filters (shipped) | SNANA_kcor variants |
|---|---|---|---|
| `CFA3K` | `CFA3K` | `CFA3K B`, `CFA3K V`, `CFA3K i`, `CFA3K r` | `Land2` |
| `LOSS2` | `KAIT1`, `KAIT2`, `KAIT3`, `KAIT4`, `NICKEL1`, `NICKEL2` | `KAIT1 Ganesh B`, `KAIT1 Ganesh I`, `KAIT1 Ganesh R`, `KAIT1 Ganesh V`, `KAIT2 Ganesh B`, `KAIT2 Ganesh I`, `KAIT2 Ganesh R`, `KAIT2 Ganesh V`, `KAIT3 Ganesh B`, `KAIT3 Ganesh I`, `KAIT3 Ganesh R`, `KAIT3 Ganesh V`, …(+28) | `KAIT_Mo`, `KAIT_Stahl` |

