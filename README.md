# Hubble-Systematics-Review-Chain

Systematics-first, *audit-style* pipeline for late-time distance-scale probes, patterned on:

- stability scans (cuts/thresholds) + correlated-cut drift nulls
- mechanism ladders (low-dimensional knobs → structured residual closure)
- injection/recovery suites
- SBC/coverage gates
- time/epoch and sky (low-ℓ) invariance tests

## Current findings (real data; 2026-02-05)

These are **real-data** runs on the public Pantheon+ / Pantheon+SH0ES tables in `data/raw/…`,
using this repo’s **linear-Gaussian audit models** (not a full end-to-end SH0ES reanalysis).

- **Ladder reproduction:** `H0_eff = 73.552 ± 1.078` vs anchor `67.4` (equivalent `|Δμ| ≈ 0.19 mag`).  
  Report: `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_predictive_score_v2.yaml`
- **Late-time non-ladder stack:** Pantheon+ (SN-only) + DESI BAO + CC gives `H0_eff = 68.208 ± 0.314` (close to anchor).  
  Report: `outputs/stack_sn_bao_cc_stress_v2/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_stress.yaml`
- **Anchor-consistency (no sirens):** adding the SH0ES ladder subset forces a large calibrator↔HF offset:  
  `calibrator_offset_mag = +0.1616 ± 0.0318` mag.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_v1.yaml`
- **Ladder calibrator holdout (CV; real data):** holding out calibrators (keeping hubble-flow fixed in TRAIN) shows a large out-of-sample improvement from calibrator-only mechanisms:  
  Δlogp ≈ +10.2 / +8.5 (diag/full-cov) for `calibrator_offset_mag`, and Δlogp ≈ +8.3 / +6.1 for calibrator time-bin offsets (`pkmjd_bins` on calibrators).  
  Survey-by-survey holdout also improves (Δlogp ≈ +3.8 / +3.2 for `calibrator_offset_mag`).  
  Reports: `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1/report.md`, `outputs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_v1.yaml`, `configs/pantheon_plus_shoes_ladder_predictive_score_cal_holdout_fullcov_v1.yaml`, `configs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_v1.yaml`, `configs/pantheon_plus_shoes_ladder_predictive_score_cal_survey_holdout_fullcov_v1.yaml`
- **Joint-stack calibrator holdout (real data):** in the *anchor-consistency* joint stack (SN-only+BAO+CC+ladder), holding out calibrators (keeping hubble-flow + other probes fixed in TRAIN) still prefers calibrator-only corrections:  
  Δlogp ≈ +5.7 / +6.2 (diag/full-cov) for `calibrator_offset_mag`.  
  A simple calibrator-only proxy `pkmjd_err_linear` (linear in `PKMJDERR`) also improves held-out calibrators (Δlogp ≈ +4.6), while a `m_b_corr_err_VPEC` linear proxy does not.  
  Survey-holdout inside the same joint stack is smaller but still improves (Δlogp ≈ +1.9 / +2.3 for calibrator time-bin offsets; +1.7 / +2.3 for `calibrator_offset_mag`).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fullcov_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_mechanism_scan_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_mechanism_scan_fullcov_v1.yaml`
- **External-prior stress test:** forcing a tight prior on `calibrator_offset_mag` degrades evidence and shifts the fit (does *not* recover `H0≈73`).  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_tight_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_tight_v1.yaml`
- **External-prior “gate” sweep (real data):** if external calibration work can truly bound calibrator-step distortions at the ~0.02 mag level, the joint anchor-consistency fit pays a large evidence penalty and cannot maintain the large ~0.16 mag calibrator↔HF offset.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_external_prior_gates_v1.yaml`
- **Covariance-implied calibration bounds (real data):** we can also derive “budget-like” bounds for per-survey and per-epoch calibrator offsets from the published Pantheon+SH0ES STAT+SYS covariance. Under these bounds (typical σ≈0.02–0.05 mag), the joint fit cannot sustain a 0.16 mag calibrator↔HF step without a large evidence penalty.  
  Report: `outputs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cov_implied_gates_v1.yaml`  
  Derivation: `scripts/derive_pantheon_shoes_cov_priors.py`
- **External calibration covariance (Brout+21 “FRAGILISTIC”; real data):** using the public zeropoint-offset covariance shipped with the Pantheon+ DataRelease (typical per-survey σ≈0.002–0.008 mag), constraining per-survey offsets does **not** remove the need for a large calibrator↔HF step (the fit still wants `calibrator_offset_mag ≈ 0.16` when allowed). Calibrator holdout predictive scoring still prefers `calibrator_offset_mag` even under these tight survey-calibration priors.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_fragilistic_gates_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_v1.yaml`  
  Derivation: `scripts/derive_pantheon_shoes_fragilistic_priors.py`
- **Calibration-only covariance gate (Pantheon+SH0ES `CALIB.cov`; real data):** using the Pantheon+SH0ES *calibration-only* covariance grouping (`sytematic_groupings/Pantheon+SH0ES_122221_CALIB.cov`) to derive prior widths gives σ(`calibrator_offset_mag`)≈0.019 mag. Under these bounds, the joint anchor-consistency fit can only support a much smaller step: `calibrator_offset_mag ≈ 0.045 ± 0.016` (tension-reduction frac ≈ 0.09 vs ≈ 0.37 when free). Calibrator-holdout predictive scoring still prefers the mechanism, but with a smaller gain (Δlogp ≈ +4.1 vs +5.7 when free).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_calibcov_gates_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_calibcov_v1.yaml`  
  Derivation: `scripts/derive_pantheon_shoes_cov_priors.py --raw-cov-path data/raw/pantheon_plus_shoes/sytematic_groupings/Pantheon+SH0ES_122221_CALIB.cov`
- **SH0ES calibrator-chain prior scale (linear-system; real data product):** the SH0ES DataRelease includes a compact linear system (`SH0ES_Data/all[LCY]_...fits`, `lstsq_results.txt`) with σ(`fivelogH0`)≈0.028 mag. Treating that as a *calibrator-chain-inspired* prior width for an additional `calibrator_offset_mag`, the joint stack under FRAGILISTIC survey priors supports only `calibrator_offset_mag ≈ 0.074 ± 0.021` (tension-reduction frac ≈ 0.16).  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_fragilistic_shoes_gates_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_fragilistic_shoes_v1.yaml`  
  Derivation: `scripts/derive_shoes_linear_system_fivelogh0_prior.py`
- **Injection mapping (calibrator step → H0 shift):** with an unmodeled calibrator-only magnitude shift injected into the ladder data, the induced bias in `delta_lnH0` follows the expected slope `d(delta_lnH0)/d(Δm)≈0.46`, so faking the full ladder-vs-anchor offset requires Δm≈0.19 mag.  
  Report: `outputs/pantheon_plus_shoes_ladder_injection_calibcov_misspec_v1/report.md`  
  Reproduce: `configs/pantheon_plus_shoes_ladder_injection_calibcov_misspec_v1.yaml`
- **Anti-overfit gates:** cross-validated predictive scoring and exact Gaussian log-evidence both *penalize* adding flexible closure terms (HF redshift splines / sky low-ℓ modes) on the ladder subset.  
  Reports: `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`, `outputs/pantheon_plus_shoes_ladder_level_sweep_v2/report.md`
- **External H0 probes as `h0_grid` (TRGB / lenses / masers; stress-test):** adding these does not remove the need for a large calibrator↔HF offset in the joint stack, and calibrator holdout improvements persist.  
  Reports: `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1/report.md`, `outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_low_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_high_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_cal_offset_extgrid_all_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_holdout_extgrid_all_v1.yaml`, `configs/stack_sn_bao_cc_plus_ladder_predictive_score_cal_survey_holdout_extgrid_all_v1.yaml`
- **Independent probe adapter (sirens, Gate-2):** selection-corrected per-event `logL(H0)` grid + metadata cut/time drift audit (experimental; upstream Gate-2 product not complete yet).  
  Report: `outputs/siren_gate2_grid_audit_v2/report.md`  
  Reproduce: `configs/siren_gate2_grid_audit_v2.yaml`

For the full status + what’s still missing vs the full ambition, start at:
- `docs/LAYMAN_SUMMARY.md`
- `docs/SPEC_STATUS.md`

## Quickstart (uses existing venvs in this workspace)

Run with the Project venv (recommended in this container):

```bash
PYTHONPATH=src ../PROJECT/.venv/bin/python -m hubble_systematics.cli --help
```

Or install editable into that venv:

```bash
../PROJECT/.venv/bin/pip install -e .
hubble-audit --help
```

Example: run the Pantheon+ audit packet (baseline + cut scan + correlated-null drift MC):

```bash
PYTHONPATH=src ../PROJECT/.venv/bin/python -m hubble_systematics.cli run configs/pantheon_plus_audit.yaml
```

Example: ladder time-invariance null (fixed-support calibrators; shuffle-within-survey null):

```bash
PYTHONPATH=src ../PROJECT/.venv/bin/python -m hubble_systematics.cli run configs/pantheon_plus_shoes_ladder_time_invariance_null_fixedcal_v3.yaml
```

Outputs are written under `outputs/<run_id>/`.

## Tests

```bash
../PROJECT/.venv/bin/python -m pytest
```
