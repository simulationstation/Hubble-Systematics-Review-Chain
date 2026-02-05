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
- **Late-time non-ladder stack:** Pantheon+ (SN-only) + DESI BAO + CC + a GR-control siren grid gives `H0_eff = 68.218 ± 0.313` (close to anchor).  
  When the ladder is included, anchor-consistency is achieved mainly via `calibrator_offset_mag = +0.1617 ± 0.0318` mag (≈ `85% ± 17%` of the full `~0.19 mag` gap).  
  Report: `outputs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_v2/report.md`  
  Reproduce: `configs/stack_sn_bao_cc_siren_plus_ladder_cal_offset_v2.yaml`
- **Anti-overfit gates:** cross-validated predictive scoring and exact Gaussian log-evidence both *penalize* adding flexible closure terms (HF redshift splines / sky low-ℓ modes) on the ladder subset.  
  Reports: `outputs/pantheon_plus_shoes_ladder_predictive_score_v2/report.md`, `outputs/pantheon_plus_shoes_ladder_level_sweep_v2/report.md`
- **Independent probe adapter (sirens, Gate-2):** selection-corrected per-event `logL(H0)` grid + metadata cut/time drift audit.  
  Report: `outputs/siren_gate2_grid_audit_v1/report.md`  
  Reproduce: `configs/siren_gate2_grid_audit_v1.yaml`

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
