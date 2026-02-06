# External Constraint Interface (Stub)

This repo uses **per-parameter Gaussian prior width overrides** (“sigma overrides”) as the standard way to
impose **external calibration constraints** on systematics terms (per survey, per epoch, etc.).

Today, the audit runner expects a JSON mapping:

```json
{
  "cal_survey_offset_5": 0.010,
  "survey_pkmjd_bin_offset_5_2": 0.015,
  "pkmjd_err_linear_mag": 0.040
}
```

and configs reference it via:

```yaml
model:
  priors:
    sigma_overrides_path: data/processed/external_calibration/your_overrides.json
```

## Why this exists

The “systematics-first” chain is only useful if the **winning/metadata-rich terms are externally bounded**.
This doc records the intended plumbing for:

- Gaia cross-calibration (per survey/instrument),
- overlap-region residuals (per survey pair / sky region),
- real zeropoint logs (per epoch/time bin).

The actual calibration products are *not* included here; this is a **stub interface** so we can plug them in
as they become available.

## Compiler script

Use:

```bash
.venv/bin/python scripts/build_sigma_overrides_from_external_constraints.py \
  --out data/processed/external_calibration/sigma_overrides_from_external_constraints_v1.json \
  references/external_constraints_stub_v1.json
```

Input formats supported:

1. `{"constraints":[ ... ]}` list format (recommended)
2. raw list format: `[ ... ]`
3. backwards-compatible mapping: `{ "param": sigma, ... }`

Each list entry may specify:

- `param` **or** `mechanism`
- `sigma`/`sigma_mag` (required)
- selectors such as `idsurvey` and `bin` (for survey/time-binned constraints)

See `references/external_constraints_stub_v1.json` for an example.

## Naming conventions (what parameters exist)

The key you constrain must match a parameter name in the design matrix. Common patterns:

- `cal_survey_offset_<idsurvey>`
- `hf_survey_offset_<idsurvey>`
- `survey_offset_<idsurvey>`
- `pkmjd_bin_offset_<k>` (time-bin index)
- `survey_pkmjd_bin_offset_<idsurvey>_<k>` (survey×time)
- metadata proxy terms:
  - `pkmjd_err_linear_mag`
  - `mwebv_linear_mag`
  - `host_mass_step_mag`
  - `c_linear_mag`
  - `x1_linear_mag`
  - `biascor_m_b_linear_mag`

These are produced by the corresponding `mechanisms:` flags in configs.

## Next (when real products land)

When we have real Gaia/overlap/zeropoint products, we should:

1. convert them into sigma overrides using the compiler script (or a dedicated parser),
2. merge them with existing k-correction / SH0ES-linear / cov-projection priors via `scripts/merge_sigma_overrides.py`,
3. rerun:
   - CID holdout predictive score,
   - injections (modeled + misspecified),
   - SBC/coverage checks.

