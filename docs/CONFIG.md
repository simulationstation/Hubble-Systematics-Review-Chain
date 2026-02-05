# Config schema (minimal)

The runner consumes a YAML file with top-level keys:

- `run`: output + task selection
- `anchor`: early-time (fixed) cosmology used as the target
- `probe`: which late-time dataset adapter to run
- `model`: mechanism ladder level + priors
- `model.shared_scale`: optional shared distance-scale stress parameters
- optional task blocks: `scan`, `null`, `injection`, `sbc`
- optional task blocks: `split_fit`, `split_null`, `group_split_null`, `hemisphere_scan`, `predictive_score`

## `run`

```yaml
run:
  run_id: my_run
  out_base: outputs
  tasks: [baseline_fit, baseline_sweep, cut_scan, correlated_cut_null, injection_suite, split_fit, split_null, group_split_null, hemisphere_scan, predictive_score, sbc, report]
```

## `anchor`

Flat ΛCDM background (used only for predictions; late-time data do *not* update these parameters in the audit flow):

```yaml
anchor:
  H0: 67.4
  Omega_m: 0.315
  Omega_k: 0.0
  rd_Mpc: 147.09
```

## `probe`

### Pantheon+ (SN)

```yaml
probe:
  name: pantheon_plus
  data_path: data/processed/pantheon_plus/pantheon_plus_sky_cosmology_stat+sys_zHD.npz
  z_min: 0.02
  z_max: 0.62
```

### Pantheon+SH0ES ladder (calibrators + Hubble-flow SNe)

Loads the public `Pantheon+SH0ES.dat` table and its full STAT+SYS covariance and selects:

- calibrators: `IS_CALIBRATOR==1`
- Hubble-flow sample: `USED_IN_SH0ES_HF==1` with configurable `z_hf_min/z_hf_max`

```yaml
probe:
  name: pantheon_plus_shoes_ladder
  raw_dat_path: data/raw/pantheon_plus_shoes/Pantheon+SH0ES.dat
  raw_cov_path: data/raw/pantheon_plus_shoes/Pantheon+SH0ES_STAT+SYS.cov
  include_calibrators: true
  include_hubble_flow: true
  z_column: zHD         # or: zCMB, etc. (as present in the table)
  z_hf_min: 0.023
  z_hf_max: 0.15
  tag: cal+hf_stat+sys_zHD   # cache key for processed NPZ
```

### BAO

```yaml
probe:
  name: bao
  data_path: data/processed/bao/bao_desi_2024_bao_all.npz
  meta_path: data/processed/bao/bao_desi_2024_bao_all.meta.json
```

### Chronometers

```yaml
probe:
  name: chronometers
  data_path: data/processed/chronometers/chronometers_BC03_all.npz
```

### H0 grid posterior (e.g., siren GR control)

Reads a grid posterior JSON with `H0_grid` and `posterior` arrays (as written by the existing
`PROJECT` siren runners). This adapter uses a Gaussian moment approximation (mean+std) for now.

```yaml
probe:
  name: h0_grid
  data_path: data/processed/sirens/gr_h0_selection_on_inv_sampling_pdf.json
  label: optional_label
  space: linear   # or: ln (treats y=ln H0 so delta_lnH0 maps exactly)
```

### Gaussian measurement (single number)

Use this for external probes that publish an approximately Gaussian constraint on `H0` or `r_d`
without needing a full table adapter (e.g. a lensing compilation, TRGB summary, etc.).

```yaml
probe:
  name: gaussian_measurement
  label: example_trgb
  quantity: H0     # or: rd
  mean: 69.8
  sigma: 1.6
  space: ln        # ln (exact delta_lnH0 mapping) or linear (small-delta approximation)
```

### Gate-2 siren grid (event-level metadata; selection-corrected)

Reads a Gate-2 JSON produced by the `Dark_Siren_Ladder_Audit` project and retains per-event metadata
(`ess_min`, `n_good_min`, event time parsed from name, and PE analysis label).

When evaluating a subset of events, it recomputes the **selection-corrected** posterior on the
shared `H0_grid`:

- `logpost(H0) = sum_i logL_i(H0) - N * log_alpha_grid(H0) + const`

It then summarizes that subset posterior as a Gaussian moment approximation in either `ln(H0)`
(default; log-normal style) or linear `H0`, enabling the same cut/epoch drift audits while keeping
the runner linear-Gaussian.

```yaml
probe:
  name: siren_gate2_grid
  data_path: data/processed/sirens/gr_h0_selection_on_inv_sampling_pdf.json
  label: gate2_gr_o3_events
  space: ln   # or: linear
```

### Stack (joint fit across probes)

```yaml
probe:
  name: stack
  stack:
    - name: pantheon_plus
      data_path: data/processed/pantheon_plus/pantheon_plus_sky_cosmology_stat+sys_zHD.npz
      model: {ladder_level: L1}
    - name: bao
      data_path: data/processed/bao/bao_desi_2024_bao_all.npz
      meta_path: data/processed/bao/bao_desi_2024_bao_all.meta.json
      model: {ladder_level: L0}
    - name: chronometers
      data_path: data/processed/chronometers/chronometers_BC03_all.npz
      model: {ladder_level: L0}
    - name: h0_grid
      data_path: data/processed/sirens/gr_h0_selection_on_inv_sampling_pdf.json
      model: {ladder_level: L0}
```

## `model`

Mechanism ladder levels (current implementation, per-probe):

- `L0`: no systematics parameters (diagnostic only)
- `L1`: global offset only
- `L2`: global offset + survey offsets (Pantheon+)
- `L3`: add smooth `z` residual closure (B-spline + second-difference penalty)
- `L4`: add low-ℓ sky modes (real spherical harmonics; regularized)

Notes:

- For `pantheon_plus_shoes_ladder`, the `delta_lnH0` shared-scale parameter (if enabled) applies only
  to Hubble-flow rows, so it acts like an “H0 tension” diagnostic relative to the anchor when
  calibrators are included.
- Additional optional mechanisms for `pantheon_plus_shoes_ladder` live under `model.mechanisms`, e.g.:
  - `calibrator_offset: true` → adds `calibrator_offset_mag` (calibrator-only shift)
  - `cal_survey_offsets: true` → adds per-survey calibrator-only shifts `cal_survey_offset_<idsurvey>` (prior: `sigma_cal_survey_offset_mag`)
  - `m_b_corr_err_linear: true` → adds `m_b_corr_err_linear_mag` (linear in `m_b_corr_err_DIAG`, z-scored on load; prior: `sigma_m_b_corr_err_linear_mag`)
    - scope via `m_b_corr_err_apply_to: hf|cal|all` (default `hf`)

## `sweep` (baseline model variants)

`baseline_sweep` runs a series of baseline fits with model overrides and reports how much each
variant reduces the inferred `delta_lnH0` relative to the first entry (a simple “percent of tension
absorbed” diagnostic).

```yaml
run:
  tasks: [baseline_sweep]

model:
  ladder_level: L2
  shared_scale: {enable: true, params: [delta_lnH0], prior_sigma: 0.5}

sweep:
  - label: baseline
    model: {}
  - label: +dust
    model:
      mechanisms: {mwebv_linear: true}
  - label: +time
    model:
      mechanisms: {pkmjd_linear: true}
```

Example:

```yaml
model:
  ladder_level: L2
  shared_scale:
    enable: true
    params: [delta_lnH0]
    prior_sigma: 0.5
  priors:
    sigma_global_offset_mag: 10.0
    sigma_survey_offset_mag: 0.2
```

## `scan` (cut scans)

```yaml
scan:
  cut_var: z
  cut_mode: leq   # leq (<=) or geq (>=)
  n_cuts: 12
  cut_min: 0.02
  cut_max: 0.62
```

For `pantheon_plus_shoes_ladder`, a natural scan is `cut_var: z_hf_max`, which keeps calibrators
always included while varying the maximum Hubble-flow redshift.

## `null` (correlated-cut drift Monte Carlo)

YAML 1.1 has an implicit `null` scalar, so to avoid parser surprises you can quote the key as
`"null":` (the runner also supports the unquoted form).

Notes:

- For most probes, `correlated_cut_null` uses a linear-Gaussian simulation (fit the max-cut model,
  inject noise, and re-fit across nested cuts).
- For `probe.name: siren_gate2_grid`, you can choose a siren-specific null:
  - `mode: event_gaussian` simulates event-level ln(H0) measurements using widths inferred from
    each event’s `logL(H0)` curve, then recomputes the selection-corrected posterior per cut.

```yaml
"null":
  n_mc: 200
  seed: 123
  drift_param: global_offset_mag
  use_diagonal_errors: true
  mode: ""   # default (linear-Gaussian); use "event_gaussian" for siren_gate2_grid
```

## `injection` (injection/recovery)

```yaml
injection:
  seed: 123
  n_mc: 200
  use_diagonal_errors: true
  mechanism: sky_dipole_mag
  frame: galactic
  axis_lon_deg: 264.021
  axis_lat_deg: 48.253
  amplitudes: [-0.10, -0.05, 0.0, 0.05, 0.10]
  param_of_interest: global_offset_mag
```

You can also provide a list to run multiple injection mechanisms in one audit packet:

```yaml
injection:
  - label: dust
    seed: 123
    mechanism: mwebv_linear_mag
    amplitudes: [-0.2, -0.1, 0.0, 0.1, 0.2]
    param_of_interest: delta_lnH0
  - label: quality
    seed: 123
    mechanism: m_b_corr_err_linear_mag
    apply_to: hf
    amplitudes: [-0.2, -0.1, 0.0, 0.1, 0.2]
    param_of_interest: delta_lnH0
  - label: cal
    seed: 123
    mechanism: calibrator_offset_mag
    amplitudes: [-0.2, -0.1, 0.0, 0.1, 0.2]
    param_of_interest: delta_lnH0
```

## `sbc` (coverage under repeated-noise truth)

```yaml
sbc:
  seed: 123
  n_rep: 256
  use_diagonal_errors: true
```

## `split_fit` (invariance by groups/bins)

```yaml
split_fit:
  ladder_level: L1
  split_var: idsurvey
  mode: categories   # or: bins + edges: [...]
  param: global_offset_mag
  use_diagonal_errors: true
```

You can also provide a list to run multiple splits:

```yaml
split_fit:
  - label: by_survey
    ladder_level: L1
    split_var: idsurvey
    mode: categories
    param: delta_lnH0
  - label: by_time
    ladder_level: L1
    split_var: pkmjd
    mode: bins
    edges: [44672.6, 53410.4, 54131.12, 54885.0, 57269.22, 59385.6]
    param: delta_lnH0
    include_nan: false
```

## `hemisphere_scan` (directional texture scan with train/test split)

```yaml
hemisphere_scan:
  seed: 123
  ladder_level: L1
  frame: galactic
  nside: 4
  param: global_offset_mag
  train_frac: 0.7
  z_match: true
  z_match_bins: 10
  use_diagonal_errors: true
  top_k: 10
```

## `split_null` (shuffle null for split-based invariance)

Monte Carlo null for a `split_fit`-style binning test. Shuffles the split variable (optionally within
groups) and recomputes a span/chi2 metric for the fitted parameter across bins.

```yaml
run:
  tasks: [split_fit, split_null]

split_null:
  seed: 123
  n_mc: 200
  ladder_level: L2
  split_var: pkmjd
  mode: bins
  edges: [44672.6, 53410.4, 54131.12, 54885.0, 57269.22, 59385.6]
  param: delta_lnH0
  use_diagonal_errors: true
  shuffle_within: idsurvey   # optional (preserves survey composition)
  include_nan: false         # if true, NaNs are treated as "always included"
```

## `predictive_score` (cross-validated log predictive density)

Fits the model on TRAIN subsets and scores the held-out TEST subset under the posterior predictive
distribution. This is a “does it generalize?” check (preferred over just adding parameters until
the in-sample chi2 improves).

Random split example:

```yaml
run:
  tasks: [predictive_score, report]

predictive_score:
  seed: 123
  mode: random
  n_rep: 50
  train_frac: 0.7
  always_include_calibrators: true
  use_diagonal_errors: true
  models:
    - label: baseline
      model: {}
    - label: +sky
      ladder_level: L4
      model:
        sky: {frame: galactic, lmin: 2, lmax: 3}
```

Group-holdout example (hold out one survey at a time):

```yaml
predictive_score:
  mode: group_holdout
  group_var: idsurvey
  min_group_n: 20
  always_include_calibrators: true
  use_diagonal_errors: true
  models:
    - label: baseline
      model: {}
```
