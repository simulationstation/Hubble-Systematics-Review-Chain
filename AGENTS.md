# Agent Instructions (Hubble-Systematics-Review-Chain)

This folder is intended to be a *runner + reporting* framework. Keep it reproducible, config-driven,
and conservative about statistical claims (calibration gates).

## Running jobs

- For long runs (null MC / SBC / injection grids), follow the detached launch paradigm used in the
  other projects:
  - Create an output directory under `outputs/`.
  - Write a `job.sh` in that directory.
  - Launch via a single top-level command using `setsid` + `taskset`, writing PID to `pid.txt` and
    logs to `run.log`.
- Default to single-thread BLAS/OpenMP to avoid oversubscription:
  - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.

## Code style

- Prefer small, testable pure functions for core math (design matrices, posteriors, drift metrics).
- Always write run metadata (`config.yaml`, `env.json`, `git_like.json`) into each output directory.
- Do not silently change defaults that impact inference; surface them in configs.

