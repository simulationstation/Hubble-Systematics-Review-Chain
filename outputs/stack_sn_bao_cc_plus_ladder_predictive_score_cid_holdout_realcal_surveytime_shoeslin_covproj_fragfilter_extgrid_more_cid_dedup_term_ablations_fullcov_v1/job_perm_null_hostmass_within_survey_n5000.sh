#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="outputs/stack_sn_bao_cc_plus_ladder_predictive_score_cid_holdout_realcal_surveytime_shoeslin_covproj_fragfilter_extgrid_more_cid_dedup_term_ablations_fullcov_v1"
PROG_JSON="${RUN_DIR}/perm_null_hostmass_within_survey_n5000.progress.json"
OUT_JSON="${RUN_DIR}/permutation_null__fields_host_mass_step_host_logmass_within_survey_n5000.json"

# Use only half the machine (0-127) and avoid BLAS oversubscription.
exec taskset -c 0-127 env \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/proxy_permutation_null.py" \
    --run-dir "${ROOT_DIR}/${RUN_DIR}" \
    --test-model "+fields+host_mass_step" \
    --permute-column host_logmass \
    --apply-to cal \
    --mode within_survey \
    --n-perm 5000 \
    --seed 123 \
    --n-proc 120 \
    --chunksize 1 \
    --progress-every 50 \
    --progress-secs 30 \
    --progress-path "${ROOT_DIR}/${PROG_JSON}" \
    --out-json "${ROOT_DIR}/${OUT_JSON}"
