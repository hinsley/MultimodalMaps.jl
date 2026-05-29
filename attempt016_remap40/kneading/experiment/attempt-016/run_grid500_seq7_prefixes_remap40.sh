#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-6}"
export ATTEMPT016_NX="${ATTEMPT016_NX:-500}"
export ATTEMPT016_NY="${ATTEMPT016_NY:-500}"
export ATTEMPT016_DELTA_X_MIN="${ATTEMPT016_DELTA_X_MIN:--1.5}"
export ATTEMPT016_DELTA_X_MAX="${ATTEMPT016_DELTA_X_MAX:--0.5}"
export ATTEMPT016_DELTA_CA_MIN="${ATTEMPT016_DELTA_CA_MIN:--45}"
export ATTEMPT016_DELTA_CA_MAX="${ATTEMPT016_DELTA_CA_MAX:--20}"
export ATTEMPT016_MAX_SEQ_LENGTH="${ATTEMPT016_MAX_SEQ_LENGTH:-7}"
export ATTEMPT016_MAP_RESOLUTION="${ATTEMPT016_MAP_RESOLUTION:-40}"
export ATTEMPT016_OUTPUT_TAG="${ATTEMPT016_OUTPUT_TAG:-grid500_seq7_prefixes_remap40}"
export ATTEMPT016_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT016_MAX_PREFIX_PLOT_LENGTH:-7}"
export JULIA_PKG_PRECOMPILE_AUTO="${JULIA_PKG_PRECOMPILE_AUTO:-0}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT016_OUTPUT_TAG}.log"

cd "${REPO_ROOT}"

echo "Running attempt-016 full scan with remap resolution ${ATTEMPT016_MAP_RESOLUTION}" | tee "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Julia command: ${JULIA_CMD:-julia}" | tee -a "${LOG_PATH}"

/usr/bin/time -p ${JULIA_CMD:-julia} --startup-file=no --project=. kneading/experiment/attempt-016/contours.jl 2>&1 | tee -a "${LOG_PATH}"
