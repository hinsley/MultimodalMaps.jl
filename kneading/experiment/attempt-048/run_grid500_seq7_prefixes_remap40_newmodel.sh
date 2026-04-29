#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-6}"
export ATTEMPT048_NX="${ATTEMPT048_NX:-500}"
export ATTEMPT048_NY="${ATTEMPT048_NY:-500}"
export ATTEMPT048_DELTA_X_MIN="${ATTEMPT048_DELTA_X_MIN:--1.5}"
export ATTEMPT048_DELTA_X_MAX="${ATTEMPT048_DELTA_X_MAX:--0.5}"
export ATTEMPT048_DELTA_CA_MIN="${ATTEMPT048_DELTA_CA_MIN:--45}"
export ATTEMPT048_DELTA_CA_MAX="${ATTEMPT048_DELTA_CA_MAX:--20}"
export ATTEMPT048_MAX_SEQ_LENGTH="${ATTEMPT048_MAX_SEQ_LENGTH:-7}"
export ATTEMPT048_MAP_RESOLUTION="${ATTEMPT048_MAP_RESOLUTION:-40}"
export ATTEMPT048_OUTPUT_TAG="${ATTEMPT048_OUTPUT_TAG:-grid500_seq7_prefixes_remap40_newmodel}"
export ATTEMPT048_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT048_MAX_PREFIX_PLOT_LENGTH:-7}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT048_OUTPUT_TAG}.log"

cd "${REPO_ROOT}"

echo "Running attempt-048 full scan with remap resolution ${ATTEMPT048_MAP_RESOLUTION}" | tee "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"

/usr/bin/time -p julia --project=. kneading/experiment/attempt-048/contours.jl 2>&1 | tee -a "${LOG_PATH}"
