#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-6}"
export ATTEMPT017_NX="${ATTEMPT017_NX:-1200}"
export ATTEMPT017_NY="${ATTEMPT017_NY:-1200}"
export ATTEMPT017_DELTA_X_MIN="${ATTEMPT017_DELTA_X_MIN:--3.2}"
export ATTEMPT017_DELTA_X_MAX="${ATTEMPT017_DELTA_X_MAX:-0.2}"
export ATTEMPT017_DELTA_CA_MIN="${ATTEMPT017_DELTA_CA_MIN:--50}"
export ATTEMPT017_DELTA_CA_MAX="${ATTEMPT017_DELTA_CA_MAX:--10}"
export ATTEMPT017_MAX_SEQ_LENGTH="${ATTEMPT017_MAX_SEQ_LENGTH:-10}"
export ATTEMPT017_MAP_RESOLUTION="${ATTEMPT017_MAP_RESOLUTION:-40}"
export ATTEMPT017_OUTPUT_TAG="${ATTEMPT017_OUTPUT_TAG:-grid1200_seq10_prefixes_remap40}"
export ATTEMPT017_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT017_MAX_PREFIX_PLOT_LENGTH:-10}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT017_OUTPUT_TAG}.log"

cd "${REPO_ROOT}"

echo "Running attempt-017 full scan with remap resolution ${ATTEMPT017_MAP_RESOLUTION}" | tee "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"

/usr/bin/time -p julia --project=. kneading/experiment/attempt-017/contours.jl 2>&1 | tee -a "${LOG_PATH}"
