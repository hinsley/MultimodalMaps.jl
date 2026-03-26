#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-6}"
export ATTEMPT020_NX="${ATTEMPT020_NX:-1200}"
export ATTEMPT020_NY="${ATTEMPT020_NY:-1200}"
export ATTEMPT020_DELTA_X_MIN="${ATTEMPT020_DELTA_X_MIN:--3.2}"
export ATTEMPT020_DELTA_X_MAX="${ATTEMPT020_DELTA_X_MAX:-0.2}"
export ATTEMPT020_DELTA_CA_MIN="${ATTEMPT020_DELTA_CA_MIN:--50}"
export ATTEMPT020_DELTA_CA_MAX="${ATTEMPT020_DELTA_CA_MAX:--10}"
export ATTEMPT020_MAX_SEQ_LENGTH="${ATTEMPT020_MAX_SEQ_LENGTH:-10}"
export ATTEMPT020_MAP_RESOLUTION="${ATTEMPT020_MAP_RESOLUTION:-40}"
export ATTEMPT020_OUTPUT_TAG="${ATTEMPT020_OUTPUT_TAG:-grid1200_seq10_tonly_sin}"
export ATTEMPT020_REPAIR_INPUT_TAG="${ATTEMPT020_REPAIR_INPUT_TAG:-${ATTEMPT020_OUTPUT_TAG}}"
export ATTEMPT020_REPAIR_OUTPUT_TAG="${ATTEMPT020_REPAIR_OUTPUT_TAG:-grid1200_seq10_tonly_sin_remap40}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT020_REPAIR_OUTPUT_TAG}.log"

cd "${REPO_ROOT}"

echo "Running attempt-020 T-only scan with SiN model" | tee "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Output tag: ${ATTEMPT020_OUTPUT_TAG}" | tee -a "${LOG_PATH}"
echo "Repair output tag: ${ATTEMPT020_REPAIR_OUTPUT_TAG}" | tee -a "${LOG_PATH}"

/usr/bin/time -p julia --project=. kneading/experiment/attempt-020/contours.jl 2>&1 | tee -a "${LOG_PATH}"

export ATTEMPT020_OUTPUT_TAG="${ATTEMPT020_REPAIR_OUTPUT_TAG}"
/usr/bin/time -p julia --project=. kneading/experiment/attempt-020/repair_failed_runs.jl 2>&1 | tee -a "${LOG_PATH}"

FINAL_PNG="${SCRIPT_DIR}/${ATTEMPT020_REPAIR_OUTPUT_TAG}_contours.png"
if [[ ! -f "${FINAL_PNG}" ]]; then
    echo "Expected final contour PNG not found: ${FINAL_PNG}" | tee -a "${LOG_PATH}"
    exit 1
fi

echo "Uploading ${FINAL_PNG} with tglfs" | tee -a "${LOG_PATH}"
tglfs upload --json "${FINAL_PNG}" 2>&1 | tee -a "${LOG_PATH}"
