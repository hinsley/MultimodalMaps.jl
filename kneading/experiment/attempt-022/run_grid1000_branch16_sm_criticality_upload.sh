#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-24}"
export ATTEMPT022_N_ALPHA="${ATTEMPT022_N_ALPHA:-1000}"
export ATTEMPT022_N_LAMBDA="${ATTEMPT022_N_LAMBDA:-1000}"
export ATTEMPT022_MAX_ZMAX="${ATTEMPT022_MAX_ZMAX:-16}"
export ATTEMPT022_T_END="${ATTEMPT022_T_END:-700.0}"
export ATTEMPT022_DT="${ATTEMPT022_DT:-0.02}"
export ATTEMPT022_OUTPUT_TAG="${ATTEMPT022_OUTPUT_TAG:-grid1000_branch16_criticality_shimizu_morioka_cpu}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT022_OUTPUT_TAG}.log"

timestamped_tee() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
    done | tee -a "${LOG_PATH}"
}

cd "${REPO_ROOT}"

{
    echo "Running attempt-022 Shimizu-Morioka criticality-only CPU scan"
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Output tag: ${ATTEMPT022_OUTPUT_TAG}"
    echo "Grid: ${ATTEMPT022_N_ALPHA} alpha x ${ATTEMPT022_N_LAMBDA} lambda"
    /usr/bin/time -p julia --project=. kneading/experiment/attempt-022/contours.jl

    FINAL_PNG="${SCRIPT_DIR}/${ATTEMPT022_OUTPUT_TAG}_contours.png"
    if [[ ! -f "${FINAL_PNG}" ]]; then
        echo "Expected final contour PNG not found: ${FINAL_PNG}"
        exit 1
    fi

    echo "Uploading ${FINAL_PNG} with tglfs"
    tglfs upload --json "${FINAL_PNG}"
} 2>&1 | timestamped_tee
