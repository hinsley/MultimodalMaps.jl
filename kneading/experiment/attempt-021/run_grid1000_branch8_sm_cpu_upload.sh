#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-12}"
export ATTEMPT021_N_ALPHA="${ATTEMPT021_N_ALPHA:-1000}"
export ATTEMPT021_N_LAMBDA="${ATTEMPT021_N_LAMBDA:-1000}"
export ATTEMPT021_MAX_ZMAX="${ATTEMPT021_MAX_ZMAX:-8}"
export ATTEMPT021_T_END="${ATTEMPT021_T_END:-350.0}"
export ATTEMPT021_DT="${ATTEMPT021_DT:-0.02}"
export ATTEMPT021_OUTPUT_TAG="${ATTEMPT021_OUTPUT_TAG:-grid1000_branch8_shimizu_morioka}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT021_OUTPUT_TAG}.log"

timestamped_tee() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
    done | tee -a "${LOG_PATH}"
}

cd "${REPO_ROOT}"

{
    echo "Running attempt-021 Shimizu-Morioka CPU scan"
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Output tag: ${ATTEMPT021_OUTPUT_TAG}"
    echo "Grid: ${ATTEMPT021_N_ALPHA} alpha x ${ATTEMPT021_N_LAMBDA} lambda"
    /usr/bin/time -p julia --project=. kneading/experiment/attempt-021/contours.jl

    FINAL_PNG="${SCRIPT_DIR}/${ATTEMPT021_OUTPUT_TAG}_contours.png"
    if [[ ! -f "${FINAL_PNG}" ]]; then
        echo "Expected final contour PNG not found: ${FINAL_PNG}"
        exit 1
    fi

    echo "Uploading ${FINAL_PNG} with tglfs"
    tglfs upload --json "${FINAL_PNG}"
} 2>&1 | timestamped_tee
