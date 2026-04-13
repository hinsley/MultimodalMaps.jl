#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-24}"
export ATTEMPT023_N_ALPHA="${ATTEMPT023_N_ALPHA:-5000}"
export ATTEMPT023_N_LAMBDA="${ATTEMPT023_N_LAMBDA:-5000}"
export ATTEMPT023_MAX_ZMAX="${ATTEMPT023_MAX_ZMAX:-16}"
export ATTEMPT023_T_END="${ATTEMPT023_T_END:-700.0}"
export ATTEMPT023_DT="${ATTEMPT023_DT:-0.02}"
export ATTEMPT023_OUTPUT_TAG="${ATTEMPT023_OUTPUT_TAG:-grid5000_branch16_criticality_shimizu_morioka_cpu}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT023_OUTPUT_TAG}.log"

timestamped_tee() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
    done | tee -a "${LOG_PATH}"
}

cd "${REPO_ROOT}"

{
    echo "Running attempt-023 Shimizu-Morioka criticality-only CPU scan"
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Output tag: ${ATTEMPT023_OUTPUT_TAG}"
    echo "Grid: ${ATTEMPT023_N_ALPHA} alpha x ${ATTEMPT023_N_LAMBDA} lambda"
    /usr/bin/time -p julia --project=. kneading/experiment/attempt-023/contours.jl

    FINAL_PNG="${SCRIPT_DIR}/${ATTEMPT023_OUTPUT_TAG}_contours.png"
    FINAL_SVG="${SCRIPT_DIR}/${ATTEMPT023_OUTPUT_TAG}_contours.svg"
    if [[ ! -f "${FINAL_PNG}" ]]; then
        echo "Expected final contour PNG not found: ${FINAL_PNG}"
        exit 1
    fi
    if [[ ! -f "${FINAL_SVG}" ]]; then
        echo "Expected final contour SVG not found: ${FINAL_SVG}"
        exit 1
    fi

    echo "Uploading ${FINAL_PNG} with tglfs"
    tglfs upload --json "${FINAL_PNG}"
    echo "Uploading ${FINAL_SVG} with tglfs"
    tglfs upload --json "${FINAL_SVG}"
} 2>&1 | timestamped_tee
