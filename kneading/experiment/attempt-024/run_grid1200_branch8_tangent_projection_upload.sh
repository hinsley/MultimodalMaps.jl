#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-24}"
export ATTEMPT024_N_ALPHA="${ATTEMPT024_N_ALPHA:-1200}"
export ATTEMPT024_N_LAMBDA="${ATTEMPT024_N_LAMBDA:-1200}"
export ATTEMPT024_MAX_ZMAX="${ATTEMPT024_MAX_ZMAX:-8}"
export ATTEMPT024_T_END="${ATTEMPT024_T_END:-200.0}"
export ATTEMPT024_DT="${ATTEMPT024_DT:-0.02}"
export ATTEMPT024_OUTPUT_TAG="${ATTEMPT024_OUTPUT_TAG:-grid1200_branch8_tangent_zproj_shimizu_morioka_cpu}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT024_OUTPUT_TAG}.log"

timestamped_tee() {
    while IFS= read -r line; do
        printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"
    done | tee -a "${LOG_PATH}"
}

cd "${REPO_ROOT}"

{
    echo "Running attempt-024 Shimizu-Morioka tangent-projection CPU scan"
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Output tag: ${ATTEMPT024_OUTPUT_TAG}"
    echo "Grid: ${ATTEMPT024_N_ALPHA} alpha x ${ATTEMPT024_N_LAMBDA} lambda"
    /usr/bin/time -p julia --project=. kneading/experiment/attempt-024/contours.jl

    FINAL_PNG="${SCRIPT_DIR}/${ATTEMPT024_OUTPUT_TAG}_contours.png"
    if [[ ! -f "${FINAL_PNG}" ]]; then
        echo "Expected final contour PNG not found: ${FINAL_PNG}"
        exit 1
    fi

    echo "Uploading ${FINAL_PNG} with tglfs"
    env -u TGLFS_UPLOAD_PASSWORD tglfs upload --json "${FINAL_PNG}"
} 2>&1 | timestamped_tee
