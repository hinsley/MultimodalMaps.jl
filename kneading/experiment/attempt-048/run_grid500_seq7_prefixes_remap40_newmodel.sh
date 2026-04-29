#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

detect_cpu_threads() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        echo 6
    fi
}

DEFAULT_THREADS="$(detect_cpu_threads)"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-${DEFAULT_THREADS}}"
DEFAULT_GC_THREADS=$((JULIA_NUM_THREADS / 4))
if (( DEFAULT_GC_THREADS < 1 )); then
    DEFAULT_GC_THREADS=1
elif (( DEFAULT_GC_THREADS > 8 )); then
    DEFAULT_GC_THREADS=8
fi
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-${DEFAULT_GC_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
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
export ATTEMPT048_INSTANTIATE="${ATTEMPT048_INSTANTIATE:-1}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT048_OUTPUT_TAG}.log"

on_shutdown() {
    {
        echo
        echo "[$(date -Is)] Received shutdown signal. Completed column files are resumable; incomplete columns will be recomputed."
        sync || true
    } | tee -a "${LOG_PATH}"
}
trap on_shutdown TERM INT HUP

cd "${REPO_ROOT}"

echo | tee -a "${LOG_PATH}"
echo "[$(date -Is)] Running attempt-048 full scan with remap resolution ${ATTEMPT048_MAP_RESOLUTION}" | tee -a "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}" | tee -a "${LOG_PATH}"
echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}" | tee -a "${LOG_PATH}"

if [[ "${ATTEMPT048_INSTANTIATE}" == "1" ]]; then
    echo "Instantiating Julia project dependencies." | tee -a "${LOG_PATH}"
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "${LOG_PATH}"
fi

/usr/bin/time -p julia --startup-file=no --project=. kneading/experiment/attempt-048/contours.jl 2>&1 | tee -a "${LOG_PATH}"
