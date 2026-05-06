#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -d "${HOME}/.juliaup/bin" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
fi

JULIA_CMD=(julia)
if julia +release --version >/dev/null 2>&1; then
    JULIA_CMD=(julia +release)
fi

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)}"
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export ATTEMPT054_NX="${ATTEMPT056_NX:-200}"
export ATTEMPT054_NY="${ATTEMPT056_NY:-200}"
export ATTEMPT054_MAX_ITER="${ATTEMPT056_MAX_ITER:-8}"
export ATTEMPT054_TMAX="${ATTEMPT056_TMAX:-1.0e5}"
export ATTEMPT054_OUTPUT_TAG="${ATTEMPT056_OUTPUT_TAG:-grid200_sscs_vtangent_dotzero_tmax1e5_iter8_ystub}"
export ATTEMPT054_DELTA_X_MIN="${ATTEMPT056_DELTA_X_MIN:--1.5}"
export ATTEMPT054_DELTA_X_MAX="${ATTEMPT056_DELTA_X_MAX:--0.5}"
export ATTEMPT054_DELTA_CA_MIN="${ATTEMPT056_DELTA_CA_MIN:--45.0}"
export ATTEMPT054_DELTA_CA_MAX="${ATTEMPT056_DELTA_CA_MAX:--20.0}"
export ATTEMPT054_DELTA_X_TICK_STEP="${ATTEMPT056_DELTA_X_TICK_STEP:-0.1}"
export ATTEMPT054_DELTA_CA_TICK_STEP="${ATTEMPT056_DELTA_CA_TICK_STEP:-5.0}"
export ATTEMPT054_MAP_RESOLUTION="${ATTEMPT056_MAP_RESOLUTION:-40}"
export ATTEMPT054_PLOT_WIDTH="${ATTEMPT056_PLOT_WIDTH:-2000}"
export ATTEMPT054_PLOT_HEIGHT="${ATTEMPT056_PLOT_HEIGHT:-1500}"
export ATTEMPT054_PLOT_PX_PER_UNIT="${ATTEMPT056_PLOT_PX_PER_UNIT:-2.0}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT054_OUTPUT_TAG}.log"

cd "${REPO_ROOT}"

{
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Running attempt-056 SSCS-event tangent-V zero-contour scan."
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia command: ${JULIA_CMD[*]}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}"
    echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}"
    echo "Grid: ${ATTEMPT054_NY} Delta Ca x ${ATTEMPT054_NX} Delta x"
    echo "Delta Ca range: [${ATTEMPT054_DELTA_CA_MIN}, ${ATTEMPT054_DELTA_CA_MAX}]"
    echo "Delta x range: [${ATTEMPT054_DELTA_X_MIN}, ${ATTEMPT054_DELTA_X_MAX}]"
    echo "Max SSCS symbols: ${ATTEMPT054_MAX_ITER}"
    echo "Tangent integration tmax: ${ATTEMPT054_TMAX}"
    echo "Output tag: ${ATTEMPT054_OUTPUT_TAG}"
} | tee -a "${LOG_PATH}"

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-056/main.jl 2>&1 | tee -a "${LOG_PATH}"
else
    time "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-056/main.jl 2>&1 | tee -a "${LOG_PATH}"
fi

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] attempt-056 complete." | tee -a "${LOG_PATH}"
