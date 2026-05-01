#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -d "${HOME}/.juliaup/bin" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
fi

detect_cpu_threads() {
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    elif command -v nproc >/dev/null 2>&1; then
        nproc
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

export ATTEMPT052_NX="${ATTEMPT052_NX:-200}"
export ATTEMPT052_NY="${ATTEMPT052_NY:-200}"
export ATTEMPT052_DELTA_X_MIN="${ATTEMPT052_DELTA_X_MIN:--1.5}"
export ATTEMPT052_DELTA_X_MAX="${ATTEMPT052_DELTA_X_MAX:--0.5}"
export ATTEMPT052_DELTA_CA_MIN="${ATTEMPT052_DELTA_CA_MIN:--45.0}"
export ATTEMPT052_DELTA_CA_MAX="${ATTEMPT052_DELTA_CA_MAX:--20.0}"
export ATTEMPT052_G_H="0.0"
export ATTEMPT052_TAU_Y="${ATTEMPT052_TAU_Y:-2.0e4}"
export ATTEMPT052_LYAP_TMAX="${ATTEMPT052_LYAP_TMAX:-1.0e5}"
export ATTEMPT052_LYAP_MIN_TIME="${ATTEMPT052_LYAP_MIN_TIME:-3.0e4}"
export ATTEMPT052_LYAP_CHECK_INTERVAL="${ATTEMPT052_LYAP_CHECK_INTERVAL:-5.0e3}"
export ATTEMPT052_OUTPUT_DIR="${SCRIPT_DIR}"
export ATTEMPT052_OUTPUT_TAG="${ATTEMPT052_OUTPUT_TAG:-grid200_lyapdim_tmax1e5_gh0p000}"
export ATTEMPT052_PLOT_WIDTH="${ATTEMPT052_PLOT_WIDTH:-1800}"
export ATTEMPT052_PLOT_HEIGHT="${ATTEMPT052_PLOT_HEIGHT:-1300}"
export ATTEMPT052_PLOT_PX_PER_UNIT="${ATTEMPT052_PLOT_PX_PER_UNIT:-2.0}"

cd "${REPO_ROOT}"

if julia +release --version >/dev/null 2>&1; then
    JULIA_CMD=(julia +release)
else
    JULIA_CMD=(julia)
fi

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT052_OUTPUT_TAG}.log"

{
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Running attempt-053 local gh=0 Lyapunov-dimension scan"
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}"
    echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}"
    echo "Grid: ${ATTEMPT052_NY} Delta Ca x ${ATTEMPT052_NX} Delta x"
    echo "Delta Ca range: [${ATTEMPT052_DELTA_CA_MIN}, ${ATTEMPT052_DELTA_CA_MAX}]"
    echo "Delta x range: [${ATTEMPT052_DELTA_X_MIN}, ${ATTEMPT052_DELTA_X_MAX}]"
    echo "g_h: ${ATTEMPT052_G_H}"
    echo "tau_y: ${ATTEMPT052_TAU_Y}"
    echo "Lyapunov Tmax: ${ATTEMPT052_LYAP_TMAX}"
    echo "Lyapunov min-time: ${ATTEMPT052_LYAP_MIN_TIME}"
} | tee -a "${LOG_PATH}"

"${JULIA_CMD[@]}" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "${LOG_PATH}"

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. \
        kneading/experiment/attempt-052/main.jl 2>&1 | tee -a "${LOG_PATH}"
else
    time "${JULIA_CMD[@]}" --startup-file=no --project=. \
        kneading/experiment/attempt-052/main.jl 2>&1 | tee -a "${LOG_PATH}"
fi
