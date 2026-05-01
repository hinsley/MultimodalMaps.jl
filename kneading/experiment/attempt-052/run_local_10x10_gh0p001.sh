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

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$(detect_cpu_threads)}"
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export ATTEMPT052_NX="${ATTEMPT052_NX:-10}"
export ATTEMPT052_NY="${ATTEMPT052_NY:-10}"
export ATTEMPT052_DELTA_X_MIN="${ATTEMPT052_DELTA_X_MIN:--1.5}"
export ATTEMPT052_DELTA_X_MAX="${ATTEMPT052_DELTA_X_MAX:--0.5}"
export ATTEMPT052_DELTA_CA_MIN="${ATTEMPT052_DELTA_CA_MIN:--45.0}"
export ATTEMPT052_DELTA_CA_MAX="${ATTEMPT052_DELTA_CA_MAX:--20.0}"
export ATTEMPT052_G_H="${ATTEMPT052_G_H:-1.0e-3}"
export ATTEMPT052_TAU_Y="${ATTEMPT052_TAU_Y:-2.0e4}"
export ATTEMPT052_OUTPUT_TAG="${ATTEMPT052_OUTPUT_TAG:-local10x10_lyapdim_tmax1e5_gh0p001}"
export ATTEMPT052_LYAP_TMAX="${ATTEMPT052_LYAP_TMAX:-1.0e5}"
export ATTEMPT052_LYAP_MIN_TIME="${ATTEMPT052_LYAP_MIN_TIME:-3.0e4}"
export ATTEMPT052_LYAP_CHECK_INTERVAL="${ATTEMPT052_LYAP_CHECK_INTERVAL:-5.0e3}"

cd "${REPO_ROOT}"

if julia +release --version >/dev/null 2>&1; then
    JULIA_CMD=(julia +release)
else
    JULIA_CMD=(julia)
fi

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "[$(timestamp)] Running attempt-052 local 10x10 Lyapunov-dimension test."
echo "Threads: ${JULIA_NUM_THREADS}; g_h=${ATTEMPT052_G_H}; tag=${ATTEMPT052_OUTPUT_TAG}"

/usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. \
    kneading/experiment/attempt-052/main.jl 2>&1 | tee "${SCRIPT_DIR}/${ATTEMPT052_OUTPUT_TAG}.log"
