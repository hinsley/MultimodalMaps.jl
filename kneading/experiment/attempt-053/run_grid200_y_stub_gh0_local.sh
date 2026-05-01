#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

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

export ATTEMPT053_NX="${ATTEMPT053_NX:-200}"
export ATTEMPT053_NY="${ATTEMPT053_NY:-200}"
export ATTEMPT053_OUTPUT_DIR="${SCRIPT_DIR}"
export ATTEMPT053_OUTPUT_TAG="${ATTEMPT053_OUTPUT_TAG:-grid200_lyapdim_tmax1e5_gh0p000_y_stub}"

julia +release --startup-file=no --project=. "${SCRIPT_DIR}/main_y_stub_gh0.jl"
