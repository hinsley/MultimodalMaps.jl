#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d "${HOME}/.juliaup/bin" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
fi

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-10}"
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export ATTEMPT053_NX=10
export ATTEMPT053_NY=10
export ATTEMPT053_OUTPUT_DIR="${SCRIPT_DIR}"
export ATTEMPT053_OUTPUT_TAG="local10x10_lyapdim_tmax1e5_gh0p000_y_stub"

julia +release --startup-file=no --project=. "${SCRIPT_DIR}/main_y_stub_gh0.jl"
