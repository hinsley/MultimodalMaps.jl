#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
export ATTEMPT054_NX=200
export ATTEMPT054_NY=200
export ATTEMPT054_MAX_ITER=8
export ATTEMPT054_TMAX=1.0e5
export ATTEMPT054_OUTPUT_TAG=grid200_tangent_ca_dotzero_tmax1e5_iter8_ystub
export ATTEMPT054_DELTA_X_MIN=-1.5
export ATTEMPT054_DELTA_X_MAX=-0.5
export ATTEMPT054_DELTA_CA_MIN=-45.0
export ATTEMPT054_DELTA_CA_MAX=-20.0
export ATTEMPT054_DELTA_X_TICK_STEP=0.1
export ATTEMPT054_DELTA_CA_TICK_STEP=5
export ATTEMPT054_PLOT_PX_PER_UNIT=2

if [[ -z "${JULIA_CMD:-}" ]]; then
    if julia +release --version >/dev/null 2>&1; then
        JULIA_CMD=(julia +release)
    else
        JULIA_CMD=(julia)
    fi
else
    # shellcheck disable=SC2206
    JULIA_CMD=(${JULIA_CMD})
fi

"${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-054/main.jl 2>&1 | tee kneading/experiment/attempt-054/${ATTEMPT054_OUTPUT_TAG}.log
