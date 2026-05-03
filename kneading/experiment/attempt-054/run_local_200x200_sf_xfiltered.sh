#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-auto}"
export ATTEMPT054_NX=200
export ATTEMPT054_NY=200
export ATTEMPT054_MAX_ITER=8
export ATTEMPT054_TMAX=1.0e5
export ATTEMPT054_OUTPUT_TAG=grid200_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered
export ATTEMPT054_DELTA_X_MIN=-1.5
export ATTEMPT054_DELTA_X_MAX=-0.5
export ATTEMPT054_DELTA_CA_MIN=-45.0
export ATTEMPT054_DELTA_CA_MAX=-20.0
export ATTEMPT054_DELTA_X_TICK_STEP=0.1
export ATTEMPT054_DELTA_CA_TICK_STEP=5.0
export ATTEMPT054_MAP_RESOLUTION=40
export ATTEMPT054_CA_MIN_V_MAX=0.0
export ATTEMPT054_PLOT_WIDTH=1600
export ATTEMPT054_PLOT_HEIGHT=1200
export ATTEMPT054_PLOT_PX_PER_UNIT=2.0

exec julia +release --project=. "${SCRIPT_DIR}/main.jl"
