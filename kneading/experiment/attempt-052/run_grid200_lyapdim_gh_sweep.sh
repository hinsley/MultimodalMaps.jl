#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ATTEMPT052_NX="${ATTEMPT052_NX:-200}"
export ATTEMPT052_NY="${ATTEMPT052_NY:-200}"
export ATTEMPT052_TAG_GRID_LABEL="${ATTEMPT052_TAG_GRID_LABEL:-grid200}"

exec "${SCRIPT_DIR}/run_grid1000_lyapdim_gh_sweep.sh"
