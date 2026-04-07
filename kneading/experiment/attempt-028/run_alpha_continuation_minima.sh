#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
unset ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION

exec julia --project=. "${SCRIPT_DIR}/continue_alpha_minima.jl"
