#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_TAG="${ATTEMPT025_OUTPUT_TAG:-grid1200_branch16_absxskip16_plot8_shimizu_morioka_cpu}"
PLOT_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}_contours.png"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT025_OUTPUT_TAG="${OUTPUT_TAG}"
export ATTEMPT025_PLOT_ITERATE_CAP="${ATTEMPT025_PLOT_ITERATE_CAP:-8}"
export ATTEMPT025_MAX_EVENT_ITERATES="${ATTEMPT025_MAX_EVENT_ITERATES:-16}"

julia --project=. "${SCRIPT_DIR}/contours.jl"
tglfs upload --json "${PLOT_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"
