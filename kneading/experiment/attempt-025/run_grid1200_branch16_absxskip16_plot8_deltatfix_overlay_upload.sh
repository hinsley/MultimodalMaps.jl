#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_TAG="${ATTEMPT025_SWEEP_TAG:-grid1200_branch16_absxskip16_plot8_shimizu_morioka_cpu}"
OUTPUT_TAG="${ATTEMPT025_OVERLAY_OUTPUT_TAG:-grid1200_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_overlay_shimizu_morioka_cpu}"
PNG_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}.png"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT025_SWEEP_TAG="${SWEEP_TAG}"
export ATTEMPT025_RUN_COLUMNS="${ATTEMPT025_RUN_COLUMNS:-false}"
export ATTEMPT025_PLOT_ITERATE_CAP="${ATTEMPT025_PLOT_ITERATE_CAP:-8}"
export ATTEMPT025_MAX_EVENT_ITERATES="${ATTEMPT025_MAX_EVENT_ITERATES:-16}"
export ATTEMPT025_OVERLAY_OUTPUT_TAG="${OUTPUT_TAG}"
export ATTEMPT025_OVERLAY_ITERATE_START="${ATTEMPT025_OVERLAY_ITERATE_START:-2}"
export ATTEMPT025_OVERLAY_ITERATE_END="${ATTEMPT025_OVERLAY_ITERATE_END:-8}"

echo "Rendering attempt-025 monochrome accepted/pruned overlay PNG from existing sweep data."
echo "Sweep source tag: ${SWEEP_TAG}"
echo "PNG output tag: ${OUTPUT_TAG}"
echo "Threads: ${JULIA_NUM_THREADS}, processed nominal iterates: 1:${ATTEMPT025_PLOT_ITERATE_CAP}, rendered overlay iterates: ${ATTEMPT025_OVERLAY_ITERATE_START}:${ATTEMPT025_OVERLAY_ITERATE_END}"

julia --project=. "${SCRIPT_DIR}/overlay_excluded_png.jl"

tglfs upload --json "${PNG_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"

echo "Uploaded ${PNG_PATH}"
