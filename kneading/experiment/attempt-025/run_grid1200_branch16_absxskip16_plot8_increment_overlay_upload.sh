#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_TAG="${ATTEMPT025_SWEEP_TAG:-grid1200_branch16_absxskip16_plot8_shimizu_morioka_cpu}"
OUTPUT_TAG="${ATTEMPT025_OUTPUT_TAG:-grid1200_branch16_absxskip16_plot8_increment_overlay_shimizu_morioka_cpu}"
PLOT_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}_contours.png"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT025_SWEEP_TAG="${SWEEP_TAG}"
export ATTEMPT025_OUTPUT_TAG="${OUTPUT_TAG}"
export ATTEMPT025_RUN_COLUMNS="${ATTEMPT025_RUN_COLUMNS:-false}"
export ATTEMPT025_DRAW_INCREMENT_OVERLAY="${ATTEMPT025_DRAW_INCREMENT_OVERLAY:-true}"
export ATTEMPT025_WRITE_INCREMENT_COUNTS="${ATTEMPT025_WRITE_INCREMENT_COUNTS:-true}"
export ATTEMPT025_WRITE_MERGED_RESULTS="${ATTEMPT025_WRITE_MERGED_RESULTS:-false}"
export ATTEMPT025_PLOT_ITERATE_CAP="${ATTEMPT025_PLOT_ITERATE_CAP:-8}"

echo "Rendering attempt-025 increment-overlay debug plot from existing sweep data."
echo "Sweep source tag: ${SWEEP_TAG}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Threads: ${JULIA_NUM_THREADS}, plotted nominal iterates: ${ATTEMPT025_PLOT_ITERATE_CAP}"

julia --project=. "${SCRIPT_DIR}/contours.jl"
tglfs upload --json "${PLOT_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"

echo "Uploaded ${PLOT_PATH}"
