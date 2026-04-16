#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_TAG="${ATTEMPT025_OUTPUT_TAG:-grid3000_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_retired_shimizu_morioka_cpu}"
OUTPUT_TAG="${ATTEMPT047_OUTPUT_TAG:-grid3000_branch16_absxskip16_plot8_forcedfirstskip_sameedge_black_red_blue_explorer_shimizu_morioka_cpu}"
HTML_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}.html"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT025_OUTPUT_TAG="${SWEEP_TAG}"
export ATTEMPT025_SWEEP_TAG="${ATTEMPT025_SWEEP_TAG:-${SWEEP_TAG}}"
export ATTEMPT025_RUN_COLUMNS="${ATTEMPT025_RUN_COLUMNS:-true}"
export ATTEMPT025_WRITE_MERGED_RESULTS="${ATTEMPT025_WRITE_MERGED_RESULTS:-false}"
export ATTEMPT025_WRITE_ITERATE_STATS="${ATTEMPT025_WRITE_ITERATE_STATS:-false}"
export ATTEMPT025_N_ALPHA="${ATTEMPT025_N_ALPHA:-3000}"
export ATTEMPT025_N_LAMBDA="${ATTEMPT025_N_LAMBDA:-3000}"
export ATTEMPT025_MAX_EVENT_ITERATES="${ATTEMPT025_MAX_EVENT_ITERATES:-16}"
export ATTEMPT025_PLOT_ITERATE_CAP="${ATTEMPT025_PLOT_ITERATE_CAP:-8}"
export ATTEMPT047_OUTPUT_TAG="${OUTPUT_TAG}"

echo "Running attempt-047 full recompute and symbolic-only explorer build."
echo "Sweep tag: ${SWEEP_TAG}"
echo "Output tag: ${OUTPUT_TAG}"
echo "Threads: ${JULIA_NUM_THREADS}"
echo "Grid: ${ATTEMPT025_N_ALPHA} x ${ATTEMPT025_N_LAMBDA}"
echo "Stored iterates: ${ATTEMPT025_MAX_EVENT_ITERATES}, plotted nominal iterates: 1:${ATTEMPT025_PLOT_ITERATE_CAP}"

julia --project=. "${SCRIPT_DIR}/run_columns_only.jl"
julia --project=. "${SCRIPT_DIR}/build_explorer.jl"

tglfs upload --json "${HTML_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"

echo "Uploaded ${HTML_PATH}"
