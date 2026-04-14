#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_TAG="${ATTEMPT046_OUTPUT_TAG:-grid2000_branch16_absxskip16_plot8_forcedfirstskip_sameedge_black_red_blue_explorer_shimizu_morioka_cpu}"
HTML_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}.html"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT046_OUTPUT_TAG="${OUTPUT_TAG}"

echo "Building attempt-046 interactive explorer from attempt-027 saved data."
echo "Output tag: ${OUTPUT_TAG}"
echo "Threads: ${JULIA_NUM_THREADS}"

julia --project=. "${SCRIPT_DIR}/build_explorer.jl"

tglfs upload --json "${HTML_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"

echo "Uploaded ${HTML_PATH}"
