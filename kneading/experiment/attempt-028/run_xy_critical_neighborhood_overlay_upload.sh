#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_TAG="${ATTEMPT028_XY_OUTPUT_TAG:-alpha_slice_critical_neighborhood_xy_overlay}"
PNG_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}.png"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT028_XY_OUTPUT_TAG="${OUTPUT_TAG}"

echo "Rendering attempt-028 XY critical-neighborhood overlay."
echo "Output tag: ${OUTPUT_TAG}"
echo "Threads: ${JULIA_NUM_THREADS}"

julia --project=. "${SCRIPT_DIR}/plot_xy_critical_neighborhood_overlay.jl"

tglfs upload --json "${PNG_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"

echo "Uploaded ${PNG_PATH}"
