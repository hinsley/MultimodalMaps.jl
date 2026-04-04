#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_TAG="${ATTEMPT025_SWEEP_TAG:-grid1200_branch16_absxskip16_plot8_shimizu_morioka_cpu}"
OUTPUT_TAG="${ATTEMPT025_GIF_OUTPUT_TAG:-grid1200_branch16_absxskip16_plot8_deltatfix_nominal_iterates_shimizu_morioka_cpu}"
FRAME_START="${ATTEMPT025_GIF_FRAME_START:-1}"
FRAME_END="${ATTEMPT025_GIF_FRAME_END:-${ATTEMPT025_PLOT_ITERATE_CAP:-8}}"
FRAME_SECONDS="${ATTEMPT025_GIF_FRAME_SECONDS:-1}"
FRAME_DIR="${SCRIPT_DIR}/${OUTPUT_TAG}_frames"
GIF_PATH="${SCRIPT_DIR}/${OUTPUT_TAG}.gif"
UPLOAD_JSON="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.json"
UPLOAD_STDERR="${SCRIPT_DIR}/${OUTPUT_TAG}_upload.stderr"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-32}"
export ATTEMPT025_SWEEP_TAG="${SWEEP_TAG}"
export ATTEMPT025_RUN_COLUMNS="${ATTEMPT025_RUN_COLUMNS:-false}"
export ATTEMPT025_PLOT_ITERATE_CAP="${ATTEMPT025_PLOT_ITERATE_CAP:-8}"
export ATTEMPT025_MAX_EVENT_ITERATES="${ATTEMPT025_MAX_EVENT_ITERATES:-16}"
export ATTEMPT025_GIF_OUTPUT_TAG="${OUTPUT_TAG}"
export ATTEMPT025_GIF_FRAME_START="${FRAME_START}"
export ATTEMPT025_GIF_FRAME_END="${FRAME_END}"
export ATTEMPT025_GIF_BLACK_CONTOURS="${ATTEMPT025_GIF_BLACK_CONTOURS:-false}"
export ATTEMPT025_GIF_SHOW_EXCLUDED="${ATTEMPT025_GIF_SHOW_EXCLUDED:-false}"

echo "Rendering attempt-025 nominal-iterate GIF from existing sweep data."
echo "Sweep source tag: ${SWEEP_TAG}"
echo "GIF output tag: ${OUTPUT_TAG}"
echo "Threads: ${JULIA_NUM_THREADS}, processed nominal iterates: 1:${ATTEMPT025_PLOT_ITERATE_CAP}, rendered frames: ${FRAME_START}:${FRAME_END}, seconds/frame: ${FRAME_SECONDS}"

rm -f "${FRAME_DIR}"/frame_*.png

julia --project=. "${SCRIPT_DIR}/nominal_iterate_gif.jl"

ffmpeg -y \
  -framerate "1/${FRAME_SECONDS}" \
  -start_number "${FRAME_START}" \
  -i "${FRAME_DIR}/frame_%02d.png" \
  -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  -loop 0 \
  "${GIF_PATH}"

tglfs upload --json "${GIF_PATH}" > "${UPLOAD_JSON}" 2> "${UPLOAD_STDERR}"

echo "Uploaded ${GIF_PATH}"
