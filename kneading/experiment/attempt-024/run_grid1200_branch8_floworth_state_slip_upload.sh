#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_TAG="${ATTEMPT024_OUTPUT_TAG:-grid1200_branch8_floworth_state_slip_shimizu_morioka_cpu}"
export ATTEMPT024_OUTPUT_TAG="$OUTPUT_TAG"
export ATTEMPT024_SWEEP_TAG="${ATTEMPT024_SWEEP_TAG:-$OUTPUT_TAG}"
export ATTEMPT024_N_ALPHA="${ATTEMPT024_N_ALPHA:-1200}"
export ATTEMPT024_N_LAMBDA="${ATTEMPT024_N_LAMBDA:-1200}"
export ATTEMPT024_MAX_EVENT_ITERATES="${ATTEMPT024_MAX_EVENT_ITERATES:-8}"
export ATTEMPT024_T_END="${ATTEMPT024_T_END:-200.0}"
export ATTEMPT024_FIG_WIDTH="${ATTEMPT024_FIG_WIDTH:-1200}"
export ATTEMPT024_FIG_HEIGHT="${ATTEMPT024_FIG_HEIGHT:-1200}"
export ATTEMPT024_PX_PER_UNIT="${ATTEMPT024_PX_PER_UNIT:-1.0}"
export ATTEMPT024_LINEWIDTH="${ATTEMPT024_LINEWIDTH:-0.35}"
export ATTEMPT024_WRITE_MERGED_RESULTS="${ATTEMPT024_WRITE_MERGED_RESULTS:-false}"

LOG_PATH="kneading/experiment/attempt-024/${OUTPUT_TAG}.log"

echo "[run] output tag: ${OUTPUT_TAG}"
echo "[run] log path: ${LOG_PATH}"

JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-24}" /usr/bin/time -p \
    julia --project=. kneading/experiment/attempt-024/contours.jl \
    2>&1 | tee "$LOG_PATH"

ZMAX_PNG="kneading/experiment/attempt-024/${OUTPUT_TAG}_zmax_contours.png"
ABSXMAX_PNG="kneading/experiment/attempt-024/${OUTPUT_TAG}_absxmax_contours.png"

echo "[upload] ${ZMAX_PNG}"
env -u TGLFS_UPLOAD_PASSWORD tglfs upload --json "$ZMAX_PNG"

echo "[upload] ${ABSXMAX_PNG}"
env -u TGLFS_UPLOAD_PASSWORD tglfs upload --json "$ABSXMAX_PNG"
