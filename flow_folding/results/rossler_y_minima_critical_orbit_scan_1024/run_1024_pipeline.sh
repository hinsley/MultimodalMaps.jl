#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RESULT_DIR}/../../.." && pwd)"
TSV="${RESULT_DIR}/coarse_scan.tsv"
TGLFS_UPLOAD_JSON="${RESULT_DIR}/coarse_scan_tglfs_upload.json"
HEATMAP_DIR="${RESULT_DIR}/heatmaps"
LOG="${RESULT_DIR}/run.log"
PNG_WIDTH="${MM_FLOW_FOLDING_PNG_WIDTH:-6400}"
PNG_HEIGHT="${MM_FLOW_FOLDING_PNG_HEIGHT:-4400}"

cd "${REPO_ROOT}"
mkdir -p "${RESULT_DIR}" "${HEATMAP_DIR}"
THREADS="${JULIA_NUM_THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)}"
export JULIA_NUM_THREADS="${THREADS}"

{
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=${REPO_ROOT}"
  echo "result_dir=${RESULT_DIR}"
  echo "julia_num_threads=${JULIA_NUM_THREADS}"
  echo "parallel_axis=columns"
  echo "column_continuation=a_forward"
  echo "column_anchor=a_min_serial"
  echo "n_c=1024"
  echo "n_a=1024"
  echo "c_range=2.0..7.0"
  echo "a_range=0.30..0.55"
  echo "b=0.3"
  echo "critical_event_index=4"
  echo "rho_range=-24.0..-1.0"
  echo "word_length=8"
  echo "orbit_transient_events=0"
  echo "initial_event_included=true"
  echo "max_time=2000.0"
  echo "dt=0.05"
  echo "png_width=${PNG_WIDTH}"
  echo "png_height=${PNG_HEIGHT}"
  echo "scan_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  /usr/bin/time -p env \
    JULIA_NUM_THREADS="${JULIA_NUM_THREADS}" \
    MM_FLOW_FOLDING_PARALLEL_AXIS=columns \
    MM_FLOW_FOLDING_NC=1024 \
    MM_FLOW_FOLDING_NA=1024 \
    MM_FLOW_FOLDING_C_MIN=2.0 \
    MM_FLOW_FOLDING_C_MAX=7.0 \
    MM_FLOW_FOLDING_A_MIN=0.30 \
    MM_FLOW_FOLDING_A_MAX=0.55 \
    MM_FLOW_FOLDING_B=0.3 \
    MM_FLOW_FOLDING_CRITICAL_EVENT_INDEX=4 \
    MM_FLOW_FOLDING_RHO_MIN=-24 \
    MM_FLOW_FOLDING_RHO_MAX=-1 \
    MM_FLOW_FOLDING_WORD_LENGTH=8 \
    MM_FLOW_FOLDING_MAX_TIME=2000 \
    MM_FLOW_FOLDING_DT=0.05 \
    MM_FLOW_FOLDING_PROGRESS_SECONDS=60 \
    MM_FLOW_FOLDING_OUTPUT="${TSV}" \
    julia --startup-file=no --project=. flow_folding/examples/rossler_y_minima_critical_orbit_scan.jl

  echo "scan_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 flow_folding/examples/verify_rossler_y_minima_critical_orbit_outputs.py "${TSV}" \
    --n-c 1024 \
    --n-a 1024

  echo "render_word_heatmap_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  /usr/bin/time -p python3 flow_folding/examples/rossler_y_minima_tangent_pngs.py "${TSV}" \
    --output-dir "${HEATMAP_DIR}" \
    --stem coarse_scan \
    --width "${PNG_WIDTH}" \
    --height "${PNG_HEIGHT}" \
    --only-heatmap \
    --write-heatmap-probe \
    --clean

  echo "render_monotone_heatmap_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  /usr/bin/time -p python3 flow_folding/examples/rossler_y_minima_tangent_pngs.py "${TSV}" \
    --output-dir "${HEATMAP_DIR}" \
    --stem coarse_scan \
    --width "${PNG_WIDTH}" \
    --height "${PNG_HEIGHT}" \
    --only-monotone-heatmap \
    --write-monotone-probe

  python3 flow_folding/examples/verify_rossler_y_minima_critical_orbit_outputs.py "${TSV}" \
    --n-c 1024 \
    --n-a 1024 \
    --png-width "${PNG_WIDTH}" \
    --png-height "${PNG_HEIGHT}" \
    --png "${HEATMAP_DIR}/coarse_scan_8bit_word_heatmap.png" \
    --png "${HEATMAP_DIR}/coarse_scan_7bit_monotone_heatmap.png" \
    --html "${HEATMAP_DIR}/coarse_scan_8bit_word_heatmap_probe.html" \
    --html "${HEATMAP_DIR}/coarse_scan_7bit_monotone_heatmap_probe.html"

  echo "tglfs_upload_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  /usr/bin/time -p tglfs upload --no-interactive --json "${TSV}" | tee "${TGLFS_UPLOAD_JSON}"
  echo "tglfs_upload_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "${LOG}"
