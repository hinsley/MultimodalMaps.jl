#!/bin/zsh
set -euo pipefail

export PATH="/Users/carterhinsley/.juliaup/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="/Users/carterhinsley"
export JULIA_DEPOT_PATH="/Users/carterhinsley/.julia"
export JULIA_PROJECT="/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl"
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHONUNBUFFERED=1

cd "$JULIA_PROJECT"

RESULT_DIR="flow_folding/results/rossler_y_minima_tangent_scan_4096"
CONTOUR_DIR="$RESULT_DIR/contours"
LOG="$RESULT_DIR/run.log"
PID_FILE="$RESULT_DIR/run.pid"
PNG_WIDTH=25600
PNG_HEIGHT=17600
JULIA_BIN="/Users/carterhinsley/.julia/juliaup/julia-1.11.8+0.aarch64.apple.darwin14/bin/julia"

mkdir -p "$CONTOUR_DIR"
echo "$$" > "$PID_FILE"
exec >> "$LOG" 2>&1

echo "pipeline_pid=$$"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "scan_grid=4096x4096"
echo "render_size=${PNG_WIDTH}x${PNG_HEIGHT}"
echo "full_tsv_schema=a,c,b,status,events,word,code,period,gamma,max_time,first_time,last_time,min_y,max_y"
echo "stream_tsv=true"
echo "julia_project=$JULIA_PROJECT"
echo "julia_depot_path=$JULIA_DEPOT_PATH"

if [ -f "$RESULT_DIR/coarse_scan.tsv.gz" ]; then
  echo "coarse_scan.tsv.gz already exists; skipping scan"
else
  if [ -f "$RESULT_DIR/coarse_scan.tsv" ]; then
    echo "refusing to overwrite existing $RESULT_DIR/coarse_scan.tsv"
    exit 2
  fi

  MM_FLOW_FOLDING_RESULTS_NAME=rossler_y_minima_tangent_scan_4096 \
  MM_FLOW_FOLDING_NC=4096 \
  MM_FLOW_FOLDING_NA=4096 \
  MM_FLOW_FOLDING_C_MIN=2.0 \
  MM_FLOW_FOLDING_C_MAX=7.0 \
  MM_FLOW_FOLDING_A_MIN=0.30 \
  MM_FLOW_FOLDING_A_MAX=0.55 \
  MM_FLOW_FOLDING_B=0.3 \
  MM_FLOW_FOLDING_WORD_LENGTH=8 \
  MM_FLOW_FOLDING_TRANSIENT_EVENTS=20 \
  MM_FLOW_FOLDING_DT=0.05 \
  MM_FLOW_FOLDING_MAX_TIME=450.0 \
  MM_FLOW_FOLDING_PROGRESS_SECONDS=30.0 \
  MM_FLOW_FOLDING_GENERATE_CONTOURS=false \
  MM_FLOW_FOLDING_WRITE_DOCS_DATA=false \
  MM_FLOW_FOLDING_STREAM_TSV=true \
  MM_FLOW_FOLDING_OUTPUT="$RESULT_DIR/coarse_scan.tsv" \
  MM_FLOW_FOLDING_RUNTIME_LOG="$RESULT_DIR/coarse_scan_runtime.tsv" \
  "$JULIA_BIN" --startup-file=no --project="$JULIA_PROJECT" flow_folding/examples/rossler_y_minima_tangent_scan.jl

  echo "scan_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  gzip -f "$RESULT_DIR/coarse_scan.tsv"
  echo "gzip_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

echo "render_word_heatmap_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 flow_folding/examples/rossler_y_minima_tangent_pngs.py "$RESULT_DIR/coarse_scan.tsv.gz" \
  --output-dir "$CONTOUR_DIR" \
  --stem coarse_scan \
  --width "$PNG_WIDTH" \
  --height "$PNG_HEIGHT" \
  --only-heatmap \
  --write-heatmap-probe
echo "render_word_heatmap_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "render_monotone_heatmap_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 flow_folding/examples/rossler_y_minima_tangent_pngs.py "$RESULT_DIR/coarse_scan.tsv.gz" \
  --output-dir "$CONTOUR_DIR" \
  --stem coarse_scan \
  --width "$PNG_WIDTH" \
  --height "$PNG_HEIGHT" \
  --only-monotone-heatmap \
  --write-monotone-probe

echo "render_finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
