#!/bin/zsh
set -euo pipefail

export PATH="/Users/carterhinsley/.juliaup/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="/Users/carterhinsley"
export JULIA_DEPOT_PATH="/Users/carterhinsley/.julia"
export JULIA_PROJECT="/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl"
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHONUNBUFFERED=1

cd "$JULIA_PROJECT"

RESULTS_NAME="${MM_FLOW_FOLDING_RESULTS_NAME:-rossler_y_minima_tangent_scan_4096}"
RESULT_DIR="${MM_FLOW_FOLDING_RESULT_DIR:-flow_folding/results/$RESULTS_NAME}"
CONTOUR_DIR="$RESULT_DIR/contours"
CHUNK_DIR="$RESULT_DIR/chunks"
LOG="$RESULT_DIR/run.log"
PID_FILE="$RESULT_DIR/run.pid"
NC="${MM_FLOW_FOLDING_NC:-4096}"
NA="${MM_FLOW_FOLDING_NA:-4096}"
C_MIN="${MM_FLOW_FOLDING_C_MIN:-2.0}"
C_MAX="${MM_FLOW_FOLDING_C_MAX:-7.0}"
A_MIN="${MM_FLOW_FOLDING_A_MIN:-0.30}"
A_MAX="${MM_FLOW_FOLDING_A_MAX:-0.55}"
B_VALUE="${MM_FLOW_FOLDING_B:-0.3}"
WORD_LENGTH="${MM_FLOW_FOLDING_WORD_LENGTH:-8}"
TRANSIENT_EVENTS="${MM_FLOW_FOLDING_TRANSIENT_EVENTS:-20}"
DT="${MM_FLOW_FOLDING_DT:-0.05}"
MAX_TIME="${MM_FLOW_FOLDING_MAX_TIME:-450.0}"
PROGRESS_SECONDS="${MM_FLOW_FOLDING_PROGRESS_SECONDS:-30.0}"
WORKERS="${MM_FLOW_FOLDING_WORKERS:-8}"
CHUNKS="${MM_FLOW_FOLDING_CHUNKS:-32}"
PNG_WIDTH="${MM_FLOW_FOLDING_PNG_WIDTH:-25600}"
PNG_HEIGHT="${MM_FLOW_FOLDING_PNG_HEIGHT:-17600}"
JULIA_BIN="/Users/carterhinsley/.julia/juliaup/julia-1.11.8+0.aarch64.apple.darwin14/bin/julia"

if [ "$CHUNKS" -gt "$NA" ]; then
  CHUNKS="$NA"
fi
if [ "$WORKERS" -gt "$CHUNKS" ]; then
  WORKERS="$CHUNKS"
fi

mkdir -p "$CONTOUR_DIR" "$CHUNK_DIR"
echo "$$" > "$PID_FILE"
exec >> "$LOG" 2>&1

echo "pipeline_pid=$$"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "scan_grid=${NC}x${NA}"
echo "render_size=${PNG_WIDTH}x${PNG_HEIGHT}"
echo "full_tsv_schema=a,c,b,status,events,word,code,period,gamma,max_time,first_time,last_time,min_y,max_y"
echo "stream_tsv=true"
echo "workers=$WORKERS"
echo "chunks=$CHUNKS"
echo "chunk_dir=$CHUNK_DIR"
echo "julia_project=$JULIA_PROJECT"
echo "julia_depot_path=$JULIA_DEPOT_PATH"

chunk_bounds() {
  local chunk="$1"
  local start_idx end_idx block_na
  start_idx=$(( chunk * NA / CHUNKS + 1 ))
  end_idx=$(( (chunk + 1) * NA / CHUNKS ))
  block_na=$(( end_idx - start_idx + 1 ))
  awk -v amin="$A_MIN" -v amax="$A_MAX" -v na="$NA" -v s="$start_idx" -v e="$end_idx" -v block="$block_na" 'BEGIN {
    if (na <= 1) {
      start_a = amin
      end_a = amax
    } else {
      step = (amax - amin) / (na - 1)
      start_a = amin + (s - 1) * step
      end_a = amin + (e - 1) * step
    }
    printf "%d\t%d\t%d\t%.17g\t%.17g\n", s, e, block, start_a, end_a
  }'
}

chunk_is_complete() {
  local chunk_tsv="$1"
  local expected_lines="$2"
  [ -f "$chunk_tsv" ] || return 1
  local actual_lines
  actual_lines=$(wc -l < "$chunk_tsv" | tr -d ' ')
  [ "$actual_lines" -eq "$expected_lines" ]
}

run_chunk() {
  local chunk="$1"
  local start_idx="$2"
  local end_idx="$3"
  local block_na="$4"
  local start_a="$5"
  local end_a="$6"
  local chunk_name
  chunk_name=$(printf "chunk_%03d" "$chunk")
  local chunk_tsv="$CHUNK_DIR/${chunk_name}.tsv"
  local chunk_runtime="$CHUNK_DIR/${chunk_name}_runtime.tsv"
  local chunk_log="$CHUNK_DIR/${chunk_name}.log"
  local expected_lines=$(( block_na * NC + 1 ))

  if chunk_is_complete "$chunk_tsv" "$expected_lines"; then
    echo "$chunk_name already complete rows=$(( expected_lines - 1 )); skipping"
    return 0
  fi
  if [ -f "$chunk_tsv" ]; then
    echo "refusing to overwrite incomplete $chunk_tsv"
    return 2
  fi

  (
    exec >> "$chunk_log" 2>&1
    echo "chunk=$chunk_name start_idx=$start_idx end_idx=$end_idx n_a=$block_na a_min=$start_a a_max=$end_a started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    MM_FLOW_FOLDING_RESULTS_NAME="$RESULTS_NAME" \
    MM_FLOW_FOLDING_NC="$NC" \
    MM_FLOW_FOLDING_NA="$block_na" \
    MM_FLOW_FOLDING_C_MIN="$C_MIN" \
    MM_FLOW_FOLDING_C_MAX="$C_MAX" \
    MM_FLOW_FOLDING_A_MIN="$start_a" \
    MM_FLOW_FOLDING_A_MAX="$end_a" \
    MM_FLOW_FOLDING_B="$B_VALUE" \
    MM_FLOW_FOLDING_WORD_LENGTH="$WORD_LENGTH" \
    MM_FLOW_FOLDING_TRANSIENT_EVENTS="$TRANSIENT_EVENTS" \
    MM_FLOW_FOLDING_DT="$DT" \
    MM_FLOW_FOLDING_MAX_TIME="$MAX_TIME" \
    MM_FLOW_FOLDING_PROGRESS_SECONDS="$PROGRESS_SECONDS" \
    MM_FLOW_FOLDING_GENERATE_CONTOURS=false \
    MM_FLOW_FOLDING_WRITE_DOCS_DATA=false \
    MM_FLOW_FOLDING_STREAM_TSV=true \
    MM_FLOW_FOLDING_OUTPUT="$chunk_tsv" \
    MM_FLOW_FOLDING_RUNTIME_LOG="$chunk_runtime" \
    "$JULIA_BIN" --startup-file=no --project="$JULIA_PROJECT" flow_folding/examples/rossler_y_minima_tangent_scan.jl
    echo "chunk=$chunk_name finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  )
}

merge_chunks() {
  local output="$RESULT_DIR/coarse_scan.tsv"
  local tmp_output="$RESULT_DIR/coarse_scan.tsv.tmp"
  rm -f "$tmp_output"
  for chunk in $(seq 0 $(( CHUNKS - 1 ))); do
    local chunk_name chunk_tsv
    chunk_name=$(printf "chunk_%03d" "$chunk")
    chunk_tsv="$CHUNK_DIR/${chunk_name}.tsv"
    if [ "$chunk" -eq 0 ]; then
      cat "$chunk_tsv" > "$tmp_output"
    else
      tail -n +2 "$chunk_tsv" >> "$tmp_output"
    fi
  done
  mv "$tmp_output" "$output"
}

if [ -f "$RESULT_DIR/coarse_scan.tsv.gz" ]; then
  echo "coarse_scan.tsv.gz already exists; skipping scan"
else
  if [ -f "$RESULT_DIR/coarse_scan.tsv" ]; then
    echo "refusing to overwrite existing $RESULT_DIR/coarse_scan.tsv"
    exit 2
  fi

  failed_marker="$CHUNK_DIR/.failed"
  rm -f "$failed_marker"
  for chunk in $(seq 0 $(( CHUNKS - 1 ))); do
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$WORKERS" ]; do
      sleep 5
    done
    IFS=$'\t' read -r start_idx end_idx block_na start_a end_a <<< "$(chunk_bounds "$chunk")"
    echo "launching chunk=$(printf "%03d" "$chunk") start_idx=$start_idx end_idx=$end_idx n_a=$block_na a_min=$start_a a_max=$end_a"
    (
      if ! run_chunk "$chunk" "$start_idx" "$end_idx" "$block_na" "$start_a" "$end_a"; then
        echo "chunk=$(printf "%03d" "$chunk") failed"
        touch "$failed_marker"
      fi
    ) &
  done

  wait
  if [ -f "$failed_marker" ]; then
    echo "one or more chunks failed"
    exit 1
  fi

  echo "merge_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  merge_chunks
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
