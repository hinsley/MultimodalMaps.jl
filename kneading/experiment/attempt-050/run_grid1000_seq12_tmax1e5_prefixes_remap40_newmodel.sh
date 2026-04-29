#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -d "${HOME}/.juliaup/bin" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
fi

detect_cpu_threads() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        echo 6
    fi
}

DEFAULT_THREADS="$(detect_cpu_threads)"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-${DEFAULT_THREADS}}"
DEFAULT_GC_THREADS=$((JULIA_NUM_THREADS / 4))
if (( DEFAULT_GC_THREADS < 1 )); then
    DEFAULT_GC_THREADS=1
elif (( DEFAULT_GC_THREADS > 8 )); then
    DEFAULT_GC_THREADS=8
fi
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-${DEFAULT_GC_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export ATTEMPT050_NX="${ATTEMPT050_NX:-1000}"
export ATTEMPT050_NY="${ATTEMPT050_NY:-1000}"
export ATTEMPT050_DELTA_X_MIN="${ATTEMPT050_DELTA_X_MIN:--1.5}"
export ATTEMPT050_DELTA_X_MAX="${ATTEMPT050_DELTA_X_MAX:--0.5}"
export ATTEMPT050_DELTA_CA_MIN="${ATTEMPT050_DELTA_CA_MIN:--45}"
export ATTEMPT050_DELTA_CA_MAX="${ATTEMPT050_DELTA_CA_MAX:--20}"
export ATTEMPT050_MAX_SEQ_LENGTH="${ATTEMPT050_MAX_SEQ_LENGTH:-12}"
export ATTEMPT050_MAP_RESOLUTION="${ATTEMPT050_MAP_RESOLUTION:-40}"
export ATTEMPT050_SSCS_TMAX="${ATTEMPT050_SSCS_TMAX:-1.0e5}"
export ATTEMPT050_OUTPUT_TAG="${ATTEMPT050_OUTPUT_TAG:-grid1000_seq12_tmax1e5_prefixes_remap40_newmodel}"
export ATTEMPT050_FILTER_OUTPUT_TAG="${ATTEMPT050_FILTER_OUTPUT_TAG:-grid1000_seq12_tmax1e5_prefixcompatible}"
export ATTEMPT050_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT050_MAX_PREFIX_PLOT_LENGTH:-12}"
export ATTEMPT050_CONTOUR_LINEWIDTH="${ATTEMPT050_CONTOUR_LINEWIDTH:-0.35}"
export ATTEMPT050_PLOT_WIDTH="${ATTEMPT050_PLOT_WIDTH:-1600}"
export ATTEMPT050_PLOT_HEIGHT="${ATTEMPT050_PLOT_HEIGHT:-1200}"
export ATTEMPT050_PLOT_PX_PER_UNIT="${ATTEMPT050_PLOT_PX_PER_UNIT:-2.0}"
export ATTEMPT050_AXIS_LABEL_SIZE="${ATTEMPT050_AXIS_LABEL_SIZE:-34}"
export ATTEMPT050_AXIS_TITLE_SIZE="${ATTEMPT050_AXIS_TITLE_SIZE:-40}"
export ATTEMPT050_TICK_LABEL_SIZE="${ATTEMPT050_TICK_LABEL_SIZE:-24}"
export ATTEMPT050_INSTANTIATE="${ATTEMPT050_INSTANTIATE:-1}"
export ATTEMPT050_GCS_URI="${ATTEMPT050_GCS_URI:-}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT050_OUTPUT_TAG}.log"
COLUMN_DIR="${SCRIPT_DIR}/${ATTEMPT050_OUTPUT_TAG}_columns"

gcs_enabled() {
    [[ -n "${ATTEMPT050_GCS_URI}" ]]
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "ATTEMPT050_GCS_URI is set, but gcloud is not available on PATH." | tee -a "${LOG_PATH}" >&2
        return 1
    fi
}

sync_gcs_checkpoints() {
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Syncing attempt-050 checkpoint artifacts to ${ATTEMPT050_GCS_URI}" | tee -a "${LOG_PATH}"
    if [[ -d "${COLUMN_DIR}" ]]; then
        gcloud storage rsync -r "${COLUMN_DIR}" "${ATTEMPT050_GCS_URI}/${ATTEMPT050_OUTPUT_TAG}_columns" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
    if [[ -f "${LOG_PATH}" ]]; then
        gcloud storage cp "${LOG_PATH}" "${ATTEMPT050_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
}

upload_gcs_final_artifacts() {
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Uploading attempt-050 final artifacts to ${ATTEMPT050_GCS_URI}" | tee -a "${LOG_PATH}"
    sync_gcs_checkpoints
    while IFS= read -r -d '' artifact_path; do
        gcloud storage cp "${artifact_path}" "${ATTEMPT050_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}"
    done < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f \( -name "${ATTEMPT050_OUTPUT_TAG}*" -o -name "${ATTEMPT050_FILTER_OUTPUT_TAG}*" \) -print0)
}

on_exit() {
    status=$?
    if (( status != 0 )); then
        echo "[$(date -Is)] Runner exiting with status ${status}; syncing resumable checkpoints if configured." | tee -a "${LOG_PATH}"
        sync_gcs_checkpoints
    fi
}

on_signal() {
    {
        echo
        echo "[$(date -Is)] Received shutdown signal. Completed column files are resumable; incomplete columns will be recomputed."
        sync || true
    } | tee -a "${LOG_PATH}"
    sync_gcs_checkpoints
}
trap on_exit EXIT
trap on_signal TERM INT HUP

cd "${REPO_ROOT}"

echo | tee -a "${LOG_PATH}"
echo "[$(date -Is)] Running attempt-050 full scan with remap resolution ${ATTEMPT050_MAP_RESOLUTION}" | tee -a "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}" | tee -a "${LOG_PATH}"
echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "SSCS integration tmax: ${ATTEMPT050_SSCS_TMAX}" | tee -a "${LOG_PATH}"
echo "Plot size: ${ATTEMPT050_PLOT_WIDTH}x${ATTEMPT050_PLOT_HEIGHT} at px_per_unit=${ATTEMPT050_PLOT_PX_PER_UNIT}; linewidth=${ATTEMPT050_CONTOUR_LINEWIDTH}" | tee -a "${LOG_PATH}"
if gcs_enabled; then
    echo "GCS artifact URI: ${ATTEMPT050_GCS_URI}" | tee -a "${LOG_PATH}"
fi

if [[ "${ATTEMPT050_INSTANTIATE}" == "1" ]]; then
    echo "Instantiating Julia project dependencies." | tee -a "${LOG_PATH}"
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "${LOG_PATH}"
fi

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -p julia --startup-file=no --project=. kneading/experiment/attempt-050/contours.jl 2>&1 | tee -a "${LOG_PATH}"
else
    echo "/usr/bin/time is not available; using shell timing fallback." | tee -a "${LOG_PATH}"
    time julia --startup-file=no --project=. kneading/experiment/attempt-050/contours.jl 2>&1 | tee -a "${LOG_PATH}"
fi

export ATTEMPT050_FILTER_RESULTS="${SCRIPT_DIR}/${ATTEMPT050_OUTPUT_TAG}_results.tsv"
export ATTEMPT050_FILTER_OUTPUT_DIR="${SCRIPT_DIR}"
echo "[$(date -Is)] Generating filtered prefix-compatible contour figure." | tee -a "${LOG_PATH}"
julia --startup-file=no --project=. kneading/experiment/attempt-050/plot_filtered_full_contours.jl 2>&1 | tee -a "${LOG_PATH}"

upload_gcs_final_artifacts
