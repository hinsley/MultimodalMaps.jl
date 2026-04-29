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
export ATTEMPT049_NX="${ATTEMPT049_NX:-1000}"
export ATTEMPT049_NY="${ATTEMPT049_NY:-1000}"
export ATTEMPT049_DELTA_X_MIN="${ATTEMPT049_DELTA_X_MIN:--1.5}"
export ATTEMPT049_DELTA_X_MAX="${ATTEMPT049_DELTA_X_MAX:--0.5}"
export ATTEMPT049_DELTA_CA_MIN="${ATTEMPT049_DELTA_CA_MIN:--45}"
export ATTEMPT049_DELTA_CA_MAX="${ATTEMPT049_DELTA_CA_MAX:--20}"
export ATTEMPT049_MAX_SEQ_LENGTH="${ATTEMPT049_MAX_SEQ_LENGTH:-12}"
export ATTEMPT049_MAP_RESOLUTION="${ATTEMPT049_MAP_RESOLUTION:-40}"
export ATTEMPT049_OUTPUT_TAG="${ATTEMPT049_OUTPUT_TAG:-grid1000_seq12_prefixes_remap40_newmodel}"
export ATTEMPT049_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT049_MAX_PREFIX_PLOT_LENGTH:-12}"
export ATTEMPT049_CONTOUR_LINEWIDTH="${ATTEMPT049_CONTOUR_LINEWIDTH:-0.35}"
export ATTEMPT049_PLOT_WIDTH="${ATTEMPT049_PLOT_WIDTH:-1600}"
export ATTEMPT049_PLOT_HEIGHT="${ATTEMPT049_PLOT_HEIGHT:-1200}"
export ATTEMPT049_PLOT_PX_PER_UNIT="${ATTEMPT049_PLOT_PX_PER_UNIT:-2.0}"
export ATTEMPT049_AXIS_LABEL_SIZE="${ATTEMPT049_AXIS_LABEL_SIZE:-34}"
export ATTEMPT049_AXIS_TITLE_SIZE="${ATTEMPT049_AXIS_TITLE_SIZE:-40}"
export ATTEMPT049_TICK_LABEL_SIZE="${ATTEMPT049_TICK_LABEL_SIZE:-24}"
export ATTEMPT049_INSTANTIATE="${ATTEMPT049_INSTANTIATE:-1}"
export ATTEMPT049_GCS_URI="${ATTEMPT049_GCS_URI:-}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT049_OUTPUT_TAG}.log"
COLUMN_DIR="${SCRIPT_DIR}/${ATTEMPT049_OUTPUT_TAG}_columns"

gcs_enabled() {
    [[ -n "${ATTEMPT049_GCS_URI}" ]]
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "ATTEMPT049_GCS_URI is set, but gcloud is not available on PATH." | tee -a "${LOG_PATH}" >&2
        return 1
    fi
}

sync_gcs_checkpoints() {
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Syncing attempt-049 checkpoint artifacts to ${ATTEMPT049_GCS_URI}" | tee -a "${LOG_PATH}"
    if [[ -d "${COLUMN_DIR}" ]]; then
        gcloud storage rsync -r "${COLUMN_DIR}" "${ATTEMPT049_GCS_URI}/${ATTEMPT049_OUTPUT_TAG}_columns" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
    if [[ -f "${LOG_PATH}" ]]; then
        gcloud storage cp "${LOG_PATH}" "${ATTEMPT049_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
}

upload_gcs_final_artifacts() {
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Uploading attempt-049 final artifacts to ${ATTEMPT049_GCS_URI}" | tee -a "${LOG_PATH}"
    sync_gcs_checkpoints
    while IFS= read -r -d '' artifact_path; do
        gcloud storage cp "${artifact_path}" "${ATTEMPT049_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}"
    done < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f -name "${ATTEMPT049_OUTPUT_TAG}*" -print0)
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
echo "[$(date -Is)] Running attempt-049 full scan with remap resolution ${ATTEMPT049_MAP_RESOLUTION}" | tee -a "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}" | tee -a "${LOG_PATH}"
echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Plot size: ${ATTEMPT049_PLOT_WIDTH}x${ATTEMPT049_PLOT_HEIGHT} at px_per_unit=${ATTEMPT049_PLOT_PX_PER_UNIT}; linewidth=${ATTEMPT049_CONTOUR_LINEWIDTH}" | tee -a "${LOG_PATH}"
if gcs_enabled; then
    echo "GCS artifact URI: ${ATTEMPT049_GCS_URI}" | tee -a "${LOG_PATH}"
fi

if [[ "${ATTEMPT049_INSTANTIATE}" == "1" ]]; then
    echo "Instantiating Julia project dependencies." | tee -a "${LOG_PATH}"
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "${LOG_PATH}"
fi

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -p julia --startup-file=no --project=. kneading/experiment/attempt-049/contours.jl 2>&1 | tee -a "${LOG_PATH}"
else
    echo "/usr/bin/time is not available; using shell timing fallback." | tee -a "${LOG_PATH}"
    time julia --startup-file=no --project=. kneading/experiment/attempt-049/contours.jl 2>&1 | tee -a "${LOG_PATH}"
fi
upload_gcs_final_artifacts
