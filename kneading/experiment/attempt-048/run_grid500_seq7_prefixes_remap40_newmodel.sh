#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

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
export ATTEMPT048_NX="${ATTEMPT048_NX:-500}"
export ATTEMPT048_NY="${ATTEMPT048_NY:-500}"
export ATTEMPT048_DELTA_X_MIN="${ATTEMPT048_DELTA_X_MIN:--1.5}"
export ATTEMPT048_DELTA_X_MAX="${ATTEMPT048_DELTA_X_MAX:--0.5}"
export ATTEMPT048_DELTA_CA_MIN="${ATTEMPT048_DELTA_CA_MIN:--45}"
export ATTEMPT048_DELTA_CA_MAX="${ATTEMPT048_DELTA_CA_MAX:--20}"
export ATTEMPT048_MAX_SEQ_LENGTH="${ATTEMPT048_MAX_SEQ_LENGTH:-7}"
export ATTEMPT048_MAP_RESOLUTION="${ATTEMPT048_MAP_RESOLUTION:-40}"
export ATTEMPT048_OUTPUT_TAG="${ATTEMPT048_OUTPUT_TAG:-grid500_seq7_prefixes_remap40_newmodel}"
export ATTEMPT048_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT048_MAX_PREFIX_PLOT_LENGTH:-7}"
export ATTEMPT048_INSTANTIATE="${ATTEMPT048_INSTANTIATE:-1}"
export ATTEMPT048_GCS_URI="${ATTEMPT048_GCS_URI:-}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT048_OUTPUT_TAG}.log"
COLUMN_DIR="${SCRIPT_DIR}/${ATTEMPT048_OUTPUT_TAG}_columns"

gcs_enabled() {
    [[ -n "${ATTEMPT048_GCS_URI}" ]]
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "ATTEMPT048_GCS_URI is set, but gcloud is not available on PATH." | tee -a "${LOG_PATH}" >&2
        return 1
    fi
}

sync_gcs_checkpoints() {
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Syncing attempt-048 checkpoint artifacts to ${ATTEMPT048_GCS_URI}" | tee -a "${LOG_PATH}"
    if [[ -d "${COLUMN_DIR}" ]]; then
        gcloud storage rsync -r "${COLUMN_DIR}" "${ATTEMPT048_GCS_URI}/${ATTEMPT048_OUTPUT_TAG}_columns" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
    if [[ -f "${LOG_PATH}" ]]; then
        gcloud storage cp "${LOG_PATH}" "${ATTEMPT048_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
}

upload_gcs_final_artifacts() {
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Uploading attempt-048 final artifacts to ${ATTEMPT048_GCS_URI}" | tee -a "${LOG_PATH}"
    sync_gcs_checkpoints
    while IFS= read -r -d '' artifact_path; do
        gcloud storage cp "${artifact_path}" "${ATTEMPT048_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}"
    done < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f -name "${ATTEMPT048_OUTPUT_TAG}*" -print0)
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
echo "[$(date -Is)] Running attempt-048 full scan with remap resolution ${ATTEMPT048_MAP_RESOLUTION}" | tee -a "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_PATH}"
echo "Julia threads: ${JULIA_NUM_THREADS}" | tee -a "${LOG_PATH}"
echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}" | tee -a "${LOG_PATH}"
echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}" | tee -a "${LOG_PATH}"
if gcs_enabled; then
    echo "GCS artifact URI: ${ATTEMPT048_GCS_URI}" | tee -a "${LOG_PATH}"
fi

if [[ "${ATTEMPT048_INSTANTIATE}" == "1" ]]; then
    echo "Instantiating Julia project dependencies." | tee -a "${LOG_PATH}"
    julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "${LOG_PATH}"
fi

/usr/bin/time -p julia --startup-file=no --project=. kneading/experiment/attempt-048/contours.jl 2>&1 | tee -a "${LOG_PATH}"
upload_gcs_final_artifacts
