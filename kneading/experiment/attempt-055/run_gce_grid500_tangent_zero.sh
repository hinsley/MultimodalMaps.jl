#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -d "${HOME}/.juliaup/bin" ]]; then
    export PATH="${HOME}/.juliaup/bin:${PATH}"
fi

JULIA_CMD=(julia)
if julia +release --version >/dev/null 2>&1; then
    JULIA_CMD=(julia +release)
fi

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-48}"
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export ATTEMPT054_NX="${ATTEMPT055_NX:-500}"
export ATTEMPT054_NY="${ATTEMPT055_NY:-500}"
export ATTEMPT054_MAX_ITER="${ATTEMPT055_MAX_ITER:-8}"
export ATTEMPT054_TMAX="${ATTEMPT055_TMAX:-1.0e5}"
export ATTEMPT054_OUTPUT_TAG="${ATTEMPT055_OUTPUT_TAG:-grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered}"
export ATTEMPT054_DELTA_X_MIN="${ATTEMPT055_DELTA_X_MIN:--1.5}"
export ATTEMPT054_DELTA_X_MAX="${ATTEMPT055_DELTA_X_MAX:--0.5}"
export ATTEMPT054_DELTA_CA_MIN="${ATTEMPT055_DELTA_CA_MIN:--45.0}"
export ATTEMPT054_DELTA_CA_MAX="${ATTEMPT055_DELTA_CA_MAX:--20.0}"
export ATTEMPT054_DELTA_X_TICK_STEP="${ATTEMPT055_DELTA_X_TICK_STEP:-0.1}"
export ATTEMPT054_DELTA_CA_TICK_STEP="${ATTEMPT055_DELTA_CA_TICK_STEP:-5.0}"
export ATTEMPT054_MAP_RESOLUTION="${ATTEMPT055_MAP_RESOLUTION:-40}"
export ATTEMPT054_CA_MIN_V_MAX="${ATTEMPT055_CA_MIN_V_MAX:-0.0}"
export ATTEMPT054_PLOT_WIDTH="${ATTEMPT055_PLOT_WIDTH:-2000}"
export ATTEMPT054_PLOT_HEIGHT="${ATTEMPT055_PLOT_HEIGHT:-1500}"
export ATTEMPT054_PLOT_PX_PER_UNIT="${ATTEMPT055_PLOT_PX_PER_UNIT:-2.0}"

export ATTEMPT055_GCS_URI="${ATTEMPT055_GCS_URI:-}"
export ATTEMPT055_INSTANTIATE="${ATTEMPT055_INSTANTIATE:-0}"

LOG_PATH="${SCRIPT_DIR}/${ATTEMPT054_OUTPUT_TAG}.log"
COLUMN_DIR="${SCRIPT_DIR}/${ATTEMPT054_OUTPUT_TAG}_columns"

gcs_enabled() {
    [[ -n "${ATTEMPT055_GCS_URI}" ]]
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "ATTEMPT055_GCS_URI is set, but gcloud is not available on PATH." | tee -a "${LOG_PATH}" >&2
        return 1
    fi
}

sync_gcs_checkpoints() {
    gcs_enabled || return 0
    require_gcloud || return 0
    echo "[$(date -Is)] Syncing attempt-055 checkpoint artifacts to ${ATTEMPT055_GCS_URI}" | tee -a "${LOG_PATH}"
    if [[ -d "${COLUMN_DIR}" ]]; then
        gcloud storage rsync -r "${COLUMN_DIR}" "${ATTEMPT055_GCS_URI}/${ATTEMPT054_OUTPUT_TAG}_columns" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
    if [[ -f "${LOG_PATH}" ]]; then
        gcloud storage cp "${LOG_PATH}" "${ATTEMPT055_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}" || true
    fi
}

upload_gcs_final_artifacts() {
    gcs_enabled || return 0
    require_gcloud || return 0
    echo "[$(date -Is)] Uploading attempt-055 final artifacts for ${ATTEMPT054_OUTPUT_TAG} to ${ATTEMPT055_GCS_URI}" | tee -a "${LOG_PATH}"
    sync_gcs_checkpoints
    while IFS= read -r -d '' artifact_path; do
        gcloud storage cp "${artifact_path}" "${ATTEMPT055_GCS_URI}/" 2>&1 | tee -a "${LOG_PATH}"
    done < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f -name "${ATTEMPT054_OUTPUT_TAG}*" -print0)
}

on_exit() {
    local status=$?
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

{
    echo "[$(date -Is)] Running attempt-055 tangent zero-contour scan."
    echo "Repo root: ${REPO_ROOT}"
    echo "Julia command: ${JULIA_CMD[*]}"
    echo "Julia threads: ${JULIA_NUM_THREADS}"
    echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}"
    echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}"
    echo "Grid: ${ATTEMPT054_NY} Delta Ca x ${ATTEMPT054_NX} Delta x"
    echo "Delta Ca range: [${ATTEMPT054_DELTA_CA_MIN}, ${ATTEMPT054_DELTA_CA_MAX}]"
    echo "Delta x range: [${ATTEMPT054_DELTA_X_MIN}, ${ATTEMPT054_DELTA_X_MAX}]"
    echo "Max iterates: ${ATTEMPT054_MAX_ITER}"
    echo "Tangent integration tmax: ${ATTEMPT054_TMAX}"
    echo "Ca-min filter: V <= ${ATTEMPT054_CA_MIN_V_MAX} and x <= saddle-focus x_eq"
    echo "Output tag: ${ATTEMPT054_OUTPUT_TAG}"
    if gcs_enabled; then
        echo "GCS artifact URI: ${ATTEMPT055_GCS_URI}"
    fi
} | tee -a "${LOG_PATH}"

if [[ "${ATTEMPT055_INSTANTIATE}" == "1" ]]; then
    echo "[$(date -Is)] Instantiating Julia project dependencies." | tee -a "${LOG_PATH}"
    "${JULIA_CMD[@]}" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()' 2>&1 | tee -a "${LOG_PATH}"
fi

if [[ -x /usr/bin/time ]]; then
    /usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-055/main.jl 2>&1 | tee -a "${LOG_PATH}"
else
    time "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-055/main.jl 2>&1 | tee -a "${LOG_PATH}"
fi

upload_gcs_final_artifacts
echo "[$(date -Is)] attempt-055 complete." | tee -a "${LOG_PATH}"
