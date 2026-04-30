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

export ATTEMPT051_NX="${ATTEMPT051_NX:-1000}"
export ATTEMPT051_NY="${ATTEMPT051_NY:-1000}"
export ATTEMPT051_DELTA_X_MIN="${ATTEMPT051_DELTA_X_MIN:--2.0}"
export ATTEMPT051_DELTA_X_MAX="${ATTEMPT051_DELTA_X_MAX:-2.0}"
export ATTEMPT051_DELTA_CA_MIN="${ATTEMPT051_DELTA_CA_MIN:--60.0}"
export ATTEMPT051_DELTA_CA_MAX="${ATTEMPT051_DELTA_CA_MAX:-20.0}"
export ATTEMPT051_MAX_SEQ_LENGTH="${ATTEMPT051_MAX_SEQ_LENGTH:-12}"
export ATTEMPT051_MAP_RESOLUTION="${ATTEMPT051_MAP_RESOLUTION:-40}"
export ATTEMPT051_SSCS_TMAX="${ATTEMPT051_SSCS_TMAX:-1.0e5}"
export ATTEMPT051_TAU_Y="${ATTEMPT051_TAU_Y:-2.0e4}"
export ATTEMPT051_MAX_PREFIX_PLOT_LENGTH="${ATTEMPT051_MAX_PREFIX_PLOT_LENGTH:-12}"
export ATTEMPT051_CONTOUR_LINEWIDTH="${ATTEMPT051_CONTOUR_LINEWIDTH:-0.35}"
export ATTEMPT051_FILTER_LINEWIDTH="${ATTEMPT051_FILTER_LINEWIDTH:-${ATTEMPT051_CONTOUR_LINEWIDTH}}"
export ATTEMPT051_PLOT_WIDTH="${ATTEMPT051_PLOT_WIDTH:-1600}"
export ATTEMPT051_PLOT_HEIGHT="${ATTEMPT051_PLOT_HEIGHT:-1200}"
export ATTEMPT051_PLOT_PX_PER_UNIT="${ATTEMPT051_PLOT_PX_PER_UNIT:-2.0}"
export ATTEMPT051_AXIS_LABEL_SIZE="${ATTEMPT051_AXIS_LABEL_SIZE:-34}"
export ATTEMPT051_AXIS_TITLE_SIZE="${ATTEMPT051_AXIS_TITLE_SIZE:-40}"
export ATTEMPT051_TICK_LABEL_SIZE="${ATTEMPT051_TICK_LABEL_SIZE:-24}"
export ATTEMPT051_INSTANTIATE="${ATTEMPT051_INSTANTIATE:-1}"
export ATTEMPT051_GCS_URI="${ATTEMPT051_GCS_URI:-}"

GH_VALUES=("0.0" "1.0e-3" "1.0e-2")
GH_LABELS=("gh0p000" "gh0p001" "gh0p01")
CURRENT_LOG_PATH=""
CURRENT_COLUMN_DIR=""
CURRENT_OUTPUT_TAG=""

gcs_enabled() {
    [[ -n "${ATTEMPT051_GCS_URI}" ]]
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "ATTEMPT051_GCS_URI is set, but gcloud is not available on PATH." | tee -a "${CURRENT_LOG_PATH}" >&2
        return 1
    fi
}

sync_gcs_checkpoints() {
    gcs_enabled || return 0
    [[ -n "${CURRENT_LOG_PATH}" ]] || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Syncing attempt-051 checkpoint artifacts to ${ATTEMPT051_GCS_URI}" | tee -a "${CURRENT_LOG_PATH}"
    if [[ -d "${CURRENT_COLUMN_DIR}" ]]; then
        gcloud storage rsync -r "${CURRENT_COLUMN_DIR}" "${ATTEMPT051_GCS_URI}/${CURRENT_OUTPUT_TAG}_columns" 2>&1 | tee -a "${CURRENT_LOG_PATH}" || true
    fi
    if [[ -f "${CURRENT_LOG_PATH}" ]]; then
        gcloud storage cp "${CURRENT_LOG_PATH}" "${ATTEMPT051_GCS_URI}/" 2>&1 | tee -a "${CURRENT_LOG_PATH}" || true
    fi
}

upload_gcs_final_artifacts() {
    local output_tag="$1"
    local filter_tag="$2"

    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(date -Is)] Uploading attempt-051 final artifacts for ${output_tag} to ${ATTEMPT051_GCS_URI}" | tee -a "${CURRENT_LOG_PATH}"
    sync_gcs_checkpoints
    while IFS= read -r -d '' artifact_path; do
        gcloud storage cp "${artifact_path}" "${ATTEMPT051_GCS_URI}/" 2>&1 | tee -a "${CURRENT_LOG_PATH}"
    done < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f \( -name "${output_tag}*" -o -name "${filter_tag}*" \) -print0)
}

on_exit() {
    local status=$?
    if (( status != 0 )); then
        if [[ -n "${CURRENT_LOG_PATH}" ]]; then
            echo "[$(date -Is)] Runner exiting with status ${status}; syncing resumable checkpoints if configured." | tee -a "${CURRENT_LOG_PATH}"
        fi
        sync_gcs_checkpoints
    fi
}

on_signal() {
    {
        echo
        echo "[$(date -Is)] Received shutdown signal. Completed column files are resumable; incomplete columns will be recomputed."
        sync || true
    } | tee -a "${CURRENT_LOG_PATH}"
    sync_gcs_checkpoints
}
trap on_exit EXIT
trap on_signal TERM INT HUP

cd "${REPO_ROOT}"

if [[ "${ATTEMPT051_INSTANTIATE}" == "1" ]]; then
    echo "[$(date -Is)] Instantiating Julia project dependencies."
    "${JULIA_CMD[@]}" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
fi

for idx in "${!GH_VALUES[@]}"; do
    gh_value="${GH_VALUES[$idx]}"
    gh_label="${GH_LABELS[$idx]}"

    export ATTEMPT051_G_H="${gh_value}"
    export ATTEMPT051_OUTPUT_TAG="grid1000_seq12_tmax1e5_${gh_label}_prefixes_remap40_newmodel"
    export ATTEMPT051_FILTER_OUTPUT_TAG="grid1000_seq12_tmax1e5_${gh_label}_prefixcompatible_tzero2to12"
    export ATTEMPT051_FILTER_RESULTS="${SCRIPT_DIR}/${ATTEMPT051_OUTPUT_TAG}_results.tsv"
    export ATTEMPT051_FILTER_OUTPUT_DIR="${SCRIPT_DIR}"

    CURRENT_OUTPUT_TAG="${ATTEMPT051_OUTPUT_TAG}"
    CURRENT_LOG_PATH="${SCRIPT_DIR}/${ATTEMPT051_OUTPUT_TAG}.log"
    CURRENT_COLUMN_DIR="${SCRIPT_DIR}/${ATTEMPT051_OUTPUT_TAG}_columns"

    {
        echo
        echo "[$(date -Is)] Running attempt-051 full scan for ${gh_label} with remap resolution ${ATTEMPT051_MAP_RESOLUTION}"
        echo "Repo root: ${REPO_ROOT}"
        echo "Julia threads: ${JULIA_NUM_THREADS}"
        echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}"
        echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}"
        echo "Grid: ${ATTEMPT051_NY} Delta Ca x ${ATTEMPT051_NX} Delta x"
        echo "Delta Ca range: [${ATTEMPT051_DELTA_CA_MIN}, ${ATTEMPT051_DELTA_CA_MAX}]"
        echo "Delta x range: [${ATTEMPT051_DELTA_X_MIN}, ${ATTEMPT051_DELTA_X_MAX}]"
        echo "g_h: ${ATTEMPT051_G_H}"
        echo "tau_y: ${ATTEMPT051_TAU_Y}"
        echo "SSCS integration tmax: ${ATTEMPT051_SSCS_TMAX}"
        echo "Plot size: ${ATTEMPT051_PLOT_WIDTH}x${ATTEMPT051_PLOT_HEIGHT} at px_per_unit=${ATTEMPT051_PLOT_PX_PER_UNIT}; linewidth=${ATTEMPT051_CONTOUR_LINEWIDTH}"
        if gcs_enabled; then
            echo "GCS artifact URI: ${ATTEMPT051_GCS_URI}"
        fi
    } | tee -a "${CURRENT_LOG_PATH}"

    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-051/contours.jl 2>&1 | tee -a "${CURRENT_LOG_PATH}"
    else
        echo "/usr/bin/time is not available; using shell timing fallback." | tee -a "${CURRENT_LOG_PATH}"
        time "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-051/contours.jl 2>&1 | tee -a "${CURRENT_LOG_PATH}"
    fi

    echo "[$(date -Is)] Generating filtered prefix-compatible contour figure for ${gh_label}." | tee -a "${CURRENT_LOG_PATH}"
    "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-051/plot_filtered_full_contours.jl 2>&1 | tee -a "${CURRENT_LOG_PATH}"

    upload_gcs_final_artifacts "${ATTEMPT051_OUTPUT_TAG}" "${ATTEMPT051_FILTER_OUTPUT_TAG}"
done

echo "[$(date -Is)] attempt-051 g_h sweep complete."
