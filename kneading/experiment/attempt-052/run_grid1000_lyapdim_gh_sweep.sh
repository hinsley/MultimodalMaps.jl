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

export ATTEMPT052_NX="${ATTEMPT052_NX:-1000}"
export ATTEMPT052_NY="${ATTEMPT052_NY:-1000}"
export ATTEMPT052_DELTA_X_MIN="${ATTEMPT052_DELTA_X_MIN:--1.5}"
export ATTEMPT052_DELTA_X_MAX="${ATTEMPT052_DELTA_X_MAX:--0.5}"
export ATTEMPT052_DELTA_CA_MIN="${ATTEMPT052_DELTA_CA_MIN:--45.0}"
export ATTEMPT052_DELTA_CA_MAX="${ATTEMPT052_DELTA_CA_MAX:--20.0}"
export ATTEMPT052_TAU_Y="${ATTEMPT052_TAU_Y:-2.0e4}"
export ATTEMPT052_LYAP_TMAX="${ATTEMPT052_LYAP_TMAX:-1.0e5}"
export ATTEMPT052_LYAP_MIN_TIME="${ATTEMPT052_LYAP_MIN_TIME:-3.0e4}"
export ATTEMPT052_LYAP_CHECK_INTERVAL="${ATTEMPT052_LYAP_CHECK_INTERVAL:-5.0e3}"
export ATTEMPT052_PLOT_WIDTH="${ATTEMPT052_PLOT_WIDTH:-1800}"
export ATTEMPT052_PLOT_HEIGHT="${ATTEMPT052_PLOT_HEIGHT:-1300}"
export ATTEMPT052_PLOT_PX_PER_UNIT="${ATTEMPT052_PLOT_PX_PER_UNIT:-2.0}"
export ATTEMPT052_GCS_URI="${ATTEMPT052_GCS_URI:-}"
if [[ -z "${ATTEMPT052_TAG_GRID_LABEL:-}" ]]; then
    if [[ "${ATTEMPT052_NX}" == "${ATTEMPT052_NY}" ]]; then
        export ATTEMPT052_TAG_GRID_LABEL="grid${ATTEMPT052_NX}"
    else
        export ATTEMPT052_TAG_GRID_LABEL="grid${ATTEMPT052_NY}x${ATTEMPT052_NX}"
    fi
fi

GH_VALUES=("0.0" "1.0e-3" "1.0e-2")
GH_LABELS=("gh0p000" "gh0p001" "gh0p01")
CURRENT_LOG_PATH=""
CURRENT_COLUMN_DIR=""
CURRENT_OUTPUT_TAG=""

timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

gcs_enabled() {
    [[ -n "${ATTEMPT052_GCS_URI}" ]]
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        echo "ATTEMPT052_GCS_URI is set, but gcloud is not available on PATH." | tee -a "${CURRENT_LOG_PATH}" >&2
        return 1
    fi
}

sync_gcs_checkpoints() {
    gcs_enabled || return 0
    [[ -n "${CURRENT_LOG_PATH}" ]] || return 0
    require_gcloud || return 0

    echo "[$(timestamp)] Syncing attempt-052 checkpoint artifacts to ${ATTEMPT052_GCS_URI}" | tee -a "${CURRENT_LOG_PATH}"
    if [[ -d "${CURRENT_COLUMN_DIR}" ]]; then
        gcloud storage rsync -r "${CURRENT_COLUMN_DIR}" "${ATTEMPT052_GCS_URI}/${CURRENT_OUTPUT_TAG}_columns" 2>&1 | tee -a "${CURRENT_LOG_PATH}" || true
    fi
    if [[ -f "${CURRENT_LOG_PATH}" ]]; then
        gcloud storage cp "${CURRENT_LOG_PATH}" "${ATTEMPT052_GCS_URI}/" 2>&1 | tee -a "${CURRENT_LOG_PATH}" || true
    fi
}

upload_gcs_final_artifacts() {
    local output_tag="$1"
    gcs_enabled || return 0
    require_gcloud || return 0

    echo "[$(timestamp)] Uploading attempt-052 final artifacts for ${output_tag} to ${ATTEMPT052_GCS_URI}" | tee -a "${CURRENT_LOG_PATH}"
    sync_gcs_checkpoints
    while IFS= read -r -d '' artifact_path; do
        gcloud storage cp "${artifact_path}" "${ATTEMPT052_GCS_URI}/" 2>&1 | tee -a "${CURRENT_LOG_PATH}"
    done < <(find "${SCRIPT_DIR}" -maxdepth 1 -type f -name "${output_tag}*" -print0)
}

on_exit() {
    local status=$?
    if (( status != 0 )); then
        if [[ -n "${CURRENT_LOG_PATH}" ]]; then
            echo "[$(timestamp)] Runner exiting with status ${status}; syncing resumable checkpoints if configured." | tee -a "${CURRENT_LOG_PATH}"
        fi
        sync_gcs_checkpoints
    fi
}

on_signal() {
    {
        echo
        echo "[$(timestamp)] Received shutdown signal. Completed column files are resumable; incomplete columns will be recomputed."
        sync || true
    } | tee -a "${CURRENT_LOG_PATH}"
    sync_gcs_checkpoints
}
trap on_exit EXIT
trap on_signal TERM INT HUP

cd "${REPO_ROOT}"

if julia +release --version >/dev/null 2>&1; then
    JULIA_CMD=(julia +release)
else
    JULIA_CMD=(julia)
fi

echo "[$(timestamp)] Instantiating Julia project dependencies."
"${JULIA_CMD[@]}" --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'

for idx in "${!GH_VALUES[@]}"; do
    gh_value="${GH_VALUES[$idx]}"
    gh_label="${GH_LABELS[$idx]}"

    export ATTEMPT052_G_H="${gh_value}"
    export ATTEMPT052_OUTPUT_TAG="${ATTEMPT052_TAG_GRID_LABEL}_lyapdim_tmax1e5_${gh_label}"
    CURRENT_OUTPUT_TAG="${ATTEMPT052_OUTPUT_TAG}"
    CURRENT_LOG_PATH="${SCRIPT_DIR}/${ATTEMPT052_OUTPUT_TAG}.log"
    CURRENT_COLUMN_DIR="${SCRIPT_DIR}/${ATTEMPT052_OUTPUT_TAG}_columns"

    {
        echo
        echo "[$(timestamp)] Running attempt-052 Lyapunov-dimension scan for ${gh_label}"
        echo "Repo root: ${REPO_ROOT}"
        echo "Julia threads: ${JULIA_NUM_THREADS}"
        echo "Julia GC threads: ${JULIA_NUM_GC_THREADS}"
        echo "OpenBLAS threads: ${OPENBLAS_NUM_THREADS}"
        echo "Grid: ${ATTEMPT052_NY} Delta Ca x ${ATTEMPT052_NX} Delta x"
        echo "Delta Ca range: [${ATTEMPT052_DELTA_CA_MIN}, ${ATTEMPT052_DELTA_CA_MAX}]"
        echo "Delta x range: [${ATTEMPT052_DELTA_X_MIN}, ${ATTEMPT052_DELTA_X_MAX}]"
        echo "g_h: ${ATTEMPT052_G_H}"
        echo "tau_y: ${ATTEMPT052_TAU_Y}"
        echo "Lyapunov Tmax: ${ATTEMPT052_LYAP_TMAX}"
        echo "Lyapunov min-time: ${ATTEMPT052_LYAP_MIN_TIME}"
        if gcs_enabled; then
            echo "GCS artifact URI: ${ATTEMPT052_GCS_URI}"
        fi
    } | tee -a "${CURRENT_LOG_PATH}"

    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. \
            kneading/experiment/attempt-052/main.jl 2>&1 | tee -a "${CURRENT_LOG_PATH}"
    else
        echo "/usr/bin/time is not available; using shell timing fallback." | tee -a "${CURRENT_LOG_PATH}"
        time "${JULIA_CMD[@]}" --startup-file=no --project=. \
            kneading/experiment/attempt-052/main.jl 2>&1 | tee -a "${CURRENT_LOG_PATH}"
    fi

    upload_gcs_final_artifacts "${ATTEMPT052_OUTPUT_TAG}"
done

echo "[$(timestamp)] attempt-052 g_h Lyapunov-dimension sweep complete."
