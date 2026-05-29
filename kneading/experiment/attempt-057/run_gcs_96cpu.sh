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

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-96}"
export JULIA_NUM_GC_THREADS="${JULIA_NUM_GC_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export ATTEMPT057_NX="${ATTEMPT057_NX:-500}"
export ATTEMPT057_NY="${ATTEMPT057_NY:-500}"
export ATTEMPT057_MAX_SEQ_LENGTH="${ATTEMPT057_MAX_SEQ_LENGTH:-20}"
export ATTEMPT057_SSCS_TMAX="${ATTEMPT057_SSCS_TMAX:-3.0e5}"
export ATTEMPT057_DELTA_X_MIN="${ATTEMPT057_DELTA_X_MIN:--1.5}"
export ATTEMPT057_DELTA_X_MAX="${ATTEMPT057_DELTA_X_MAX:--0.5}"
export ATTEMPT057_DELTA_CA_MIN="${ATTEMPT057_DELTA_CA_MIN:--45.0}"
export ATTEMPT057_DELTA_CA_MAX="${ATTEMPT057_DELTA_CA_MAX:--20.0}"
export ATTEMPT057_DELTA_X_TICK_STEP="${ATTEMPT057_DELTA_X_TICK_STEP:-0.1}"
export ATTEMPT057_DELTA_CA_TICK_STEP="${ATTEMPT057_DELTA_CA_TICK_STEP:-5.0}"
export ATTEMPT057_U0_V="${ATTEMPT057_U0_V:--30.0}"
export ATTEMPT057_U0_X_OFFSET="${ATTEMPT057_U0_X_OFFSET:--1.0e-4}"

BUCKET="${ATTEMPT057_GCS_BUCKET:-gs://carter-kneading-attempt057}"
GH_VALUES=(${ATTEMPT057_GH_VALUES:-0.0 0.001 0.01})

gh_tag() {
    case "$1" in
        0|0.0|0.00|0.000) printf "gh0p000" ;;
        0.001|.001) printf "gh0p001" ;;
        0.01|.01|0.010) printf "gh0p010" ;;
        *) printf "gh%s" "$1" | tr '.-' 'pm' ;;
    esac
}

cd "${REPO_ROOT}"

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Running attempt-057 arbitrary-IC SSCS scans."
echo "Repo root: ${REPO_ROOT}"
echo "Julia command: ${JULIA_CMD[*]}"
echo "Threads: JULIA_NUM_THREADS=${JULIA_NUM_THREADS}, JULIA_NUM_GC_THREADS=${JULIA_NUM_GC_THREADS}, OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "Grid: ${ATTEMPT057_NY} Delta Ca x ${ATTEMPT057_NX} Delta x"
echo "Max SSCS symbols: ${ATTEMPT057_MAX_SEQ_LENGTH}, tmax=${ATTEMPT057_SSCS_TMAX}"
echo "GCS bucket: ${BUCKET}"
echo "g_h values, in order: ${GH_VALUES[*]}"

for gh in "${GH_VALUES[@]}"; do
    tag="grid500_arbitrary_ic_$(gh_tag "${gh}")_seq20_tmax3e5"
    log_path="${SCRIPT_DIR}/${tag}.log"
    export ATTEMPT057_GH="${gh}"
    export ATTEMPT057_OUTPUT_TAG="${tag}"

    {
        echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Starting g_h=${gh}, tag=${tag}"
        echo "Output tag: ${ATTEMPT057_OUTPUT_TAG}"
    } | tee -a "${log_path}"

    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -p "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-057/main.jl 2>&1 | tee -a "${log_path}"
    else
        time "${JULIA_CMD[@]}" --startup-file=no --project=. kneading/experiment/attempt-057/main.jl 2>&1 | tee -a "${log_path}"
    fi

    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Uploading ${tag} artifacts." | tee -a "${log_path}"
    gcloud storage cp --recursive \
        "${SCRIPT_DIR}/${tag}_columns" \
        "${SCRIPT_DIR}/${tag}_results.tsv" \
        "${SCRIPT_DIR}/${tag}_summary.txt" \
        "${log_path}" \
        "${BUCKET}/attempt-057/" 2>&1 | tee -a "${log_path}"

    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Completed g_h=${gh}." | tee -a "${log_path}"
done

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] attempt-057 complete."
