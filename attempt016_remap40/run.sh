#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${JULIA_CMD:-}" ]]; then
    if julia +release --startup-file=no --version >/dev/null 2>&1; then
        export JULIA_CMD="julia +release"
    else
        export JULIA_CMD="julia"
    fi
fi

export JULIA_PKG_PRECOMPILE_AUTO="${JULIA_PKG_PRECOMPILE_AUTO:-0}"

${JULIA_CMD} --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
exec "${ROOT_DIR}/kneading/experiment/attempt-016/run_grid500_seq7_prefixes_remap40.sh"
