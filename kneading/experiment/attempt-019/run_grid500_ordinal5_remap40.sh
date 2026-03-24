#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export ATTEMPT019_NX="${ATTEMPT019_NX:-500}"
export ATTEMPT019_NY="${ATTEMPT019_NY:-500}"
export ATTEMPT019_OUTPUT_TAG="${ATTEMPT019_OUTPUT_TAG:-grid500_ordinal5}"
export ATTEMPT019_REPAIR_INPUT_TAG="${ATTEMPT019_REPAIR_INPUT_TAG:-${ATTEMPT019_OUTPUT_TAG}}"

/usr/bin/time -p julia --project=. kneading/experiment/attempt-019/contours.jl

export ATTEMPT019_OUTPUT_TAG="${ATTEMPT019_REPAIR_OUTPUT_TAG:-grid500_ordinal5_remap40}"
export ATTEMPT019_MAP_RESOLUTION="${ATTEMPT019_MAP_RESOLUTION:-40}"
/usr/bin/time -p julia --project=. kneading/experiment/attempt-019/repair_failed_runs.jl
