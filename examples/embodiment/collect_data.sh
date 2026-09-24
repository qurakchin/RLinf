#!/usr/bin/env bash
set -eo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/collect_real_data.py"
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1

CONFIG_NAME="${1:-realworld_collect_data}"
if [ "$#" -gt 0 ]; then
    shift
fi

LOG_DIR="${RLINF_LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/"
     --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}" "$@")
echo "Using Python at $(command -v python)"
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
