#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

source "${SCRIPT_DIR}/runtime-env.sh"

FRONTEND_HOST=${EDGE_DIT_FRONTEND_HOST:-127.0.0.1}

MANAGER_ARGS=("$@")
if [[ " ${MANAGER_ARGS[*]} " != *" --auto-start-profile "* ]]; then
  MANAGER_ARGS=(--auto-start-profile "${EDGE_DIT_DEFAULT_PROFILE:-flux-dev}" "${MANAGER_ARGS[@]}")
fi

MANAGER_PID=""
SHUTTING_DOWN=0

cleanup() {
  SHUTTING_DOWN=1
  if [[ -n "${MANAGER_PID}" ]] && kill -0 "${MANAGER_PID}" 2>/dev/null; then
    kill "${MANAGER_PID}" 2>/dev/null || true
    wait "${MANAGER_PID}" 2>/dev/null || true
  fi
}

run_manager_loop() {
  while true; do
    bash "${SCRIPT_DIR}/run-runtime-manager.sh" "${MANAGER_ARGS[@]}" &
    MANAGER_PID=$!
    local exit_code=0
    wait "${MANAGER_PID}" || exit_code=$?
    MANAGER_PID=""

    if [[ ${SHUTTING_DOWN} -eq 1 ]]; then
      return 0
    fi

    echo "[python-server-console] runtime manager exited with ${exit_code}; restarting in 2s" >&2
    sleep 2
  done
}

trap cleanup EXIT INT TERM

run_manager_loop &
SUPERVISOR_PID=$!

npm run dev -- --host "${FRONTEND_HOST}" --port 5173
VITE_EXIT_CODE=$?

cleanup
wait "${SUPERVISOR_PID}" 2>/dev/null || true
exit "${VITE_EXIT_CODE}"
