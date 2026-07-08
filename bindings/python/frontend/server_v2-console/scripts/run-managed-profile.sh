#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROFILE_SLUG=${1:?usage: run-managed-profile.sh <profile-slug> [extra server args]}
shift

source "${SCRIPT_DIR}/runtime-env.sh"

PROFILE_PATH="${APP_ROOT}/runtime/profiles/${PROFILE_SLUG}.json"
if [[ ! -f "${PROFILE_PATH}" ]]; then
  echo "error: unknown managed profile '${PROFILE_SLUG}' at ${PROFILE_PATH}" >&2
  exit 1
fi

exec "${EDGE_DIT_PYTHON_BIN}" "${APP_ROOT}/runtime/managed_server_v2.py" --profile "${PROFILE_PATH}" "$@"
