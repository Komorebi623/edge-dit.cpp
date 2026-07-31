#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

: "${EDGE_DIT_LIBRARY:?set EDGE_DIT_LIBRARY to the libedgedit.so path}"
: "${EDGE_DIT_MODEL_PATH:?set EDGE_DIT_MODEL_PATH to the model directory}"

EDGE_DIT_PYTHON=${EDGE_DIT_PYTHON:-/usr/bin/python3}
EDGE_DIT_SERVER_KIND=${EDGE_DIT_SERVER_KIND:-image}

if [[ "${EDGE_DIT_SERVER_KIND}" == "video" ]]; then
  EDGE_DIT_CONFIG=${EDGE_DIT_CONFIG:-"${REPO_ROOT}/bindings/python/examples/wan_t2v_smoke_config.json"}
  EDGE_DIT_OUTPUT=${EDGE_DIT_OUTPUT:-/tmp/edge_dit_python_server_wan_smoke.gif}
else
  EDGE_DIT_CONFIG=${EDGE_DIT_CONFIG:-"${REPO_ROOT}/bindings/python/examples/flux_smoke_config.json"}
  EDGE_DIT_OUTPUT=${EDGE_DIT_OUTPUT:-/tmp/edge_dit_python_server_smoke.png}
fi

export PYTHONPATH="${REPO_ROOT}/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
export EDGE_DIT_LIBRARY
export EDGE_DIT_MODEL_PATH

"${EDGE_DIT_PYTHON}" \
  "${REPO_ROOT}/bindings/python/examples/server_smoke.py" \
  --kind "${EDGE_DIT_SERVER_KIND}" \
  --config "${EDGE_DIT_CONFIG}" \
  --output "${EDGE_DIT_OUTPUT}"
