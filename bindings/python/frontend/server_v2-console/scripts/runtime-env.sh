#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${APP_ROOT}/../../../.." && pwd)

export EDGE_DIT_REPO_ROOT="${EDGE_DIT_REPO_ROOT:-${REPO_ROOT}}"
export EDGE_DIT_PYTHON_BIN="${EDGE_DIT_PYTHON_BIN:-/usr/bin/python3}"
export EDGE_DIT_LIBRARY="${EDGE_DIT_LIBRARY:-${EDGE_DIT_REPO_ROOT}/build-cuda-shared/bin/libedgedit.so}"
export CUDNN_ROOT="${CUDNN_ROOT:-/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cudnn}"
export EDGE_DIT_DEPENDENCY_DIRS="${EDGE_DIT_DEPENDENCY_DIRS:-/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cublas/lib:/home/yangminghong/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${EDGE_DIT_REPO_ROOT}/build-cuda-shared/bin}"
export PYTHONPATH="${EDGE_DIT_REPO_ROOT}/bindings/python/src${PYTHONPATH:+:${PYTHONPATH}}"
