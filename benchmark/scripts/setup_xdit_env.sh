#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XDIT_REPO="${XDIT_REPO:-$(cd "${REPO_ROOT}/.." && pwd)/xDiT}"
DISTVAE_REF="${DISTVAE_REF:-git+https://github.com/xdit-project/DistVAE.git@6d8025c96b9975d45badae12d8ed9b6422e34e2c}"
YUNCHANG_VERSION="${YUNCHANG_VERSION:-0.6.4}"

echo "Using Python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "Using xDiT repo: ${XDIT_REPO}"

"${PYTHON_BIN}" -m pip install --no-deps "yunchang==${YUNCHANG_VERSION}"
"${PYTHON_BIN}" -m pip install --no-deps "${DISTVAE_REF}"

PYTHONPATH="${XDIT_REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - <<'PY'
import importlib

for module in ["xfuser", "yunchang", "distvae"]:
    importlib.import_module(module)
    print(f"{module}: import ok")
PY
