#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# The ggml Metal backend is Apple-only: it links against the Metal /
# Foundation frameworks and can only be built and run on macOS.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: the Metal backend can only be built on macOS (uname -s reports '$(uname -s)')." >&2
  echo "       Use scripts/build_cuda.sh, scripts/build_vulkan.sh or scripts/build_cpu.sh on this host." >&2
  exit 1
fi

CMAKE_BIN=${CMAKE_BIN:-cmake}
BUILD_DIR=${BUILD_DIR:-build-metal}

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

${CMAKE_BIN} -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_METAL=ON

${CMAKE_BIN} --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/bin/ed-cli" --help
