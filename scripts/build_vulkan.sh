#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CMAKE_BIN=${CMAKE_BIN:-/usr/bin/cmake}
BUILD_DIR=${BUILD_DIR:-build-vulkan}
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ggml-vulkan compiles GLSL shaders to SPIR-V at build time and needs glslc
# on PATH. If a Vulkan SDK is installed in a non-standard location, point
# VULKAN_SDK at it so its bin/, headers and loader are picked up.
BUILD_PATH=${CLEAN_PATH}
if [[ -n "${VULKAN_SDK:-}" ]]; then
  BUILD_PATH=${VULKAN_SDK}/bin:${BUILD_PATH}
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

if ! env PATH=${BUILD_PATH} command -v glslc >/dev/null 2>&1; then
  echo "warning: glslc not found on PATH; ggml-vulkan shader compilation will fail." >&2
  echo "         Install the Vulkan SDK (shaderc) or set VULKAN_SDK=/path/to/sdk." >&2
fi

env PATH=${BUILD_PATH} ${CMAKE_BIN} -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_VULKAN=ON

env PATH=${BUILD_PATH} ${CMAKE_BIN} --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/bin/ed-cli" --help
