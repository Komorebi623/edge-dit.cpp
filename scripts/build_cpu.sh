#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_DIR="${BUILD_DIR:-build-cpu}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if ! command -v "${CMAKE_BIN}" >/dev/null 2>&1 && [[ ! -x "${CMAKE_BIN}" ]]; then
  echo "error: CMake was not found. Install CMake or set CMAKE_BIN=/path/to/cmake." >&2
  exit 1
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf -- "${BUILD_DIR}"
fi

# oneDNN (bf16 AMX matmul) — 自动启用 if third_party/onednn 已构建。
# 先跑 scripts/build_onednn.sh 生成 install。也可用 ED_ONEDNN=0 强制关闭。
ONEDNN_ARGS=()
ONEDNN_PREFIX="${ROOT_DIR}/third_party/onednn/install"
if [[ "${ED_ONEDNN:-1}" != "0" && -d "${ONEDNN_PREFIX}" ]]; then
  for d in "${ONEDNN_PREFIX}/lib64/cmake/dnnl" "${ONEDNN_PREFIX}/lib/cmake/dnnl"; do
    if [[ -d "${d}" ]]; then
      ONEDNN_ARGS=(-DGGML_ONEDNN=ON -Ddnnl_DIR="${d}")
      echo "[build_cpu] oneDNN enabled: dnnl_DIR=${d}"
      echo "[build_cpu] 运行时需: LD_LIBRARY_PATH=${ONEDNN_PREFIX}/lib64"
      break
    fi
  done
fi
[[ ${#ONEDNN_ARGS[@]} -eq 0 ]] && echo "[build_cpu] oneDNN 未启用 (third_party/onednn/install 不存在; 先跑 scripts/build_onednn.sh)"

"${CMAKE_BIN}" -S . -B "${BUILD_DIR}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  -DED_BUILD_EXAMPLES=ON \
  "${ONEDNN_ARGS[@]}"

"${CMAKE_BIN}" --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/bin/ed-cli" --help
