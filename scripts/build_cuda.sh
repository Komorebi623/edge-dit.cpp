#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
CMAKE_BIN=${CMAKE_BIN:-/usr/bin/cmake}
BUILD_DIR=${BUILD_DIR:-build-cuda}
CLEAN_PATH=${CUDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

if [[ -z "${CUDA_ARCHITECTURES:-}" ]]; then
  CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')"
fi

if [[ -z "${CUDA_ARCHITECTURES}" || ! "${CUDA_ARCHITECTURES}" =~ ^[0-9]+$ ]]; then
  # Buildable fallback for common datacenter/workstation NVIDIA GPUs when
  # no driver-visible GPU is available during configure.
  CUDA_ARCHITECTURES="75;80;86;89;90"
fi

echo "CUDA_ROOT=${CUDA_ROOT}"
echo "CMAKE_BIN=${CMAKE_BIN}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
CUDA_HOME=${CUDA_ROOT} \
CUDA_PATH=${CUDA_ROOT} \
CUDAToolkit_ROOT=${CUDA_ROOT} \
PATH=${CLEAN_PATH} \
${CMAKE_BIN} -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_CUDA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=${CUDA_ROOT} \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT} \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}"

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
PATH=${CLEAN_PATH} \
${CMAKE_BIN} --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/bin/ed-cli" --help
