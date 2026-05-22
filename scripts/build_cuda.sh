#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

rm -rf build-cuda

CUDA_ROOT=/usr/local/cuda-13.0
CMAKE_BIN=/export/home/liuyiming54/miniconda3/envs/hicache/bin/cmake

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
CUDA_HOME=${CUDA_ROOT} \
CUDA_PATH=${CUDA_ROOT} \
CUDAToolkit_ROOT=${CUDA_ROOT} \
PATH=${CUDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
${CMAKE_BIN} -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DLDIT_BUILD_EXAMPLES=ON \
  -DLDIT_GGML_CUDA=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=${CUDA_ROOT} \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT} \
  -DCMAKE_CUDA_ARCHITECTURES=native

env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
PATH=${CUDA_ROOT}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
${CMAKE_BIN} --build build-cuda -j

./build-cuda/bin/ld-cli --help
