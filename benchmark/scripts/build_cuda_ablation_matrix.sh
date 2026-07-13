#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

build_variant() {
  local name="$1"
  local build_dir="$2"
  local cudnn_sdpa="$3"
  local cuda_norm="$4"
  local cuda_rope="$5"
  local cuda_modulation="$6"

  echo "==> Building CUDA ablation variant: ${name} (${build_dir})"
  BUILD_DIR="${build_dir}" \
  ED_BUILD_PROFILE=performance \
  ED_ENABLE_CUDNN_SDPA="${cudnn_sdpa}" \
  ED_ENABLE_CUDA_NORM="${cuda_norm}" \
  ED_ENABLE_CUDA_ROPE="${cuda_rope}" \
  ED_ENABLE_CUDA_MODULATION="${cuda_modulation}" \
  ED_INSTALL_CUDNN="${cudnn_sdpa}" \
  ED_INSTALL_CUDNN_FRONTEND="${cudnn_sdpa}" \
  bash scripts/build_cuda.sh
}

build_variant generic build-cuda-generic OFF OFF OFF OFF
build_variant norm build-cuda-norm OFF ON OFF OFF
build_variant norm_rope build-cuda-norm-rope OFF ON ON OFF
build_variant norm_rope_mod build-cuda-norm-rope-mod OFF ON ON ON
build_variant full_optimized build-cuda ON ON ON ON

echo "CUDA ablation build matrix complete."
