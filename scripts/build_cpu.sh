#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CMAKE_BIN=${CMAKE_BIN:-/usr/bin/cmake}
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# oneDNN (bf16 AMX matmul) — 自动启用 if third_party/onednn 已构建。
# 先跑 scripts/build_onednn.sh 生成 install。也可用 ED_ONEDNN=0 强制关闭。
ONEDNN_ARGS=()
ONEDNN_PREFIX="$ROOT_DIR/third_party/onednn/install"
if [ "${ED_ONEDNN:-1}" != "0" ] && [ -d "$ONEDNN_PREFIX" ]; then
  for d in "$ONEDNN_PREFIX/lib64/cmake/dnnl" "$ONEDNN_PREFIX/lib/cmake/dnnl"; do
    if [ -d "$d" ]; then
      ONEDNN_ARGS=(-DGGML_ONEDNN=ON -Ddnnl_DIR="$d")
      echo "[build_cpu] oneDNN enabled: dnnl_DIR=$d"
      echo "[build_cpu] 运行时需: LD_LIBRARY_PATH=$(dirname "$d")/../.. (即 $ONEDNN_PREFIX/lib64)"
      break
    fi
  done
fi
[ ${#ONEDNN_ARGS[@]} -eq 0 ] && echo "[build_cpu] oneDNN 未启用 (third_party/onednn/install 不存在; 先跑 scripts/build_onednn.sh)"

env PATH=${CLEAN_PATH} ${CMAKE_BIN} -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON \
  "${ONEDNN_ARGS[@]}"

env PATH=${CLEAN_PATH} ${CMAKE_BIN} --build build-cpu -j

./build-cpu/bin/ed-cli --help
