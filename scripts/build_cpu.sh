#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CMAKE_BIN=${CMAKE_BIN:-/usr/bin/cmake}
CLEAN_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

env PATH=${CLEAN_PATH} ${CMAKE_BIN} -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DED_BUILD_EXAMPLES=ON

env PATH=${CLEAN_PATH} ${CMAKE_BIN} --build build-cpu -j

./build-cpu/bin/ed-cli --help
