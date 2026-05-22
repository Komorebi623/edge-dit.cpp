#!/usr/bin/env bash
set -e

cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLDIT_BUILD_EXAMPLES=ON

cmake --build build-cpu -j

./build-cpu/smoke_ggml