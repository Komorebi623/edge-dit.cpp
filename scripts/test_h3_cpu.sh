#!/usr/bin/env bash
# GPU-free developer regression; never stops/inspects someone else's process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
cmake -S . -B build-h3-cpu -DED_GGML_CUDA=OFF -DED_ENABLE_CUDNN_SDPA=OFF \
    -DED_ENABLE_CUDA_SAGE_ATTN=OFF -DED_BUILD_H3_CUDA_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-h3-cpu -j "${H3_BUILD_JOBS:-4}"
ctest --test-dir build-h3-cpu --output-on-failure
python3 -m unittest discover -s scripts/tests -p 'test_h3_cuda_workflow.py' -v
python3 scripts/fetch_h3_references.py --self-test
# Uses tiny CPU tensors only; available in the established dev container.
python3 tests/test_h3_lora_merge.py
# New-server/build configuration safety regressions; tiny fake inputs only.
python3 -m unittest discover -s tests -p 'test_h3_cuda_build.py' -v
python3 -m unittest discover -s tests -p 'test_h3_server_preflight.py' -v
python3 -m unittest discover -s tests -p 'test_h3_request_cleanup.py' -v
python3 -m unittest discover -s tests -p 'test_h3_4090_workflow.py' -v
python3 -m unittest discover -s tests -p 'test_h3_gpu_memory_monitor.py' -v
python3 -m unittest discover -s tests -p 'test_h3_gpu_parity_guard.py' -v
python3 -m unittest discover -s tests -p 'test_h3_build_source_stability.py' -v
