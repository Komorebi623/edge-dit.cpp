# H3 fusion numerical regression — 2026-10-11

## Scope

Only `tests/h3_cuda_fusion_test.cpp` and this progress file are written.
No model/shared-backend/CMake changes, GPU execution/discovery, downloads,
process inspection or process termination. Uses existing `build-merge-cpu`
libraries and a small standalone executable under `/tmp`.

## Coverage

- Actual F32 ggml CUSTOM modulation vs `x + x * scale + shift` CPU graph.
- Actual F32 ggml CUSTOM residual gate vs `residual + x * gate` CPU graph.
- Actual production `modulate_segments` / `gated_residual_segments`, five
  unequal spans, nonmonotonic modality/timestep rows, six AdaLN fields and
  three modalities; materialized baseline vs view/custom fused execution.
- 2D/3D/4D broadcasts, singleton and divisor/repeat broadcasts, different
  hidden widths, equal-size inputs, nonzero-offset padded/strided views of
  inputs and broadcast parameters. Batch sizes 1 and 2.
- Actual ggml SwiGLU vs split + SiLU + multiply CPU graph; dense 3D and
  non-contiguous 2D views. No new implementation of the operator.
- One/two CPU threads and changed-input replay of every graph.
- Absolute tolerance `1e-6`, finite results, output shape/type assertions,
  actual custom-node counts (no silent baseline fallback), CPU supports-op
  and compute-status checks. Direct modulation/residual also checked against
  a scalar oracle **in addition to**, not instead of, actual graph execution.
- Unsupported dimension broadcasts, residual mismatch, F16/I32, and null
  inputs must return rejection; unsupported positive graph cases fail.

CPU registration/init is direct from the linked CPU backend, not global
backend enumeration or `load_all`. Dynamic-backend builds deliberately fail
compilation rather than silently discovering GPU plugins. Production CPU
custom segment ops use scoped legacy flags because request-scoped fused_ops
is intentionally CUDA-only. These tests **do not validate CUDA kernels or
CUDA request-scoped dispatch**. QKV/RMSNorm/partial-RoPE expansion deferred
for bounded delivery.

## Standalone compile / run (existing CPU build, no CMake modifications)

```bash
ROOT=/export/home/wangtianyang.21/code1/edge-dit.cpp
c++ -std=c++17 -O2 -DGGML_MAX_NAME=128 -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/src/core" \
  -I"$ROOT/third_party" -I"$ROOT/third_party/ggml/include" \
  -I"$ROOT/third_party/ggml/src" "$ROOT/tests/h3_cuda_fusion_test.cpp" \
  -L"$ROOT/build-merge-cpu/lib" -L"$ROOT/build-merge-cpu/bin" \
  -Wl,-rpath,"$ROOT/build-merge-cpu/bin" \
  -ledgedit -lggml -lggml-cpu -lggml-base -pthread -o /tmp/h3-cpu-fusion-test
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 /tmp/h3-cpu-fusion-test
```

## Result

**DONE — files ready for the main thread to add its CMake test target.**

- Compile: **PASS**, exit 0, using the command above and existing CPU-only
  `build-merge-cpu` (`GGML_BACKEND_DL=OFF`, `GGML_CUDA=OFF`).
- Run: **PASS**, exit 0, **56 actual-graph numerical comparisons** plus
  unsupported-input rejection checks. `worst_max_abs=0`, tolerance `1e-6`.
- Every positive graph executes twice with changed inputs; both one-thread
  and two-thread configurations pass. Segment graphs each contain exactly
  five fused modulation and five fused residual CUSTOM nodes.
- Runtime binary: `/tmp/h3-cpu-fusion-test`.
- CPU-only result; **CUDA NOT validated**. Existing project-wide CTest,
  workflow/fetch/LoRA runs are outside this test's result and were not rerun.

Final test summary:

```text
PASS unsupported shapes/types/null rejected (no fallback)
PASS CPU-only actual ggml graphs: 56 comparisons; tolerance=1e-06; worst_max_abs=0. CUDA NOT validated.
```

Integration: main thread can add `h3_cuda_fusion_test` from this source and
link `edgedit`, `ggml`, `ggml-cpu` (standalone additionally names `ggml-base`).
Ensure `GGML_MAX_NAME=128` matches the existing library ABI; the standalone
command supplies it explicitly. No changes to CMake were made here.
