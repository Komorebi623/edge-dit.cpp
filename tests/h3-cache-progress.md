# H3 model-side fusion / dual-block cache progress — 2026-10-11

## Implemented

- Request-local `CudaOptimizationOptions`, no global environment mutation.
- Fused segment modulation/residual-gate custom operators, segment/QKV/RoPE
  views, SwiGLU, final-layer modulation. Legacy flags retain their off behavior.
- Real dual-block cache graph: front `[0, Fn)` executes, RF = front - stack input.
  Every nonempty text/condition-video/condition-audio/target-audio/target-video
  span receives its own in-graph relative-L1 guard. Only scalars leave CUDA.
- Miss: starts from current front, computes middle `[Fn, L-Bn)`, captures
  middle_output - current_front, then executes real suffix `[L-Bn, L)` and heads.
- Hit: starts from current front, adds the stored middle delta, then executes
  real suffix and heads. RF anchor stays paired with the last computed delta.
- Reuses existing cache operators (`cache.difference`, `cache.rel_l1`,
  `cache.weighted_blend`, `cache.copy`), TapRegistry graph expansion, GGMLRunner
  compute/named device handoff, and RunnerCacheDeviceStore. No output-level cache.
- Independent conditional/unconditional persistent CUDA slots and policy state.
- Warmup, fractional window, max consecutive hits, non-finite guards, partition
  validation, request reset, repeated/skipped steps and total-step changes.
- Conditioning identity hashes tensor bytes/shapes, tags, keyframes, reference
  blocks, video/audio shift and latent shape; in-place changes invalidate.
- `begin_cuda_step(int,int,bool)` requires one call per cached CFG branch.
- Default-off reset/configure does not invalidate baseline persistent graphs.
  Graph-affecting option changes and actual cache-state reset do invalidate.

## Capability / explicit restrictions

- Runner static `cache_dit=true` describes implemented graph integration.
- `fused_ops=true` only when `ED_ENABLE_CUDA_MODULATION` is compiled.
- Public configuration/computation require an actual CUDA backend.
- Cache explicitly rejects parameter offload, parallel/segmented graph cuts,
  `ED_MINIMAX_H3_REUSE_GRAPH`, debug-target outputs, and mutable weight adapters.
- AdaLN/Sage/sparse are not enabled by this worker. Main integrates separate
  workers; existing H3 scale remains 1/128 and head dimension remains 128.
- No GPU initialized, no GPU tests run, no process stopped. CUDA execution,
  memory/performance and production-weight quality remain unvalidated.

## CPU validation

- `/tmp/h3_cuda_cache_test`: compiled C++17 with existing CPU-only ggml libraries
  and `build-merge-cpu/lib/libedgedit.a`; exit 0.
- Actual tiny 4-block H3 graph uses head_dim=128 and original 1/128 scale.
- Full and split-miss numerical outputs agree; exact-RF hit agrees.
- Block-node counts: full 4, probe 1, miss remainder 3; hit remainder 1.
- CFG independence, conditioning hash mutation, request reset, total_steps
  change, skipped/repeated steps, all modality guards, window, max consecutive,
  zero/nonfinite RF norm and invalid partitions covered.
- Cache-off after request reset agrees with full H3 graph.
- `git diff --check` passed for model header.
- CPU H3 pipeline syntax check passed before later concurrent pipeline updates;
  final syntax log records the combined state for main-thread follow-up.

## Files written by this worker

1. `src/dit_models/models/minimax_h3_full.hpp`
2. `src/dit_models/models/minimax_h3_cuda_cache.hpp`
3. `tests/h3_cuda_cache_test.cpp`
4. `tests/h3-cache-progress.md` (explicitly permitted additional log)

Logs: `/tmp/h3-cache-test-build-20261011.log`,
`/tmp/h3-cache-test-run-20261011.log`, `/tmp/h3-cache-final-syntax-20261011.log`.
Main-thread files were read only, not edited by this worker.
