# H3 integrated CPU-only tiny runtime regression — 2026-10-11

## Delivered / write scope

Only these files were written by this worker:

1. `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3_runtime_optimization_test.cpp`
2. `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3-runtime-progress.md`

No model, pipeline, CMake source, shared backend, existing test, or another
worker's files were changed. Main added the CMake target and fixed the model
callback described below. Executables/build/run logs are in container `/tmp`.
No GPU enumeration/execution, Sage custom execution, process stop, weights
fetch, dependency installation, or commit was performed.

## Actual integrated runtime coverage

- Architecture detection fixture follows `tiny_weights` / `CpuCacheTestRunner`
  in the existing cache test: **four real H3 blocks, hidden dimension 8, one
  head with head dimension 128**, original H3 attention behavior.
- Tiny packed video/audio input, fixed sigma schedule, and deterministic
  initialization of **all real `params_ctx` tensors**, including implicit
  AdaLN weights/biases. Semantic names come from `get_param_tensors`; tensor
  header names are not assumed initialized before graph construction.
- Actual default `MiniMaxH3Runner::compute()` compared with integrated
  `prepare_cuda_schedule()` / `begin_cuda_step()` / `compute()` all-sigma
  ExactAdaLN prefill/import. No `get_context()` / `compute()` override, mock
  forward, output cache, or callback fix inside the test.
- Both TimeEmbedder/SiLU and curve-table/no-SiLU H3 architectures. All outputs
  are finite and packed-shape-preserving; comparison accepts abs/relative 2e-5,
  additionally reports bitexact comparisons. Fixture output must change with
  sigma to rule out a vacuous regression.
- All future sigma slots are complete after the first evaluation. Every block
  and final-layer projection has its expected `[144, rows, 1, 1]` or
  `[16, rows, 1, 1]` shape and finite native F32 readback. An independent real
  projection-only execution is compared **bit-for-bit** for every sigma,
  all four blocks, and the final layer.
- Hit graph inspection: all four real block boundaries remain; AdaLN GEMMs
  fall from **five to zero**; time-embedding GEMMs disappear; total node count
  decreases. Inspection builds only for counts; numerical comparisons execute
  the actual full graph through public runner `compute()`.
- Conditional/unconditional independent cache objects and identities; real
  video/audio references; in-place reference/context changes; text-tag run
  changes; keyframe identity; audio/video sigma shifts; reference order and
  source indices. Conditional refresh must not disturb unconditional state.
- Request reset clears cache pointers, staging, step metadata and schedule;
  missing schedule after reset fails; new schedule/full-weight identity works.
- Cache-off through public configuration restores baseline numerical output
  and original projection graph, with the shared CPU context callback still
  empty. Public optimization configuration rejects CPU; only explicit
  **test-local `model.cuda_optimizations` assignment** opts into low-level CPU.
- Invalid/out-of-range step, NaN/Inf/empty timestep, fixed schedule/identity/actual
  sigma, incompatible cache and graph
  **projection shapes**, stale token, one-byte contract budget, and real runner
  **one-MiB prefill budget rejection**. Assert-aborting malformed raw latent
  inputs are deliberately not invoked. Failed prefill must stay within its
  retained budget, leave model imports disabled, and recover after reset.

## Independent CPU registration

The shared `ggml_extend_backend.hpp` CPU wrappers can trigger backend discovery
and conflict with `ggml-cpu.h` C linkage. This test uses GNU/Clang symbol aliases
to call CPU library registration/init/thread-count symbols directly, explicitly
registering only CPU. No `init_by_type`, `load_all`, or `dev_count` in the test.
The supplied development build has `ED_GGML_CUDA=OFF`, `GGML_BACKEND_DL=OFF`.
The standalone alias declarations require a GNU-compatible compiler (validated
with the container GNU C++ compiler); this is not a CUDA/MSVC portability claim.

## Model issue reported to main and resolved by main

Initial real execution successfully prefetched projections but failed on the
first fullgraph import with `MiniMax-H3 exact AdaLN projection unavailable`.
`build_graph()` had not supplied `runner_ctx.bind_backend_tensor_data`, which
`GraphInputs::import()` requires. Main added an AdaLN/sparse-only binding to
runner `set_backend_tensor_data()` at the H3 graph context boundary. The test
was rerun against that **real model fix**, without a masked test override.
A test-only block-counting error (inherited `(view)/(cont)` names counted as
boundaries) was also fixed by requiring the exact `|hidden_states` suffix.
No further model bug was found in the covered actual flows.

## Final real validation

Development container: `wty-edgedit-dev`, repo `/workspace/edge-dit.cpp`.
Built and linked against host CPU libraries in `build-h3-cpu/bin` and existing
`build-h3-cpu/lib/libedgedit.a` using the main-owned CMake target:

```sh
docker exec wty-edgedit-dev bash -lc 'cd /workspace/edge-dit.cpp && cmake --build build-h3-cpu --target h3-runtime_optimization-test -j 2'
docker exec wty-edgedit-dev bash -lc 'cd /workspace/edge-dit.cpp && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 ctest --test-dir build-h3-cpu -R runtime --repeat until-fail:3 -V'
```

Final results at **2026-10-11 19:49:57 Asia/Shanghai** (three consecutive runs,
all exit 0, against the latest step-safety/reset/SHA model changes):

```text
PASS actual tiny H3 TimeEmbedder/SiLU all-sigma default/cache; nodes=662/642; AdaLN GEMMs=5/0; native projection readback bitexact
PASS actual tiny H3 curves/no-SiLU all-sigma default/cache; nodes=657/642; AdaLN GEMMs=5/0; native projection readback bitexact
PASS CFG isolation, video/audio references, source/order/text-tag mutation, reset/new request
EXPECT prefill budget rejection (the runner may log post compute callback failed)
PASS invalid schedule/sigma/shape, stale token, real runner small-budget failure/recovery
H3 integrated runtime optimization tests passed (172163 checks; 24 actual fullgraph comparisons, 24 bitexact; CPU only, no GPU enumeration/Sage execution)
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.20 sec
```

The single `post compute callback failed` log per run is **expected** from the
one-MiB budget negative case, not a swallowed numerical error: the test requires
`prefill failed` as the exception reason and verifies recovery afterwards.

Logs (inside container):

- `/tmp/h3-runtime-configure-20261011.log`
- `/tmp/h3-runtime-cmake-build-20261011.log`
- `/tmp/h3-runtime-ctest-20261011.log` (final verbose three-run record)
- `/tmp/h3-runtime-build-20261011.log`, `/tmp/h3-runtime-run-20261011.log`
  (latest direct compile/run at 2026-10-11 19:49:18 Asia/Shanghai,
  also real fullgraph; final CTest validates the same source)

`git diff --check` for the two delivered files passed. Runtime target only was
rerun here; no claim to have rerun the main's other ten tests, CUDA execution,
production-weight quality, or GPU memory/performance validation.
