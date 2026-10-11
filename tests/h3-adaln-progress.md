# H3 CUDA exact AdaLN worker — handoff (2026-10-11)

## Scope / completion

Worker wrote only:

- `/export/home/wangtianyang.21/code1/edge-dit.cpp/src/dit_models/models/minimax_h3_cuda_adaln_cache.hpp`
- `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3_cuda_adaln_test.cpp`
- `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3-adaln-progress.md`

No edits to main H3 model, pipeline, CMake, parser, shared backend, or other workers' files. No GPU execution, weights download, process stop, dependency install, or git commit. Test executables/logs are in `/tmp`. Independent header is off by default and has no static initialization/environment mutation. It depends only on ggml headers; the optional RunnerBridge is instantiated after ggml_extend.hpp is available.

## Implemented, not merely a configuration flag

- Entire layer linear outputs, including bias/adapter contribution: blocks `[hidden*6*3, timestep_rows, 1, 1]`, final `[hidden*2, timestep_rows, 1, 1]`.
- Exact FP32-bit timestep/sigma identity, including signed zero. No quantization, approximate neighbouring-step reuse, tolerance comparison, or curve interpolation in cache lookup. Existing H3 table interpolation is performed by the ORIGINAL producer before capture, not used as a cache approximation.
- Fixed full schedule and scheduler identity; ordered timesteps; semantic token spans, reference indices, text-tag modality runs; final video/audio row selection; flow shifts; full checkpoint identity and reload generation; adapter identity/generation; CFG identity/branch; architecture and numerical/backend identity.
- Strict request reset and stale-token rejection. Cache values are immutable until reset. Caller must re-bind once each evaluation if ANY identity may change; do not reuse tokens across mutable model state.
- Retained metadata + F32 payload + graph-owned staging descriptors/payloads obey one `Options.max_bytes` limit and `max_entries`. This is NOT a total process RSS cap: caller contract, temporary identity encoding, graph metadata, allocator overhead, runner/backend transient allocations and device leaves are outside it. No LRU surprises or weight release.
- Graph interception uses pinned output pointers; synchronous native F32 readback before graph-buffer release; shape/type/stride/finite validation. Non-F32 cache paths fail closed, never convert.
- Hit graph gets a stable, budgeted host staging copy; resetting the cache cannot invalidate an executing graph. Cache must outlive its GraphInputs objects; no shared concurrent graph builds on one worker.
- `projection_or_compute` does not call the producer on a hit: no SiLU/linear/bias/adapter projection nodes. Misses preserve the original producer.
- All-schedule projection-only prefill through existing Runner prepare/execute/post-readback, selecting only reachable params for partial offload. Explicit CUDA gate for production; low-level primitives deliberately allow CPU synthetic tests.

Diagnostic tensor names: `h3_exact_adaln.block.<index>` / `h3_exact_adaln.final`. Capture uses exact tensor pointers, not string name lookup. Runner may rename its final-result node; this does not invalidate capture.

## Minimal integration — model hook

Namespace: `MiniMaxH3::ExactAdaLN` (below abbreviated `EA`). Main worker owns model/pipeline integration.

Model/forward fields: `EA::Cache*`, `EA::Cache::StepToken`, `EA::GraphInputs*`, projection index. Block indices `0..num_layers-1`; final index `num_layers`. Add public projection accessors to blocks/final for warmup; expose a model dispatcher to call those accessors without transformer forward.

```cpp
// Replace each original adaln->forward(ctx, t_emb), keeping defaults disabled:
auto* mods = cache && inputs
    ? EA::projection_or_compute(*cache, token, projection_index, ctx, *inputs,
          [&] { return adaln->forward(ctx, t_emb); })
    : adaln->forward(ctx, t_emb);

// GGMLRunner::get_context does NOT fill this callback by default:
ctx.bind_backend_tensor_data = [this](ggml_tensor* t, const void* p) {
    set_backend_tensor_data(t, p);
};
```

GraphInputs must survive the RETURN of build_graph, until its graph finishes execution. A local GraphInputs inside the graph-builder lambda is incorrect. Keep it in runner/compute scope; clear model non-owning fields and destroy staging after execution. Do not pair with shape-only compiled graph reuse unless imported leaves are refreshed for the exact new token/sigma.

## Minimal integration — Runner / all-step warmup

Actual GGMLRunner params/prepare/execute APIs are protected. No shared changes are necessary:

```cpp
using H3ExactRunnerBase = EA::RunnerBridge<GGMLRunner>;
struct MiniMaxH3Runner : H3ExactRunnerBase {
    // Initialize H3ExactRunnerBase(backend, offload_params_to_cpu), not GGMLRunner.
    // All existing inherited Runner APIs stay available.
};
```

Alternatively implement the six `exact_*` thin forwarding methods shown in RunnerBridge inside the existing H3 Runner, avoiding a base change.

```cpp
// Caller supplies build(i, token) -> ggml_cgraph*.
EA::Status prefill_request(Runner& runner, EA::Cache& cache,
                          const EA::RequestContract& contract,
                          int n_threads, Build build);

// Single-plan warmup helper, used by prefill_request:
EA::Status run_projection_warmup(Runner& runner, EA::Cache& cache,
                                EA::Cache::StepToken token,
                                int n_threads, Build build);
```

Example producer:

```cpp
EA::prefill_request(*this, branch_cache, branch_contract, n_threads,
    [&](size_t i, EA::Cache::StepToken token) {
        const auto& plan = branch_contract.plans[i];
        // Extract existing H3 build_graph timestep input construction unchanged:
        // native -> timestep_embedding + make_input;
        // curves -> lower/upper/fractions + make_input.
        auto ti = make_exact_time_inputs(plan.timesteps); // main worker defines
        auto ctx = get_context();
        auto* emb = model.time_embedding(&ctx, ti.features, ti.lower,
                                         ti.upper, ti.fractions);
        return EA::build_projection_graph(compute_ctx, branch_cache, token,
            H3_GRAPH_SIZE, [&](size_t p) { return model.project_adaln(&ctx, emb, p); });
    });
```

`make_exact_time_inputs` / `model.project_adaln` are the main worker's small extraction/accessor hooks, not functions secretly implemented by this header. Warmup does NOT need packed latents, transformer attention, VDN/VSA, or large test checkpoints.

For lazy补漏 capture, use `capture_pending(token, backend)` in a successful post-compute callback BEFORE free; after `compute(..., free=true)` returns is too late. Fully prefilled ordinary forwards require no miss capture. Failure paths discard_pending/reset. Complete prefill must be checked; partial coverage is not advertised as full support.

## Minimal integration — pipeline request

Accepted main-thread design: `prepare_cuda_schedule(sigmas, scheduleridentity, weightidentity)` at generate start; first compute PER CFG branch constructs that branch's actual context/tags layouts for ALL schedule steps, builds one branch contract, pre-fills all projections, then serves the actual step. Two branches may hold two independent Cache instances; this header does not mandate sharing CFG payloads.

```cpp
EA::RequestContract c;
c.full_weights_identity = full_dit_checkpoint_manifest_sha; // ALL source shards/config
c.load_generation = immutable_model_generation;            // 1 only if truly immutable
c.adapter_identity = "none"; c.adapter_generation = 0;      // no adapter only
c.cfg_identity = branch_guidance_conditioning_identity;
c.math_identity = device_precision_tf32_tp_kernel_identity;
c.schedule_identity = scheduler_identity;
c.video_shift = video_shift; c.audio_shift = audio_shift;
c.timestep_input_dim = config.timestep_input_dim;
c.time_embed_hidden = config.time_embed_hidden_size;
c.time_embed_dim = config.time_embed_dim;
c.curve_grid = config.adaln_curve_grid;
c.projections = EA::h3_projections(config.hidden_size, config.num_layers);

// Store exact sigmas as the Runner receives them: account for *1000 /1000
// roundtrip if pipeline uses that representation. Include terminal endpoint.
c.sigmas = actual_full_runner_sigma_schedule;
for (size_t i = 0; i + 1 < c.sigmas.size(); ++i) {
    const float sigma = c.sigmas[i];
    const float tv = 1.f - sigma;
    const float ta = 1.f - MiniMaxH3::time_shift_sigma(sigma, video_shift, audio_shift);
    auto layout = make_actual_branch_layout(tv, ta); // existing build_layout
    c.plans.push_back(EA::plan_from_layout(i, sigma, cfg_role, layout));
}

EA::begin_cuda_request(branch_cache, c, runtime_backend);
// Immediately prefill_request(...): not just lazy per-sigma misses.
// Actual compute:
auto plan = EA::plan_from_layout(step, actual_sigma, cfg_role, actual_layout);
EA::Cache::StepToken token;
EA::Status valid = branch_cache.bind_step(c, plan_index, plan, &token);
// Only use hook when valid == Hit; layer lookup may still independently miss.
```

Exported layout helper is `plan_from_layout`, not `make_h3_step_plan`; main thread may wrap/alias it. It intersects semantic sequence segments with modulation spans to preserve text-tag runs and reference source indices. H3 SequenceKind enum order is documented in the helper.

Main thread can parse `ED_H3_ADALN_MAX_MIB` with default 4096 and pass `Options{true, mib*1024*1024, max_entries}`. Standalone header default remains 256 MiB/off. The cap is HOST storage/staging, not a GPU allocator cap. Two branch caches each receiving 4096 MiB mean two separate limits (potentially 8192 MiB host total). GPU imported leaves/transient projections and resident weights remain additional. This worker does not release weights and does not claim an 11 GiB VRAM reduction or measured H3 CUDA speedup. Request prefill moves first-use projections earlier; it is not itself a reduction of total first-use GEMMs. Include warmup in any end-to-end benchmark.

## Source inspection / provenance

Read current local `minimax_h3_full.hpp` TimeEmbedder/AdaLayerNormModulation/time_embedding/build_layout/curve construction, and GGMLRunner get_context/make_input/params allocation/prepare/execute/post-compute/offload interfaces. Reference files were supplied locally, not downloaded by this worker:

- `/tmp/edge-dit-h3-research-20261011/sglang--python__sglang__multimodal_gen__runtime__models__dits__minimax_h3_adaln_cache.py`
  SHA256 `a43ce678b043cd0e4b0241f2b2f6c93513de397d69218e05831a610dfb1f632a`
- `/tmp/edge-dit-h3-research-20261011/sglang--python__sglang__multimodal_gen__runtime__models__dits__minimax_h3.py`
  SHA256 `ced7b679ee05f75045646cb81a436088557b90f23d3be980aba13f816ac23f0d`
- `/tmp/edge-dit-h3-research-20261011/vllm-omni--vllm_omni__diffusion__models__minimax_h3__adaln_cache.py`
  SHA256 `7b83cf4a6d1c957fb103a2f0a8c2333e26e500ab6d051b8d5ea2448aff7917cc`
- `/tmp/edge-dit-h3-research-20261011/vllm-omni--tests__diffusion__models__minimax_h3__test_adaln_cache.py`
  SHA256 `733e3d5a9c87fd43c9490412c50032d49ff14594c63ef2f8891f5b965e3a64f0`

Relevant reference facts: SGLang `_plan_key` encodes exact FP32 bits, distinguishes native/curve embed paths, and stores complete per-layer/final projections; vLLM-Omni cache validation conservatively binds schedule, weights, adapters and math. This implementation is independently written C++ using local GGML and adds explicit role/layout/request-lifetime guards.

## Reproducible CPU-only commands

Run from `/export/home/wangtianyang.21/code1/edge-dit.cpp`. Uses EXISTING CPU GGML libraries, no configure/build of CUDA baseline and no dependency fetch.

```sh
c++ -std=c++17 -Wall -Wextra -Werror -DGGML_MAX_NAME=128 \
  -Isrc -Ithird_party/ggml/include tests/h3_cuda_adaln_test.cpp \
  -Lbuild-merge-cpu/bin -Wl,-rpath,$PWD/build-merge-cpu/bin \
  -lggml -lggml-cpu -lggml-base -o /tmp/h3_cuda_adaln_test
/tmp/h3_cuda_adaln_test

c++ -std=c++17 -O0 -g -DH3_ADALN_TEST_REAL_RUNNER \
  -D__STATIC_INLINE__=inline -DGGML_MAX_NAME=128 -DGGML_USE_CPU \
  -Iinclude -Isrc -Isrc/core -Ithird_party -Ithird_party/ggml/src \
  -Ithird_party/ggml/include tests/h3_cuda_adaln_test.cpp \
  build-merge-cpu/lib/libedgedit.a -Lbuild-merge-cpu/bin \
  -Wl,-rpath,$PWD/build-merge-cpu/bin -lggml -lggml-cpu -lggml-base \
  -pthread -o /tmp/h3_cuda_adaln_real_test
/tmp/h3_cuda_adaln_real_test
```

The optional real-runner test includes the full current H3 header and instantiates the bridge with actual GGMLRunner, TimeEmbedder and AdaLayerNormModulation; it does not run the full transformer. Native tests use actual H3 sinusoidal timestep_embedding/TimeEmbedder/SiLU. Curve tests use the same table/get_rows/lerp GGML formula and actual no-SiLU AdaLayerNormModulation. Both warm up EVERY plan and test complete coverage before denoising.

## Final test logs

Standalone strict compile/run, exit 0 (`/tmp/h3_cuda_adaln_test.log`):

```text
PASS identity: same exact hit; 23 identity/layout/schedule misses; reset; shape; finite
PASS validation/budget: finite plan; missing identity; overflow; exact ceiling; metadata/entry cap
PASS layout adapter: text tags/reference roles/final rows; staging budget; signed-zero bits
PASS import budget: graph staging payload ceiling fails closed
PASS tensor rejection: dtype/stride/shape/null backend/stale readback
PASS CPU tiny native(SiLU): full 2-layer+final GEMM/readback/import; bit exact; zero hit GEMMs; reset-safe staging
PASS CPU tiny curves(no SiLU): full 2-layer+final GEMM/readback/import; bit exact; zero hit GEMMs; reset-safe staging
H3 exact AdaLN tests passed (191 checks; CPU only, no weights downloaded)
```

Real H3/Runner CPU tiny compile/run, exit 0 (`/tmp/h3_cuda_adaln_real_test.log`; compiler log `/tmp/h3_cuda_adaln_real_build.log`):

```text
src/core/backend/ggml/ggml_extend.hpp:7106 [debug] H3-AdaLN-CPU-tiny params backend buffer size =   0.00 MB(RAM) (10 tensors)
src/core/backend/ggml/ggml_extend.hpp:7106 [debug] H3-AdaLN-CPU-tiny params backend buffer size =   0.00 MB(RAM) (11 tensors)
PASS identity: same exact hit; 23 identity/layout/schedule misses; reset; shape; finite
PASS validation/budget: finite plan; missing identity; overflow; exact ceiling; metadata/entry cap
PASS layout adapter: text tags/reference roles/final rows; staging budget; signed-zero bits
PASS import budget: graph staging payload ceiling fails closed
PASS tensor rejection: dtype/stride/shape/null backend/stale readback
PASS CPU tiny native(SiLU): full 2-layer+final GEMM/readback/import; bit exact; zero hit GEMMs; reset-safe staging
PASS CPU tiny curves(no SiLU): full 2-layer+final GEMM/readback/import; bit exact; zero hit GEMMs; reset-safe staging
PASS REAL H3 TimeEmbedder/SiLU AdaLayerNormModulation + GGMLRunner all-sigma prefill (CPU tiny)
PASS REAL H3 curve table/no SiLU AdaLayerNormModulation + GGMLRunner all-sigma prefill (CPU tiny)
H3 exact AdaLN tests passed (2659 checks; CPU only, no weights downloaded)
```

Additional memory tools attempted, NOT reported as passed:

- ASan/UBSan build could not link: `/usr/bin/ld: cannot find -lasan`. No install attempted.
- Valgrind could not start due to stripped dynamic linker/missing glibc debug redirection support. Details in `/tmp/h3_cuda_adaln_valgrind.log`. No process memory correctness claim based on this failed run.

`git diff --check` returned exit 0. The test suite caught and fixed a test-only lifetime issue: reading an unpinned hit intermediate after gallocr reused its buffer; tests now pin and read the actual copied graph outputs. Production miss capture already pins outputs.

Handoff complete. Main-thread integration, parser/CMake wiring, CUDA compile/device execution/performance validation remain main-thread work. This worker stops writing after this handoff.
