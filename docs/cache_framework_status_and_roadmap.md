# Cache Framework Status Summary and Forward-Goal Analysis

> For the step-caching subsystem of `edge-dit.cpp`
>
> - Document status: status review + roadmap recommendations
> - Branch: `refactor/cache_framework`
> - Trigger: a code review targeting the cache framework
> - Intended audience: cache-algorithm developers, Runtime/Backend developers

---

## 1. Background and Scope

`refactor/cache_framework` refactors step-caching from the old `CacheController` into a declarative
`CacheEngine`. This document does two things:

1. **Take stock of the status**: which capabilities have actually landed, and what the strengths and gaps are respectively;
2. **Lay out the forward goal**: treat "landing the declarative engine" as the recommended direction, analyze its advantages and problems in depth, and list alternative directions.

This document uses the declarative architecture described in `docs/cache_framework_redesign.md` as the **target state**, measuring "already landed
vs. gaps". All "not load-bearing / dead code" assertions in the text have been verified by code search, with `file:line` attached.

> **Latest progress (2026-07-15): the decoupling of the cache layer from the model layer is complete.** A batch of commits after 07-14
> finished the "dehook" work that was originally listed as "weeks-scale, terminal state": the model forward no longer
> knows any cache concept, and instead calls the conditional `tap()` of `TapRegistry` at structural landmarks; the old
> `CacheGraphScope` seam has been **deleted wholesale**, and the `ED_CACHE_SUBSTEP`
> gate disappeared along with it -- **the substep loop became the only execution path**. §2.6, §4, §5 below have been rewritten accordingly;
> in §5's three-layer roadmap, the first two layers (StateManager / Operator) plus the original "Slice 4 dehook" have all landed,
> leaving only layer 3 (GraphRepository) and the SP+cache composition undone.
>
> **Follow-up (late 07): every method now has an on-device path.** SenCache reuse was routed onto the device
> single-residual slot and TaylorSeer's feature-history reuse onto the device ring, so no method is disabled on device-only runners (Flux/Qwen) any more — only SP still limits them
> to Output granularity (§4.1). The DiCache probe was pinned to no-return, and `CacheGraphLowering` was folded into `CacheEngine` as `run_substep_loop`
>; the removed `ED_*_GPU` toggles leave the device path as the only path.

> **Terminology: load-bearing vs. scaffolding.** "Load-bearing" means the component is called on the real inference path and removing it would change behavior;
> "scaffolding" means code that is implemented but currently has no caller, and removing it does not affect any existing functionality. One of the core conclusions of this document
> is precisely to distinguish which parts of the declarative design are load-bearing and which are scaffolding.

---

## 2. Status Summary (already landed)

### 2.1 Architecture

- `CacheEngine` (`src/core/optimization/cache/runtime/cache_engine.{hpp,cpp}`) replaces
  `CacheController`, keeping the `init / enabled / begin_step / end_step / run_branch /
  log_summary` interface; the end of the header transitions with the alias `using CacheController = CacheEngine;`,
  so pipelines connect just by changing the type name.
- 6 pipelines have connected to `run_branch`: flux, flux_kontext, qwen_image, qwen_image_edit,
  sd3, wan (`src/dit_models/pipelines/*_pipeline.cpp`).

### 2.2 Capability negotiation is real (load-bearing)

`CacheEngine::init` calls `validate_requirements` (`runtime/capability_negotiation.cpp`)
**before** `policy_->compile(...)`. An unsupported combination will explicitly
`LOG_WARN("cache disabled: ...")` and return false, replacing the old "silent full-compute"
fallback. The error message lists the sites the model actually exposes (`exposed_sites`), which is diagnosable.

### 2.3 8 policies migrated to `ICachePolicy`

`policy/policy_factory.cpp` dispatches: null / EasyCache / UCache / DBCache+CacheDiT
(sharing `condition_policy`) / TaylorSeer / MagCache / DiCache / SenCache, each one file in
`policy/policies/`.

### 2.4 Three granularities (load-bearing)

In `run_substep_loop` of `cache_engine.cpp` (the former `CacheGraphLowering::execute_substeps`, folded
into `CacheEngine` with no behavior change), dispatch is done by `SubstepOpKind` (the policy's
`next_substep()` produces a `SubstepPlan` substep by substep, which the middle layer translates into a tap-driven runner pass):

- **Output**: black box, uses only `hooks.full()` + host-side output diff/reuse, also usable under SP;
- **Feature**: captures/injects residuals at the ModelIn/ModelOut anchors via `TapRegistry` (MagCache, TaylorSeer,
  SenCache);
- **Probe**: runs a shallow prefix (`stop_after`) and then decides (DiCache).

### 2.5 Performance optimizations are genuinely effective

See `docs/cache_quality_benchmark_report.md` for details (H200 / 50 steps / Flux+Qwen):

- on-GPU MagCache feature reuse, DiCache GPU probe rewrite, `ED_CACHE_COMPILED_GRAPHS`
  build-once graph reuse;
- skip cost dropped from ~40ms/skip to ~1ms, verified byte-identical, quality unchanged;
- Flux: MagCache 2.05x, DiCache default 1.24x (near-lossless, PSNR ~40).
- **2026-07-14 update**: the default on-GPU feature reuse for Flux and Qwen now uniformly goes through the **declarative device slot**
  (`CacheStateManager` device backend), having replaced the legacy `DiCacheGpuState` path; device slot vs.
  host-declarative verification is **byte-identical (PSNR 100 dB)**, with skip counts bitwise-identical (see §5.5).
  Note: this round of Qwen verification requires `ED_QWEN_SINGLE_FUSED_ATTENTION=0` (the default fused attention produces
  a pure-white image in the current CUDA-13 environment, which is a pre-existing environment issue unrelated to cache).

### 2.6 cache/model decoupling complete: TapRegistry replaces the seam (load-bearing, 2026-07-15)

The original design doc listed "fully dehook" as the terminal state, a weeks-scale Slice 4. **It is already done**:

- **The dependency direction has been cleanly inverted**: the cache layer (`src/core/optimization/cache/`) includes **zero concrete
  model headers** (flux/qwen/wan/mmdit), depending only on the abstractions exposed by the model side -- `anchor` / `cache_site` /
  `model_topology` / `model_schema` / `tap_registry` / `model_cache_contract`. Verified by grep.
- **The model forward no longer knows about cache**: the forward of the 4 models (flux/qwen_image/mmdit/wan) calls the conditional
  `tap()` of `TapRegistry` (`ctx->tap_registry`) at structural landmarks -- a no-op unless requested, not pinning
  buffers, gallocr unaware. Which tensors the cache needs to read is decided by the per-substep `SubstepPlan.taps`,
  not by model enumeration fields. inject (reuse injection) also goes through the registry: the forward replaces the stream
  at `inject_at(i)` and jumps to `inject_resume()`.
- **The `CacheGraphScope` seam has been deleted wholesale**: the old fixed `*_node`
  fields, `kCache*Name` constants, `expand_cache_scope_nodes`, the runner's `cache_scope_` member +
  `set_cache_scope`, and the `DiCacheGpuState` standalone GPU state path have all been removed. Only a pure result struct
  `sd::DiffusionCacheResult` remains (host-readback output/feature/before/probe + DiCache scalar).
- **substep is the only path**: the `ED_CACHE_SUBSTEP` gate disappeared, `run_branch` unconditionally goes through
  `run_substep_loop` (`cache_engine.cpp:786`); the old `execute` and its
  large hook if/else have been deleted. All 8 methods implement `next_substep()`.
- **DiCache probe has also been tap-ified**: the checklist item once marked "probe not migrated" is complete -- the probe's
  cross-step persistent tensors (prev_probe/probe_prev1/2) are threaded into indicator lowering via
  `TapRegistry::ProbeMetricOperands`, and delta_y/delta_x/gamma are reduced in-graph then read back as scalars.

---

## 3. Strengths

1. **The decoupling goal is achieved**: adding a cache method = writing one `ICachePolicy`; adding a model = exposing one
   `DiTModelCacheContract` + calling `TapRegistry::tap()` at the forward's structural landmarks (ModelIn/
   BlockOut[i]/ModelOut). It escapes the "number of models × number of methods" combinatorial explosion, and the model side no longer includes or
   knows any cache concept (see §2.6).
2. **Failure is explicit**: under SP-parallel or when the graph cannot be cut, Feature/Probe methods are correctly rejected by capability negotiation,
   rather than silently running as no-cache.
3. **CacheProgram is dumpable** (`ED_DUMP_CACHE_PROGRAM`), policy decisions are observable.
4. **The GPU path is faster with quality unchanged**: skip ~1ms, byte-identical.
5. **CFG-parallel + cache is usable**: cond/uncond each use `condition_key` (condition struct address) +
   `CacheBranch` tag to isolate state (`cache_slot.hpp:26`, `cache_state_manager.cpp:8`
   `key_of`, `magcache_policy.cpp` `branch_for`). A cache hit only skips the local forward, and
   `cfg_all_gather` still executes unconditionally to keep synchronization (aligned with `CLAUDE.md`).

---

## 4. Weaknesses / Gaps

### 4.1 Problems in the load-bearing layer

- **Coarse-grained Option-A topology**: `dit_model_cache_contract.cpp` exposes only a single `BLOCK_STACK`
  segment (input projection / whole-block stack / output projection). No per-block, attention/FFN, or token-level
  site. The `CachePhase::{FULL_ANCHOR,CORRECTION,REINTEGRATION}` in `cache_program.hpp`
  are only placeholders; in practice only `FORWARD/PROBE` are used (stages three/four of the `redesign` doc are not done).
- **Feature/Probe are entirely disabled under SP-parallel**: `feature_cache_available()` is defined as
  `!can_attempt_graph_cut_segmented_compute()` (`ggml_extend.hpp:6991-6992`), and the latter is true whenever
  "any process group is enabled" -- so pure SP also disables the tap capture path (the pipeline passes
  `cache_seam_available = !cfg_parallel && feature_cache_available()` into `CacheEngine::init`),
  leaving only Output granularity under SP. This is the direct manifestation that the engine's two big selling points (multi-GPU parallel + step caching) currently cannot be stacked,
  and is the **biggest remaining capability gap** after decoupling.
- **~~hook coupling is fragile~~ (resolved along with dehook)**: `gpu_metric`/`branch_key` once silently zeroed out all Feature
  cache (MagCache/TaylorSeer/SenCache), running them as 0/N. That fragility stemmed from the old `CacheGraphScope`'s
  fixed `*_node` fields + dual-write path; the dehook in §2.6 has deleted that seam, and taps are actively registered by the model per request set,
  so the structural cause of this class of regression is eliminated. This item is retained as a historical lesson: when touching tap/lowering, still accept by "skip count +
  PSNR ≥ floor".
- **The cache path once had a GPU memory leak** (fixed); long runs still need chunked generation to work around it.

### 4.2 The declarative machinery: load-bearing vs. still-scaffolding (2026-07-15 update, all grep-verified)

The design doc's three most ambitious declarative machines have advanced from "only the decision layer landed" -- **the first two are now load-bearing**:

- **`CacheStateManager` is load-bearing (history ring + device slot backend)**: `CacheEngine::init` passes
  `state_` into `run_substep_loop` (`cache_engine.cpp:786`), and the runner's
  `ActionInterpreter` really calls `state_.read_history` / `state_.write` / `state_.rotate_history`
  (`cache_engine.cpp`), with the device-slot path calling `state.read` + `state.alloc_device_entry`.
  TaylorSeer's feature-history ring (host, or on-device on a GPU runner with a store) and MagCache/SenCache's **device-side single residual slot** all run on top of it.
  What is still scaffolding: `commit_step` (`cache_state_manager.cpp`) / `rollback_step`
  are still **explicit no-ops** -- but this is **honest**: the whole codebase has no `rollback_step` caller, `commit_step` is only
  called once unconditionally by `end_step`, and no cache method aborts midway (failure takes the synchronous full-compute
  fallback, not involving state rollback). The so-called "transactional rollback safety" currently has no trigger point; it is an interface placeholder reserved for future fine-grained/token-cache
  work, not a load-bearing capability. Each policy still holds its own scalar decision state (`states_` map), which is **decision
  logic** (belongs in the policy), separated from the **tensor storage** that StateManager manages.
- **`CacheOperatorRegistry` + operators are load-bearing**: `CacheEngine` instantiates the registry and calls
  `register_builtin_cache_operators` (`cache_engine.cpp:701`, previously dead code); the runner's
  DIFFERENCE/PREDICT/BLEND actions are routed via `operators_.find(...)` to `DifferenceOperator/
  LinearPredictOperator/WeightedBlendOperator::apply_host`.
  EasyCache/UCache's output diff and TaylorSeer's history-extrapolation blend are all executed by operators.
- **Still no `GraphRepository` / multiple static-graph variants**: `GraphRepository`, `GraphVariantKey`,
  `get_or_compile` have **zero hits** in `src/`, `examples/`, and all docs except redesign.
  The design doc's core idea "at runtime, choose among multiple pre-compiled static graph variants" **is not implemented**. The actual execution is
  tap-driven runner-pass dispatch: `run_substep_loop` branches by `SubstepPlan.op.kind` to call
  `hooks.full` and the tap-driven `hooks.substep_capture/substep_probe/substep_inject_*` (device
  or host). So `CacheProgram`/`GraphVariantPlan` are "decision descriptors + action sequences" (telling the lowering
  which substep to produce and which slot/operator actions to execute), not compilation artifacts.

> **Summary**: among the declarative components, `CacheProgram`/`GraphVariantPlan` (descriptors + actions),
> the `CacheRunnerHooks` lowering, the `CacheStateManager` history/device-slot backend, and `CacheOperatorRegistry`
> are all load-bearing -- the 8 methods (including Flux/Qwen's default device-side GPU reuse) genuinely run on StateManager+Operator;
> and the model side is fully dehooked (§2.6). What is still scaffolding: only `GraphRepository`/multi-variant compilation (not built),
> and the transactional semantics of `commit_step`/`rollback_step` (honest no-ops, no trigger point, reserved for the future).

---

## 5. Recommended Forward Goal: Land the Declarative Engine

**Direction**: actually connect the three scaffolding pieces in §4.2, delivering the core promises of `cache_framework_redesign.md`.
**Status (2026-07-14)**: the first two layers have landed as load-bearing, leaving only layer 3.

### 5.1 Concrete meaning (layered, each layer independently accepted by "skip count + PSNR ≥ floor")

1. **StateManager becomes load-bearing ✅ done**: `CacheStateManager` is already the real state backend for TaylorSeer / DiCache /
   MagCache / Qwen (host history ring + device-side single residual slot), replacing the private `states_` of these methods
   at the tensor-storage level. Transactional commit/rollback are still honest no-ops (no trigger point, see §4.2).
2. **Operator-izing the math ✅ done**: residual diff / prediction extrapolation / blend are routed to `CacheOperator`,
   policies reference operator ids, and `register_builtin_cache_operators` is genuinely called.
3. **GraphRepository + variant compilation ⬜ not done**: introduce real FULL/REUSE/PREDICT static-graph variant compilation and
   caching, and fuse it with the build-once graph reuse of `ED_CACHE_COMPILED_GRAPHS` (both want to "build fewer graphs").
   This is the only remaining declarative layer, and the natural downstream of SP composition / fine-grained.

### 5.2 Benefits already delivered (layers 1-2) and remaining value (layer 3)

Already delivered:
- the 8 methods + Flux/Qwen default device-side GPU reuse run on StateManager+Operator, and `CacheProgram`
  is no longer just a decision descriptor;
- operator-ization makes onboarding new methods faster (compose existing operators, rather than inlining new math each time);
- the device slot compresses skip cost to ~1ms with byte-identical verification (see §2.5, §5.5).

Remaining value (layer 3):
- lays the foundation for fine-grained / trajectory correction / token cache (needs real multi-variant static graphs);
- merging with compiled-graph reuse may further reduce graph-build overhead;
- transactional rollback safety currently has **no trigger point**, and is not a benefit -- unless a method that aborts midway appears in the future.

### 5.3 Problems / Risks (not avoided)

- **Layers 1-2 have proven "scaffolding can become load-bearing"**: device-slot acceptance is byte-identical (PSNR 100 dB), skip counts are bitwise-identical,
  proving that "connecting the scaffolding" does bring concrete benefits (skip ~1ms, a single GPU reuse path). This dissolves the original concern of "pure tidying,
  no user-visible benefit" -- this document still argues to **treat landing as a means, not an end**, binding each layer to a concrete benefit.
- **Over-engineering risk (layer 3)**: `GraphRepository` / multi-variant overlaps in responsibility with the existing `PlanCache` /
  `compute_reuse` (`ED_CACHE_COMPILED_GRAPHS`) on GGML static graphs. The boundary must be clarified before landing,
  otherwise it is two sets of "graph cache" fighting each other (see the graph-build overhead analysis in `perf_gap_vs_diffusers.md`).
- **High regression risk**: layer 3 touches the most fragile graph-cut / PlanCache subsystem. The hook-coupling surface that once caused incidents
  has been resolved along with the dehook in §2.6 (seam deleted, taps actively registered per request set), but tap/lowering still needs care,
  and where touched should be accepted by "skip count + PSNR ≥ floor".
- **Deferred benefit**: only layer 3 (GraphRepository) is directly tied to user-perceptible capabilities like SP composition and fine-graining.

### 5.4 The first minimal slice (done, as a proof point that "scaffolding can be load-bearing")

Original plan: connect `CacheStateManager` only as the history backend for **TaylorSeer** (depth `n_derivatives+1`),
a small change surface with clear boundaries, as a proof point. **Actual result**: that slice + the subsequent 7 methods + Flux/Qwen
default device-side GPU reuse all landed and were verified equivalent (skip counts bitwise-identical + PSNR ≥ floor; device slot vs. host
even reaches byte-identical 100 dB). The proof point holds -- scaffolding can indeed be load-bearing, and it brought concrete benefits (skip ~1ms, a single
GPU reuse path), so layers 1-2 have been advanced to completion. **The next decision point moves to layer 3**: GraphRepository and
`ED_CACHE_COMPILED_GRAPHS` overlap in responsibility, and the boundary must be clarified before landing (§5.3), otherwise it is better to stop.

### 5.5 Implementation progress (2026-07-15, layers 1-2 + model dehook done and verified)

> **Acceptance-baseline correction**: §5.1/§5.4 originally wrote "byte-identical", which measurement disproves -- this engine's CUDA backend is **frame-to-frame
> nondeterministic** (the same no-cache config run twice yields PSNR ≈ 35 dB, not 100%). So equivalence acceptance is changed to
> **① skip counts bitwise-identical (exact reproduction, directly catching decision regressions) + ② output PSNR ≥ the
> ~35 dB noise floor of the no-cache self-comparison**. md5 comparison is the wrong method.

**Landed and verified equivalent (8/8 methods + default device-side GPU reuse + model dehook):**

- **Capability wiring**: `CacheEngine` instantiates `CacheOperatorRegistry` and calls
  `register_builtin_cache_operators` (previously dead code), passing `state_` + `operators_` + optional
  `ICacheDeviceStore` into `run_substep_loop`.
- **Unified substep path**: all 8 methods implement `next_substep()`, producing `SubstepPlan` substep by substep;
  the middle layer dispatches by `SubstepOpKind` to a tap-driven runner pass. The `ED_CACHE_SUBSTEP` gate and the old
  `execute`'s large hook if/else have been deleted.
- **Output (EasyCache/UCache/DBCache/CacheDiT)**: `OutputCompute` (FULL then DIFFERENCE→slot) /
  `OutputReuse` (LOAD+BLEND→output). Skip counts unchanged.
- **Feature path (MagCache/TaylorSeer/SenCache)**: `FeatureCompute`/`FeatureReuse`, served on the device
  ring/slot when a GPU store is wired (MagCache/SenCache single residual; TaylorSeer feature-history ring:
  device ring + ROTATE_HISTORY + per-step reuse_coeffs blend) and on the host ring on
  CPU/SP runners. SenCache was routed onto the device slot too, fixing its earlier
  fake skips on device-only runners.
- **Probe (DiCache)**: `stop_after` shallow prefix + residual ring; probe has been tap-ified (delta_y/
  delta_x/gamma are reduced in-graph via `TapRegistry::ProbeMetricOperands` then read back as scalars). The
  probe pass is pinned to no-return: it reads only the indicator scalars, so the
  discardable model-output D2H copy is skipped (the dead `SubstepPlan.input`/`InputSource` was dropped too).
- **Default device-side GPU reuse (MagCache/SenCache/TaylorSeer on a GPU runner) is load-bearing and verified**:
  `CacheStateManager` owns a per-slot persistent device buffer (`RunnerCacheDeviceStore`,
  `ggml_extend.hpp:1975`), storing residuals d2d via `substep_capture` and reusing them in-graph via `substep_inject_slot`'s
  `ggml_add(x_before, slot)`. The device path is now the only path — the `ED_*_GPU` toggles were removed
  and both Flux **and Qwen** long since migrated off the legacy `DiCacheGpuState` path
  to the device slot. **Acceptance (2026-07-14)**: device slot vs. host-declarative is **byte-identical (PSNR 100 dB)** --
  Flux MagCache 30/50, Qwen MagCache 36/50, skip counts bitwise-identical, and the two paths vs. no-cache are
  28.4 dB / 25.2 dB respectively (consistent with the benchmark report).
- **Model dehook done (§2.6)**: the `CacheGraphScope` seam was deleted wholesale,
  the 4 model forwards were changed to `TapRegistry` conditional taps, and the cache layer has zero dependency on concrete model headers.

**Remaining work:**

- **`GraphRepository` / multiple static-graph variants**: layer 3 of the redesign, not built. This is the natural downstream of SP + cache composition
  and fine-grained cut points.
- **SP + cache composition**: under SP, Feature/Probe are still rejected by capability negotiation (§4.1), leaving only Output granularity.
  After decoupling, this is the **biggest user-perceptible gap**.
- **Transactional semantics**: honest no-op, no trigger point (see §4.2). Reserved for the future, not current debt.

**Conclusion**: the declarative engine has moved from "scaffolding" to "load-bearing" -- the 8 methods + Flux/Qwen default device-side GPU reuse
genuinely run on StateManager+Operator, and `CacheProgram` is no longer just a decision descriptor; moreover **the model side is fully dehooked**
(the Slice 4 originally listed as "weeks-scale terminal state" is done, the seam deleted, substep the only path). What is still scaffolding is only
`GraphRepository` (not built) and the transactional semantics (honest no-op). Each step continues to be accepted by "skip count + PSNR ≥ floor"
(CUDA is frame-to-frame nondeterministic, md5 comparison is wrong), and where tap/lowering is touched, keep the fallback preferentially.

---

## 6. Alternative Directions (brief, not recommended)

- **SP + cache composition**: highest value, best fits the engine's dual selling points (multi-GPU + step caching currently cannot be stacked). The difficulty = making the
  graph-cut planner preserve the tap's named intermediate outputs across segments -- `ggml_extend.hpp:6990`
  explicitly documents this conflict: "mid-graph capture is not preserved across segments". This is the natural downstream of "landing the
  declarative engine" layer 3 (GraphRepository/variant compilation), best done afterward. Output-granularity caching
  can in theory catch the SP ride earlier (no tap capture needed).
- **Consolidate and harden**: delete or explicitly annotate the remaining scaffolding, refresh the full benchmark (the Qwen row is still old data).
  Low risk, adds no capability, can serve as a prerequisite for any direction. (Note: the once-fragile hook coupling has been resolved along with the dehook in §2.6.)
- **Fine-grained cut points**: expand the topology from a single `BLOCK_STACK` to per-block / attention·FFN / token level,
  unlocking partial-compute and token cache. Research frontier, high ceiling, but every model needs contract + seam changes,
  and the test-matrix combination explodes.

---

## 7. Reference Documents (references not repeated)

| Document | Content |
|---|---|
| `cache_framework_design.md` | Analysis of the upstream xllm `dit_cache` framework and its DiT integration; §13 proposes the next-gen interface |
| `cache_framework_redesign.md` | Declarative-refactor **design draft** (the target state partially implemented on this branch) |
| `cache_quality_benchmark_report.md` | PSNR/SSIM/LPIPS + speed of MagCache/DiCache vs. no-cache |
| `perf_gap_vs_diffusers.md` | Root causes of this engine being slower than diffusers (step-by-step graph build, no CUDA Graph, GPU↔CPU round trips) |
| `perf_improvement_plan.md` | Execution plan targeting the above root causes |
