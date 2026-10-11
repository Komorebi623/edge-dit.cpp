# H3-only SubBlock progress — 2026-10-11

## Delivered / scope

Only these three files were written for this task:

- `/export/home/wangtianyang.21/code1/edge-dit.cpp/src/dit_models/models/minimax_h3_cuda_sparse.hpp`
- `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3_cuda_sparse_test.cpp`
- `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3-sparse-progress.md`

No edits to the main model, CMake or shared backend. No GPU execution, model
weights, process termination, downloads or duplicate source trees. Tests link
already-existing CPU libraries; small executables are under `/tmp`.

## Minimal integration contract

```cpp
#include "dit_models/models/minimax_h3_cuda_sparse.hpp"

// Keep config.graph_inputs with the graph until its LAST execution/replay.
// The runner binding callback registers raw pointers; it does not copy bytes.
MiniMaxH3::SparseAttentionConfig config; // block=64, keep=.5, warmup=2
config.sparse_block_tokens = main_options.sparse_block_tokens;
config.sparse_keep_ratio = main_options.sparse_keep_ratio;
config.sparse_warmup_steps = main_options.sparse_warmup_steps;

MiniMaxH3::SparseAttentionLayout layout;
int64_t offset = 0;
for (const auto& packed_segment : packed_segments) {
    const int64_t end = offset + packed_segment.actual_token_count;
    // Adapt your model's enum here; do NOT infer layout from total lengths.
    // TARGET_VIDEO -> TargetVideo
    // TEXT -> Text
    // TARGET_AUDIO / CONDITION_AUDIO -> Audio
    // CONDITION_VIDEO / reference video/image -> Reference
    // protected query/conditioning tokens -> Query
    const auto kind = classify_packed_segment(packed_segment);
    if (end > offset) layout.spans.push_back({offset, end, kind});
    offset = end;
}

// After H3 Q/K normalization and RoPE, before out_proj:
// q:[128,L,H,1], k:[128,L,H,1], v:[128,H,L,1], all F32.
// Existing non-contiguous QKV views are supported if dimension 0 is contiguous.
// Only invoke after YOUR warmup/step/layer gating; this helper owns no counters.
ggml_tensor* out = MiniMaxH3::h3_subblock_attention(
    ctx, q, k, v, heads, 1.f / 128.f, config, layout);
if (!out) {
    // Surface unsupported configuration/op explicitly. Do not label a silent
    // baseline fallback as sparse success. Caller decides its explicit policy.
    throw std::runtime_error("H3 SubBlock unavailable for this graph/backend");
}
// out:[128*H,L,1,1], same layout as ggml_ext_attention_ext; pass to out_proj.
```

`classify_packed_segment` / `packed_segments` above are integration pseudocode,
not new dependencies. Spans are **absolute packed-sequence [begin,end)**, must
cover every real token exactly once, in order. A reference video starting at any
offset is protected by tagging its span `Reference`; only `TargetVideo` can be
pruned. Reference queries are also dense. Empty spans should be omitted. Block
numbering is local to eligible video spans, never spans a protected boundary.
All eligible spans share the video-block candidate list for each head.

`SparseAttentionConfig` stays independent of the evolving main options type.
`sparse_topk=-1` means `ceil(keep_ratio * video_blocks)`; explicit topk 0, a
nonpositive/NaN keep ratio, topk beyond video blocks, non-Q64/K64 block size,
unknown/gapped/overlapping layout, missing target video, wrong dimension/shape,
wrong kv_scale or missing graph-input ownership/binding fail closed. Defaults
include warmup=2 only as metadata; warmup/layer/step remain caller-controlled.

**F32 inputs only in this graph implementation.** BF16 K scaling in the existing
baseline rounds in BF16 before its F16 cast; `get_rows` returns F32. F16 V takes
a different shared-helper preparation branch. Those dtype cases deliberately
return nullptr until explicitly matched, rather than claiming exact baseline
semantics. In particular, do not route the main model's low-memory BF16 path
through this version. Batch must be 1; no GQA, cross attention, additional masks,
or head sharing is implied.

## Algorithm / numerical semantics

Read the local upstream router and README, without copying their source tree:

- `/tmp/edge-dit-h3-research-20261011/sglang--python__sglang__multimodal_gen__runtime__layers__attention__backends__subblock_sparse__router.py`
- `/tmp/edge-dit-h3-research-20261011/sglang--python__sglang__multimodal_gen__runtime__layers__attention__backends__subblock_sparse__README.md`

For each **head separately** and each target-video Q64 block:

1. Pool real Q/K tokens into four 16-token means; partial cells divide by their
   **real** token count. Empty cells do not participate.
2. Score every valid query/key subcell pair with
   `dot(q_mean,k_mean) / sqrt(128)` and take stable log-sum-exp over pairs, yielding
   one F32 score per candidate video block. This is SubBlock mass estimation,
   not a shared-head average or a single whole-block mean dot product.
3. `ggml_top_k` selects the per-head/per-Q-block video KV budget dynamically on
   the graph. Expand block IDs through an I32 token table using `ggml_get_rows`,
   then gather K and V using those token indices. There is no activation readback
   or host-side fixed route. Top-k order is unspecified; boundary ties can select
   different equal-score blocks on different backends. CPU reference ties prefer
   lower block IDs; no tiny score bias disguises that difference.
4. Append **every** nonvideo KV, including text, target/condition audio, reference
   video/image and protected query segments. Nonvideo Q chunks attend the whole
   original KV sequence. Every real Q row gets an output, in original order.
5. Only block/token index tables and a small 16-pair router mask use padding;
   no full Q/K/V sequence padding. Ragged selected KV tails gather a safe real
   index and attach `-inf` validity masks so duplicate placeholders receive zero
   softmax mass. Masks cover selected chunks, not a full L×L dense attention mask.
6. Each head/chunk calls the existing `ggml_ext_attention_ext`, with the original
   **kv_scale=1/128**, not a custom attention implementation. Flash mode uses its
   K/V scaling, F16 preparation, `softmax_scale / kv_scale` and output ×128.
   Masked flash is preflighted, generated ops are checked with
   `ggml_backend_supports_op`, and requested flash must remain a FLASH_ATTN_EXT
   graph rather than silently falling back to a matmul path. Unsupported ops
   return nullptr, including TOP_K width limits on CUDA builds without CUB.

Important correction: H3's helper argument `1/128` is **kv_scale**, not softmax
scale. The router and mathematical reference use **1/sqrt(128)**. In ideal F32
math the KV factors cancel. Flash's F16 casts (and kernel accumulation/reordering)
do not cancel rounding, so the F32 reference is not a bitwise flash emulator.
The tests inspect actual FLASH_ATTN_EXT parameters and preceding K/V scale/cast
nodes, in addition to comparing numerical outputs to the real baseline helper.

## Real CPU tests (not only an oracle)

Completed using the existing CPU build, with two CPU threads. The graph tests
explicitly initialize the linked CPU device; no default GPU plugin loading.

```text
reference: all-selected max_abs=1.71363e-07
CPU ggml:
  all-selected vs F32 reference max_abs=1.2666e-07
  all-selected vs existing dense ggml max_abs=5.96046e-08
  selected sparse vs F32 reference max_abs=1.11759e-07
  all-selected flash vs existing flash baseline max_abs=0.00062272
H3 SubBlock sparse tests passed (CPU only)
```

CPU flash comparison tolerance is 1e-3, not bitwise equality. Short masked
chunks may select the CPU vector path (with F16 PV accumulation), unlike the
full baseline's tiled path; selected KV order also changes accumulation. The
semantic scale/cast checks remain strict. No GPU agreement/performance claim.

Coverage includes true SubBlock vs whole-block mean discrimination, opposite
head routing, nondivisible 65-token blocks, omitted padded subcells, gathered
KV tails, late audio/reference/query retention, mixed/interleaved layouts,
all Q updates, all-selected dense agreement, head_dim128/scale checks, large
finite stable-softmax logits, malformed configuration/layout, BF16 rejection,
missing binding, and simulated unsupported TOP_K/flash returning nullptr.
Projected-QKV **strided views** are exercised, not only contiguous tensors.
Each real sparse graph is also replayed after changing K (negating it), then
compared with a freshly computed F32 reference: router/gather stays dynamic.

### Exact CPU graph compile/run command used

```bash
ROOT=/export/home/wangtianyang.21/code1/edge-dit.cpp
c++ -std=c++17 -O1 -DGGML_MAX_NAME=128 \
  -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/src/core" \
  -I"$ROOT/third_party" -I"$ROOT/third_party/ggml/include" \
  "$ROOT/tests/h3_cuda_sparse_test.cpp" \
  "$ROOT/build-merge-cpu/lib/libedgedit.a" \
  -L"$ROOT/build-merge-cpu/bin" -Wl,-rpath,"$ROOT/build-merge-cpu/bin" \
  -lggml -lggml-cpu -lggml-base -pthread -o /tmp/h3-sparse-graph-test
/tmp/h3-sparse-graph-test
```

For the main thread's opt-in target, `PRIVATE edgedit ggml ggml-cpu` provides the
existing utils function (`sd_backend_is`) and CPU libraries; includes must also
cover `src`, `src/core`, `third_party`, and ggml's include directory because most
of edgedit's internal include paths are PRIVATE. `GGML_MAX_NAME=128` must match
the build. The test includes the test-only backend-impl header by its local
path, so no extra ggml/src include is needed. **Do not define reference-only**
on that target if you want graph execution. Existing CMake edits belong to the
main thread, not this task.

For `GGML_BACKEND_DL` builds, the test takes the explicit CPU module path as its
sole argument and loads only that module (not all backend plugins). Do not link
a MODULE target directly. This DL branch was not executed here.

### Standalone reference command (no torch, no ggml)

```bash
ROOT=/export/home/wangtianyang.21/code1/edge-dit.cpp
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  -DMINIMAX_H3_SPARSE_REFERENCE_ONLY -I"$ROOT/src" \
  "$ROOT/tests/h3_cuda_sparse_test.cpp" -o /tmp/h3-sparse-reference-test
/tmp/h3-sparse-reference-test
```

This command passed. ASan/UBSan build attempts could not link because this host
has no `libasan` / `libubsan`; sanitizer execution is **not** claimed. No runtime
or dependencies were installed to work around that limitation.

## Limits / honest performance status

This is a correctness/integration ggml-ops implementation, not the upstream
fused Triton/CuTe attention kernel and not a claimed throughput optimization.
Graph size is O(heads × video_query_blocks) chunks with many small pool/LSE/topk,
gather and attention ops. Per-chunk score scratch is O(video_blocks), but if a
scheduler retains all intermediates/gathered KV, graph memory can still be large;
profile/reserve against the caller's memory budget before enabling on long H3
sequences. No large-model/GPU allocation or end-to-end rendering was tested.
Graph capacity, layer/step gates, capability exposure and practical memory limits
remain with the integrating main thread. Static input buffers must be released
with the owning graph; failed attempts may leave allocated metadata in its ggml
context, so discard a failed graph build instead of repeatedly accumulating it.
