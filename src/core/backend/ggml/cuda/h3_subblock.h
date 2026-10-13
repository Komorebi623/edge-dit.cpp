#pragma once

// MiniMax-H3-only SubBlock adapter. No global attention flag, no CPU fallback.
// Arithmetic reference: SGLang 53295c697255a5e8ddb263f98ab7e047e2700ad2
// subblock_sparse/router.py + kernels.py (see tests/h3_subblock_integration.md).
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

struct ggml_backend_cuda_context;

namespace edgedit::h3_subblock {
constexpr uint32_t router_tag = 0x48335352u;    // H3SR
constexpr uint32_t attention_tag = 0x48335341u ^ 0x100u; // distinct from H3 Sage tag
constexpr int dimension = 128;
constexpr int block_size = 64;
constexpr int cell_size = 16;
constexpr int max_blocks = 1024;
constexpr int max_heads = 64;
constexpr int max_sequence = 65536;
constexpr size_t router_limit = size_t(192) << 20;
constexpr size_t gather_limit = size_t(128) << 20;
constexpr int max_batch = 64;

struct Capacity {
    size_t means_bytes = 0;    // Q and K means together
    size_t scores_bytes = 0;
    size_t route_bytes = 0;
    size_t batch_bytes = 0;    // explicit gather scratch, excludes stock flash's own scratch
    int batch = 0;
    int key_slots = 0;
};
// A real allocation bound, not a graph-cap increase. Checked before allocation.
inline bool capacity(int64_t sequence, int64_t heads, int64_t blocks,
                     int64_t topk, int64_t protected_tokens, Capacity* out) {
    if (!out || sequence < 1 || sequence > max_sequence || heads < 1 || heads > max_heads ||
        blocks < 1 || blocks > max_blocks || topk < 1 || topk > blocks ||
        protected_tokens < 0 || protected_tokens >= sequence ||
        blocks > sequence - protected_tokens ||
        sequence - protected_tokens > blocks * block_size) return false;
    Capacity c;
    c.means_bytes = size_t(heads * blocks * 4 * dimension) * sizeof(float) * 2;
    c.scores_bytes = size_t(heads * blocks * blocks) * sizeof(float);
    c.route_bytes = size_t(heads * blocks * topk) * sizeof(int32_t);
    if (c.means_bytes + c.scores_bytes > router_limit) return false;
    c.key_slots = int(std::max(sequence, topk * block_size + protected_tokens) + 63) / 64 * 64;
    // Q F32, K/V F16, per-batch mask F16 [NK,64], output F32.
    const size_t one = size_t(2 * dimension * block_size) * sizeof(float) +
                       size_t(2 * dimension + block_size) * c.key_slots * sizeof(uint16_t);
    if (one > gather_limit) return false;
    c.batch = int(std::min(size_t(max_batch), gather_limit / one));
    c.batch_bytes = one * size_t(c.batch);
    *out = c;
    return true;
}

struct Traffic {
    uint64_t sparse_jobs = 0, dense_jobs = 0;
    uint64_t sparse_batches = 0, dense_batches = 0;
    uint64_t kv_stage_bytes = 0, mask_stage_bytes = 0, q_stage_bytes = 0;
};
// Per attention layer / forward: stage *writes*, not resident memory or a
// bandwidth/time prediction. K/V source F32 reads, Flash rereads and final
// output are additional traffic. Tails are padded exactly as in the launcher.
inline bool traffic(int64_t sequence, int64_t heads, int64_t blocks, int64_t topk,
                    int64_t protected_tokens, int64_t protected_chunks, Traffic* out) {
    Capacity c;
    if (!out || !capacity(sequence,heads,blocks,topk,protected_tokens,&c) ||
        protected_chunks < 0 || protected_chunks > protected_tokens ||
        (protected_tokens==0) != (protected_chunks==0)) return false;
    Traffic t;
    t.sparse_jobs = uint64_t(heads*blocks);
    t.dense_jobs = uint64_t(heads*protected_chunks);
    t.sparse_batches = (t.sparse_jobs+c.batch-1)/c.batch;
    t.dense_batches = (t.dense_jobs+c.batch-1)/c.batch;
    const uint64_t sparse_slots = uint64_t((topk*64+protected_tokens+63)/64*64);
    const uint64_t dense_slots = uint64_t((sequence+63)/64*64);
    const uint64_t slots = t.sparse_jobs*sparse_slots+t.dense_jobs*dense_slots;
    t.kv_stage_bytes = slots*2*dimension*sizeof(uint16_t);
    t.mask_stage_bytes = slots*block_size*sizeof(uint16_t);
    t.q_stage_bytes = (t.sparse_jobs+t.dense_jobs)*block_size*dimension*sizeof(float);
    *out=t;
    return true;
}

// Same layout as ggml's CUSTOM params; inspect packed tag, never dereference
// userdata. The callback aborts deliberately: CPU scheduling is not a fallback.
struct CustomParams { ggml_custom_op_t fun; int n_tasks; void* userdata; };
inline uint32_t tag(const ggml_tensor* t) {
    if (!t || t->op != GGML_OP_CUSTOM) return 0;
    CustomParams p{};
    static_assert(sizeof(p) <= GGML_MAX_OP_PARAMS, "CUSTOM ABI");
    std::memcpy(&p, t->op_params, sizeof(p));
    return uint32_t(reinterpret_cast<uintptr_t>(p.userdata));
}
inline void no_cpu(ggml_tensor*, int, int, void*) {
    GGML_ABORT("H3 SubBlock native CUDA op cannot execute on CPU");
}
inline ggml_tensor* router(ggml_context* c, ggml_tensor* q, ggml_tensor* k,
                           ggml_tensor* blocks, int topk) {
    ggml_tensor* args[] = {q, k, blocks};
    auto* t = ggml_custom_4d(c, GGML_TYPE_I32, topk, blocks->ne[1], q->ne[2], 1,
                           args, 3, no_cpu, 1, reinterpret_cast<void*>(uintptr_t(router_tag)));
    ggml_set_name(t, "h3.subblock.native.perhead.topk");
    return t;
}
inline ggml_tensor* attention(ggml_context* c, ggml_tensor* q, ggml_tensor* k,
                              ggml_tensor* v, ggml_tensor* route, ggml_tensor* blocks,
                              ggml_tensor* protected_chunks, ggml_tensor* protected_ids) {
    // blocks/protected_chunks: I32 [2,count], each pair [begin,end).
    // protected_ids: I32 [max(1,protected_count)]; dummy -1 if no protected tokens.
    ggml_tensor* args[] = {q, k, v, route, blocks, protected_chunks, protected_ids};
    auto* t = ggml_custom_4d(c, GGML_TYPE_F32, dimension * q->ne[2], q->ne[1], 1, 1,
                           args, 7, no_cpu, 1, reinterpret_cast<void*>(uintptr_t(attention_tag)));
    ggml_set_name(t, "h3.subblock.native.attention");
    return t;
}
} // namespace edgedit::h3_subblock

// Device-free shape validation, callable in tests. Device capability is checked
// by supported() before the graph enters CUDA; execution uses that same GPU.
bool ed_cuda_h3_subblock_shape_supported(const ggml_tensor* dst);
bool ed_cuda_h3_subblock_supported(int device, const ggml_tensor* dst);
bool ed_cuda_h3_subblock_compute(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
