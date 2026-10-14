#pragma once
// SPDX-License-Identifier: Apache-2.0
// Bounded native ggml compression primitives, NOT full/production VSA. No CUDA
// routing, top-k, fine kernel, checkpoint qualification, or model feature switch.
// Contract: FastVideo video_sparse_attn_h3.py _pool_tiles / compression branch;
// Geometry is shared with the independent CPU oracle; no oracle math is called.
#include "ggml.h"
#include "minimax_h3_vsa_reference.hpp"
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>

namespace MiniMaxH3 { namespace VSAGraph {
inline constexpr int64_t max_tokens = 65536, max_tiles = 2048;
inline constexpr int64_t max_heads = 64, max_head_dim = 128;
inline constexpr size_t default_budget_bytes = size_t(4) * 1024 * 1024 * 1024;
inline constexpr int max_graph_nodes = 64; // excludes caller's input producers
inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
// One owner per graph, alive through all executions/replays. The callback has
// GGMLRunnerContext::bind_backend_tensor_data semantics: registers, NOT copies.
// CPU contexts with real storage copy immediately and need no callback.
struct GraphInputs {
    std::function<void(ggml_tensor*, const void*)> bind_backend_tensor_data;
    std::vector<std::unique_ptr<uint8_t[]>> buffers;
};
struct Result {
    ggml_tensor *pooled_q, *pooled_k, *pooled_v; // F32 [D,T,H]
    ggml_tensor *scores, *probabilities;         // F32 [key T,query T,H]
    ggml_tensor *compressed_tiles;              // F32 [D,T,H]
    ggml_tensor *compressed_rows, *output;      // F32 [D*H,S], original packed order
};
namespace detail {
inline size_t add(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, "VSA graph byte estimate overflow");
    return a+b;
}
inline size_t mul(size_t a, size_t b) {
    require(!a || b <= std::numeric_limits<size_t>::max()/a, "VSA graph byte estimate overflow");
    return a*b;
}
inline void shape(const ggml_tensor* t, int64_t a, int64_t b, int64_t c = 1) {
    require(t && t->type == GGML_TYPE_F32, "VSA graph requires F32 tensors");
    require(t->ne[0] == a && t->ne[1] == b && t->ne[2] == c && t->ne[3] == 1,
            "VSA graph layout mismatch (batch must be one)");
    require(t->nb[0] == sizeof(float), "VSA graph channel axis must be contiguous");
    for (size_t stride : t->nb)
        require(stride <= INT32_MAX, "VSA graph ggml permute stride exceeds int32");
}
template<class T>
inline ggml_tensor* constant(ggml_context* ctx, GraphInputs& owner, ggml_type type,
                             int64_t a, int64_t b, const std::vector<T>& values) {
    auto* t = ggml_new_tensor_2d(ctx, type, a, b);
    const size_t bytes = values.size() * sizeof(T);
    if (t->data) std::memcpy(t->data, values.data(), bytes);
    else {
        auto data = std::make_unique<uint8_t[]>(bytes);
        std::memcpy(data.get(), values.data(), bytes);
        const void* ptr = data.get();
        owner.buffers.push_back(std::move(data));
        ggml_set_input(t);
        owner.bind_backend_tensor_data(t, ptr);
    }
    return t;
}
// Input is [D,H,S]. Flatten heads into channels for ONE gather, not H gathers
// or an H-fold index map. Add an actual zero row; padding never samples token 0.
inline ggml_tensor* pool(ggml_context* ctx, ggml_tensor* x, ggml_tensor* slots,
                         ggml_tensor* mean_scale, int64_t d, int64_t h, int64_t s, int64_t t) {
    auto* rows = ggml_reshape_2d(ctx, ggml_cont(ctx, x), d*h, s);
    auto* tiled = ggml_get_rows(ctx, ggml_pad(ctx, rows, 0,1,0,0), slots);
    tiled = ggml_reshape_4d(ctx, tiled, d,h,64,t);
    // [D,H,64,T] -> contiguous [64,D,T,H], ggml_mean reduces axis zero.
    tiled = ggml_cont(ctx, ggml_permute(ctx, tiled, 1,3,0,2));
    auto* mean = ggml_reshape_3d(ctx, ggml_mean(ctx, tiled), d,t,h);
    return ggml_mul(ctx, mean, mean_scale); // mean * (64 / valid_count)
}
} // namespace detail

// Conservative sum of ALL newly owned tensor payloads (no allocator reuse),
// constant source copies, alignment and tensor metadata. Excludes caller inputs,
// graph container and backend workspace: NOT a peak-VRAM/graph-cut planner.
// A caller can explicitly raise the default budget for metadata-only planning.
inline size_t estimate_bytes(const VSAReference::Geometry& g, int64_t d, int64_t h) {
    require(g.tile_elems == 64, "VSA graph currently supports tile64 only");
    require(g.packed_tokens > 0 && g.packed_tokens <= size_t(max_tokens) &&
            g.tiles() > 0 && g.tiles() <= size_t(max_tiles), "VSA graph geometry bound");
    require(d > 0 && d <= max_head_dim && h > 0 && h <= max_heads, "VSA graph head/dimension bound");
    using detail::add; using detail::mul;
    const size_t t = g.tiles(), p = mul(t,64), dh = mul(size_t(d),size_t(h));
    const size_t x = mul(dh,g.packed_tokens), padded = mul(dh,p), pooled = mul(dh,t);
    const size_t scores = mul(size_t(h),mul(t,t));
    // This ggml version uses int temporaries for permute byte strides.
    require(mul(add(padded,dh),sizeof(float)) <= INT32_MAX, "VSA graph ggml permute stride exceeds int32");
    // Three pools: 6X+6P+6C+3DH; score/softmax: 3U;
    // pooled-V transpose/compress/transpose: 3C; broadcast/mul/add: 3X.
    size_t floats = add(add(mul(9,x),mul(6,padded)),add(mul(9,pooled),mul(3,scores)));
    floats = add(floats,mul(3,dh));
    const size_t constants = mul(add(add(p,g.packed_tokens),t),sizeof(int32_t));
    const size_t metadata = mul(max_graph_nodes+3,add(ggml_tensor_overhead(),GGML_MEM_ALIGN));
    return add(add(mul(floats,sizeof(float)),mul(2,constants)),metadata);
}

// Finite F32 activations, ggml axes fastest first: Q/K=[D,S,H], V=[D,H,S],
// raw_gate/fine_attention=[D*H,S]. Channel-contiguous strided views are accepted.
// Fine attention is INJECTED, already in packed order; this builds no fine path.
// All logical tiles participate (including text/audio/reference/video); no
// sparsity mask, dummy transport partner, sigmoid, or deletion of global tokens.
// O(1) graph nodes independent of H/T/S; storage O(D*H*T*64 + H*T*T).
// Hard bounds reject before graph allocation; D=1..128 allows small CPU tests,
// not a claim that a future native CUDA fine kernel supports these dimensions.
using FineBuilder = std::function<ggml_tensor*(ggml_tensor* scores)>;
inline Result build_with_fine(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                    ggml_tensor* raw_gate, const FineBuilder& fine_builder,
                    const VSAReference::Geometry& geometry, GraphInputs& inputs,
                    size_t budget_bytes = default_budget_bytes) {
    require(ctx && q, "VSA graph null context/query");
    require(!ggml_get_no_alloc(ctx) || bool(inputs.bind_backend_tensor_data),
            "VSA graph no-alloc context needs constant binding callback");
    const int64_t d = q->ne[0], h = q->ne[2];
    require(estimate_bytes(geometry,d,h) <= budget_bytes, "VSA graph tensor byte budget exceeded");
    const int64_t s = int64_t(geometry.packed_tokens), t = int64_t(geometry.tiles()), p = t*64;
    detail::shape(q,d,s,h); detail::shape(k,d,s,h); detail::shape(v,d,h,s);
    detail::shape(raw_gate,d*h,s);
    require(bool(fine_builder), "VSA graph needs explicit fine attention builder");
    VSAReference::validate_geometry(geometry);

    std::vector<int32_t> slots(static_cast<size_t>(p), static_cast<int32_t>(s)), row_tile(size_t(s), 0);
    std::vector<float> scales(size_t(t), 0.f);
    for (size_t i = 0; i < slots.size(); ++i)
        if (geometry.source_index[i] != VSAReference::invalid_token)
            slots[i] = int32_t(geometry.source_index[i]);
    for (size_t i = 0; i < row_tile.size(); ++i)
        row_tile[i] = int32_t(geometry.untile_combined_index[i] / 64);
    for (size_t i = 0; i < scales.size(); ++i)
        scales[i] = 64.f / float(geometry.variable_block_sizes[i]);
    auto* gather = detail::constant(ctx, inputs, GGML_TYPE_I32, p,1, slots);
    auto* restore = detail::constant(ctx, inputs, GGML_TYPE_I32, s,1, row_tile);
    auto* divisor = detail::constant(ctx, inputs, GGML_TYPE_F32, 1,t, scales);

    Result out{};
    out.pooled_q = detail::pool(ctx, ggml_permute(ctx,q,0,2,1,3), gather,divisor,d,h,s,t);
    out.pooled_k = detail::pool(ctx, ggml_permute(ctx,k,0,2,1,3), gather,divisor,d,h,s,t);
    out.pooled_v = detail::pool(ctx, v, gather,divisor,d,h,s,t);
    auto* dot = ggml_mul_mat(ctx, out.pooled_k, out.pooled_q);
    ggml_mul_mat_set_prec(dot, GGML_PREC_F32);
    out.scores = ggml_scale(ctx, dot, 1.f/std::sqrt(float(d)));
    out.probabilities = ggml_soft_max(ctx, out.scores);
    auto* vt = ggml_cont(ctx, ggml_permute(ctx,out.pooled_v,1,0,2,3)); // [T,D,H]
    out.compressed_tiles = ggml_mul_mat(ctx, vt, out.probabilities);
    ggml_mul_mat_set_prec(out.compressed_tiles, GGML_PREC_F32);
    auto* ct = ggml_cont(ctx, ggml_permute(ctx,out.compressed_tiles,0,2,1,3));
    ct = ggml_reshape_2d(ctx, ct, d*h,t);
    out.compressed_rows = ggml_get_rows(ctx, ct, restore);
    auto* fine_attention = fine_builder(out.scores);
    detail::shape(fine_attention,d*h,s);
    out.output = ggml_add(ctx, fine_attention, ggml_mul(ctx, out.compressed_rows, raw_gate));
    return out;
}
// Existing injected-fine CPU API stays unchanged. The callback overload lets
// native routing consume exactly the same pooled scores without a graph cycle.
inline Result build(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                    ggml_tensor* raw_gate, ggml_tensor* fine_attention,
                    const VSAReference::Geometry& geometry, GraphInputs& inputs,
                    size_t budget_bytes = default_budget_bytes) {
    require(q, "VSA graph null query");
    detail::shape(fine_attention,q->ne[0]*q->ne[2],int64_t(geometry.packed_tokens));
    return build_with_fine(ctx,q,k,v,raw_gate,[&](ggml_tensor*){ return fine_attention; },geometry,inputs,budget_bytes);
}
}} // namespace MiniMaxH3::VSAGraph
