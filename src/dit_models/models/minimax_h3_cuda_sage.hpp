#pragma once

// H3 graph-side adapter only. This does not implement attention or enable the
// process-wide Sage gate. CUDA must explicitly accept the H3 userdata tag.
#include "backend/ggml/ed_ggml_sage_attn_ext.hpp"
#include "backend/ggml/cuda/ed_cuda_sage_attn.h"
#include "dit_models/models/minimax_h3_cuda_options.hpp"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <string>

namespace MiniMaxH3 {

// Shared ABI: low 32 bits = H3-scoped, explicit opt-in; high 31 bits
// = n_head, exactly like SAGQ. No pointer to model/request-owned storage.
constexpr uint32_t kH3SageCustomMagic = edgedit::ggml_ext::kSageAttnH3CustomMagic;
constexpr int64_t kH3SageHeadDim = 128;
constexpr float kH3SageKvScale = 1.0f / 128.0f;
constexpr float kH3SageOutputScale = 128.0f;
static_assert(sizeof(uintptr_t) >= 8, "H3 Sage userdata requires a 64-bit host");
static_assert(kH3SageCustomMagic != edgedit::ggml_ext::kSageAttnCustomMagic,
              "H3 must not share the global Sage tag");

inline float h3_sage_sm_scale() { return 1.0f / std::sqrt(float(kH3SageHeadDim)); }

inline bool h3_sage_fp8_enabled() {
    const char* value = std::getenv("ED_H3_SAGE_PV_FP8");
    return value && value[0] && std::strcmp(value,"0") && std::strcmp(value,"false") &&
           std::strcmp(value,"FALSE") && std::strcmp(value,"off") && std::strcmp(value,"OFF");
}

inline void* h3_sage_params_to_userdata(int64_t n_head, bool fp8 = false) {
    if (n_head <= 0 || n_head > std::numeric_limits<int32_t>::max()) return nullptr;
    const uintptr_t packed = uintptr_t(fp8 ? ED_CUDA_H3_SAGE_FP8_MAGIC : kH3SageCustomMagic) | (uintptr_t(n_head) << 32);
    return reinterpret_cast<void*>(packed);
}

inline bool h3_sage_params_valid(const edgedit::ggml_ext::SageAttnCustomParams& params) {
    return edgedit::ggml_ext::sage_attn_h3_params_valid(params) ||
           (params.magic == ED_CUDA_H3_SAGE_FP8_MAGIC && params.n_head > 0);
}

enum class H3SageScope { TokenRefiner, DiT };
enum class H3SageRoute { Dense, Sage, Reject };

struct H3SageRequest {
    bool enabled = false; // supplied from parsed ED_H3_SAGE_ATTN, never ED_SAGE_ATTN
    bool is_cuda = false;
    bool flash_attn = false;
    bool subblock_enabled = false; // caller must pass its ACTUAL execution mode
    bool pv_fp8 = false; // unvalidated alternative shared PV path
    bool dense_fallback = false; // prohibit hybrid/dense recomputation being reported as Sage
    H3SageScope scope = H3SageScope::TokenRefiner; // missing layer metadata stays dense
    int layer_idx = -1;
    int total_layers = -1;
};

struct H3SageDecision {
    H3SageRoute route;
    const char* reason;
};

inline H3SageDecision h3_sage_decide(const H3SageRequest& request) {
    if (!request.enabled) return {H3SageRoute::Dense, "ED_H3_SAGE_ATTN is off"};
    if (!request.is_cuda) return {H3SageRoute::Reject, "ED_H3_SAGE_ATTN requires CUDA"};
    if (request.subblock_enabled)
        return {H3SageRoute::Reject, "H3 Sage + subblock is not validated"};
    if (request.pv_fp8)
        return {H3SageRoute::Reject, "H3 Sage + ED_SAGE_PV_FP8 is not validated"};
    if (request.dense_fallback)
        return {H3SageRoute::Reject, "H3 Sage forbids ED_SAGE_FALLBACK; no hidden dense recomputation"};
    if (request.scope == H3SageScope::TokenRefiner)
        return {H3SageRoute::Dense, "H3 token refiner stays dense"};
    if (request.total_layers <= 0 || request.layer_idx < 0 ||
        request.layer_idx >= request.total_layers)
        return {H3SageRoute::Reject, "H3 Sage needs valid DiT layer metadata"};
    // Initial rollout: ALWAYS preserve first and last DiT block. Do not inherit
    // ED_SAGE_SKIP_FIRST/LAST or use pe != nullptr to infer token-refiner scope.
    if (request.layer_idx == 0 || request.layer_idx == request.total_layers - 1)
        return {H3SageRoute::Dense, "H3 first/last DiT block stays dense"};
    if (!request.flash_attn) return {H3SageRoute::Reject, "H3 Sage requires flash attention; refusing silent dense"};
    return {H3SageRoute::Sage, "H3-scoped Sage candidate"};
}

inline bool h3_sage_cuda_backend(ggml_backend_t backend) {
    if (backend == nullptr) return false;
    const char* name = ggml_backend_name(backend);
    if (name == nullptr || std::strncmp(name, "CUDA", 4) != 0) return false;
    for (const char* p = name + 4; *p != '\0'; ++p)
        if (*p < '0' || *p > '9') return false;
    return true;
}

inline H3SageRequest h3_sage_request(const CudaOptimizationOptions* options,
                                    ggml_backend_t backend, bool flash_attn,
                                    H3SageScope scope, int layer_idx, int total_layers,
                                    bool subblock_enabled) {
    H3SageRequest request;
    request.enabled = options != nullptr && options->sage_attn;
    request.is_cuda = h3_sage_cuda_backend(backend);
    request.flash_attn = flash_attn;
    request.scope = scope;
    request.layer_idx = layer_idx;
    request.total_layers = total_layers;
    request.subblock_enabled = subblock_enabled || (options != nullptr && options->sparse_attn);
    request.pv_fp8 = edgedit::ggml_ext::sage_pv_fp8_enabled();
    request.dense_fallback = edgedit::ggml_ext::sage_fallback_enabled(kH3SageHeadDim);
    return request;
}

// Contract at H3 Attention::forward after QK norm/partial RoPE:
// q/k [128, L, heads*batch, 1]; v [128, heads, L, batch], or the existing
// packed [128, L, heads, 1] layout for batch=1. No mask, GQA, BF16 or cross-attn.
inline bool h3_sage_shape_supported(const ggml_tensor* q, const ggml_tensor* k,
                                    const ggml_tensor* v, int64_t n_head,
                                    const ggml_tensor* mask = nullptr) {
    if (mask != nullptr || !edgedit::ggml_ext::sage_attn_shape_supported(q, k, v, n_head))
        return false;
    if (h3_sage_params_to_userdata(n_head) == nullptr || q->ne[0] != kH3SageHeadDim ||
        q->ne[3] != 1 || k->ne[3] != 1 || q->ne[1] != k->ne[1]) return false;
    // CUDA grid.z flattens heads*batch; sequence arithmetic uses int and u32
    // strides (including ceil-to-128/64 padding). Guard before any narrowing.
    constexpr int64_t max_seq = std::numeric_limits<uint32_t>::max() / kH3SageHeadDim;
    if (q->ne[1] > max_seq || q->ne[1] > std::numeric_limits<int>::max() - 127 ||
        q->ne[2] > 65535) return false;
    for (const ggml_tensor* t : {q, k, v})
        if (t->nb[0] != ggml_type_size(t->type)) return false;
    return true;
}

inline ggml_tensor* h3_sage_cont_f32(ggml_context* ctx, ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_F32) tensor = ggml_cast(ctx, tensor, GGML_TYPE_F32);
    // ggml_is_contiguous ignores singleton-axis strides; SCALE requires the
    // stricter padded-1D invariant even for batch=1 offset views.
    if (!ggml_is_contiguous(tensor) || tensor->nb[2] != tensor->nb[1] * tensor->ne[1] ||
        tensor->nb[3] != tensor->nb[2] * tensor->ne[2]) tensor = ggml_cont(ctx, tensor);
    return tensor;
}

// CPU currently claims generic CUSTOM ops are supported, contrary to the Sage
// header's no-op-stub comment. An accidental CPU scheduling must fail loudly,
// not return an uncomputed tensor. This is NOT an attention implementation.
inline void h3_sage_cpu_custom_op_stub(ggml_tensor*, int, int, void*) {
    GGML_ABORT("H3-scoped Sage attention requires its CUDA dispatch; CPU execution is forbidden");
}

// GRAPH CONSTRUCTION ONLY: safe to inspect on CPU, never compute this custom
// node on CPU. The runtime entry below additionally probes the CUDA backend.
inline ggml_tensor* h3_sage_custom_graph(ggml_context* ctx, ggml_tensor* q,
                                        ggml_tensor* k, ggml_tensor* v, int64_t n_head, bool fp8 = false) {
    if (ctx == nullptr || !h3_sage_shape_supported(q, k, v, n_head)) return nullptr;
    const int64_t batch = q->ne[2] / n_head;
    const int64_t sequence = q->ne[1];
    // run_sage<T> chooses T from Q and also reinterprets K/V as T. Its shared
    // shape predicate allows mixed types; normalize ALL THREE here to F32.
    q = h3_sage_cont_f32(ctx, q);
    k = h3_sage_cont_f32(ctx, k);
    if (!edgedit::ggml_ext::sage_attn_v_is_4d(v, kH3SageHeadDim, n_head, sequence, batch))
        v = ggml_permute(ctx, v, 0, 2, 1, 3); // packed -> token-major
    v = h3_sage_cont_f32(ctx, v);
    v = ggml_scale(ctx, v, kH3SageKvScale);
    ggml_set_name(v, "minimax_h3.sage.v_conditioned");

    // Important: Q and K are NOT scaled. The existing kernel's fixed
    // 1/sqrt(128) is already H3's effective logit scale. Scaling K by 1/128
    // would require an unavailable scale override, or compensating Q by 128.
    // Remove the algebraically cancelling K conditioning instead; retain V
    // conditioning through the kernel's F16 PV/output and undo it afterwards.
    ggml_tensor* args[] = {q, k, v};
    auto raw = ggml_custom_4d(ctx, GGML_TYPE_F32, kH3SageHeadDim, n_head * batch,
                              sequence, 1, args, 3, h3_sage_cpu_custom_op_stub, 1,
                              h3_sage_params_to_userdata(n_head, fp8));
    ggml_set_name(raw, fp8 ? "minimax_h3.sage.int8qk_fp8pv_fp32fp16.h3_scoped" : "minimax_h3.sage.int8qk_f16pv.h3_scoped");
    return raw;
}

inline ggml_tensor* h3_sage_output_graph(ggml_context* ctx, ggml_tensor* raw,
                                        int64_t n_head) {
    if (ctx == nullptr || raw == nullptr || n_head <= 0 ||
        raw->type != GGML_TYPE_F32 || raw->ne[0] != kH3SageHeadDim ||
        raw->ne[1] <= 0 || raw->ne[1] % n_head != 0 || raw->ne[2] <= 0 ||
        raw->ne[3] != 1 || !ggml_is_contiguous(raw)) return nullptr;
    const int64_t batch = raw->ne[1] / n_head;
    const int64_t sequence = raw->ne[2];
    auto out = ggml_scale(ctx, raw, kH3SageOutputScale);
    // raw [d, heads*batch, L, 1] -> [d, heads, batch, L] -> [d, heads, L, batch]
    // Do not copy shared flash's batch=1 view_3d, which loses the batch axis.
    if (batch > 1) {
        out = ggml_reshape_4d(ctx, out, kH3SageHeadDim, n_head, batch, sequence);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 1, 3, 2));
    }
    out = ggml_reshape_3d(ctx, out, kH3SageHeadDim * n_head, sequence, batch);
    ggml_set_name(out, "minimax_h3.sage.output");
    return out;
}

struct H3SageBuildResult {
    H3SageRoute route = H3SageRoute::Dense;
    ggml_tensor* output = nullptr;
    std::string reason;
};

// The caller must throw/report Reject, and call the UNCHANGED dense path only
// for Dense. A backend without the H3 tag patch cannot silently run
// the no-op CPU stub or report that Sage was enabled.
inline H3SageBuildResult h3_sage_try_build(ggml_context* ctx, ggml_backend_t backend,
                                         ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                                         int64_t n_head, H3SageRequest request,
                                         const ggml_tensor* mask = nullptr) {
    request.is_cuda = h3_sage_cuda_backend(backend); // never trust a caller's claim
    request.pv_fp8 = edgedit::ggml_ext::sage_pv_fp8_enabled();
    request.dense_fallback = edgedit::ggml_ext::sage_fallback_enabled(kH3SageHeadDim);
    const auto decision = h3_sage_decide(request);
    if (decision.route != H3SageRoute::Sage)
        return {decision.route, nullptr, decision.reason};
    if (ctx == nullptr) return {H3SageRoute::Reject, nullptr, "H3 Sage needs a graph context"};
    if (!h3_sage_shape_supported(q, k, v, n_head, mask))
        return {H3SageRoute::Reject, nullptr, "H3 Sage shape/type/mask is unsupported; refusing silent dense"};
    const bool fp8 = h3_sage_fp8_enabled();
    auto raw = h3_sage_custom_graph(ctx, q, k, v, n_head, fp8);
    if (raw == nullptr || !ggml_backend_supports_op(backend, raw))
        return {H3SageRoute::Reject, nullptr,
                fp8 ? "H3F8 dispatch unavailable: requires CUDA compiler >=12.8 and SM89; no dense fallback" :
                "H3-scoped Sage CUDA dispatch unavailable; build Sage and apply the H3SA tag patch"};
    auto output = h3_sage_output_graph(ctx, raw, n_head);
    if (output == nullptr) return {H3SageRoute::Reject, nullptr, "H3 Sage output layout is invalid"};
    return {H3SageRoute::Sage, output, fp8 ?
        "H3 per-warp INT8-QK / E4M3-PV / FP32+FP16-accum, V scale_max=2.25, smooth-K only" :
        "H3 per-warp INT8-QK / FP16-PV / FP32-accum, smooth-K only; no dense fallback"};
}

} // namespace MiniMaxH3
