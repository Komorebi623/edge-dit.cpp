#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdlib>

namespace edgedit::ggml_ext {

constexpr uint32_t kAttentionVPrepCustomMagic = 0x45565050u; // "EVPP"
constexpr uint32_t kAttentionPairPackCustomMagic = 0x45505150u; // "EPQP"
constexpr uint32_t kAttentionQKVPairPackCustomMagic = 0x45514b50u; // "EQKP"

struct AttentionVPrepCustomParams {
    uint32_t magic = kAttentionVPrepCustomMagic;
    int32_t v_is_seq_major = 0;
};

struct AttentionPairPackCustomParams {
    uint32_t magic = kAttentionPairPackCustomMagic;
    int32_t n_head = 0;
};

struct AttentionQKVPairPackCustomParams {
    uint32_t magic = kAttentionQKVPairPackCustomMagic;
    int32_t n_head = 0;
};

inline bool attention_v_prep_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_ATTENTION_V_PREP");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline bool attention_pair_pack_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_ATTENTION_PAIR_PACK");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline bool attention_qkv_pair_pack_enabled() {
    const char* env = std::getenv("ED_DISABLE_CUDA_ATTENTION_QKV_PAIR_PACK");
    return !(env != nullptr && std::atoi(env) != 0);
}

inline AttentionVPrepCustomParams attention_v_prep_params_from_userdata(void* userdata) {
    AttentionVPrepCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.v_is_seq_major = static_cast<int32_t>((packed >> 32) & 0xffu);
    return params;
}

inline void* attention_v_prep_params_to_userdata(bool v_is_seq_major) {
    uintptr_t packed = static_cast<uintptr_t>(kAttentionVPrepCustomMagic);
    packed |= (static_cast<uintptr_t>(v_is_seq_major ? 1u : 0u) << 32);
    return reinterpret_cast<void*>(packed);
}

inline AttentionPairPackCustomParams attention_pair_pack_params_from_userdata(void* userdata) {
    AttentionPairPackCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.n_head = static_cast<int32_t>((packed >> 32) & 0x7fffffffu);
    return params;
}

inline void* attention_pair_pack_params_to_userdata(int64_t n_head) {
    uintptr_t packed = static_cast<uintptr_t>(kAttentionPairPackCustomMagic);
    packed |= (static_cast<uintptr_t>(n_head) << 32);
    return reinterpret_cast<void*>(packed);
}

inline AttentionQKVPairPackCustomParams attention_qkv_pair_pack_params_from_userdata(void* userdata) {
    AttentionQKVPairPackCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    params.n_head = static_cast<int32_t>((packed >> 32) & 0x7fffffffu);
    return params;
}

inline void* attention_qkv_pair_pack_params_to_userdata(int64_t n_head) {
    uintptr_t packed = static_cast<uintptr_t>(kAttentionQKVPairPackCustomMagic);
    packed |= (static_cast<uintptr_t>(n_head) << 32);
    return reinterpret_cast<void*>(packed);
}

inline bool attention_v_prep_params_valid(const AttentionVPrepCustomParams& params) {
    return params.magic == kAttentionVPrepCustomMagic &&
           (params.v_is_seq_major == 0 || params.v_is_seq_major == 1);
}

inline bool attention_pair_pack_params_valid(const AttentionPairPackCustomParams& params) {
    return params.magic == kAttentionPairPackCustomMagic && params.n_head > 0;
}

inline bool attention_qkv_pair_pack_params_valid(const AttentionQKVPairPackCustomParams& params) {
    return params.magic == kAttentionQKVPairPackCustomMagic && params.n_head > 0;
}

inline float attention_tensor_f32_at(const ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    return *reinterpret_cast<const float*>(ptr);
}

inline void attention_tensor_f16_set(ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    GGML_ASSERT(t->type == GGML_TYPE_F16);
    *reinterpret_cast<ggml_fp16_t*>(ptr) = ggml_fp32_to_fp16(v);
}

inline bool attention_v_prep_shape_supported(const ggml_tensor* v, bool v_is_seq_major) {
    if (!attention_v_prep_enabled() || v == nullptr || v->type != GGML_TYPE_F32) {
        return false;
    }
    if (v->ne[0] <= 0 || v->ne[1] <= 0 || v->ne[2] <= 0 || v->ne[3] <= 0) {
        return false;
    }
    const int64_t d_head = v->ne[0];
    const int64_t seq = v_is_seq_major ? v->ne[1] : v->ne[2];
    const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
    return d_head > 0 && seq > 0 && n_head > 0;
}

inline bool attention_pair_pack_shape_supported(const ggml_tensor* first,
                                                const ggml_tensor* second,
                                                int64_t n_head) {
    if (!attention_pair_pack_enabled() ||
        first == nullptr ||
        second == nullptr ||
        first->type != GGML_TYPE_F32 ||
        second->type != GGML_TYPE_F32 ||
        n_head <= 0) {
        return false;
    }
    if (first->ne[0] <= 0 || first->ne[1] <= 0 || first->ne[2] <= 0 ||
        second->ne[0] <= 0 || second->ne[1] <= 0 || second->ne[2] <= 0) {
        return false;
    }
    if (first->ne[0] != second->ne[0] ||
        first->ne[2] != second->ne[2] ||
        first->ne[3] != 1 ||
        second->ne[3] != 1 ||
        first->ne[0] % n_head != 0) {
        return false;
    }
    return true;
}

inline bool attention_qkv_pair_pack_shape_supported(const ggml_tensor* q_first,
                                                    const ggml_tensor* k_first,
                                                    const ggml_tensor* v_first,
                                                    const ggml_tensor* q_second,
                                                    const ggml_tensor* k_second,
                                                    const ggml_tensor* v_second,
                                                    int64_t n_head) {
    if (!attention_qkv_pair_pack_enabled() ||
        !attention_pair_pack_shape_supported(q_first, q_second, n_head) ||
        !attention_pair_pack_shape_supported(k_first, k_second, n_head) ||
        !attention_pair_pack_shape_supported(v_first, v_second, n_head)) {
        return false;
    }
    if (q_first->ne[2] != 1 ||
        q_second->ne[2] != 1 ||
        q_first->ne[0] != k_first->ne[0] ||
        q_first->ne[0] != v_first->ne[0] ||
        q_first->ne[1] != k_first->ne[1] ||
        q_first->ne[1] != v_first->ne[1] ||
        q_second->ne[0] != k_second->ne[0] ||
        q_second->ne[0] != v_second->ne[0] ||
        q_second->ne[1] != k_second->ne[1] ||
        q_second->ne[1] != v_second->ne[1]) {
        return false;
    }
    return true;
}

inline void attention_v_prep_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const AttentionVPrepCustomParams params = attention_v_prep_params_from_userdata(userdata);
    GGML_ASSERT(attention_v_prep_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr);
    const ggml_tensor* v = dst->src[0];
    const bool v_is_seq_major = params.v_is_seq_major != 0;
    GGML_ASSERT(attention_v_prep_shape_supported(v, v_is_seq_major));
    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    const int64_t d_head = v->ne[0];
    const int64_t seq = v_is_seq_major ? v->ne[1] : v->ne[2];
    const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
    const int64_t batch = v->ne[3];
    GGML_ASSERT(dst->ne[0] == d_head && dst->ne[1] == seq && dst->ne[2] == n_head * batch);

    // Parallelize over flattened (batch, head, seq): each writes an independent dst
    // row of d_head, so a static ith/nth split is race-free. Was single-threaded
    // (ith!=0 return) — a CUDA-era no-op that cost ~1 of 96 cores on CPU.
    const int64_t work = batch * n_head * seq;
    for (int64_t idx = ith; idx < work; idx += nth) {
        const int64_t s = idx % seq;
        const int64_t h = (idx / seq) % n_head;
        const int64_t b = idx / (seq * n_head);
        for (int64_t d = 0; d < d_head; ++d) {
            const float value = v_is_seq_major ?
                                    attention_tensor_f32_at(v, d, s, h, b) :
                                    attention_tensor_f32_at(v, d, h, s, b);
            attention_tensor_f16_set(dst, d, s, h + b * n_head, 0, value);
        }
    }
}

inline void attention_pair_pack_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const AttentionPairPackCustomParams params = attention_pair_pack_params_from_userdata(userdata);
    GGML_ASSERT(attention_pair_pack_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr);
    const ggml_tensor* first = dst->src[0];
    const ggml_tensor* second = dst->src[1];
    const int64_t n_head = params.n_head;
    GGML_ASSERT(attention_pair_pack_shape_supported(first, second, n_head));
    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    const int64_t d_head = first->ne[0] / n_head;
    const int64_t first_seq = first->ne[1];
    const int64_t second_seq = second->ne[1];
    const int64_t total_seq = first_seq + second_seq;
    const int64_t batch = first->ne[2];
    GGML_ASSERT(dst->ne[0] == d_head && dst->ne[1] == total_seq && dst->ne[2] == n_head && dst->ne[3] == batch);

    // Parallelize over flattened (batch, head, total_seq); race-free per dst row.
    const int64_t work = batch * n_head * total_seq;
    for (int64_t idx = ith; idx < work; idx += nth) {
        const int64_t s = idx % total_seq;
        const int64_t h = (idx / total_seq) % n_head;
        const int64_t b = idx / (total_seq * n_head);
        const ggml_tensor* src = s < first_seq ? first : second;
        const int64_t src_s = s < first_seq ? s : s - first_seq;
        for (int64_t d = 0; d < d_head; ++d) {
            const float value = attention_tensor_f32_at(src, d + h * d_head, src_s, b, 0);
            attention_tensor_f16_set(dst, d, s, h, b, value);
        }
    }
}

inline void attention_qkv_pair_pack_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const AttentionQKVPairPackCustomParams params = attention_qkv_pair_pack_params_from_userdata(userdata);
    GGML_ASSERT(attention_qkv_pair_pack_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);
    GGML_ASSERT(dst->src[3] != nullptr && dst->src[4] != nullptr && dst->src[5] != nullptr);
    const ggml_tensor* q_first = dst->src[0];
    const ggml_tensor* k_first = dst->src[1];
    const ggml_tensor* v_first = dst->src[2];
    const ggml_tensor* q_second = dst->src[3];
    const ggml_tensor* k_second = dst->src[4];
    const ggml_tensor* v_second = dst->src[5];
    const int64_t n_head = params.n_head;
    GGML_ASSERT(attention_qkv_pair_pack_shape_supported(q_first, k_first, v_first, q_second, k_second, v_second, n_head));
    GGML_ASSERT(dst->type == GGML_TYPE_F16);

    const int64_t d_head = q_first->ne[0] / n_head;
    const int64_t first_seq = q_first->ne[1];
    const int64_t second_seq = q_second->ne[1];
    const int64_t total_seq = first_seq + second_seq;
    GGML_ASSERT(dst->ne[0] == d_head && dst->ne[1] == total_seq && dst->ne[2] == n_head && dst->ne[3] == 3);

    const ggml_tensor* firsts[3] = { q_first, k_first, v_first };
    const ggml_tensor* seconds[3] = { q_second, k_second, v_second };
    // Parallelize over flattened (plane=q/k/v, head, total_seq); race-free per dst row.
    const int64_t work = 3 * n_head * total_seq;
    for (int64_t idx = ith; idx < work; idx += nth) {
        const int64_t s = idx % total_seq;
        const int64_t h = (idx / total_seq) % n_head;
        const int64_t plane = idx / (total_seq * n_head);
        const ggml_tensor* src = s < first_seq ? firsts[plane] : seconds[plane];
        const int64_t src_s = s < first_seq ? s : s - first_seq;
        for (int64_t d = 0; d < d_head; ++d) {
            const float value = attention_tensor_f32_at(src, d + h * d_head, src_s, 0, 0);
            attention_tensor_f16_set(dst, d, s, h, plane, value);
        }
    }
}

inline ggml_tensor* attention_v_prep_custom_f16(ggml_context* ctx, ggml_tensor* v, bool v_is_seq_major) {
    if (!attention_v_prep_shape_supported(v, v_is_seq_major)) {
        return nullptr;
    }
    ggml_tensor* args[] = { v };
    const int64_t d_head = v->ne[0];
    const int64_t seq = v_is_seq_major ? v->ne[1] : v->ne[2];
    const int64_t n_head = v_is_seq_major ? v->ne[2] : v->ne[1];
    const int64_t batch = v->ne[3];
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      d_head,
                                      seq,
                                      n_head * batch,
                                      1,
                                      args,
                                      1,
                                      attention_v_prep_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      attention_v_prep_params_to_userdata(v_is_seq_major));
    ggml_set_name(out, "ed_fused_attention_v_f16");
    return out;
}

inline ggml_tensor* attention_pair_pack_custom_f16(ggml_context* ctx,
                                                   ggml_tensor* first,
                                                   ggml_tensor* second,
                                                   int64_t n_head) {
    if (!attention_pair_pack_shape_supported(first, second, n_head)) {
        return nullptr;
    }
    ggml_tensor* args[] = { first, second };
    const int64_t d_head = first->ne[0] / n_head;
    const int64_t total_seq = first->ne[1] + second->ne[1];
    const int64_t batch = first->ne[2];
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      d_head,
                                      total_seq,
                                      n_head,
                                      batch,
                                      args,
                                      2,
                                      attention_pair_pack_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      attention_pair_pack_params_to_userdata(n_head));
    ggml_set_name(out, "ed_fused_attention_pair_pack_f16");
    return out;
}

inline ggml_tensor* attention_qkv_pair_pack_custom_f16(ggml_context* ctx,
                                                       ggml_tensor* q_first,
                                                       ggml_tensor* k_first,
                                                       ggml_tensor* v_first,
                                                       ggml_tensor* q_second,
                                                       ggml_tensor* k_second,
                                                       ggml_tensor* v_second,
                                                       int64_t n_head) {
    if (!attention_qkv_pair_pack_shape_supported(q_first, k_first, v_first, q_second, k_second, v_second, n_head)) {
        return nullptr;
    }
    ggml_tensor* args[] = { q_first, k_first, v_first, q_second, k_second, v_second };
    const int64_t d_head = q_first->ne[0] / n_head;
    const int64_t total_seq = q_first->ne[1] + q_second->ne[1];
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F16,
                                      d_head,
                                      total_seq,
                                      n_head,
                                      3,
                                      args,
                                      6,
                                      attention_qkv_pair_pack_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      attention_qkv_pair_pack_params_to_userdata(n_head));
    ggml_set_name(out, "ed_fused_attention_qkv_pair_pack_f16");
    return out;
}

} // namespace edgedit::ggml_ext
