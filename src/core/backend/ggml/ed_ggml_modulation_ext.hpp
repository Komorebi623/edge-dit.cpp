#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>

namespace edgedit::ggml_ext {

constexpr uint32_t kFusedModulateCustomMagic = 0x45464d4fu; // "EFMO"
constexpr uint32_t kFusedResidualGateCustomMagic = 0x45465247u; // "EFRG"
constexpr uint32_t kFusedWhereSelectCustomMagic = 0x45465753u; // "EFWS"

struct FusedModulateCustomParams {
    uint32_t magic = kFusedModulateCustomMagic;
};

struct FusedResidualGateCustomParams {
    uint32_t magic = kFusedResidualGateCustomMagic;
};

struct FusedWhereSelectCustomParams {
    uint32_t magic = kFusedWhereSelectCustomMagic;
};

inline FusedModulateCustomParams fused_modulate_params_from_userdata(void* userdata) {
    FusedModulateCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* fused_modulate_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kFusedModulateCustomMagic));
}

inline bool fused_modulate_params_valid(const FusedModulateCustomParams& params) {
    return params.magic == kFusedModulateCustomMagic;
}

inline FusedResidualGateCustomParams fused_residual_gate_params_from_userdata(void* userdata) {
    FusedResidualGateCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* fused_residual_gate_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kFusedResidualGateCustomMagic));
}

inline bool fused_residual_gate_params_valid(const FusedResidualGateCustomParams& params) {
    return params.magic == kFusedResidualGateCustomMagic;
}

inline FusedWhereSelectCustomParams fused_where_select_params_from_userdata(void* userdata) {
    FusedWhereSelectCustomParams params;
    uintptr_t packed = reinterpret_cast<uintptr_t>(userdata);
    params.magic = static_cast<uint32_t>(packed & 0xffffffffu);
    return params;
}

inline void* fused_where_select_params_to_userdata() {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(kFusedWhereSelectCustomMagic));
}

inline bool fused_where_select_params_valid(const FusedWhereSelectCustomParams& params) {
    return params.magic == kFusedWhereSelectCustomMagic;
}

inline bool fused_modulate_broadcast_dim_supported(int64_t src, int64_t dst) {
    return src > 0 && dst > 0 && dst % src == 0;
}

inline bool fused_modulate_shape_supported(const ggml_tensor* x,
                                           const ggml_tensor* shift,
                                           const ggml_tensor* scale) {
    if (x == nullptr || shift == nullptr || scale == nullptr) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || shift->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (x->ne[i] <= 0 || shift->ne[i] <= 0 || scale->ne[i] <= 0) {
            return false;
        }
        if (!fused_modulate_broadcast_dim_supported(shift->ne[i], x->ne[i]) ||
            !fused_modulate_broadcast_dim_supported(scale->ne[i], x->ne[i])) {
            return false;
        }
    }
    return true;
}

inline bool fused_residual_gate_shape_supported(const ggml_tensor* residual,
                                                const ggml_tensor* x,
                                                const ggml_tensor* gate) {
    if (residual == nullptr || x == nullptr || gate == nullptr) {
        return false;
    }
    if (residual->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (residual->ne[i] <= 0 || x->ne[i] <= 0 || gate->ne[i] <= 0) {
            return false;
        }
        if (residual->ne[i] != x->ne[i] ||
            !fused_modulate_broadcast_dim_supported(gate->ne[i], x->ne[i])) {
            return false;
        }
    }
    return true;
}

// out[i] = mod_0[i] + index[i] * (mod_1[i] - mod_0[i]) == where(index==0, mod_0, mod_1)
// where each of index/mod_0/mod_1 broadcasts (per-dim ne==1 or ==out_ne) into out.
inline bool fused_where_select_shape_supported(const ggml_tensor* out,
                                               const ggml_tensor* index,
                                               const ggml_tensor* mod_0,
                                               const ggml_tensor* mod_1) {
    if (out == nullptr || index == nullptr || mod_0 == nullptr || mod_1 == nullptr) {
        return false;
    }
    if (out->type != GGML_TYPE_F32 || index->type != GGML_TYPE_F32 ||
        mod_0->type != GGML_TYPE_F32 || mod_1->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (out->ne[i] <= 0 || index->ne[i] <= 0 || mod_0->ne[i] <= 0 || mod_1->ne[i] <= 0) {
            return false;
        }
        if (!fused_modulate_broadcast_dim_supported(index->ne[i], out->ne[i]) ||
            !fused_modulate_broadcast_dim_supported(mod_0->ne[i], out->ne[i]) ||
            !fused_modulate_broadcast_dim_supported(mod_1->ne[i], out->ne[i])) {
            return false;
        }
    }
    return true;
}

inline float fused_modulate_tensor_f32_at_broadcast(const ggml_tensor* t,
                                                    int64_t i0,
                                                    int64_t i1,
                                                    int64_t i2,
                                                    int64_t i3) {
    const int64_t j0 = i0 % t->ne[0];
    const int64_t j1 = i1 % t->ne[1];
    const int64_t j2 = i2 % t->ne[2];
    const int64_t j3 = i3 % t->ne[3];
    const char* base = static_cast<const char*>(t->data);
    const char* ptr = base + j0 * t->nb[0] + j1 * t->nb[1] + j2 * t->nb[2] + j3 * t->nb[3];
    return *reinterpret_cast<const float*>(ptr);
}

inline void fused_modulate_tensor_f32_set(ggml_tensor* t,
                                          int64_t i0,
                                          int64_t i1,
                                          int64_t i2,
                                          int64_t i3,
                                          float v) {
    char* base = static_cast<char*>(t->data);
    char* ptr = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    *reinterpret_cast<float*>(ptr) = v;
}

inline void fused_modulate_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FusedModulateCustomParams params = fused_modulate_params_from_userdata(userdata);
    GGML_ASSERT(fused_modulate_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);
    const ggml_tensor* x = dst->src[0];
    const ggml_tensor* shift = dst->src[1];
    const ggml_tensor* scale = dst->src[2];
    GGML_ASSERT(fused_modulate_shape_supported(x, shift, scale));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        GGML_ASSERT(dst->ne[i] == x->ne[i]);
    }

    // Parallelize over flattened (i3,i2,i1) rows; each row writes independent dst.
    // Was single-threaded (ith!=0 return): CUDA-era no-op, 1/96 cores on CPU.
    const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t ne0 = x->ne[0];
    for (int64_t r = ith; r < rows; r += nth) {
        const int64_t i1 = r % x->ne[1];
        const int64_t i2 = (r / x->ne[1]) % x->ne[2];
        const int64_t i3 = r / (x->ne[1] * x->ne[2]);
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const float xv = fused_modulate_tensor_f32_at_broadcast(x, i0, i1, i2, i3);
            const float sv = fused_modulate_tensor_f32_at_broadcast(scale, i0, i1, i2, i3);
            const float bv = fused_modulate_tensor_f32_at_broadcast(shift, i0, i1, i2, i3);
            const float prod = xv * sv;
            const float y = xv + prod;
            fused_modulate_tensor_f32_set(dst, i0, i1, i2, i3, y + bv);
        }
    }
}

inline void fused_residual_gate_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FusedResidualGateCustomParams params = fused_residual_gate_params_from_userdata(userdata);
    GGML_ASSERT(fused_residual_gate_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);
    const ggml_tensor* residual = dst->src[0];
    const ggml_tensor* x = dst->src[1];
    const ggml_tensor* gate = dst->src[2];
    GGML_ASSERT(fused_residual_gate_shape_supported(residual, x, gate));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        GGML_ASSERT(dst->ne[i] == x->ne[i]);
    }

    // Parallelize over flattened (i3,i2,i1) rows; race-free per dst row.
    const int64_t rows = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t ne0 = x->ne[0];
    for (int64_t r = ith; r < rows; r += nth) {
        const int64_t i1 = r % x->ne[1];
        const int64_t i2 = (r / x->ne[1]) % x->ne[2];
        const int64_t i3 = r / (x->ne[1] * x->ne[2]);
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const float rv = fused_modulate_tensor_f32_at_broadcast(residual, i0, i1, i2, i3);
            const float xv = fused_modulate_tensor_f32_at_broadcast(x, i0, i1, i2, i3);
            const float gv = fused_modulate_tensor_f32_at_broadcast(gate, i0, i1, i2, i3);
            const float prod = xv * gv;
            fused_modulate_tensor_f32_set(dst, i0, i1, i2, i3, rv + prod);
        }
    }
}

inline void fused_where_select_cpu_custom_op(ggml_tensor* dst, int ith, int nth, void* userdata) {
    const FusedWhereSelectCustomParams params = fused_where_select_params_from_userdata(userdata);
    GGML_ASSERT(fused_where_select_params_valid(params));
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[1] != nullptr && dst->src[2] != nullptr);
    const ggml_tensor* index = dst->src[0];
    const ggml_tensor* mod_0 = dst->src[1];
    const ggml_tensor* mod_1 = dst->src[2];
    GGML_ASSERT(fused_where_select_shape_supported(dst, index, mod_0, mod_1));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
    const int64_t ne0 = dst->ne[0];
    for (int64_t r = ith; r < rows; r += nth) {
        const int64_t i1 = r % dst->ne[1];
        const int64_t i2 = (r / dst->ne[1]) % dst->ne[2];
        const int64_t i3 = r / (dst->ne[1] * dst->ne[2]);
        for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const float iv = fused_modulate_tensor_f32_at_broadcast(index, i0, i1, i2, i3);
            const float m0 = fused_modulate_tensor_f32_at_broadcast(mod_0, i0, i1, i2, i3);
            const float m1 = fused_modulate_tensor_f32_at_broadcast(mod_1, i0, i1, i2, i3);
            fused_modulate_tensor_f32_set(dst, i0, i1, i2, i3, m0 + iv * (m1 - m0));
        }
    }
}

inline ggml_tensor* fused_modulate_custom(ggml_context* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* shift,
                                          ggml_tensor* scale) {
    if (!fused_modulate_shape_supported(x, shift, scale)) {
        return nullptr;
    }
    ggml_tensor* args[] = { x, shift, scale };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      x->ne[0],
                                      x->ne[1],
                                      x->ne[2],
                                      x->ne[3],
                                      args,
                                      3,
                                      fused_modulate_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      fused_modulate_params_to_userdata());
    ggml_set_name(out, "ed_fused_modulate");
    return out;
}

inline ggml_tensor* fused_residual_gate_custom(ggml_context* ctx,
                                               ggml_tensor* residual,
                                               ggml_tensor* x,
                                               ggml_tensor* gate) {
    if (!fused_residual_gate_shape_supported(residual, x, gate)) {
        return nullptr;
    }
    ggml_tensor* args[] = { residual, x, gate };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      x->ne[0],
                                      x->ne[1],
                                      x->ne[2],
                                      x->ne[3],
                                      args,
                                      3,
                                      fused_residual_gate_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      fused_residual_gate_params_to_userdata());
    ggml_set_name(out, "ed_fused_residual_gate");
    return out;
}

// out[hidden, n_tok, N] = where(index==0, mod_0, mod_1) = mod_0 + index*(mod_1-mod_0).
// index broadcasts as [1, n_tok, N]; mod_0/mod_1 broadcast as [hidden, 1, 1].
// out_ne gives the materialized output dims. Returns nullptr if shapes unsupported.
inline ggml_tensor* fused_where_select_custom(ggml_context* ctx,
                                              ggml_tensor* index,
                                              ggml_tensor* mod_0,
                                              ggml_tensor* mod_1,
                                              const int64_t out_ne[4]) {
    ggml_tensor* args[] = { index, mod_0, mod_1 };
    ggml_tensor* out = ggml_custom_4d(ctx,
                                      GGML_TYPE_F32,
                                      out_ne[0],
                                      out_ne[1],
                                      out_ne[2],
                                      out_ne[3],
                                      args,
                                      3,
                                      fused_where_select_cpu_custom_op,
                                      GGML_N_TASKS_MAX,
                                      fused_where_select_params_to_userdata());
    if (!fused_where_select_shape_supported(out, index, mod_0, mod_1)) {
        return nullptr;
    }
    ggml_set_name(out, "ed_fused_where_select");
    return out;
}

} // namespace edgedit::ggml_ext
