#include "dit_models/components/common/modulation.hpp"

#ifdef ED_ENABLE_CUDA_MODULATION
#include "backend/ggml/ed_ggml_modulation_ext.hpp"
#endif

namespace dit {

ggml_tensor* modulate(ggml_context* ctx,
                      ggml_tensor* x,
                      ggml_tensor* shift,
                      ggml_tensor* scale,
                      bool skip_reshape) {
    if (!skip_reshape) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
        shift = ggml_reshape_3d(ctx, shift, shift->ne[0], 1, shift->ne[1]);
    }
#ifdef ED_ENABLE_CUDA_MODULATION
    // Fuse x + x*scale + shift into one CUDA kernel; falls back to split ops
    // (and on CPU) when shapes are unsupported.
    if (auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx, x, shift, scale)) {
        return fused;
    }
#endif
    // x + x*scale + shift == x*(1+scale) + shift. The (1+scale) is a scale_bias
    // on the small broadcast tensor [dim,1,N], turning two full-size elementwise
    // ops (mul + add) plus a second add into one full-size mul + one add. On
    // Vulkan (no fused-modulation custom op) this drops one big elementwise pass
    // per modulate; on CPU/CUDA it is one fewer op and numerically identical.
    ggml_tensor* scale_plus_one = ggml_scale_bias(ctx, scale, 1.0f, 1.0f);
    x = ggml_mul(ctx, x, scale_plus_one);
    x = ggml_add(ctx, x, shift);
    return x;
}

ggml_tensor* modulate(ggml_context* ctx,
                      ggml_tensor* x,
                      ggml_tensor* scale,
                      bool skip_reshape) {
    if (!skip_reshape) {
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);
    }
    // x + x*scale == x*(1+scale): one full-size mul instead of mul + add
    // (the 1+scale scale_bias runs on the small broadcast tensor).
    x = ggml_mul(ctx, x, ggml_scale_bias(ctx, scale, 1.0f, 1.0f));
    return x;
}

ggml_tensor* residual_gate(ggml_context* ctx,
                           ggml_tensor* residual,
                           ggml_tensor* x,
                           ggml_tensor* gate,
                           bool skip_reshape) {
    if (!skip_reshape) {
        gate = ggml_reshape_3d(ctx, gate, gate->ne[0], 1, gate->ne[1]);
    }
#ifdef ED_ENABLE_CUDA_MODULATION
    if (auto fused = edgedit::ggml_ext::fused_residual_gate_custom(ctx, residual, x, gate)) {
        return fused;
    }
#endif
    return ggml_add(ctx, residual, ggml_mul(ctx, x, gate));
}

}  // namespace dit
