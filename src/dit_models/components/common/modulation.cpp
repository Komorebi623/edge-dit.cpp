#include "dit_models/components/common/modulation.hpp"

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
    x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
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
    x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
    return x;
}

}  // namespace dit
