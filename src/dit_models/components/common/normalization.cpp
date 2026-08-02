#include "dit_models/components/common/normalization.hpp"

#include "backend/ggml/ed_ggml_norm_ext.hpp"

#include <utility>

namespace dit {

RMSNorm::RMSNorm(
    int64_t hidden_size,
    float eps,
    std::string weight_name,
    bool use_model_weight_type,
    bool cast_output_to_input_type)
    : hidden_size(hidden_size),
      eps(eps),
      weight_name(std::move(weight_name)),
      use_model_weight_type(use_model_weight_type),
      cast_output_to_input_type(cast_output_to_input_type) {
}

void RMSNorm::init_params(
    ggml_context* ctx,
    const String2TensorStorage& tensor_storage_map,
    const std::string prefix) {
    ggml_type wtype = GGML_TYPE_F32;
    if (use_model_weight_type) {
        wtype = get_type(prefix + weight_name, tensor_storage_map, GGML_TYPE_F32);
        if (wtype == GGML_TYPE_F32 && weight_name != "weight") {
            wtype = get_type(prefix + "weight", tensor_storage_map, wtype);
        }
        if (wtype == GGML_TYPE_F32 && weight_name != "scale") {
            wtype = get_type(prefix + "scale", tensor_storage_map, wtype);
        }
        if (wtype != GGML_TYPE_F32 && wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_BF16) {
            wtype = GGML_TYPE_F32;
        }
    }
    params[weight_name] = ggml_new_tensor_1d(ctx, wtype, hidden_size);
}

ggml_tensor* RMSNorm::forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
    const ggml_type input_type = x->type;
    ggml_tensor* w = params[weight_name];
    if (cast_output_to_input_type &&
        (input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16)) {
        x = ggml_cast(ctx->ggml_ctx, x, GGML_TYPE_F32);
    }
    x = ggml_rms_norm(ctx->ggml_ctx, x, eps);
    if (cast_output_to_input_type &&
        (input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16)) {
        x = ggml_cast(ctx->ggml_ctx, x, input_type);
    }
    x = ggml_mul(ctx->ggml_ctx, x, w);
    if (cast_output_to_input_type &&
        (input_type == GGML_TYPE_F16 || input_type == GGML_TYPE_BF16) &&
        x->type != input_type) {
        x = ggml_cast(ctx->ggml_ctx, x, input_type);
    }
    return x;
}

ggml_tensor* RMSNorm::forward_f16(GGMLRunnerContext* ctx, ggml_tensor* x) {
    ggml_tensor* w = params[weight_name];
    if (auto fused = edgedit::ggml_ext::rms_norm_mul_f16_custom(ctx->ggml_ctx, x, w, eps)) {
        return fused;
    }
    return ggml_cast(ctx->ggml_ctx, forward(ctx, x), GGML_TYPE_F16);
}

QKNorm::QKNorm(
    int64_t dim,
    float eps,
    std::string weight_name,
    bool use_model_weight_type,
    bool cast_output_to_input_type) {
    blocks["query_norm"] = std::make_shared<RMSNorm>(dim, eps, weight_name, use_model_weight_type, cast_output_to_input_type);
    blocks["key_norm"]   = std::make_shared<RMSNorm>(dim, eps, weight_name, use_model_weight_type, cast_output_to_input_type);
}

ggml_tensor* QKNorm::query_norm(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);
    return norm->forward(ctx, x);
}

ggml_tensor* QKNorm::key_norm(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);
    return norm->forward(ctx, x);
}

ggml_tensor* QKNorm::query_norm_f16(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["query_norm"]);
    return norm->forward_f16(ctx, x);
}

ggml_tensor* QKNorm::key_norm_f16(GGMLRunnerContext* ctx, ggml_tensor* x) {
    auto norm = std::dynamic_pointer_cast<RMSNorm>(blocks["key_norm"]);
    return norm->forward_f16(ctx, x);
}

}  // namespace dit
