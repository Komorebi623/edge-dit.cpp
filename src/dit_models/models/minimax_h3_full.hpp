#ifndef __ED_DIT_MODELS_MODELS_MINIMAX_H3_FULL_HPP__
#define __ED_DIT_MODELS_MODELS_MINIMAX_H3_FULL_HPP__

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "backend/ggml/ggml_graph_cut.h"
#include "dit_models/components/common/common_dit.hpp"
#include "backend/ggml/ed_ggml_modulation_ext.hpp"
#include "backend/ggml/ggml_extend.hpp"
#include "dit_models/components/common/rope.hpp"
#include "dit_models/models/minimax_h3_cuda_options.hpp"
#include "dit_models/models/minimax_h3_cuda_cache.hpp"
#include "dit_models/models/minimax_h3_cuda_cache_offload.hpp"
#include "dit_models/models/minimax_h3_cuda_sparse.hpp"
#include "dit_models/models/minimax_h3_cuda_sage.hpp"
#include "dit_models/models/minimax_h3_cuda_adaln_cache.hpp"
#include "dit_models/models/minimax_h3_sha256.hpp"
#include "optimization/cache/operator/cache_operator_registry.hpp"

namespace MiniMaxH3 {

    constexpr int H3_GRAPH_SIZE          = 131072;
    constexpr float FRAME_RESCALE        = 5.f / 3.f;
    constexpr float VISUAL_COND_TIMESTEP = 0.999f;

    static bool h3_env_flag_enabled(const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] != '\0' && std::atoi(value) != 0;
    }

    static bool h3_env_flag_enabled_or_default(const char* name, bool default_enabled) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_enabled;
        }
        return std::atoi(value) != 0;
    }

    // Runtime gate, not a process-wide environment override. CUDA backend names
    // are CUDA0, CUDA1, ...; ROCm/Vulkan/Metal/CPU are deliberately excluded.
    static bool h3_cuda_backend(ggml_backend_t backend) {
        if (backend == nullptr) {
            return false;
        }
        const char* name = ggml_backend_name(backend);
        return name != nullptr && std::strncmp(name, "CUDA", 4) == 0 &&
               (name[4] == '\0' || (name[4] >= '0' && name[4] <= '9'));
    }

    static bool h3_cuda_fused_enabled(const CudaOptimizationOptions* options,
                                      ggml_backend_t backend) {
        return options != nullptr && options->fused_ops && h3_cuda_backend(backend);
    }

    // Preserve legacy custom-op behavior, including its CPU implementation. New
    // request-scoped fusion must be accepted by CUDA, never silently scheduled
    // onto the host when the CUDA modulation extension was not built.
    static bool h3_fused_op_supported(ggml_tensor* op,
                                      bool legacy_enabled,
                                      ggml_backend_t backend) {
        return op != nullptr && (legacy_enabled ||
               (h3_cuda_backend(backend) && ggml_backend_supports_op(backend, op)));
    }

    static bool h3_trace_first_step_enabled() {
        if (!h3_env_flag_enabled("ED_MINIMAX_H3_TRACE")) {
            return false;
        }
        const char* step = std::getenv("ED_MINIMAX_H3_CURRENT_STEP");
        return step == nullptr || std::atoi(step) == 0;
    }


    struct Config {
        int64_t hidden_size              = 5376;
        int64_t num_layers               = 50;
        int64_t token_refiner_num_layers = 2;
        int64_t num_attention_heads      = 56;
        int64_t attention_head_dim       = 128;
        int64_t ffn_hidden_size          = 14336;
        int64_t video_latent_channels    = 24;
        int64_t audio_latent_channels    = 32;
        int64_t text_dim                 = 5120;
        int64_t timestep_input_dim       = 256;
        int64_t time_embed_hidden_size   = 5376;
        int64_t time_embed_dim           = 2688;
        int64_t rope_inv_freq_len        = 16;
        int64_t adaln_curve_grid         = 0;
        int patch_t                      = 1;
        int patch_h                      = 2;
        int patch_w                      = 2;
        float norm_eps                   = 1e-5f;
        float qk_norm_eps                = 1e-5f;
        float final_norm_eps             = 1e-5f;

        bool uses_adaln_curves() const {
            return adaln_curve_grid > 0;
        }

        static int64_t count_blocks(const String2TensorStorage& tensors,
                                    const std::string& prefix) {
            std::set<int> indices;
            for (const auto& [name, _] : tensors) {
                if (!starts_with(name, prefix)) {
                    continue;
                }
                size_t begin = prefix.size();
                size_t end   = name.find('.', begin);
                if (end != std::string::npos) {
                    indices.insert(std::atoi(name.substr(begin, end - begin).c_str()));
                }
            }
            return static_cast<int64_t>(indices.size());
        }

        static Config detect_from_weights(const String2TensorStorage& tensors,
                                          const std::string& prefix) {
            Config config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = tensors.find(prefix + "." + suffix);
                return it == tensors.end() ? nullptr : &it->second;
            };

            if (const auto* weight = find("video_patch_proj.weight")) {
                config.video_latent_channels = weight->ne[0] / 4;
                config.hidden_size           = weight->ne[1];
            }
            if (const auto* weight = find("audio_patch_proj.weight")) {
                config.audio_latent_channels = weight->ne[0];
            }
            config.num_layers               = count_blocks(tensors, prefix + ".blocks.");
            config.token_refiner_num_layers = count_blocks(tensors, prefix + ".token_refiner.blocks.");
            if (const auto* weight = find("blocks.0.attn.q_norm.weight")) {
                config.attention_head_dim = weight->ne[0];
            }
            if (const auto* weight = find("blocks.0.attn.qkv_proj.weight")) {
                int64_t qkv_out_dim = weight->ne[1];
                if (const auto* k_weight = find("blocks.0.attn.qkv_proj.weight.1")) {
                    qkv_out_dim += k_weight->ne[1];
                }
                if (const auto* v_weight = find("blocks.0.attn.qkv_proj.weight.2")) {
                    qkv_out_dim += v_weight->ne[1];
                }
                config.num_attention_heads = qkv_out_dim / (3 * config.attention_head_dim);
            }
            if (const auto* weight = find("blocks.0.mlp.fc1.weight")) {
                config.ffn_hidden_size = weight->ne[1] / 2;
            }
            if (const auto* weight = find("condition_proj.weight")) {
                config.text_dim = weight->ne[0];
            }
            if (const auto* table = find("adaln_t_table")) {
                config.time_embed_dim   = table->ne[0];
                config.adaln_curve_grid = table->ne[1];
            } else {
                if (const auto* weight = find("time_embedder.proj_in.weight")) {
                    config.timestep_input_dim     = weight->ne[0];
                    config.time_embed_hidden_size = weight->ne[1];
                }
                if (const auto* weight = find("time_embedder.proj_out.weight")) {
                    config.time_embed_dim = weight->ne[1];
                }
            }
            if (const auto* inv_freq = find("rope.inv_freq")) {
                config.rope_inv_freq_len = inv_freq->ne[0];
            }

            LOG_DEBUG("minimax_h3: layers=%" PRId64 ", hidden=%" PRId64 ", heads=%" PRId64
                      ", head_dim=%" PRId64 ", ffn=%" PRId64 ", adaln_curve=%" PRId64,
                      config.num_layers,
                      config.hidden_size,
                      config.num_attention_heads,
                      config.attention_head_dim,
                      config.ffn_hidden_size,
                      config.adaln_curve_grid);
            return config;
        }
    };

    static float time_shift_sigma(float sigma, float from_shift, float to_shift) {
        float base = sigma / (from_shift + sigma * (1.f - from_shift));
        return to_shift * base / (1.f + (to_shift - 1.f) * base);
    }

    static float time_shift_slope(float sigma, float from_shift, float to_shift) {
        float base = sigma / (from_shift + sigma * (1.f - from_shift));
        float a    = 1.f + (from_shift - 1.f) * base;
        float b    = 1.f + (to_shift - 1.f) * base;
        return to_shift * a * a / (from_shift * b * b);
    }

    struct TimeEmbedder : public GGMLBlock {
        TimeEmbedder(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
            blocks["proj_in"]  = std::make_shared<Linear>(input_dim, hidden_dim, true, true);
            blocks["proj_out"] = std::make_shared<Linear>(hidden_dim, output_dim, true, true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto proj_in  = std::dynamic_pointer_cast<Linear>(blocks["proj_in"]);
            auto proj_out = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);
            return proj_out->forward(ctx, ggml_silu(ctx->ggml_ctx, proj_in->forward(ctx, x)));
        }
    };

    struct MLP : public UnaryBlock {
        int64_t hidden_size;
        int64_t ffn_hidden_size;

        MLP(int64_t hidden_size,
            int64_t ffn_hidden_size)
            : hidden_size(hidden_size),
              ffn_hidden_size(ffn_hidden_size) {
            blocks["fc1"] = std::make_shared<Linear>(hidden_size,
                                                      ffn_hidden_size * 2,
                                                      false,
                                                      false,
                                                      true,
                                                      1.f / 128.f,
                                                      true);
            blocks["fc2"] = std::make_shared<Linear>(ffn_hidden_size,
                                                      hidden_size,
                                                      false,
                                                      false,
                                                      true,
                                                      1.f / 128.f,
                                                      true);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            return forward(ctx, x, nullptr);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x,
                             const CudaOptimizationOptions* options) {
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
            if (h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_MLP_INPUT_SCALE")) {
                fc1->set_scale(1.f);
                fc2->set_scale(1.f);
            }
            const bool use_mlp_tensor_cores = h3_env_flag_enabled("ED_MINIMAX_H3_MLP_FP16_CUBLAS");
            const bool use_fc1_tensor_cores = use_mlp_tensor_cores ||
                                              h3_env_flag_enabled("ED_MINIMAX_H3_FC1_FP16_CUBLAS") ||
                                              h3_env_flag_enabled_or_default("ED_MINIMAX_H3_FC1_F32_OUTPUT", true);
            const bool use_fc2_tensor_cores = use_mlp_tensor_cores || h3_env_flag_enabled("ED_MINIMAX_H3_FC2_FP16_CUBLAS");
            fc1->set_force_prec_f32(!use_fc1_tensor_cores);
            fc2->set_force_prec_f32(!use_fc2_tensor_cores);
            auto forward_chunk = [&](ggml_tensor* chunk) {
                auto fc1_output = fc1->forward(ctx, chunk);
                if (!h3_cuda_fused_enabled(options, ctx->backend) &&
                    h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_SWIGLU_FUSION")) {
                    auto uv = ggml_ext_chunk(ctx->ggml_ctx, fc1_output, 2, 0);
                    return fc2->forward(ctx, ggml_mul(ctx->ggml_ctx,
                                                      ggml_silu(ctx->ggml_ctx, uv[0]),
                                                      uv[1]));
                }
                return fc2->forward(ctx, ggml_swiglu(ctx->ggml_ctx, fc1_output));
            };

            const int64_t sequence = x->ne[1];
            if (ctx->max_graph_vram_bytes == 0 || sequence <= 32768) {
                return forward_chunk(x);
            }

            constexpr size_t min_working_bytes = static_cast<size_t>(512) * 1024 * 1024;
            const size_t persistent_bytes = ggml_nbytes(x) * 2 + min_working_bytes;
            const size_t working_bytes = ctx->max_graph_vram_bytes > persistent_bytes
                                             ? ctx->max_graph_vram_bytes - persistent_bytes
                                             : min_working_bytes;
            const size_t bytes_per_token = static_cast<size_t>(3 * ffn_hidden_size + hidden_size) * sizeof(float);
            int64_t chunk_tokens = static_cast<int64_t>(working_bytes / std::max<size_t>(bytes_per_token, 1));
            const int64_t max_chunk_tokens = ctx->max_graph_vram_bytes <= static_cast<size_t>(20) * 1024 * 1024 * 1024
                                                 ? 8192
                                                 : 32768;
            chunk_tokens = std::clamp<int64_t>((chunk_tokens / 256) * 256, 2048, max_chunk_tokens);
            if (chunk_tokens >= sequence) {
                return forward_chunk(x);
            }

            ggml_tensor* output = nullptr;
            for (int64_t start = 0; start < sequence; start += chunk_tokens) {
                const int64_t end = std::min(sequence, start + chunk_tokens);
                auto chunk = ggml_ext_slice(ctx->ggml_ctx, x, 1, start, end, true);
                auto chunk_output = forward_chunk(chunk);
                output = output == nullptr ? chunk_output : ggml_concat(ctx->ggml_ctx, output, chunk_output, 1);
            }
            return output;
        }
    };

    static ggml_tensor* attention_layout(ggml_context* ctx, ggml_tensor* x) {
        x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
        return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
    }

    static ggml_tensor* apply_partial_rope(ggml_context* ctx,
                                           ggml_tensor* x,
                                           ggml_tensor* pe,
                                           const CudaOptimizationOptions* options = nullptr,
                                           ggml_backend_t backend = nullptr) {
        const bool fused_ops = h3_cuda_fused_enabled(options, backend);
        int64_t rot_dim = pe->ne[2] * 2;
        GGML_ASSERT(rot_dim <= x->ne[0]);
        const bool use_slice_views = fused_ops || !h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_ROPE_SLICE_VIEW");
        auto rotated = Rope::apply_rope(ctx,
                                        ggml_ext_slice(ctx, x, 0, 0, rot_dim, !use_slice_views),
                                        pe,
                                        false);
        if (rot_dim == x->ne[0]) {
            return rotated;
        }
        ggml_tensor* tail = nullptr;
        if ((fused_ops || h3_env_flag_enabled("ED_MINIMAX_H3_ROPE_TAIL_VIEW")) && x->ne[3] == 1) {
            tail = ggml_view_3d(ctx,
                                x,
                                x->ne[0] - rot_dim,
                                x->ne[2],
                                x->ne[1],
                                x->nb[2],
                                x->nb[1],
                                static_cast<size_t>(rot_dim) * x->nb[0]);
        } else {
            tail = attention_layout(ctx, ggml_ext_slice(ctx, x, 0, rot_dim, x->ne[0], !use_slice_views));
        }
        return ggml_concat(ctx, rotated, tail, 0);
    }

    static std::vector<ggml_tensor*> h3_qkv_views(ggml_context* ctx,
                                                  ggml_tensor* qkv,
                                                  int64_t head_dim,
                                                  int64_t heads,
                                                  int64_t sequence,
                                                  int64_t batch) {
        const int64_t inner = head_dim * heads;
        std::vector<ggml_tensor*> views;
        views.reserve(3);
        for (int i = 0; i < 3; ++i) {
            views.push_back(ggml_view_4d(ctx,
                                         qkv,
                                         head_dim,
                                         heads,
                                         sequence,
                                         batch,
                                         head_dim * qkv->nb[0],
                                         qkv->nb[1],
                                         qkv->nb[2],
                                         static_cast<size_t>(i) * inner * qkv->nb[0]));
        }
        return views;
    }

    struct Attention : public GGMLBlock {
        int64_t heads;
        int64_t head_dim;

        Attention(int64_t hidden_size,
                  int64_t heads,
                  int64_t head_dim,
                  float eps)
            : heads(heads), head_dim(head_dim) {
            int64_t inner      = heads * head_dim;
            blocks["qkv_proj"] = std::make_shared<Linear>(hidden_size, inner * 3, false);
            blocks["q_norm"]   = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["k_norm"]   = std::make_shared<RMSNorm>(head_dim, eps);
            blocks["out_proj"] = std::make_shared<Linear>(inner, hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* pe = nullptr,
                             const CudaOptimizationOptions* options = nullptr,
                             const SparseAttentionLayout* sparse_layout = nullptr,
                             const SparseAttentionConfig* sparse_config = nullptr,
                             int layer_idx = -1, int total_layers = -1,
                             bool optimized_step = false) {
            auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv_proj"]);
            auto q_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
            auto k_norm   = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);
            auto out_proj = std::dynamic_pointer_cast<Linear>(blocks["out_proj"]);

            int64_t sequence = x->ne[1];
            int64_t batch    = x->ne[2] * x->ne[3];
            const int64_t inner = heads * head_dim;
            const size_t attention_activation_bytes =
                static_cast<size_t>(sequence) * static_cast<size_t>(batch) *
                    static_cast<size_t>(3 * x->ne[0] + 6 * inner) * sizeof(float) +
                static_cast<size_t>(1) * 1024 * 1024 * 1024;
            const bool low_memory_attention = ctx->max_graph_vram_bytes > 0 &&
                                              sequence > 32768 &&
                                              attention_activation_bytes > ctx->max_graph_vram_bytes;
            std::vector<ggml_tensor*> qkv;
            if (low_memory_attention) {
                qkv.reserve(3);
                for (int projection = 0; projection < 3; ++projection) {
                    auto projected = qkv_proj->forward_output_slice(ctx, x, projection * inner, inner);
                    projected = ggml_reshape_4d(ctx->ggml_ctx, projected, head_dim, heads, sequence, batch);
                    qkv.push_back(projected);
                }
            } else {
                auto qkv_out = qkv_proj->forward(ctx, x);
                if (!h3_cuda_fused_enabled(options, ctx->backend) &&
                    h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_QKV_VIEW")) {
                    qkv = ggml_ext_chunk(ctx->ggml_ctx, qkv_out, 3, 0);
                    qkv[0] = ggml_reshape_4d(ctx->ggml_ctx, qkv[0], head_dim, heads, sequence, batch);
                    qkv[1] = ggml_reshape_4d(ctx->ggml_ctx, qkv[1], head_dim, heads, sequence, batch);
                    qkv[2] = ggml_reshape_4d(ctx->ggml_ctx, qkv[2], head_dim, heads, sequence, batch);
                } else {
                    qkv = h3_qkv_views(ctx->ggml_ctx, qkv_out, head_dim, heads, sequence, batch);
                }
            }
            auto q           = qkv[0];
            auto k           = qkv[1];
            auto v           = qkv[2];
            q                = q_norm->forward(ctx, q);
            k                = k_norm->forward(ctx, k);
            if (pe != nullptr) {
                q = apply_partial_rope(ctx->ggml_ctx, q, pe, options, ctx->backend);
                k = apply_partial_rope(ctx->ggml_ctx, k, pe, options, ctx->backend);
            } else {
                q = attention_layout(ctx->ggml_ctx, q);
                k = attention_layout(ctx->ggml_ctx, k);
            }
            if (low_memory_attention) {
                q = ggml_cast(ctx->ggml_ctx, q, GGML_TYPE_BF16);
                k = ggml_cast(ctx->ggml_ctx, k, GGML_TYPE_BF16);
                v = ggml_cast(ctx->ggml_ctx, v, GGML_TYPE_BF16);
            }
            ggml_tensor* out = nullptr;
            if (options && h3_cuda_backend(ctx->backend) && (optimized_step || options->sage_attn) &&
                layer_idx > 0 && layer_idx + 1 < total_layers) {
                if (options->sparse_attn) {
                    if (!sparse_layout || !sparse_config)
                        throw std::runtime_error("MiniMax-H3 SubBlock missing packed layout");
                    out = h3_subblock_attention(ctx, q, k, v, heads, 1.f / 128.f, *sparse_config, *sparse_layout);
                    if (!out) throw std::runtime_error("MiniMax-H3 SubBlock graph unsupported by selected backend");
                } else if (options->sage_attn) {
                    H3SageRequest request;
                    request.enabled = true; request.scope = H3SageScope::DiT;
                    request.layer_idx = layer_idx; request.total_layers = total_layers;
                    request.flash_attn = ctx->flash_attn_enabled;
                    const auto built = h3_sage_try_build(ctx->ggml_ctx, ctx->backend, q, k, v, heads, request);
                    if (built.route == H3SageRoute::Reject) throw std::runtime_error(built.reason);
                    out = built.output;
                }
            }
            if (!out) out = ggml_ext_attention_ext(ctx->ggml_ctx,
                                              ctx->backend,
                                              q,
                                              k,
                                              v,
                                              static_cast<int>(heads),
                                              nullptr,
                                              true,
                                              ctx->flash_attn_enabled,
                                              1.f / 128.f,
                                              sequence < 65536);
            return out_proj->forward(ctx, out);
        }
    };

    struct TokenRefinerBlock : public GGMLBlock {
        TokenRefinerBlock(const Config& config) {
            blocks["norm1"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"] = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]  = std::make_shared<Attention>(config.hidden_size,
                                                         config.num_attention_heads,
                                                         config.attention_head_dim,
                                                         config.qk_norm_eps);
            blocks["mlp"]   = std::make_shared<MLP>(config.hidden_size, config.ffn_hidden_size);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x,
                             const CudaOptimizationOptions* options = nullptr) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            x          = ggml_add(ctx->ggml_ctx, x, attn->forward(ctx, norm1->forward(ctx, x), nullptr, options));
            return ggml_add(ctx->ggml_ctx, x, mlp->forward(ctx, norm2->forward(ctx, x), options));
        }
    };

    struct TokenRefiner : public GGMLBlock {
        int64_t num_layers;

        explicit TokenRefiner(const Config& config)
            : num_layers(config.token_refiner_num_layers) {
            for (int64_t i = 0; i < num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TokenRefinerBlock>(config);
            }
            blocks["final_norm"] = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x,
                             const CudaOptimizationOptions* options = nullptr) {
            for (int64_t i = 0; i < num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TokenRefinerBlock>(blocks["blocks." + std::to_string(i)]);
                x          = block->forward(ctx, x, options);
            }
            return std::dynamic_pointer_cast<RMSNorm>(blocks["final_norm"])->forward(ctx, x);
        }
    };

    struct AdaLayerNormModulation : public GGMLBlock {
        int64_t hidden_size;
        int expand;
        int modalities;
        bool apply_silu;

        AdaLayerNormModulation(int64_t time_dim,
                               int64_t hidden_size,
                               int expand,
                               int modalities,
                               bool apply_silu,
                               bool force_f32)
            : hidden_size(hidden_size),
              expand(expand),
              modalities(modalities),
              apply_silu(apply_silu) {
            blocks["linear"] = std::make_shared<Linear>(time_dim,
                                                        hidden_size * expand * modalities,
                                                        true,
                                                        force_f32);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            if (apply_silu) {
                t_emb = ggml_silu(ctx->ggml_ctx, t_emb);
            }
            return std::dynamic_pointer_cast<Linear>(blocks["linear"])->forward(ctx, t_emb);
        }
    };

    struct TokenModulationSpan {
        int64_t start;
        int64_t end;
        int64_t modulation_row;
    };

    enum class SequenceKind {
        TEXT,
        CONDITION_VIDEO,
        CONDITION_AUDIO,
        TARGET_AUDIO,
        TARGET_VIDEO,
    };

    struct SequenceSegment {
        int64_t start;
        int64_t end;
        SequenceKind kind;
        int32_t source_index = -1;
    };

    static std::vector<ggml_tensor*> modulation_row(ggml_context* ctx,
                                                    ggml_tensor* projection,
                                                    int64_t hidden_size,
                                                    int expand,
                                                    int modalities,
                                                    int64_t row,
                                                    bool fused_ops = false) {
        int64_t timestep_rows = projection->ne[1];
        auto reshaped         = ggml_reshape_2d(ctx,
                                                projection,
                                                hidden_size * expand,
                                                timestep_rows * modalities);
        const bool use_views  = fused_ops || !h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_SEGMENT_VIEW");
        auto selected         = ggml_ext_slice(ctx, reshaped, 1, row, row + 1, !use_views);
        return ggml_ext_chunk(ctx, selected, expand, 0, !use_views);
    }

    static ggml_tensor* modulate_segments(ggml_context* ctx,
                                          ggml_tensor* x,
                                          ggml_tensor* projection,
                                          const std::vector<TokenModulationSpan>& segments,
                                          int64_t hidden_size,
                                          int expand,
                                          int modalities,
                                          int shift_index,
                                          int scale_index,
                                          const CudaOptimizationOptions* options = nullptr,
                                          ggml_backend_t backend = nullptr) {
        const bool fused_ops = h3_cuda_fused_enabled(options, backend);
        ggml_tensor* out = nullptr;
        for (const auto& segment : segments) {
            auto mods = modulation_row(ctx,
                                       projection,
                                       hidden_size,
                                       expand,
                                       modalities,
                                       segment.modulation_row,
                                       fused_ops);
            const bool use_views = fused_ops || !h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_SEGMENT_VIEW");
            auto part = ggml_ext_slice(ctx, x, 1, segment.start, segment.end, !use_views);
            const bool legacy_fused = h3_env_flag_enabled("ED_MINIMAX_H3_FUSED_SEGMENT_MODULATION") ||
                                      h3_env_flag_enabled("ED_MINIMAX_H3_FUSED_SEGMENT_MODULATE");
            if (fused_ops || legacy_fused) {
                auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx, part, mods[shift_index], mods[scale_index]);
                if (h3_fused_op_supported(fused, legacy_fused, backend)) {
                    part = fused;
                } else {
                    part = ggml_add(ctx,
                                    ggml_add(ctx, part, ggml_mul(ctx, part, mods[scale_index])),
                                    mods[shift_index]);
                }
            } else {
                part = ggml_add(ctx,
                                ggml_add(ctx, part, ggml_mul(ctx, part, mods[scale_index])),
                                mods[shift_index]);
            }
            out       = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    static ggml_tensor* gated_residual_segments(ggml_context* ctx,
                                                ggml_tensor* x,
                                                ggml_tensor* update,
                                                ggml_tensor* projection,
                                                const std::vector<TokenModulationSpan>& segments,
                                                int64_t hidden_size,
                                                int gate_index,
                                                const CudaOptimizationOptions* options = nullptr,
                                                ggml_backend_t backend = nullptr) {
        const bool fused_ops = h3_cuda_fused_enabled(options, backend);
        ggml_tensor* out = nullptr;
        for (const auto& segment : segments) {
            auto mods = modulation_row(ctx, projection, hidden_size, 6, 3, segment.modulation_row, fused_ops);
            const bool use_views = fused_ops || !h3_env_flag_enabled("ED_MINIMAX_H3_DISABLE_SEGMENT_VIEW");
            auto base = ggml_ext_slice(ctx, x, 1, segment.start, segment.end, !use_views);
            auto add  = ggml_ext_slice(ctx, update, 1, segment.start, segment.end, !use_views);
            ggml_tensor* part = nullptr;
            const bool legacy_fused = h3_env_flag_enabled("ED_MINIMAX_H3_FUSED_SEGMENT_MODULATION") ||
                                      h3_env_flag_enabled("ED_MINIMAX_H3_FUSED_SEGMENT_RESIDUAL_GATE");
            if (fused_ops || legacy_fused) {
                auto fused = edgedit::ggml_ext::fused_residual_gate_custom(ctx, base, add, mods[gate_index]);
                if (h3_fused_op_supported(fused, legacy_fused, backend)) {
                    part = fused;
                }
            }
            if (part == nullptr) {
                part = ggml_add(ctx, base, ggml_mul(ctx, add, mods[gate_index]));
            }
            out       = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    struct TransformerBlock : public GGMLBlock {
        Config config;

        explicit TransformerBlock(const Config& config)
            : config(config) {
            blocks["norm1"]      = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["norm2"]      = std::make_shared<RMSNorm>(config.hidden_size, config.norm_eps);
            blocks["attn"]       = std::make_shared<Attention>(config.hidden_size,
                                                         config.num_attention_heads,
                                                         config.attention_head_dim,
                                                         config.qk_norm_eps);
            blocks["mlp"]        = std::make_shared<MLP>(config.hidden_size,
                                                  config.ffn_hidden_size);
            blocks["adaln_proj"] = std::make_shared<AdaLayerNormModulation>(config.time_embed_dim,
                                                                            config.hidden_size,
                                                                            6,
                                                                            3,
                                                                            !config.uses_adaln_curves(),
                                                                            config.uses_adaln_curves());
        }

        ggml_tensor* project_adaln(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            return std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"])->forward(ctx, t_emb);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* t_emb,
                             const std::vector<TokenModulationSpan>& segments,
                             ggml_tensor* pe,
                             const std::string& cut_group,
                             const std::string& debug_target = {},
                             ggml_tensor** debug_output = nullptr,
                             const CudaOptimizationOptions* options = nullptr,
                             ggml_tensor* cached_modulation = nullptr,
                             const SparseAttentionLayout* sparse_layout = nullptr,
                             const SparseAttentionConfig* sparse_config = nullptr,
                             int layer_idx = -1, int total_layers = -1,
                             bool optimized_step = false) {
            auto norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm1"]);
            auto norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["norm2"]);
            auto attn  = std::dynamic_pointer_cast<Attention>(blocks["attn"]);
            auto mlp   = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto adaln = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto mods  = cached_modulation ? cached_modulation : adaln->forward(ctx, t_emb);
            if (debug_output != nullptr && debug_target == "block0_adaln") {
                *debug_output = mods;
                return x;
            }
            if (debug_output != nullptr && starts_with(debug_target, "block0_mod_")) {
                const int component = std::atoi(debug_target.substr(std::strlen("block0_mod_")).c_str());
                if (component >= 0 && component < 6 && !segments.empty()) {
                    auto selected = modulation_row(ctx->ggml_ctx,
                                                   mods,
                                                   config.hidden_size,
                                                   6,
                                                   3,
                                                   segments.front().modulation_row);
                    *debug_output = selected[component];
                    return x;
                }
            }

            auto h = modulate_segments(ctx->ggml_ctx,
                                       norm1->forward(ctx, x),
                                       mods,
                                       segments,
                                       config.hidden_size,
                                       6,
                                       3,
                                       0,
                                       1,
                                       options,
                                       ctx->backend);
            if (debug_output != nullptr && debug_target == "block0_attn_input") {
                *debug_output = h;
                return x;
            }
            auto attn_output = attn->forward(ctx, h, pe, options, sparse_layout, sparse_config,
                                             layer_idx, total_layers, optimized_step);
            if (debug_output != nullptr && debug_target == "block0_attn_output") {
                *debug_output = attn_output;
                return x;
            }
            x      = gated_residual_segments(ctx->ggml_ctx,
                                             x,
                                             attn_output,
                                             mods,
                                             segments,
                                             config.hidden_size,
                                             2,
                                             options,
                                             ctx->backend);
            if (debug_output != nullptr && debug_target == "block0_after_attn") {
                *debug_output = x;
                return x;
            }
            if (ctx->max_graph_vram_bytes > 0 && !cut_group.empty()) {
                sd::ggml_graph_cut::mark_graph_cut(x, cut_group + ".attn.output", "hidden_states");
            }
            h      = modulate_segments(ctx->ggml_ctx,
                                       norm2->forward(ctx, x),
                                       mods,
                                       segments,
                                       config.hidden_size,
                                       6,
                                       3,
                                       3,
                                       4,
                                       options,
                                       ctx->backend);
            if (debug_output != nullptr && debug_target == "block0_mlp_input") {
                *debug_output = h;
                return x;
            }
            auto mlp_output = mlp->forward(ctx, h, options);
            if (debug_output != nullptr && debug_target == "block0_mlp_output") {
                *debug_output = mlp_output;
                return x;
            }
            return gated_residual_segments(ctx->ggml_ctx,
                                           x,
                                           mlp_output,
                                           mods,
                                           segments,
                                           config.hidden_size,
                                           5,
                                           options,
                                           ctx->backend);
        }
    };

    struct FinalLayer : public GGMLBlock {
        Config config;

        explicit FinalLayer(const Config& config)
            : config(config) {
            int64_t video_dim    = config.video_latent_channels * config.patch_t * config.patch_h * config.patch_w;
            blocks["norm"]       = std::make_shared<RMSNorm>(config.hidden_size, config.final_norm_eps);
            blocks["adaln_proj"] = std::make_shared<AdaLayerNormModulation>(config.time_embed_dim,
                                                                            config.hidden_size,
                                                                            2,
                                                                            1,
                                                                            !config.uses_adaln_curves(),
                                                                            config.uses_adaln_curves());
            blocks["video_out"]  = std::make_shared<Linear>(config.hidden_size, video_dim, true, true);
            blocks["audio_out"]  = std::make_shared<Linear>(config.hidden_size, config.audio_latent_channels, true, true);
        }

        ggml_tensor* project_adaln(GGMLRunnerContext* ctx, ggml_tensor* t_emb) {
            return std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"])->forward(ctx, t_emb);
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* x,
                                                      ggml_tensor* t_emb,
                                                      const TokenModulationSpan& video,
                                                      const TokenModulationSpan& audio,
                                                      const CudaOptimizationOptions* options = nullptr,
                                                      ggml_tensor* cached_modulation = nullptr) {
            auto norm      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm"]);
            auto adaln     = std::dynamic_pointer_cast<AdaLayerNormModulation>(blocks["adaln_proj"]);
            auto video_out = std::dynamic_pointer_cast<Linear>(blocks["video_out"]);
            auto audio_out = std::dynamic_pointer_cast<Linear>(blocks["audio_out"]);
            auto mods      = cached_modulation ? cached_modulation : adaln->forward(ctx, t_emb);
            auto apply     = [&](const TokenModulationSpan& segment) {
                const bool fused_ops = h3_cuda_fused_enabled(options, ctx->backend);
                auto row   = modulation_row(ctx->ggml_ctx, mods, config.hidden_size, 2, 1, segment.modulation_row, fused_ops);
                auto value = norm->forward(ctx, ggml_ext_slice(ctx->ggml_ctx, x, 1, segment.start, segment.end));
                if (fused_ops) {
                    auto fused = edgedit::ggml_ext::fused_modulate_custom(ctx->ggml_ctx, value, row[0], row[1]);
                    if (h3_fused_op_supported(fused, false, ctx->backend)) {
                        return fused;
                    }
                }
                return ggml_add(ctx->ggml_ctx,
                                    ggml_add(ctx->ggml_ctx, value, ggml_mul(ctx->ggml_ctx, value, row[1])),
                                    row[0]);
            };
            return {video_out->forward(ctx, apply(video)),
                    audio_out->forward(ctx, apply(audio))};
        }
    };

    struct MiniMaxH3Transformer3DModel : public GGMLBlock {
        Config config;
        CudaOptimizationOptions cuda_optimizations;
        CudaDualBlockGraphPass* cuda_cache_pass = nullptr;
        int cuda_step = -1, cuda_total_steps = 0;
        bool cuda_conditional = true;
        SparseAttentionConfig sparse_config;
        ExactAdaLN::Cache* adaln_cache = nullptr;
        ExactAdaLN::Cache::StepToken adaln_token;
        ExactAdaLN::GraphInputs* adaln_graph_inputs = nullptr;

        explicit MiniMaxH3Transformer3DModel(const Config& config)
            : config(config) {
            int64_t video_dim          = config.video_latent_channels * config.patch_t * config.patch_h * config.patch_w;
            blocks["video_patch_proj"] = std::make_shared<Linear>(video_dim, config.hidden_size, true, true);
            blocks["audio_patch_proj"] = std::make_shared<Linear>(config.audio_latent_channels, config.hidden_size, true, true);
            blocks["condition_proj"]   = std::make_shared<Linear>(config.text_dim, config.hidden_size, true);
            if (!config.uses_adaln_curves()) {
                blocks["time_embedder"] = std::make_shared<TimeEmbedder>(config.timestep_input_dim,
                                                                         config.time_embed_hidden_size,
                                                                         config.time_embed_dim);
            }
            blocks["token_refiner"] = std::make_shared<TokenRefiner>(config);
            for (int64_t i = 0; i < config.num_layers; ++i) {
                blocks["blocks." + std::to_string(i)] = std::make_shared<TransformerBlock>(config);
            }
            blocks["final_layer"] = std::make_shared<FinalLayer>(config);
        }

        void init_params(ggml_context* ctx,
                         const String2TensorStorage& tensors = {},
                         const std::string prefix            = "") override {
            GGMLBlock::init_params(ctx, tensors, prefix);
            params["rope.inv_freq"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, config.rope_inv_freq_len);
            if (config.uses_adaln_curves()) {
                params["adaln_t_table"] = ggml_new_tensor_2d(ctx,
                                                             GGML_TYPE_F32,
                                                             config.time_embed_dim,
                                                             config.adaln_curve_grid);
            }
        }

        ggml_tensor* project_adaln(GGMLRunnerContext* ctx, ggml_tensor* t_emb, size_t layer) {
            if (layer < static_cast<size_t>(config.num_layers))
                return std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(layer)])->project_adaln(ctx, t_emb);
            return std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"])->project_adaln(ctx, t_emb);
        }

        ggml_tensor* refine_context(GGMLRunnerContext* ctx, ggml_tensor* context) {
            if (context->ne[0] == config.hidden_size) {
                return context;
            }
            GGML_ASSERT(context->ne[0] == config.text_dim);
            auto condition_proj = std::dynamic_pointer_cast<Linear>(blocks["condition_proj"]);
            auto token_refiner  = std::dynamic_pointer_cast<TokenRefiner>(blocks["token_refiner"]);
            return token_refiner->forward(ctx, condition_proj->forward(ctx, context), &cuda_optimizations);
        }

        ggml_tensor* time_embedding(GGMLRunnerContext* ctx,
                                    ggml_tensor* timestep_features,
                                    ggml_tensor* curve_indices,
                                    ggml_tensor* curve_upper_indices,
                                    ggml_tensor* curve_fractions) {
            if (!config.uses_adaln_curves()) {
                return std::dynamic_pointer_cast<TimeEmbedder>(blocks["time_embedder"])->forward(ctx, timestep_features);
            }
            auto lower = ggml_get_rows(ctx->ggml_ctx, params["adaln_t_table"], curve_indices);
            auto upper = ggml_get_rows(ctx->ggml_ctx, params["adaln_t_table"], curve_upper_indices);
            return ggml_add(ctx->ggml_ctx,
                            lower,
                            ggml_mul(ctx->ggml_ctx,
                                     ggml_sub(ctx->ggml_ctx, upper, lower),
                                     curve_fractions));
        }

        ggml_tensor* build_rope(GGMLRunnerContext* ctx,
                                ggml_tensor* position_ids) {
            auto inv            = ggml_reshape_2d(ctx->ggml_ctx,
                                                  params["rope.inv_freq"],
                                                  config.rope_inv_freq_len,
                                                  1);
            ggml_tensor* angles = nullptr;
            for (int axis = 0; axis < 3; ++axis) {
                auto pos          = ggml_ext_slice(ctx->ggml_ctx, position_ids, 0, axis, axis + 1);
                auto expanded_inv = ggml_repeat_4d(ctx->ggml_ctx,
                                                   inv,
                                                   inv->ne[0],
                                                   pos->ne[1],
                                                   1,
                                                   1);
                auto a            = ggml_mul(ctx->ggml_ctx, expanded_inv, pos);
                angles            = angles == nullptr ? a : ggml_concat(ctx->ggml_ctx, angles, a, 0);
            }
            auto c  = ggml_reshape_4d(ctx->ggml_ctx, ggml_cos(ctx->ggml_ctx, angles), 1, angles->ne[0], angles->ne[1], 1);
            auto s  = ggml_reshape_4d(ctx->ggml_ctx, ggml_sin(ctx->ggml_ctx, angles), 1, angles->ne[0], angles->ne[1], 1);
            auto ns = ggml_neg(ctx->ggml_ctx, s);
            auto pe = ggml_concat(ctx->ggml_ctx, c, ns, 0);
            pe      = ggml_concat(ctx->ggml_ctx, pe, s, 0);
            pe      = ggml_concat(ctx->ggml_ctx, pe, c, 0);
            return ggml_reshape_4d(ctx->ggml_ctx, pe, 2, 2, angles->ne[0], angles->ne[1]);
        }

        static ggml_tensor* cache_math(GGMLRunnerContext* ctx, CudaDualBlockGraphPass& pass,
                                        const char* id, const std::vector<ggml_tensor*>& inputs,
                                        const std::vector<float>& weights = {}) {
            const auto* op = pass.operators->find(id);
            edgedit::cache::GraphLoweringContext lowering;
            lowering.ctx = ctx->ggml_ctx;
            edgedit::cache::CacheOperatorParams params;
            params.floats = weights;
            std::vector<ggml_tensor*> outputs;
            if (!op || !op->lower(lowering, inputs, params, &outputs) || outputs.empty()) {
                throw std::runtime_error("MiniMax-H3: cannot lower cache operator");
            }
            return outputs.front();
        }

        static void cache_pin(GGMLRunnerContext* ctx, CudaDualBlockGraphPass& pass,
                              ggml_tensor* tensor, const std::string& name) {
            if (!ctx->tap_registry) throw std::logic_error("MiniMax-H3: missing cache tap registry");
            auto extensions = ctx->tap_registry->extensions();
            edgedit::cache::GraphExtension extension;
            extension.op = pass.operators->find("cache.copy");
            extension.extra_inputs = {tensor};
            extension.output_name = name;
            extensions.push_back(std::move(extension));
            ctx->tap_registry->set_extensions(std::move(extensions));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      ggml_tensor* video,
                                                      ggml_tensor* audio,
                                                      ggml_tensor* context,
                                                      const std::vector<ggml_tensor*>& condition_videos,
                                                      const std::vector<ggml_tensor*>& condition_audios,
                                                      ggml_tensor* position_ids,
                                                      ggml_tensor* timestep_features,
                                                      ggml_tensor* curve_indices,
                                                      ggml_tensor* curve_upper_indices,
                                                      ggml_tensor* curve_fractions,
                                                      const std::vector<TokenModulationSpan>& segments,
                                                      const std::vector<SequenceSegment>& sequence_segments,
                                                      const TokenModulationSpan& video_segment,
                                                      const TokenModulationSpan& audio_segment,
                                                      ggml_tensor* audio_slope,
                                                      const std::string& debug_target = {},
                                                      ggml_tensor** debug_output = nullptr) {
            auto video_proj = std::dynamic_pointer_cast<Linear>(blocks["video_patch_proj"]);
            auto audio_proj = std::dynamic_pointer_cast<Linear>(blocks["audio_patch_proj"]);
            const bool remainder = cuda_cache_pass &&
                                   cuda_cache_pass->phase == CudaDualBlockGraphPass::Phase::Remainder;
            ggml_tensor* h = remainder ? cuda_cache_pass->front : nullptr;
            const int64_t target_audio_length = audio->ne[0];
            const int64_t target_audio_batch = audio->ne[3];
            if (!remainder) {
                std::vector<std::pair<int64_t, int64_t>> condition_video_ranges;
                ggml_tensor* video_rows = nullptr;
                int64_t video_offset    = 0;
                for (auto condition : condition_videos) {
                    auto rows = DiT::patchify_3d(ctx->ggml_ctx,
                                                 condition,
                                                 config.patch_t,
                                                 config.patch_h,
                                                 config.patch_w,
                                                 1,
                                                 true);
                    condition_video_ranges.push_back({video_offset, video_offset + rows->ne[1]});
                    video_offset += rows->ne[1];
                    video_rows = video_rows == nullptr ? rows : ggml_concat(ctx->ggml_ctx, video_rows, rows, 1);
                }
                auto target_video_rows                         = DiT::patchify_3d(ctx->ggml_ctx,
                                                                                  video,
                                                                                  config.patch_t,
                                                                                  config.patch_h,
                                                                                  config.patch_w,
                                                                                  1,
                                                                                  true);
                std::pair<int64_t, int64_t> target_video_range = {video_offset, video_offset + target_video_rows->ne[1]};
                video_rows                                     = video_rows == nullptr ? target_video_rows
                                                                                       : ggml_concat(ctx->ggml_ctx, video_rows, target_video_rows, 1);
                auto video_embeds                              = video_proj->forward(ctx, video_rows);

                auto pack_audio_rows = [&](ggml_tensor* value) {
                    value = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, value, 2, 0, 1, 3));
                    return ggml_reshape_3d(ctx->ggml_ctx,
                                           value,
                                           value->ne[0],
                                           value->ne[1] * value->ne[2],
                                           value->ne[3]);
                };
                std::vector<std::pair<int64_t, int64_t>> condition_audio_ranges;
                ggml_tensor* audio_rows = nullptr;
                int64_t audio_offset    = 0;
                for (auto condition : condition_audios) {
                    auto rows = pack_audio_rows(condition);
                    condition_audio_ranges.push_back({audio_offset, audio_offset + rows->ne[1]});
                    audio_offset += rows->ne[1];
                    audio_rows = audio_rows == nullptr ? rows : ggml_concat(ctx->ggml_ctx, audio_rows, rows, 1);
                }
                audio                                          = pack_audio_rows(audio);
                std::pair<int64_t, int64_t> target_audio_range = {audio_offset, audio_offset + audio->ne[1]};
                audio_rows                                     = audio_rows == nullptr ? audio : ggml_concat(ctx->ggml_ctx, audio_rows, audio, 1);
                auto audio_embeds                              = audio_proj->forward(ctx, audio_rows);
                context                                        = refine_context(ctx, context);

                auto append    = [&](ggml_tensor* value) {
                    h = h == nullptr ? value : ggml_concat(ctx->ggml_ctx, h, value, 1);
                };
                for (const auto& sequence : sequence_segments) {
                    if (sequence.kind == SequenceKind::TEXT) {
                        append(context);
                    } else if (sequence.kind == SequenceKind::CONDITION_VIDEO) {
                        GGML_ASSERT(sequence.source_index >= 0 &&
                                    sequence.source_index < static_cast<int32_t>(condition_video_ranges.size()));
                        auto range = condition_video_ranges[static_cast<size_t>(sequence.source_index)];
                        append(ggml_ext_slice(ctx->ggml_ctx, video_embeds, 1, range.first, range.second));
                    } else if (sequence.kind == SequenceKind::CONDITION_AUDIO) {
                        GGML_ASSERT(sequence.source_index >= 0 &&
                                    sequence.source_index < static_cast<int32_t>(condition_audio_ranges.size()));
                        auto range = condition_audio_ranges[static_cast<size_t>(sequence.source_index)];
                        append(ggml_ext_slice(ctx->ggml_ctx, audio_embeds, 1, range.first, range.second));
                    } else if (sequence.kind == SequenceKind::TARGET_AUDIO) {
                        append(ggml_ext_slice(ctx->ggml_ctx,
                                              audio_embeds,
                                              1,
                                              target_audio_range.first,
                                              target_audio_range.second));
                    } else {
                        append(ggml_ext_slice(ctx->ggml_ctx,
                                              video_embeds,
                                              1,
                                              target_video_range.first,
                                              target_video_range.second));
                    }
                }
            }
            GGML_ASSERT(h != nullptr);

            auto t_emb = time_embedding(ctx,
                                        timestep_features,
                                        curve_indices,
                                        curve_upper_indices,
                                        curve_fractions);
            if (debug_output != nullptr) {
                if (debug_target == "temb") {
                    *debug_output = t_emb;
                    return {nullptr, nullptr};
                }
                if (debug_target == "packed_input") {
                    *debug_output = h;
                    return {nullptr, nullptr};
                }
            }
            auto pe    = build_rope(ctx, position_ids);
            SparseAttentionLayout sparse_layout;
            if (cuda_optimizations.sparse_attn) {
                for (const auto& span : sequence_segments) {
                    auto kind = span.kind == SequenceKind::TARGET_VIDEO ? SparseTokenKind::TargetVideo
                        : span.kind == SequenceKind::TEXT ? SparseTokenKind::Text
                        : (span.kind == SequenceKind::TARGET_AUDIO || span.kind == SequenceKind::CONDITION_AUDIO)
                            ? SparseTokenKind::Audio : SparseTokenKind::Reference;
                    sparse_layout.spans.push_back({span.start, span.end, kind});
                }
                std::string reason;
                if (!h3_subblock_graph_budget(sparse_layout, config.num_attention_heads,
                                             config.num_layers, H3_GRAPH_SIZE, &reason))
                    throw std::invalid_argument(reason);
            }
            const bool optimized_step = cuda_step >= cuda_optimizations.sparse_warmup_steps &&
                                        cuda_step + 1 < cuda_total_steps;
            auto* cache_pass = cuda_cache_pass;
            const int64_t middle_end = cache_pass ? config.num_layers - cache_pass->back_blocks : 0;
            ggml_tensor* stack_input = h;
            ggml_tensor* middle_input = h;
            int64_t first_block = remainder ? cache_pass->front_blocks : 0;
            if (remainder && cache_pass->reuse_middle) {
                h = cache_math(ctx, *cache_pass, "cache.weighted_blend",
                               {h, cache_pass->middle_delta}, {1.f, 1.f});
                first_block = middle_end;
            }
            for (int64_t i = first_block; i < config.num_layers; ++i) {
                auto block = std::dynamic_pointer_cast<TransformerBlock>(blocks["blocks." + std::to_string(i)]);
                const std::string cut_group = "minimax_h3.blocks." + std::to_string(i);
                ggml_tensor* cached_modulation = nullptr;
                if (adaln_cache && adaln_graph_inputs) {
                    ExactAdaLN::Shape expected{};
                    auto status = adaln_cache->expected_shape(adaln_token, static_cast<size_t>(i), &expected);
                    if (status == ExactAdaLN::Status::Hit)
                        status = adaln_graph_inputs->import(ctx, adaln_token, static_cast<size_t>(i), expected, &cached_modulation);
                    if (status != ExactAdaLN::Status::Hit)
                        throw std::runtime_error("MiniMax-H3 exact AdaLN projection unavailable");
                }
                h          = block->forward(ctx,
                                            h,
                                            t_emb,
                                            segments,
                                            pe,
                                            cut_group,
                                            i == 0 ? debug_target : std::string(),
                                            i == 0 ? debug_output : nullptr,
                                            &cuda_optimizations, cached_modulation,
                                            &sparse_layout, &sparse_config,
                                            static_cast<int>(i), static_cast<int>(config.num_layers), optimized_step);
                if (debug_output != nullptr && *debug_output != nullptr) {
                    return {nullptr, nullptr};
                }
                if (debug_output != nullptr && debug_target == "block0" && i == 0) {
                    *debug_output = h;
                    return {nullptr, nullptr};
                }
                sd::ggml_graph_cut::mark_graph_cut(h,
                                                   cut_group,
                                                   "hidden_states");
                if (cache_pass && !remainder && i + 1 == cache_pass->front_blocks) {
                    // RF = CURRENT real front - CURRENT stack input, not latent
                    // output diff. Pin current front/RF for device-only handoff.
                    cache_pass->current_rf = cache_math(ctx, *cache_pass, "cache.difference", {stack_input, h});
                    cache_pin(ctx, *cache_pass, h, "h3_cache_front");
                    cache_pin(ctx, *cache_pass, cache_pass->current_rf, "h3_cache_rf");
                    cache_pass->indicator_names.clear();
                    if (cache_pass->reference_rf &&
                        ggml_are_same_shape(cache_pass->reference_rf, cache_pass->current_rf)) {
                        for (size_t span = 0; span < sequence_segments.size(); ++span) {
                            const auto& segment = sequence_segments[span];
                            if (segment.end <= segment.start) continue;
                            auto current = ggml_ext_slice(ctx->ggml_ctx, cache_pass->current_rf,
                                                          1, segment.start, segment.end, false);
                            auto reference = ggml_ext_slice(ctx->ggml_ctx, cache_pass->reference_rf,
                                                            1, segment.start, segment.end, false);
                            const std::string name = "h3_rf_span_" + std::to_string(span);
                            auto metric = cache_math(ctx, *cache_pass, "cache.rel_l1", {current, reference});
                            cache_pin(ctx, *cache_pass, metric, "cache_ind:" + name);
                            cache_pass->indicator_names.push_back(name);
                        }
                    }
                    return {h, nullptr}; // probe never executes middle, suffix or output head
                }
                if (cache_pass && remainder && !cache_pass->reuse_middle && i + 1 == middle_end) {
                    auto delta = cache_math(ctx, *cache_pass, "cache.difference", {middle_input, h});
                    cache_pin(ctx, *cache_pass, delta, "h3_cache_middle_delta");
                }
            }

            auto final_layer = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);
            ggml_tensor* final_modulation = nullptr;
            if (adaln_cache && adaln_graph_inputs) {
                ExactAdaLN::Shape expected{};
                auto status = adaln_cache->expected_shape(adaln_token, static_cast<size_t>(config.num_layers), &expected);
                if (status == ExactAdaLN::Status::Hit)
                    status = adaln_graph_inputs->import(ctx, adaln_token, static_cast<size_t>(config.num_layers), expected, &final_modulation);
                if (status != ExactAdaLN::Status::Hit)
                    throw std::runtime_error("MiniMax-H3 exact AdaLN final projection unavailable");
            }
            auto output      = final_layer->forward(ctx, h, t_emb, video_segment, audio_segment, &cuda_optimizations, final_modulation);
            auto video_out   = DiT::unpatchify_3d(ctx->ggml_ctx,
                                                  output.first,
                                                  video->ne[2] / config.patch_t,
                                                  video->ne[1] / config.patch_h,
                                                  video->ne[0] / config.patch_w,
                                                  config.patch_t,
                                                  config.patch_h,
                                                  config.patch_w,
                                                  true);
            auto audio_out   = ggml_reshape_4d(ctx->ggml_ctx,
                                               output.second,
                                               config.audio_latent_channels,
                                               target_audio_length,
                                               2,
                                               target_audio_batch);
            audio_out        = ggml_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, audio_out, 1, 2, 0, 3));
            video_out        = ggml_ext_scale(ctx->ggml_ctx, video_out, -1.f);
            audio_out        = ggml_mul(ctx->ggml_ctx, audio_out, ggml_neg(ctx->ggml_ctx, audio_slope));
            return {video_out, audio_out};
        }
    };

    struct PackedSequenceLayout {
        std::vector<float> positions;
        std::vector<TokenModulationSpan> segments;
        std::vector<SequenceSegment> sequence_segments;
        TokenModulationSpan video_segment{};
        TokenModulationSpan audio_segment{};
        std::vector<float> timesteps;
    };

    static float video_span(int64_t frame) {
        static const int spans[5] = {1, 4, 4, 4, 4};
        return FRAME_RESCALE * spans[frame % 5];
    }

    static std::vector<float> spatial_axis(int64_t dim,
                                           float sqrt_area) {
        int64_t count = dim / 2;
        float ratio   = static_cast<float>(dim) / sqrt_area;
        std::vector<float> result(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            result[static_cast<size_t>(i)] =
                (static_cast<float>(i) * (ratio / count) + (1.f - ratio) * 0.5f) * 32.f;
        }
        return result;
    }

    static int find_or_add_timestep(std::vector<float>* values, float value) {
        auto it = std::find(values->begin(), values->end(), value);
        if (it != values->end()) {
            return static_cast<int>(it - values->begin());
        }
        values->push_back(value);
        return static_cast<int>(values->size() - 1);
    }

    static PackedSequenceLayout build_layout(int64_t text_len,
                                             int64_t latent_t,
                                             int64_t latent_h,
                                             int64_t latent_w,
                                             int64_t audio_t,
                                             const std::vector<sd::Tensor<float>>& condition_videos,
                                             const std::vector<sd::Tensor<float>>& condition_audios,
                                             const sd::Tensor<int32_t>& keyframe_indices,
                                             const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
                                             const sd::Tensor<int32_t>& text_tags,
                                             float video_t,
                                             float audio_timestep) {
        PackedSequenceLayout layout;
        float sqrt_area    = std::sqrt(static_cast<float>(latent_h * latent_w));
        auto h_axis        = spatial_axis(latent_h, sqrt_area);
        auto w_axis        = spatial_axis(latent_w, sqrt_area);
        int64_t frame_rows = static_cast<int64_t>(h_axis.size() * w_axis.size());
        int64_t row        = 0;

        auto append_position = [&](float t, float h, float w) {
            layout.positions.push_back(t);
            layout.positions.push_back(h);
            layout.positions.push_back(w);
        };
        for (int64_t i = 0; i < text_len; ++i) {
            append_position(static_cast<float>(i), 0.f, 0.f);
        }
        layout.sequence_segments.push_back({0, text_len, SequenceKind::TEXT});

        int video_time_row           = find_or_add_timestep(&layout.timesteps, video_t);
        int audio_time_row           = find_or_add_timestep(&layout.timesteps, audio_timestep);
        int condition_time_row       = find_or_add_timestep(&layout.timesteps,
                                                            std::max(video_t, VISUAL_COND_TIMESTEP));
        int audio_condition_time_row = find_or_add_timestep(&layout.timesteps,
                                                            std::max(audio_timestep, 1.f));

        if (text_len > 0 && text_tags.empty()) {
            layout.segments.push_back({0, text_len, video_time_row * 3 + 1});
        } else if (text_len > 0) {
            int64_t run_start = 0;
            int current_tag   = text_tags[0];
            for (int64_t i = 1; i <= text_len; ++i) {
                int tag = i < text_len ? text_tags[i] : -1;
                if (i == text_len || tag != current_tag) {
                    layout.segments.push_back({run_start,
                                               i,
                                               video_time_row * 3 + current_tag});
                    run_start   = i;
                    current_tag = tag;
                }
            }
        }
        row = text_len;

        auto condition_spatial_axes = [&](const sd::Tensor<float>& condition) {
            float area = std::sqrt(static_cast<float>(condition.shape()[0] * condition.shape()[1]));
            return std::make_pair(spatial_axis(condition.shape()[1], area),
                                  spatial_axis(condition.shape()[0], area));
        };
        auto append_video_positions = [&](const sd::Tensor<float>& condition,
                                          float cursor) {
            auto axes = condition_spatial_axes(condition);
            for (int64_t t = 0; t < condition.shape()[2]; ++t) {
                for (float h : axes.first) {
                    for (float w : axes.second) {
                        append_position(cursor, h, w);
                    }
                }
                cursor += video_span(t);
            }
            return cursor;
        };
        auto append_audio_positions = [&](int64_t length,
                                          float cursor,
                                          float w_low,
                                          float w_high) {
            for (int channel = 0; channel < 2; ++channel) {
                float w = channel == 0 ? w_low : w_high;
                for (int64_t t = 0; t < length; ++t) {
                    append_position(cursor + static_cast<float>(t), 0.f, w);
                }
            }
        };

        float cursor = static_cast<float>(text_len);
        if (reference_blocks.empty()) {
            float video_duration = 0.f;
            for (int64_t t = 0; t < latent_t; ++t) {
                video_duration += video_span(t);
            }
            for (size_t index = 0; index < condition_videos.size(); ++index) {
                const auto& condition = condition_videos[index];
                auto axes             = condition_spatial_axes(condition);
                int64_t count         = condition.shape()[2] *
                                static_cast<int64_t>(axes.first.size() * axes.second.size());
                bool is_first    = keyframe_indices.empty() || keyframe_indices[static_cast<int64_t>(index)] == 0;
                float keyframe_t = is_first ? static_cast<float>(text_len)
                                            : static_cast<float>(text_len) + video_duration - FRAME_RESCALE;
                for (int64_t t = 0; t < condition.shape()[2]; ++t) {
                    for (float h : axes.first) {
                        for (float w : axes.second) {
                            append_position(keyframe_t, h, w);
                        }
                    }
                }
                layout.sequence_segments.push_back({row,
                                                    row + count,
                                                    SequenceKind::CONDITION_VIDEO,
                                                    static_cast<int32_t>(index)});
                layout.segments.push_back({row, row + count, condition_time_row * 3});
                row += count;
            }
        } else {
            for (const auto& block : reference_blocks) {
                const sd::Tensor<float>* ref_video = nullptr;
                const sd::Tensor<float>* ref_audio = nullptr;
                if (block.video_index >= 0) {
                    GGML_ASSERT(block.video_index < static_cast<int32_t>(condition_videos.size()));
                    ref_video = &condition_videos[static_cast<size_t>(block.video_index)];
                }
                if (block.audio_index >= 0) {
                    GGML_ASSERT(block.audio_index < static_cast<int32_t>(condition_audios.size()));
                    ref_audio = &condition_audios[static_cast<size_t>(block.audio_index)];
                }

                float block_end = cursor;
                if (block.kind == MiniMaxH3ReferenceKind::AUDIO ||
                    block.kind == MiniMaxH3ReferenceKind::VIDEO_AUDIO) {
                    GGML_ASSERT(ref_audio != nullptr);
                    float w_low  = w_axis.front();
                    float w_high = w_axis.back();
                    if (ref_video != nullptr) {
                        auto axes = condition_spatial_axes(*ref_video);
                        w_low     = axes.second.front();
                        w_high    = axes.second.back();
                    }
                    int64_t count = ref_audio->shape()[0] * 2;
                    append_audio_positions(ref_audio->shape()[0], cursor, w_low, w_high);
                    layout.sequence_segments.push_back({row,
                                                        row + count,
                                                        SequenceKind::CONDITION_AUDIO,
                                                        block.audio_index});
                    layout.segments.push_back({row,
                                               row + count,
                                               audio_condition_time_row * 3 + 2});
                    row += count;
                    block_end = std::max(block_end, cursor + static_cast<float>(ref_audio->shape()[0]));
                }

                if (block.kind != MiniMaxH3ReferenceKind::AUDIO) {
                    GGML_ASSERT(ref_video != nullptr);
                    auto axes     = condition_spatial_axes(*ref_video);
                    int64_t count = ref_video->shape()[2] *
                                    static_cast<int64_t>(axes.first.size() * axes.second.size());
                    float video_end = append_video_positions(*ref_video, cursor);
                    layout.sequence_segments.push_back({row,
                                                        row + count,
                                                        SequenceKind::CONDITION_VIDEO,
                                                        block.video_index});
                    layout.segments.push_back({row, row + count, condition_time_row * 3});
                    row += count;
                    block_end = block.kind == MiniMaxH3ReferenceKind::IMAGE
                                    ? std::max(block_end, cursor + 1.f)
                                    : std::max(block_end, video_end);
                }
                cursor = block_end;
            }
        }

        int64_t audio_start = row;
        append_audio_positions(audio_t, cursor, w_axis.front(), w_axis.back());
        layout.audio_segment = {audio_start, row + audio_t * 2, audio_time_row};
        layout.sequence_segments.push_back({audio_start,
                                            row + audio_t * 2,
                                            SequenceKind::TARGET_AUDIO});
        layout.segments.push_back({audio_start,
                                   row + audio_t * 2,
                                   audio_time_row * 3 + 2});
        row += audio_t * 2;

        int64_t video_start = row;
        for (int64_t t = 0; t < latent_t; ++t) {
            for (float h : h_axis) {
                for (float w : w_axis) {
                    append_position(cursor, h, w);
                }
            }
            cursor += video_span(t);
        }
        int64_t video_rows   = latent_t * frame_rows;
        layout.video_segment = {video_start, video_start + video_rows, video_time_row};
        layout.sequence_segments.push_back({video_start,
                                            video_start + video_rows,
                                            SequenceKind::TARGET_VIDEO});
        layout.segments.push_back({video_start,
                                   video_start + video_rows,
                                   video_time_row * 3});
        return layout;
    }

    struct MiniMaxH3Runner : public CacheOffload::RunnerBridge<ExactAdaLN::RunnerBridge<GGMLRunner>> {
        using ResearchRunnerBase = CacheOffload::RunnerBridge<ExactAdaLN::RunnerBridge<GGMLRunner>>;
        Config config;
        MiniMaxH3Transformer3DModel model;
        struct AdaLNBranch {
            std::unique_ptr<ExactAdaLN::Cache> cache;
            ExactAdaLN::RequestContract contract;
        };
        std::array<AdaLNBranch, 2> adaln_branches;
        std::unique_ptr<ExactAdaLN::GraphInputs> adaln_inputs;
        std::vector<float> adaln_sigmas;
        std::string adaln_weights_identity, adaln_schedule_identity;
        uint64_t adaln_request_generation = 0;
        CudaDualBlockCachePolicy cuda_cache_policy;
        std::unique_ptr<RunnerCacheDeviceStore> cuda_cache_store;
        std::unique_ptr<CacheOffload::State> cuda_cache_offload_state;
        edgedit::cache::CacheOperatorRegistry cuda_cache_operators;
        struct CudaCacheBranchTensors {
            ggml_tensor* rf = nullptr;
            ggml_tensor* delta = nullptr;
        };
        std::array<CudaCacheBranchTensors, 2> cuda_cache_branches{};
        sd::Tensor<float> video_input_cache;
        sd::Tensor<float> audio_input_cache;
        sd::Tensor<float> position_input_cache;
        sd::Tensor<float> timestep_feature_input_cache;
        sd::Tensor<float> audio_slope_input_cache;
        sd::Tensor<int32_t> curve_index_input_cache;
        sd::Tensor<int32_t> curve_upper_index_input_cache;
        sd::Tensor<float> curve_fraction_input_cache;

        MiniMaxH3Runner(ggml_backend_t backend,
                        const String2TensorStorage& tensors,
                        const std::string& prefix                           = "model.diffusion_model",
                        bool offload_params_to_cpu = false)
            : ResearchRunnerBase(backend, offload_params_to_cpu),
              config(Config::detect_from_weights(tensors, prefix)),
              model(config) {
            model.init(params_ctx, tensors, prefix);
            edgedit::cache::register_builtin_cache_operators(&cuda_cache_operators);
        }

        std::string get_desc() override {
            return "minimax_h3";
        }

        static CudaOptimizationCapabilities cuda_optimization_capabilities() {
            // Dual-block execution is implemented; execution-mode restrictions
            // are checked by configure and compute, never silently downgraded.
            CudaOptimizationCapabilities capabilities;
            capabilities.cache_dit = true;
            capabilities.sparse_attn = true;
            capabilities.adaln_cache = true;
#if defined(ED_ENABLE_CUDA_SAGE_ATTN)
            capabilities.sage_attn = true;
#endif
#if defined(ED_ENABLE_CUDA_MODULATION)
            capabilities.fused_ops = true;
#endif
            return capabilities;
        }

        void configure_cuda_optimizations(const CudaOptimizationOptions& options) {
            std::string error;
            if (!validate_cuda_optimization_options(options,
                                                     h3_cuda_backend(runtime_backend),
                                                     cuda_optimization_capabilities(), &error)) {
                throw std::invalid_argument(error);
            }
            if (options.cache_dit) validate_cuda_cache_execution_mode();
            validate_cuda_research_execution_mode(options);
            cuda_cache_policy.configure(options, config.num_layers);
            // Changing graph-affecting fusion invalidates even when turning it
            // off. Identical default-off configure/reset leaves baseline reuse
            // safety and performance under the original runner lifecycle.
            if (model.cuda_optimizations.fused_ops != options.fused_ops ||
                model.cuda_optimizations.cache_dit != options.cache_dit ||
                model.cuda_optimizations.adaln_cache != options.adaln_cache ||
                model.cuda_optimizations.sparse_attn != options.sparse_attn ||
                model.cuda_optimizations.sage_attn != options.sage_attn) {
                invalidate_persistent_graph();
                reset_graph_cut_run_cache();
            }
            reset_cuda_optimization_state();
            model.cuda_optimizations = options;
            model.sparse_config.sparse_keep_ratio = options.sparse_keep_ratio;
            model.sparse_config.sparse_block_tokens = options.sparse_block_tokens;
        }

        void begin_cuda_step(int step, int total_steps, bool conditional) {
            if (model.cuda_optimizations.any_enabled() && (step < 0 || total_steps <= 0 || step >= total_steps))
                throw std::invalid_argument("MiniMax-H3 invalid optimization step metadata");
            model.cuda_step = step; model.cuda_total_steps = total_steps; model.cuda_conditional = conditional;
            if (model.cuda_optimizations.cache_dit) cuda_cache_policy.begin_step(step, total_steps, conditional);
        }

        bool serial_cuda_offload_mode() const {
            if (!h3_cuda_backend(runtime_backend) || !ggml_backend_is_cpu(params_backend) ||
                params_backend == runtime_backend || params_on_runtime_backend || phase_params_pinned_ ||
                !max_graph_vram_bytes || weight_adapter || (process_group_ && process_group_->enabled()) ||
                h3_env_flag_enabled("ED_MINIMAX_H3_REUSE_GRAPH") ||
                std::getenv("ED_MINIMAX_H3_DEBUG_TARGET") != nullptr || h3_env_flag_enabled("ED_ASYNC_OFFLOAD")) return false;
#if defined(ED_ENABLE_ASYNC_OFFLOAD)
            if (async_offload_level() != 0) return false;
#endif
            auto* device = ggml_backend_get_device(runtime_backend);
            auto* registry = device ? ggml_backend_dev_backend_reg(device) : nullptr;
            return registry && std::string(ggml_backend_reg_name(registry)) == "CUDA" &&
                   ggml_backend_reg_dev_count(registry) == 1 &&
                   ggml_backend_reg_dev_get(registry, 0) == device;
        }

        void validate_cuda_cache_execution_mode() const {
            if (!h3_cuda_backend(runtime_backend))
                throw std::invalid_argument("MiniMax-H3: dual-block cache requires CUDA");
            if (params_backend != runtime_backend) {
                if (!serial_cuda_offload_mode())
                    throw std::invalid_argument("MiniMax-H3: dual-block offload requires immutable CPU parameters, single CUDA, explicit budget, no async/adapter/reuse");
            } else if (can_attempt_graph_cut_segmented_compute()) {
                throw std::invalid_argument("MiniMax-H3: dual-block cache cannot use parallel graph cuts");
            }
            if (h3_env_flag_enabled("ED_MINIMAX_H3_REUSE_GRAPH"))
                throw std::invalid_argument("MiniMax-H3: dual-block cache cannot use ED_MINIMAX_H3_REUSE_GRAPH");
            if (std::getenv("ED_MINIMAX_H3_DEBUG_TARGET") != nullptr)
                throw std::invalid_argument("MiniMax-H3: dual-block cache cannot use a debug-target output");
            if (weight_adapter)
                throw std::invalid_argument("MiniMax-H3: dual-block cache does not support mutable weight adapters");
        }

        void validate_cuda_research_execution_mode(const CudaOptimizationOptions& options) const {
            if (options.adaln_cache) {
                if (params_backend != runtime_backend && (options.cache_dit || options.sage_attn || options.sparse_attn))
                    throw std::invalid_argument("MiniMax-H3 AdaLN offload requires independent single-method validation");
                if (weight_adapter || (process_group_ && process_group_->enabled()) ||
                    h3_env_flag_enabled("ED_MINIMAX_H3_REUSE_GRAPH"))
                    throw std::invalid_argument("MiniMax-H3 exact AdaLN requires immutable single-device parameters and graph reuse disabled");
                if (params_backend != runtime_backend) {
                    // Explicit engineering contract, not deletion of the resident guard:
                    // layer-major prefill + request-owned imports + serial graph cuts.
                    if (!serial_cuda_offload_mode())
                        throw std::invalid_argument("MiniMax-H3 exact AdaLN offload requires resident immutable CPU weights, single CUDA runtime, explicit segment budget, no async/reuse/debug");
                } else if (can_attempt_graph_cut_segmented_compute()) {
                    throw std::invalid_argument("MiniMax-H3 exact AdaLN requires immutable single-device parameters");
                }
            }
            if (options.sparse_attn || options.sage_attn) {
                if (weight_adapter || (process_group_ && process_group_->enabled()) ||
                    h3_env_flag_enabled("ED_MINIMAX_H3_REUSE_GRAPH"))
                    throw std::invalid_argument("MiniMax-H3 attention requires resident immutable single-device parameters and graph reuse disabled");
                if (params_backend != runtime_backend) {
                    if (!serial_cuda_offload_mode() || options.adaln_cache || options.cache_dit ||
                        (options.sage_attn && options.sparse_attn))
                        throw std::invalid_argument("MiniMax-H3 attention offload requires resident immutable CPU weights, single CUDA and one method, no async/reuse");
#ifndef ED_ENABLE_CUDA_H3_SUBBLOCK
                    if (options.sparse_attn)
                        throw std::invalid_argument("MiniMax-H3 SubBlock offload requires bounded native CUDA build");
#endif
                } else if (can_attempt_graph_cut_segmented_compute()) {
                    throw std::invalid_argument("MiniMax-H3 attention requires immutable single-device execution");
                }
            }
            if (options.cache_dit && params_backend != runtime_backend &&
                (options.adaln_cache || options.sage_attn || options.sparse_attn))
                throw std::invalid_argument("MiniMax-H3 Cache-DiT offload requires independent single-method validation");

        }

        void validate_cuda_sparse_capacity(const DiffusionParams& params) {
            if (!model.cuda_optimizations.sparse_attn) return;
            // Packed A/V retains the target-video spatial geometry; do not
            // copy large latent buffers merely to reject an expanded graph.
            const auto& shape = params.x->shape();
            static const std::vector<sd::Tensor<float>> empty;
            static const std::vector<MiniMaxH3ReferenceBlock> empty_refs;
            static const sd::Tensor<int32_t> empty_int;
            const auto layout = build_layout(params.context->shape()[1], shape[2], shape[1], shape[0],
                                            params.minimax_audio_length,
                                            params.ref_latents ? *params.ref_latents : empty,
                                            params.minimax_reference_audio_latents ? *params.minimax_reference_audio_latents : empty,
                                            params.minimax_keyframe_indices ? *params.minimax_keyframe_indices : empty_int,
                                            params.minimax_reference_blocks ? *params.minimax_reference_blocks : empty_refs,
                                            params.minimax_text_token_tags ? *params.minimax_text_token_tags : empty_int,
                                            0.f, 0.f);
            SparseAttentionLayout sparse;
            for (const auto& span : layout.sequence_segments)
                sparse.spans.push_back({span.start, span.end,
                    span.kind == SequenceKind::TARGET_VIDEO ? SparseTokenKind::TargetVideo : SparseTokenKind::Reference});
            std::string reason;
            if (!h3_subblock_graph_budget(sparse, config.num_attention_heads, config.num_layers, H3_GRAPH_SIZE, &reason))
                throw std::invalid_argument(reason);
        }

        void reset_cuda_optimization_state() noexcept {
            // Never carry a graph assembled with another request's options or
            // bound inputs into the next generation. Options themselves survive
            // this reset; AdaLN schedule tables are rebuilt for each request.
            const bool has_cache_state = model.cuda_optimizations.cache_dit ||
                                         model.cuda_cache_pass != nullptr || cuda_cache_store != nullptr ||
                                         model.cuda_optimizations.adaln_cache ||
                                         model.cuda_optimizations.sparse_attn || model.cuda_optimizations.sage_attn;
            model.cuda_cache_pass = nullptr;
            if (has_cache_state) {
                set_tap_registry(nullptr);
                invalidate_persistent_graph();
                reset_graph_cut_run_cache();
            }
            cuda_cache_policy.reset();
            cuda_cache_branches = {};
            cuda_cache_store.reset();
            if (cuda_cache_offload_state) cuda_cache_offload_state->reset();
            model.adaln_cache = nullptr; model.adaln_graph_inputs = nullptr; model.adaln_token = {};
            adaln_inputs.reset(); adaln_branches = {}; adaln_sigmas.clear();
            ++adaln_request_generation;
            model.cuda_step = -1; model.cuda_total_steps = 0;
            model.sparse_config.graph_inputs.reset();
        }

        void prepare_cuda_schedule(const std::vector<float>& sigmas,
                                    const std::string& scheduler_identity,
                                    const std::string& full_weights_identity) {
            if (!model.cuda_optimizations.adaln_cache) return;
            if (sigmas.size() < 2 || scheduler_identity.empty() || full_weights_identity.empty())
                throw std::invalid_argument("MiniMax-H3 exact AdaLN requires a complete fixed schedule and full-weight identity");
            adaln_sigmas = sigmas;
            for (size_t i = 0; i + 1 < sigmas.size(); ++i)
                adaln_sigmas[i] = std::clamp((sigmas[i] * 1000.f) / 1000.f, 1e-6f, 1.f);
            adaln_schedule_identity = scheduler_identity;
            adaln_weights_identity = full_weights_identity;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors,
                               const std::string& prefix) {
            model.get_param_tensors(tensors, prefix);
        }

        std::pair<sd::Tensor<float>, sd::Tensor<float>> split_av_latents(const sd::Tensor<float>& packed,
                                                                         int audio_length) const {
            GGML_ASSERT(packed.dim() == 4 || packed.dim() == 5);
            int64_t spatial      = packed.shape()[0] * packed.shape()[1] * packed.shape()[2];
            int64_t video_values = spatial * config.video_latent_channels;
            sd::Tensor<float> video({packed.shape()[0],
                                     packed.shape()[1],
                                     packed.shape()[2],
                                     config.video_latent_channels,
                                     1});
            std::copy_n(packed.data(), static_cast<size_t>(video_values), video.data());
            if (audio_length <= 0) {
                return {video, {}};
            }
            int64_t audio_values = audio_length * 2 * config.audio_latent_channels;
            GGML_ASSERT(packed.numel() >= video_values + audio_values);
            sd::Tensor<float> audio({audio_length, 2, config.audio_latent_channels, 1});
            std::copy_n(packed.data() + video_values,
                        static_cast<size_t>(audio_values),
                        audio.data());
            return {video, audio};
        }

        size_t measure_compute_buffer_at(int latent_width, int latent_height, int frames) {
            if (latent_width <= 0 || latent_height <= 0 || frames <= 0) {
                return 0;
            }
            const int latent_frames = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
            const int audio_length = std::max(1, static_cast<int>(std::lround(frames * 40.0 / 24.0)));
            sd::Tensor<float> video = sd::zeros<float>(
                {latent_width, latent_height, latent_frames, config.video_latent_channels, 1});
            sd::Tensor<float> audio = sd::zeros<float>(
                {audio_length, 2, config.audio_latent_channels, 1});
            const int64_t spatial_size = video.shape()[0] * video.shape()[1] * video.shape()[2];
            const int64_t extra_channels = (audio.numel() + spatial_size - 1) / spatial_size;
            std::vector<int64_t> packed_shape = video.shape();
            packed_shape[3] += extra_channels;
            sd::Tensor<float> packed = sd::zeros<float>(packed_shape);
            std::copy_n(video.data(), video.numel(), packed.data());
            std::copy_n(audio.data(), audio.numel(), packed.data() + video.numel());
            sd::Tensor<float> timestep({1}, {1000.0f});
            sd::Tensor<float> context = sd::zeros<float>({config.text_dim, 256, 1});
            const sd::Tensor<int32_t> empty_int;
            std::vector<sd::Tensor<float>> condition_videos;
            condition_videos.push_back(sd::zeros<float>(video.shape()));
            condition_videos.push_back(sd::zeros<float>(
                {latent_width, latent_height, 2, config.video_latent_channels, 1}));
            std::vector<sd::Tensor<float>> condition_audios;
            condition_audios.push_back(sd::zeros<float>(audio.shape()));
            condition_audios.push_back(sd::zeros<float>(audio.shape()));
            const std::vector<MiniMaxH3ReferenceBlock> reference_blocks = {
                {MiniMaxH3ReferenceKind::VIDEO_AUDIO, 0, 0},
                {MiniMaxH3ReferenceKind::IMAGE, 1, -1},
                {MiniMaxH3ReferenceKind::AUDIO, -1, 1},
            };
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(packed,
                                   timestep,
                                   context,
                                   condition_videos,
                                   condition_audios,
                                   empty_int,
                                   empty_int,
                                   reference_blocks,
                                   audio_length,
                                   12.0f,
                                   3.0f);
            };
            return measure_compute_buffer(get_graph);
        }

        ggml_tensor* merge_av_latents(ggml_context* ctx,
                                      ggml_tensor* video,
                                      ggml_tensor* audio) const {
            int64_t divisor = video->ne[0] * video->ne[1] * video->ne[2];
            int64_t values  = ggml_nelements(audio);
            int64_t padding = (divisor - values % divisor) % divisor;
            audio           = ggml_reshape_4d(ctx, ggml_cont(ctx, audio), values, 1, 1, 1);
            if (padding > 0) {
                audio = ggml_ext_pad(ctx, audio, static_cast<int>(padding), 0, 0, 0);
            }
            audio = ggml_reshape_4d(ctx,
                                    audio,
                                    video->ne[0],
                                    video->ne[1],
                                    video->ne[2],
                                    (values + padding) / divisor);
            return ggml_concat(ctx, video, audio, 3);
        }

        ggml_cgraph* build_adaln_projection_graph(ExactAdaLN::Cache& cache,
                                                 ExactAdaLN::Cache::StepToken token,
                                                 const ExactAdaLN::StepPlan& plan,
                                                 size_t first_projection = 0,
                                                 size_t projection_count = std::numeric_limits<size_t>::max()) {
            ggml_tensor* features = nullptr;
            ggml_tensor* lower = nullptr;
            ggml_tensor* upper = nullptr;
            ggml_tensor* fraction = nullptr;
            if (config.uses_adaln_curves()) {
                std::vector<int32_t> lows(plan.timesteps.size()), highs(plan.timesteps.size());
                std::vector<float> fractions(plan.timesteps.size());
                for (size_t i = 0; i < plan.timesteps.size(); ++i) {
                    const float value = std::clamp(plan.timesteps[i], 0.f, 1.f) * (config.adaln_curve_grid - 1);
                    const int index = std::min(static_cast<int>(std::floor(value)), static_cast<int>(config.adaln_curve_grid - 2));
                    lows[i] = index; highs[i] = index + 1; fractions[i] = value - index;
                }
                curve_index_input_cache = sd::Tensor<int32_t>({static_cast<int64_t>(lows.size())}, lows);
                curve_upper_index_input_cache = sd::Tensor<int32_t>({static_cast<int64_t>(highs.size())}, highs);
                curve_fraction_input_cache = sd::Tensor<float>({1, static_cast<int64_t>(fractions.size())}, fractions);
                lower = make_input(curve_index_input_cache); upper = make_input(curve_upper_index_input_cache);
                fraction = make_input(curve_fraction_input_cache);
            } else {
                timestep_feature_input_cache = sd::Tensor<float>(
                    {config.timestep_input_dim, static_cast<int64_t>(plan.timesteps.size())},
                    timestep_embedding(plan.timesteps, static_cast<int>(config.timestep_input_dim), 10000, true, 1.f));
                features = make_input(timestep_feature_input_cache);
            }
            auto ctx = get_context();
            auto* embedding = model.time_embedding(&ctx, features, lower, upper, fraction);
            return ExactAdaLN::build_projection_graph(compute_ctx, cache, token,
                projection_count == 1 ? 256 : H3_GRAPH_SIZE,
                [&](size_t index) {
                    return model.project_adaln(&ctx, embedding, index);
                }, first_projection, projection_count);
        }

        ExactAdaLN::Status prefill_adaln_layer_major(ExactAdaLN::Cache& cache,
                                                       const ExactAdaLN::RequestContract& contract,
                                                       int n_threads) {
                    std::vector<sd::Tensor<float>> owned_features, owned_fractions;
                    std::vector<sd::Tensor<int32_t>> owned_lower, owned_upper;
                    const int64_t start = ggml_time_ms();
                    auto status = ExactAdaLN::prefill_request_layer_major(*this, cache, contract, n_threads,
                        [&](size_t projection, const std::vector<size_t>& plans,
                            const std::vector<ExactAdaLN::Cache::StepToken>& tokens) -> ggml_cgraph* {
                            owned_features.clear(); owned_fractions.clear(); owned_lower.clear(); owned_upper.clear();
                            if (plans.empty() || plans.size() > (H3_GRAPH_SIZE - 1) / 256)
                                throw std::invalid_argument("MiniMax-H3 AdaLN prefill schedule exceeds graph budget");
                            auto* merged = new_graph_custom(plans.size() * 256);
                            for (size_t i : plans) {
                                auto* graph = build_adaln_projection_graph(cache, tokens[i],
                                                                         contract.plans[i], projection, 1);
                                if (!graph || ggml_graph_n_nodes(graph) == 0) return nullptr;
                                ggml_build_forward_expand(merged, ggml_graph_node(graph, -1));
                                // make_input registered raw data pointers. Moves keep their
                                // allocations alive for ALL timestep graphs in this segment.
                                owned_features.push_back(std::move(timestep_feature_input_cache));
                                owned_fractions.push_back(std::move(curve_fraction_input_cache));
                                owned_lower.push_back(std::move(curve_index_input_cache));
                                owned_upper.push_back(std::move(curve_upper_index_input_cache));
                            }
                            return merged;
                        });
                    LOG_INFO("minimax-h3 AdaLN layer-major prefill: projections=%zu plans=%zu time=%lldms host-cache=%zu bytes; weights preserved",
                             contract.projections.size(), contract.plans.size(),
                             (long long)(ggml_time_ms() - start), cache.used_bytes());
                    return status;
        }

        bool exact_projection_params_supported(const std::vector<ggml_tensor*>& params) const {
            if (!ExactAdaLN::RunnerBridge<GGMLRunner>::exact_projection_params_supported(params)) return false;
            if (params_backend == runtime_backend) return true;
            size_t bytes = 0;
            for (auto* tensor : params) {
                const size_t n = ggml_nbytes(tensor);
                if (n > max_graph_vram_bytes || bytes > max_graph_vram_bytes - n) return false;
                bytes += n;
            }
            // This is the explicit weight planning bound; CUDA workspace/peak is
            // measured separately, never represented as a hard max-VRAM cap.
            return bytes > 0 && bytes <= max_graph_vram_bytes;
        }

        sd::Tensor<float> compute_adaln_offloaded(get_graph_cb_t build, int threads,
                                                 const DiffusionParams& params) {
            validate_cuda_research_execution_mode(model.cuda_optimizations);
            ggml_cgraph* graph = nullptr;
            try {
                if (!prepare_compute_graph(build, &graph) || !graph)
                    throw std::runtime_error("MiniMax-H3 AdaLN offload graph build failed");
                GraphCutPlan plan;
                if (!resolve_graph_cut_plan(graph, &plan) || !plan.valid || plan.segments.empty())
                    throw std::runtime_error("MiniMax-H3 AdaLN offload graph plan failed");
                sd::ggml_graph_cut::clear_comm_marks();
                for (const auto& segment : plan.segments) {
                    if (!segment.comm_ops.empty())
                        throw std::invalid_argument("MiniMax-H3 AdaLN offload cannot use parallel graph cuts");
                    auto weights = sd::ggml_graph_cut::runtime_param_tensors(graph, segment, get_desc().c_str());
                    // Existing executor treats empty selection as ALL model weights.
                    // Reject before any stage instead of silently triggering that path.
                    if (weights.empty())
                        throw std::invalid_argument("MiniMax-H3 AdaLN offload empty-weight segment would stage the entire model");
                }
                auto output = compute_with_graph_cuts<float>(graph, plan, threads, true, false);
                if (!output) throw std::runtime_error("MiniMax-H3 AdaLN offload segment execution failed");
                return restore_trailing_singleton_dims(std::move(*output), params.x->dim());
            } catch (...) {
                exact_release_projection_graph();
                reset_cuda_optimization_state();
                throw;
            }
        }

        void prepare_adaln_branch(int n_threads, const DiffusionParams& params) {
            if (!model.cuda_optimizations.adaln_cache) return;
            if (adaln_sigmas.empty() || model.cuda_step < 0 || model.cuda_total_steps <= 0 ||
                model.cuda_step >= model.cuda_total_steps || params.timesteps == nullptr || params.timesteps->empty() ||
                !std::isfinite((*params.timesteps)[0]) ||
                static_cast<size_t>(model.cuda_total_steps) + 1 != adaln_sigmas.size())
                throw std::invalid_argument("MiniMax-H3 exact AdaLN missing fixed schedule or step metadata");
            const float actual_sigma = std::clamp((*params.timesteps)[0] / 1000.f, 1e-6f, 1.f);
            if (ExactAdaLN::float_bits(actual_sigma) !=
                ExactAdaLN::float_bits(adaln_sigmas[static_cast<size_t>(model.cuda_step)]))
                throw std::invalid_argument("MiniMax-H3 exact AdaLN actual timestep differs from fixed schedule");
            const int branch = model.cuda_conditional ? 1 : 0;
            auto& state = adaln_branches[branch];
            const auto split = split_av_latents(*params.x, params.minimax_audio_length);
            static const std::vector<sd::Tensor<float>> empty_conditions;
            static const std::vector<MiniMaxH3ReferenceBlock> empty_refs;
            static const sd::Tensor<int32_t> empty_int;
            const auto& cv = params.ref_latents ? *params.ref_latents : empty_conditions;
            const auto& ca = params.minimax_reference_audio_latents ? *params.minimax_reference_audio_latents : empty_conditions;
            const auto& refs = params.minimax_reference_blocks ? *params.minimax_reference_blocks : empty_refs;
            const auto& tags = params.minimax_text_token_tags ? *params.minimax_text_token_tags : empty_int;
            const auto& keys = params.minimax_keyframe_indices ? *params.minimax_keyframe_indices : empty_int;
            ExactAdaLN::RequestContract actual;
            actual.full_weights_identity = adaln_weights_identity;
            actual.adapter_identity = "none"; actual.load_generation = adaln_request_generation;
            actual.cfg_identity = cuda_cache_identity(params) + (model.cuda_conditional ? ":conditional" : ":unconditional");
            actual.math_identity = std::string(ggml_backend_name(runtime_backend)) + ":ggml-native-projection:f32";
            actual.schedule_identity = adaln_schedule_identity;
            actual.video_shift = params.minimax_video_sigma_shift; actual.audio_shift = params.minimax_audio_sigma_shift;
            actual.timestep_input_dim = config.timestep_input_dim;
            actual.time_embed_hidden = config.time_embed_hidden_size;
            actual.time_embed_dim = config.time_embed_dim; actual.curve_grid = config.adaln_curve_grid;
            actual.sigmas = adaln_sigmas;
            actual.projections = ExactAdaLN::h3_projections(config.hidden_size, static_cast<size_t>(config.num_layers));
            for (size_t step = 0; step + 1 < actual.sigmas.size(); ++step) {
                const float sigma = actual.sigmas[step];
                const auto layout = build_layout(params.context->shape()[1], split.first.shape()[2],
                    split.first.shape()[1], split.first.shape()[0], params.minimax_audio_length,
                    cv, ca, keys, refs, tags, 1.f - sigma,
                    1.f - time_shift_sigma(sigma, actual.video_shift, actual.audio_shift));
                actual.plans.push_back(ExactAdaLN::plan_from_layout(step, sigma,
                    model.cuda_conditional ? ExactAdaLN::CfgRole::Conditional : ExactAdaLN::CfgRole::Unconditional, layout));
            }
            ExactAdaLN::Cache::StepToken token;
            const auto index = static_cast<size_t>(model.cuda_step);
            auto status = state.cache ? state.cache->bind_step(actual, index, actual.plans[index], &token)
                                      : ExactAdaLN::Status::Stale;
            if (status != ExactAdaLN::Status::Hit) {
                model.adaln_cache = nullptr; model.adaln_graph_inputs = nullptr; adaln_inputs.reset();
                state.cache = std::make_unique<ExactAdaLN::Cache>(ExactAdaLN::Options{
                    true, static_cast<size_t>(model.cuda_optimizations.adaln_max_mib) * 1024 * 1024, 16384});
                state.contract = actual;
                status = state.cache->begin_request(actual);
                if (status != ExactAdaLN::Status::Stored) throw std::runtime_error("MiniMax-H3 exact AdaLN contract/budget rejected");
                if (params_backend != runtime_backend) {
                    status = prefill_adaln_layer_major(*state.cache, state.contract, n_threads);
                } else {
                    status = ExactAdaLN::prefill_request(*this, *state.cache, state.contract, n_threads,
                        [&](size_t i, auto current) { return build_adaln_projection_graph(*state.cache, current, state.contract.plans[i]); });
                }
                if (status != ExactAdaLN::Status::Stored) throw std::runtime_error("MiniMax-H3 exact AdaLN prefill failed");
                status = state.cache->bind_step(actual, index, actual.plans[index], &token);
            }
            if (status != ExactAdaLN::Status::Hit) throw std::runtime_error("MiniMax-H3 exact AdaLN step contract mismatch");
            model.adaln_cache = state.cache.get(); model.adaln_token = token;
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& packed,
                                 const sd::Tensor<float>& timestep,
                                 const sd::Tensor<float>& context_tensor,
                                 const std::vector<sd::Tensor<float>>& condition_videos,
                                 const std::vector<sd::Tensor<float>>& condition_audios,
                                 const sd::Tensor<int32_t>& text_tags,
                                 const sd::Tensor<int32_t>& keyframe_indices,
                                 const std::vector<MiniMaxH3ReferenceBlock>& reference_blocks,
                                 int audio_length,
                                 float video_shift,
                                 float audio_shift) {
            if (model.cuda_optimizations.sparse_attn)
                model.sparse_config.graph_inputs = std::make_shared<SparseAttentionGraphInputs>();
            if (model.adaln_cache) {
                adaln_inputs = std::make_unique<ExactAdaLN::GraphInputs>(*model.adaln_cache,
                                                                      static_cast<size_t>(config.num_layers + 1));
                model.adaln_graph_inputs = adaln_inputs.get();
            }
            auto split        = split_av_latents(packed, audio_length);
            video_input_cache = std::move(split.first);
            audio_input_cache = std::move(split.second);
            GGML_ASSERT(!audio_input_cache.empty());
            GGML_ASSERT(!context_tensor.empty());

            auto video   = make_input(video_input_cache);
            auto audio   = make_input(audio_input_cache);
            auto context = make_input(context_tensor);
            std::vector<ggml_tensor*> condition_inputs;
            condition_inputs.reserve(condition_videos.size());
            for (const auto& condition : condition_videos) {
                condition_inputs.push_back(make_input(condition));
            }
            std::vector<ggml_tensor*> audio_condition_inputs;
            audio_condition_inputs.reserve(condition_audios.size());
            for (const auto& condition : condition_audios) {
                audio_condition_inputs.push_back(make_input(condition));
            }

            float sigma_v = std::clamp(timestep[0] / 1000.f, 1e-6f, 1.f);
            float t_v     = 1.f - sigma_v;
            float t_a     = 1.f - time_shift_sigma(sigma_v, video_shift, audio_shift);
            auto layout   = build_layout(context_tensor.shape()[1],
                                         video_input_cache.shape()[2],
                                         video_input_cache.shape()[1],
                                         video_input_cache.shape()[0],
                                         audio_length,
                                         condition_videos,
                                         condition_audios,
                                         keyframe_indices,
                                         reference_blocks,
                                         text_tags,
                                         t_v,
                                         t_a);

            if (h3_trace_first_step_enabled()) {
                auto reference_kind_name = [](MiniMaxH3ReferenceKind kind) {
                    switch (kind) {
                        case MiniMaxH3ReferenceKind::IMAGE: return "image";
                        case MiniMaxH3ReferenceKind::VIDEO: return "video";
                        case MiniMaxH3ReferenceKind::VIDEO_AUDIO: return "video_audio";
                        case MiniMaxH3ReferenceKind::AUDIO: return "audio";
                    }
                    return "unknown";
                };
                std::ostringstream ref_stream;
                for (size_t index = 0; index < reference_blocks.size(); ++index) {
                    const auto& block = reference_blocks[index];
                    if (index > 0) {
                        ref_stream << ";";
                    }
                    ref_stream << index << ":" << reference_kind_name(block.kind)
                               << "(v=" << block.video_index
                               << ",a=" << block.audio_index << ")";
                }
                LOG_INFO("minimax-h3 trace dit layout: text=%lld target_video=%lldx%lldx%lld target_audio=%d condition_videos=%zu condition_audios=%zu reference_blocks=%zu [%s] packed_tokens=%lld sequence_segments=%zu segments=%zu timesteps=%zu",
                         static_cast<long long>(context_tensor.shape()[1]),
                         static_cast<long long>(video_input_cache.shape()[2]),
                         static_cast<long long>(video_input_cache.shape()[1]),
                         static_cast<long long>(video_input_cache.shape()[0]),
                         audio_length,
                         condition_videos.size(),
                         condition_audios.size(),
                         reference_blocks.size(),
                         ref_stream.str().c_str(),
                         static_cast<long long>(layout.positions.size() / 3),
                         layout.sequence_segments.size(),
                         layout.segments.size(),
                         layout.timesteps.size());
                for (size_t index = 0; index < condition_videos.size(); ++index) {
                    const auto& condition = condition_videos[index];
                    LOG_INFO("minimax-h3 trace condition_video[%zu]: shape=%lldx%lldx%lldx%lldx%lld",
                             index,
                             static_cast<long long>(condition.shape()[0]),
                             static_cast<long long>(condition.shape()[1]),
                             static_cast<long long>(condition.shape()[2]),
                             static_cast<long long>(condition.shape()[3]),
                             static_cast<long long>(condition.shape()[4]));
                }
                for (size_t index = 0; index < condition_audios.size(); ++index) {
                    const auto& condition = condition_audios[index];
                    LOG_INFO("minimax-h3 trace condition_audio[%zu]: shape=%lldx%lldx%lld",
                             index,
                             static_cast<long long>(condition.shape()[0]),
                             static_cast<long long>(condition.shape()[1]),
                             static_cast<long long>(condition.shape()[2]));
                }
            }

            position_input_cache = sd::Tensor<float>(
                {3, static_cast<int64_t>(layout.positions.size() / 3)},
                layout.positions);
            auto positions = make_input(position_input_cache);

            ggml_tensor* timestep_features   = nullptr;
            ggml_tensor* curve_indices       = nullptr;
            ggml_tensor* curve_upper_indices = nullptr;
            ggml_tensor* curve_fractions     = nullptr;
            if (config.uses_adaln_curves()) {
                std::vector<int32_t> indices(layout.timesteps.size());
                std::vector<int32_t> upper_indices(layout.timesteps.size());
                std::vector<float> fractions(layout.timesteps.size());
                for (size_t i = 0; i < layout.timesteps.size(); ++i) {
                    float position   = std::clamp(layout.timesteps[i], 0.f, 1.f) * (config.adaln_curve_grid - 1);
                    int index        = std::min(static_cast<int>(std::floor(position)),
                                                static_cast<int>(config.adaln_curve_grid - 2));
                    indices[i]       = index;
                    upper_indices[i] = index + 1;
                    fractions[i]     = position - index;
                }
                curve_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(indices.size())},
                    indices);
                curve_upper_index_input_cache = sd::Tensor<int32_t>(
                    {static_cast<int64_t>(upper_indices.size())},
                    upper_indices);
                curve_fraction_input_cache = sd::Tensor<float>(
                    {1, static_cast<int64_t>(fractions.size())},
                    fractions);
                curve_indices       = make_input(curve_index_input_cache);
                curve_upper_indices = make_input(curve_upper_index_input_cache);
                curve_fractions     = make_input(curve_fraction_input_cache);
            } else {
                timestep_feature_input_cache = sd::Tensor<float>(
                    {config.timestep_input_dim, static_cast<int64_t>(layout.timesteps.size())},
                    timestep_embedding(layout.timesteps,
                                       static_cast<int>(config.timestep_input_dim),
                                       10000,
                                       true,
                                       1.f));
                timestep_features = make_input(timestep_feature_input_cache);
            }

            auto runner_ctx = get_context();
            if (model.cuda_optimizations.adaln_cache || model.cuda_optimizations.sparse_attn) {
                runner_ctx.bind_backend_tensor_data = [this](ggml_tensor* tensor, const void* data) {
                    set_backend_tensor_data(tensor, data);
                };
            }
            audio_slope_input_cache = sd::Tensor<float>(
                {1},
                {time_shift_slope(sigma_v, video_shift, audio_shift)});
            auto audio_slope_tensor = make_input(audio_slope_input_cache);
            const char* debug_target_env = std::getenv("ED_MINIMAX_H3_DEBUG_TARGET");
            const std::string debug_target = debug_target_env == nullptr ? std::string() : std::string(debug_target_env);
            ggml_tensor* debug_output = nullptr;
            auto output     = model.forward(&runner_ctx,
                                            video,
                                            audio,
                                            context,
                                            condition_inputs,
                                            audio_condition_inputs,
                                            positions,
                                            timestep_features,
                                            curve_indices,
                                            curve_upper_indices,
                                            curve_fractions,
                                            layout.segments,
                                            layout.sequence_segments,
                                            layout.video_segment,
                                            layout.audio_segment,
                                            audio_slope_tensor,
                                            debug_target,
                                            &debug_output);
            auto merged     = model.cuda_cache_pass &&
                              model.cuda_cache_pass->phase == CudaDualBlockGraphPass::Phase::Probe
                                  ? output.first
                                  : (debug_output == nullptr
                                         ? merge_av_latents(compute_ctx, output.first, output.second)
                                         : debug_output);
            auto graph      = new_graph_custom(H3_GRAPH_SIZE);
            ggml_build_forward_expand(graph, merged);
            return graph;
        }

        std::string cuda_cache_identity(const DiffusionParams& params) const {
            // Values, not object addresses: independently allocated but equal
            // conditioning is stable; in-place mutations invalidate immediately.
            Sha256 hash;
            auto bytes = [&](const void* data, size_t size) { hash.update(data, size); };
            auto tensor = [&](const auto* value) {
                const bool present = value != nullptr;
                bytes(&present, sizeof(present));
                if (!value) return;
                const auto dim = value->dim();
                bytes(&dim, sizeof(dim));
                for (auto extent : value->shape()) bytes(&extent, sizeof(extent));
                bytes(value->data(), static_cast<size_t>(value->numel()) * sizeof(*value->data()));
            };
            for (auto extent : params.x->shape()) bytes(&extent, sizeof(extent));
            tensor(params.context);
            tensor(params.minimax_text_token_tags);
            tensor(params.minimax_keyframe_indices);
            bytes(&params.minimax_audio_length, sizeof(params.minimax_audio_length));
            bytes(&params.minimax_video_sigma_shift, sizeof(params.minimax_video_sigma_shift));
            bytes(&params.minimax_audio_sigma_shift, sizeof(params.minimax_audio_sigma_shift));
            auto conditions = [&](const std::vector<sd::Tensor<float>>* values) {
                const size_t count = values ? values->size() : 0;
                bytes(&count, sizeof(count));
                if (values) for (const auto& value : *values) tensor(&value);
            };
            conditions(params.ref_latents);
            conditions(params.minimax_reference_audio_latents);
            const size_t refs = params.minimax_reference_blocks ? params.minimax_reference_blocks->size() : 0;
            bytes(&refs, sizeof(refs));
            if (params.minimax_reference_blocks) for (const auto& ref : *params.minimax_reference_blocks) {
                bytes(&ref.kind, sizeof(ref.kind));
                bytes(&ref.video_index, sizeof(ref.video_index));
                bytes(&ref.audio_index, sizeof(ref.audio_index));
            }
            return hash.finish();
        }

        sd::Tensor<float> compute_cuda_dual_block_cache(int n_threads,
                                                        const DiffusionParams& params,
                                                        const get_graph_cb_t& get_graph) {
            cuda_cache_policy.require_step();
            cuda_cache_policy.bind_identity(cuda_cache_identity(params));
            if (params_backend != runtime_backend) {
                // Reserve independently owned front/RF/delta snapshots within the
                // explicit planning budget; it is not a hard total VRAM ceiling.
                const size_t reserve = std::min<size_t>(max_graph_vram_bytes / 3, size_t(4) << 30);
                if (!reserve) throw std::invalid_argument("MiniMax-H3 Cache-DiT needs a nonzero snapshot budget");
                if (!cuda_cache_offload_state)
                    cuda_cache_offload_state = std::make_unique<CacheOffload::State>(runtime_backend, reserve);
                CacheOffload::Options execution; execution.enabled = true;
                CacheOffload::Statistics stats;
                auto result = CacheOffload::compute(*this, *cuda_cache_offload_state, cuda_cache_policy,
                    model.cuda_optimizations, cuda_cache_operators, get_graph,
                    [&](auto* pass) { model.cuda_cache_pass = pass; }, n_threads, execution, &stats);
                std::ostringstream rf;
                for (size_t i=0;i<stats.per_span_relative_l1.size();++i) {
                    if (i) rf << ',';
                    rf << stats.per_span_relative_l1[i];
                }
                LOG_INFO("minimax-h3 Cache-DiT offload: step=%d branch=%d hit=%d probe_segments=%zu remainder_segments=%zu H2D_weights=%zu bytes cache_peak=%zu bytes RF_spans=%zu",
                    model.cuda_step, cuda_cache_policy.branch_index(), stats.hit, stats.probe_segments,
                    stats.remainder_segments, stats.parameter_bytes, cuda_cache_offload_state->arena.peak_bytes(),
                    stats.per_span_relative_l1.size());
                LOG_INFO("minimax-h3 Cache-DiT RF: step=%d threshold=%.6g span_rel_l1=[%s]",
                         model.cuda_step, model.cuda_optimizations.cache_threshold, rf.str().c_str());
                return restore_trailing_singleton_dims(std::move(result), params.x->dim());
            }
            if (!cuda_cache_store) cuda_cache_store = std::make_unique<RunnerCacheDeviceStore>(runtime_backend);
            const int branch = cuda_cache_policy.branch_index();
            auto& state = cuda_cache_branches[branch];
            CudaDualBlockGraphPass pass;
            pass.front_blocks = model.cuda_optimizations.cache_front_blocks;
            pass.back_blocks = model.cuda_optimizations.cache_back_blocks;
            pass.operators = &cuda_cache_operators;
            pass.reference_rf = cuda_cache_policy.eligible() ? state.rf : nullptr;
            edgedit::cache::TapRegistry taps;
            model.cuda_cache_pass = &pass;
            set_tap_registry(&taps);
            auto cleanup = [&]() {
                model.cuda_cache_pass = nullptr;
                set_tap_registry(nullptr);
                reset_graph_cut_run_cache();
                invalidate_persistent_graph();
            };
            auto capture = [&](const char* name, int slot) -> ggml_tensor* {
                ggml_tensor* source = get_cache_tensor_by_name(name);
                if (!source) throw std::runtime_error(std::string("MiniMax-H3: missing cache capture ") + name);
                std::vector<int64_t> shape(source->ne, source->ne + GGML_MAX_DIMS);
                auto* destination = static_cast<ggml_tensor*>(cuda_cache_store->ensure_entry(branch + 1, slot, shape));
                if (!destination || !copy_named_cache_tensor_to(name, destination))
                    throw std::runtime_error("MiniMax-H3: device cache capture allocation/copy failed");
                return destination;
            };
            try {
                // Only scalar guards leave the device; never RF/front/middle
                // tensors. The transient run cache is copied into persistent
                // slots BEFORE the next compute resets its graph/context.
                reset_graph_cut_run_cache();
                auto probe = GGMLRunner::compute<float>(get_graph, n_threads, false, true);
                if (!probe) throw std::runtime_error("MiniMax-H3: front probe compute failed");
                pass.front = capture("h3_cache_front", 0);
                std::vector<float> metrics;
                for (const auto& name : pass.indicator_names) {
                    auto* scalar = get_cache_tensor_by_name("cache_ind:" + name);
                    float value = std::numeric_limits<float>::infinity();
                    if (scalar) ggml_backend_tensor_get(scalar, &value, 0, sizeof(value));
                    metrics.push_back(value);
                }
                const bool hit = state.delta && cuda_cache_policy.hit(metrics) &&
                                 ggml_are_same_shape(state.delta, pass.front);
                // On hits keep the RF anchor paired with the stored middle delta,
                // rather than tracking adjacent probes and allowing drift.
                if (!hit) state.rf = capture("h3_cache_rf", 1);
                reset_graph_cut_run_cache();
                taps.clear();
                pass.phase = CudaDualBlockGraphPass::Phase::Remainder;
                pass.reuse_middle = hit;
                pass.middle_delta = hit ? state.delta : nullptr;
                auto output = GGMLRunner::compute<float>(get_graph, n_threads, false, false);
                if (!output) throw std::runtime_error("MiniMax-H3: middle/suffix compute failed");
                if (!hit) state.delta = capture("h3_cache_middle_delta", 2);
                cuda_cache_policy.finish(hit, true);
                cleanup();
                return restore_trailing_singleton_dims(std::move(*output), params.x->dim());
            } catch (...) {
                cuda_cache_policy.abort();
                cleanup();
                throw;
            }
        }

        sd::Tensor<float> compute(int n_threads,
                                  const DiffusionParams& params) {
            GGML_ASSERT(params.x != nullptr && params.timesteps != nullptr && params.context != nullptr);
            // Recheck dynamic state before prefill/build/replay; configure-time
            // checks cannot protect later environment/adapter/group changes.
            validate_cuda_research_execution_mode(model.cuda_optimizations);
            if (model.cuda_optimizations.cache_dit) validate_cuda_cache_execution_mode();
            validate_cuda_sparse_capacity(params);
            prepare_adaln_branch(n_threads, params);
            static const std::vector<sd::Tensor<float>> empty_conditions;
            static const std::vector<MiniMaxH3ReferenceBlock> empty_reference_blocks;
            const auto& conditions       = params.ref_latents == nullptr ? empty_conditions : *params.ref_latents;
            const auto& audio_conditions = params.minimax_reference_audio_latents == nullptr
                                               ? empty_conditions
                                               : *params.minimax_reference_audio_latents;
            const auto& reference_blocks = params.minimax_reference_blocks == nullptr
                                               ? empty_reference_blocks
                                               : *params.minimax_reference_blocks;
            const sd::Tensor<int32_t> empty_int;
            get_graph_cb_t get_graph = [&]() -> ggml_cgraph* {
                return build_graph(*params.x,
                                   *params.timesteps,
                                   *params.context,
                                   conditions,
                                   audio_conditions,
                                   params.minimax_text_token_tags == nullptr ? empty_int : *params.minimax_text_token_tags,
                                   params.minimax_keyframe_indices == nullptr ? empty_int : *params.minimax_keyframe_indices,
                                   reference_blocks,
                                   params.minimax_audio_length,
                                   params.minimax_video_sigma_shift,
                                   params.minimax_audio_sigma_shift);
            };
            if (model.cuda_optimizations.adaln_cache && params_backend != runtime_backend) {
                if (model.cuda_optimizations.cache_dit || model.cuda_optimizations.sage_attn || model.cuda_optimizations.sparse_attn)
                    throw std::invalid_argument("MiniMax-H3 AdaLN offload is single-method until combined-path validation");
                return compute_adaln_offloaded(get_graph, n_threads, params);
            }
            if (model.cuda_optimizations.cache_dit) {
                validate_cuda_cache_execution_mode();
                return compute_cuda_dual_block_cache(n_threads, params, get_graph);
            }
            if (h3_env_flag_enabled("ED_MINIMAX_H3_REUSE_GRAPH") &&
                std::getenv("ED_MINIMAX_H3_DEBUG_TARGET") == nullptr) {
                auto split_for_reuse = split_av_latents(*params.x, params.minimax_audio_length);
                video_input_cache = std::move(split_for_reuse.first);
                audio_input_cache = std::move(split_for_reuse.second);
                const float sigma_v = std::clamp((*params.timesteps)[0] / 1000.f, 1e-6f, 1.f);
                const float t_v     = 1.f - sigma_v;
                const float t_a     = 1.f - time_shift_sigma(sigma_v,
                                                              params.minimax_video_sigma_shift,
                                                              params.minimax_audio_sigma_shift);
                auto layout_for_reuse = build_layout(params.context->shape()[1],
                                                     video_input_cache.shape()[2],
                                                     video_input_cache.shape()[1],
                                                     video_input_cache.shape()[0],
                                                     params.minimax_audio_length,
                                                     conditions,
                                                     audio_conditions,
                                                     params.minimax_keyframe_indices == nullptr ? empty_int : *params.minimax_keyframe_indices,
                                                     reference_blocks,
                                                     params.minimax_text_token_tags == nullptr ? empty_int : *params.minimax_text_token_tags,
                                                     t_v,
                                                     t_a);
                position_input_cache = sd::Tensor<float>(
                    {3, static_cast<int64_t>(layout_for_reuse.positions.size() / 3)},
                    layout_for_reuse.positions);
                if (config.uses_adaln_curves()) {
                    std::vector<float> fractions(layout_for_reuse.timesteps.size());
                    for (size_t i = 0; i < layout_for_reuse.timesteps.size(); ++i) {
                        float position = std::clamp(layout_for_reuse.timesteps[i], 0.f, 1.f) * (config.adaln_curve_grid - 1);
                        int index      = std::min(static_cast<int>(std::floor(position)),
                                                  static_cast<int>(config.adaln_curve_grid - 2));
                        fractions[i]   = position - index;
                    }
                    curve_fraction_input_cache = sd::Tensor<float>(
                        {1, static_cast<int64_t>(fractions.size())},
                        fractions);
                } else {
                    timestep_feature_input_cache = sd::Tensor<float>(
                        {config.timestep_input_dim, static_cast<int64_t>(layout_for_reuse.timesteps.size())},
                        timestep_embedding(layout_for_reuse.timesteps,
                                           static_cast<int>(config.timestep_input_dim),
                                           10000,
                                           true,
                                           1.f));
                }
                audio_slope_input_cache = sd::Tensor<float>(
                    {1},
                    {time_shift_slope(sigma_v,
                                      params.minimax_video_sigma_shift,
                                      params.minimax_audio_sigma_shift)});
                std::vector<const sd::Tensor<float>*> ordered_inputs;
                ordered_inputs.reserve(8 + conditions.size() + audio_conditions.size());
                ordered_inputs.push_back(&video_input_cache);
                ordered_inputs.push_back(&audio_input_cache);
                ordered_inputs.push_back(params.context);
                for (const auto& condition : conditions) {
                    ordered_inputs.push_back(&condition);
                }
                for (const auto& condition : audio_conditions) {
                    ordered_inputs.push_back(&condition);
                }
                ordered_inputs.push_back(&position_input_cache);
                if (config.uses_adaln_curves()) {
                    ordered_inputs.push_back(&curve_fraction_input_cache);
                } else {
                    ordered_inputs.push_back(&timestep_feature_input_cache);
                }
                ordered_inputs.push_back(&audio_slope_input_cache);
                if (auto out = GGMLRunner::compute_reuse<float>(get_graph, ordered_inputs, n_threads, false)) {
                    return restore_trailing_singleton_dims(std::move(*out), params.x->dim());
                }
            }
            return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph,
                                                                              n_threads,
                                                                              false,
                                                                              false),
                                                   params.x->dim());
        }
    };

}  // namespace MiniMaxH3

#endif  // __ED_DIT_MODELS_MODELS_MINIMAX_H3_FULL_HPP__
