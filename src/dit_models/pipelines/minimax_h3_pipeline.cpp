#include "dit_models/pipelines/minimax_h3_pipeline.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/components/autoencoders/minimax_h3_vae.hpp"
#include "dit_models/components/autoencoders/minimax_h3_audio_vae.hpp"
#include "dit_models/components/text_encoders/llm.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include "utils/rng_philox.hpp"
#include "utils/util.h"

namespace edgedit {
namespace {

bool set_minimax_error(std::string* error, const char* message) {
    if (error != nullptr) {
        *error = message;
    }
    LOG_ERROR("%s", message);
    return false;
}

uint8_t h3_to_u8(float value) {
    return static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

int64_t h3_resolve_seed(int64_t seed) {
    return seed >= 0 ? seed : static_cast<int64_t>(std::time(nullptr));
}

float h3_discrete_flow_sigma(int step, int steps, float shift) {
    const float timestep = 999.0f - 999.0f * static_cast<float>(step) / static_cast<float>(steps);
    const float unit_timestep = (timestep + 1.0f) / 1000.0f;
    return shift == 1.0f ? unit_timestep
                         : shift * unit_timestep / (1.0f + (shift - 1.0f) * unit_timestep);
}

struct MiniMaxH3DetectedConfig {
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t token_refiner_num_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t attention_head_dim = 0;
    int64_t ffn_hidden_size = 0;
    int64_t video_latent_channels = 0;
    int64_t audio_latent_channels = 0;
    int64_t text_dim = 0;
    int64_t adaln_curve_grid = 0;
};

int64_t count_minimax_blocks(const String2TensorStorage& tensors, const std::string& prefix) {
    std::set<int> indices;
    for (const auto& item : tensors) {
        const std::string& name = item.first;
        if (!starts_with(name, prefix)) {
            continue;
        }
        const size_t begin = prefix.size();
        const size_t end = name.find('.', begin);
        if (end != std::string::npos) {
            indices.insert(std::atoi(name.substr(begin, end - begin).c_str()));
        }
    }
    return static_cast<int64_t>(indices.size());
}

MiniMaxH3DetectedConfig detect_minimax_config(const String2TensorStorage& tensors) {
    MiniMaxH3DetectedConfig config;
    const std::string prefix = "model.diffusion_model";
    auto find = [&](const std::string& suffix) -> const TensorStorage* {
        auto it = tensors.find(prefix + "." + suffix);
        return it == tensors.end() ? nullptr : &it->second;
    };
    if (const auto* weight = find("video_patch_proj.weight")) {
        config.video_latent_channels = weight->ne[0] / 4;
        config.hidden_size = weight->ne[1];
    }
    if (const auto* weight = find("audio_patch_proj.weight")) {
        config.audio_latent_channels = weight->ne[0];
    }
    config.num_layers = count_minimax_blocks(tensors, prefix + ".blocks.");
    config.token_refiner_num_layers = count_minimax_blocks(tensors, prefix + ".token_refiner.blocks.");
    if (const auto* weight = find("blocks.0.attn.q_norm.weight")) {
        config.attention_head_dim = weight->ne[0];
    }
    if (const auto* weight = find("blocks.0.attn.qkv_proj.weight")) {
        config.num_attention_heads = weight->ne[1] / (3 * config.attention_head_dim);
    }
    if (const auto* weight = find("blocks.0.mlp.fc1.weight")) {
        config.ffn_hidden_size = weight->ne[1] / 2;
    }
    if (const auto* weight = find("condition_proj.weight")) {
        config.text_dim = weight->ne[0];
    }
    if (const auto* table = find("adaln_t_table")) {
        config.adaln_curve_grid = table->ne[1];
    }
    return config;
}

}  // namespace

MiniMaxH3Pipeline::MiniMaxH3Pipeline(SDVersion version)
    : version_(version) {
}

MiniMaxH3Pipeline::~MiniMaxH3Pipeline() {
    if (sentinel_ctx_ != nullptr) {
        ggml_free(sentinel_ctx_);
        sentinel_ctx_ = nullptr;
        sentinel_tensor_ = nullptr;
    }
}

bool MiniMaxH3Pipeline::prepare(const ed_context_params_t& params,
                                ModelRuntime& runtime,
                                const ModelLoader& loader,
                                PipelineTensorRegistry& registry,
                                std::string* error) {
    (void)params;
    runtime_ = &runtime;
    registry.clear();
    const MiniMaxH3DetectedConfig config = detect_minimax_config(loader.get_tensor_storage_map());
    if (config.num_layers <= 0 || config.hidden_size <= 0 ||
        config.video_latent_channels <= 0 || config.audio_latent_channels <= 0) {
        return set_minimax_error(error, "MiniMax-H3 diffusion transformer signature is incomplete");
    }

    const bool diffusion_offload = runtime.dit_offload_params_to_cpu();
    diffusion_ = std::make_unique<MiniMaxH3::MiniMaxH3Runner>(runtime.backend(),
                                                              loader.get_tensor_storage_map(),
                                                              "model.diffusion_model",
                                                              diffusion_offload);
    diffusion_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
    diffusion_->set_flash_attention_enabled(runtime.flash_attention());
    if (auto process_group = runtime.graph_process_group_ref()) {
        diffusion_->set_process_group(process_group);
    }

    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors(), "model.diffusion_model");

    const bool text_offload = runtime.clip_offload_params_to_cpu();
    conditioner_ = std::make_unique<LLM::LLMEmbedder>(LLM::LLMArch::QWEN3,
                                                       runtime.clip_backend(),
                                                       text_offload,
                                                       loader.get_tensor_storage_map(),
                                                       "text_encoders.llm",
                                                       false);
    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors(), "text_encoders.llm");

    vae_ = std::make_unique<MiniMaxH3VAE::MiniMaxH3VideoVAERunner>(runtime.vae_backend(),
                                                                    runtime.vae_offload_params_to_cpu(),
                                                                    loader.get_tensor_storage_map(),
                                                                    "first_stage_model");
    vae_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
    vae_->set_flash_attention_enabled(runtime.flash_attention());
    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), "first_stage_model");

    audio_vae_ = std::make_unique<MiniMaxH3Audio::AudioVAERunner>(runtime.vae_backend(),
                                                                    runtime.vae_offload_params_to_cpu(),
                                                                    loader.get_tensor_storage_map(),
                                                                    "audio_vae");
    audio_vae_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
    audio_vae_->set_flash_attention_enabled(runtime.flash_attention());
    audio_vae_->alloc_params_buffer();
    audio_vae_->get_param_tensors(registry.tensors(), "audio_vae");

    registry.ignore_prefix("first_stage_model.encoder.");
    registry.ignore_prefix("text_encoders.llm.visual.");
    registry.ignore_prefix("text_encoders.llm.");

    if (sentinel_ctx_ == nullptr) {
        ggml_init_params init_params{};
        init_params.mem_size = ggml_tensor_overhead() + 1024;
        init_params.mem_buffer = nullptr;
        init_params.no_alloc = false;
        sentinel_ctx_ = ggml_init(init_params);
        if (sentinel_ctx_ == nullptr) {
            return set_minimax_error(error, "failed to allocate MiniMax-H3 sentinel tensor context");
        }
        sentinel_tensor_ = ggml_new_tensor_1d(sentinel_ctx_, GGML_TYPE_F32, 1);
    }
    registry.add("__ed_minimax_h3_sentinel.weight", sentinel_tensor_);

    ready_ = true;
    LOG_INFO("MiniMax-H3 detected: layers=%lld token_refiner_layers=%lld hidden=%lld heads=%lld head_dim=%lld ffn=%lld video_latent=%lld audio_latent=%lld text_dim=%lld adaln_curve_grid=%lld",
             (long long)config.num_layers,
             (long long)config.token_refiner_num_layers,
             (long long)config.hidden_size,
             (long long)config.num_attention_heads,
             (long long)config.attention_head_dim,
             (long long)config.ffn_hidden_size,
             (long long)config.video_latent_channels,
             (long long)config.audio_latent_channels,
             (long long)config.text_dim,
             (long long)config.adaln_curve_grid);
    return true;
}

void MiniMaxH3Pipeline::mark_ready() {
    ready_ = true;
}

ed_status_t MiniMaxH3Pipeline::generate_image(const ed_image_generation_params_t*,
                                              ed_image_batch_t*,
                                              std::string* error) {
    set_minimax_error(error, "MiniMax-H3 supports video generation only");
    return ED_STATUS_UNSUPPORTED;
}

bool MiniMaxH3Pipeline::build_text_context(const char* prompt,
                                           sd::Tensor<float>* context,
                                           sd::Tensor<int32_t>* token_tags,
                                           std::string* error) {
    if (context == nullptr || token_tags == nullptr || conditioner_ == nullptr || runtime_ == nullptr) {
        return set_minimax_error(error, "MiniMax-H3 text conditioner is not initialized");
    }
    const std::string text = "<|im_start|>user\n" +
                             std::string(prompt == nullptr ? "" : prompt) +
                             "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> tokens = conditioner_->tokenizer->tokenize(text, nullptr, true, 0, 4096, false);
    if (tokens.empty()) {
        return set_minimax_error(error, "MiniMax-H3 prompt tokenization produced no tokens");
    }
    sd::Tensor<int32_t> ids({static_cast<int64_t>(tokens.size())}, tokens);
    *context = conditioner_->model.compute(runtime_->n_threads(), ids, {}, {}, {50});
    if (context->empty()) {
        return set_minimax_error(error, "MiniMax-H3 text encoder compute failed");
    }
    *token_tags = sd::Tensor<int32_t>({context->shape()[1]},
                                      std::vector<int32_t>(static_cast<size_t>(context->shape()[1]), 1));
    return true;
}

ed_status_t MiniMaxH3Pipeline::decode_video_latent(const sd::Tensor<float>& latent,
                                                    int requested_frames,
                                                    ed_video_t* out,
                                                    std::string* error) {
    sd::Tensor<float> vae_latent = vae_->diffusion_to_vae_latents(latent);
    ed_tiling_params_t tiling = runtime_->vae_tiling();
    if (tiling.force_disable) {
        tiling.enabled = false;
    }
    sd::Tensor<float> video = vae_->decode(runtime_->n_threads(), vae_latent, tiling, true);
    if (video.empty() || video.dim() != 5) {
        set_minimax_error(error, "MiniMax-H3 video VAE decode failed");
        return ED_STATUS_GENERATION_FAILED;
    }
    const int decoded_frames = static_cast<int>(video.shape()[2]);
    const int frames_count = std::max(decoded_frames, requested_frames);
    ed_image_t* frames = static_cast<ed_image_t*>(std::calloc(static_cast<size_t>(frames_count), sizeof(ed_image_t)));
    if (frames == nullptr) {
        set_minimax_error(error, "failed to allocate MiniMax-H3 output frames");
        return ED_STATUS_OUT_OF_MEMORY;
    }
    const size_t width = static_cast<size_t>(video.shape()[0]);
    const size_t height = static_cast<size_t>(video.shape()[1]);
    const size_t channels = static_cast<size_t>(video.shape()[3]);
    const size_t pixels = width * height;
    for (int frame = 0; frame < frames_count; ++frame) {
        frames[frame].width = static_cast<int>(width);
        frames[frame].height = static_cast<int>(height);
        frames[frame].channels = static_cast<int>(channels);
        frames[frame].data = static_cast<uint8_t*>(std::malloc(pixels * channels));
        if (frames[frame].data == nullptr) {
            for (int index = 0; index < frame; ++index) std::free(frames[index].data);
            std::free(frames);
            set_minimax_error(error, "failed to allocate MiniMax-H3 frame pixels");
            return ED_STATUS_OUT_OF_MEMORY;
        }
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            for (size_t channel = 0; channel < channels; ++channel) {
                frames[frame].data[pixel * channels + channel] =
                    h3_to_u8(video.index(pixel % width,
                                         pixel / width,
                                         std::min(frame, decoded_frames - 1),
                                         channel,
                                         0));
            }
        }
    }
    out->frames = frames;
    out->frame_count = frames_count;
    return ED_STATUS_OK;
}

bool MiniMaxH3Pipeline::decode_audio_latent(const sd::Tensor<float>& latent,
                                            ed_video_t* out,
                                            std::string* error) {
    if (audio_vae_ == nullptr || latent.empty()) {
        return set_minimax_error(error, "MiniMax-H3 audio VAE is not initialized");
    }
    sd::Tensor<float> waveform = audio_vae_->decode(runtime_->n_threads(), latent);
    if (waveform.empty() || waveform.dim() != 4 || waveform.shape()[1] != 2) {
        return set_minimax_error(error, "MiniMax-H3 audio VAE decode failed");
    }
    const int64_t sample_count = waveform.shape()[0];
    const int channels = static_cast<int>(waveform.shape()[1]);
    if (sample_count <= 0 || sample_count > std::numeric_limits<int>::max()) {
        return set_minimax_error(error, "MiniMax-H3 audio sample count is invalid");
    }
    float* samples = static_cast<float*>(std::malloc(static_cast<size_t>(sample_count) * channels * sizeof(float)));
    if (samples == nullptr) {
        return set_minimax_error(error, "failed to allocate MiniMax-H3 audio output");
    }
    for (int64_t sample = 0; sample < sample_count; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
            samples[sample * channels + channel] = std::clamp(waveform.index(sample, channel, 0, 0), -1.0f, 1.0f);
        }
    }
    out->audio = samples;
    out->audio_sample_count = static_cast<int>(sample_count);
    out->audio_channels = channels;
    out->audio_sample_rate = 32000;
    return true;
}

ed_status_t MiniMaxH3Pipeline::generate_video(const ed_video_generation_params_t* params,
                                              ed_video_t* out,
                                              std::string* error) {
    if (out == nullptr || params == nullptr) {
        set_minimax_error(error, "MiniMax-H3 video parameters or output are null");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    out->frames = nullptr;
    out->frame_count = 0;
    if (!ready_ || runtime_ == nullptr || !diffusion_ || !conditioner_ || !vae_) {
        set_minimax_error(error, "MiniMax-H3 pipeline is not ready");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (params->width <= 0 || params->height <= 0 || params->frames <= 0 ||
        params->width % 32 != 0 || params->height % 32 != 0) {
        set_minimax_error(error, "MiniMax-H3 width and height must be positive multiples of 32");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const int frames = std::max(5, params->frames);
    if (frames % 17 != 5) {
        set_minimax_error(error, "MiniMax-H3 frame count must satisfy 17k + 5");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    sd::Tensor<float> context;
    sd::Tensor<int32_t> token_tags;
    if (!build_text_context(params->prompt, &context, &token_tags, error)) return ED_STATUS_GENERATION_FAILED;
    const int latent_frames = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
    const int audio_length = std::max(1, static_cast<int>(std::lround(static_cast<double>(frames) * 40.0 / 24.0)));
    const int latent_width = params->width / 16;
    const int latent_height = params->height / 16;
    const int64_t video_spatial = static_cast<int64_t>(latent_width) * latent_height * latent_frames;
    sd::Tensor<float> video = sd::zeros<float>({latent_width, latent_height, latent_frames, 24, 1});
    sd::Tensor<float> audio = sd::zeros<float>({audio_length, 2, 32, 1});
    const int64_t audio_channels = (audio.numel() + video_spatial - 1) / video_spatial;
    sd::Tensor<float> packed({latent_width,
                              latent_height,
                              latent_frames,
                              24 + audio_channels,
                              1});
    auto rng = std::make_shared<PhiloxRNG>(static_cast<uint64_t>(h3_resolve_seed(params->seed)));
    packed = sd::randn_like<float>(packed, rng);
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    const float video_sigma_shift = params->sample.flow_shift > 0.0f ? params->sample.flow_shift : 12.0f;
    for (int step = 0; step < steps; ++step) {
        const float sigma = h3_discrete_flow_sigma(step, steps, video_sigma_shift);
        const float sigma_next = step + 1 == steps ? 0.0f
                                                    : h3_discrete_flow_sigma(step + 1, steps, video_sigma_shift);
        sd::Tensor<float> timestep({1}, {sigma * 1000.0f});
        DiffusionParams diffusion_params{};
        diffusion_params.x = &packed;
        diffusion_params.timesteps = &timestep;
        diffusion_params.context = &context;
        diffusion_params.minimax_text_token_tags = &token_tags;
        diffusion_params.minimax_audio_length = audio_length;
        diffusion_params.minimax_video_sigma_shift = video_sigma_shift;
        diffusion_params.minimax_audio_sigma_shift = 3.0f;
        sd::Tensor<float> velocity = diffusion_->compute(runtime_->n_threads(), diffusion_params);
        if (velocity.empty()) {
            set_minimax_error(error, "MiniMax-H3 diffusion compute failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        packed += velocity * (sigma_next - sigma);
    }
    auto av = diffusion_->split_av_latents(packed, audio_length);
    ed_status_t status = decode_video_latent(av.first, frames, out, error);
    if (status != ED_STATUS_OK) {
        return status;
    }
    if (!decode_audio_latent(av.second, out, error)) {
        ed_free_video(out);
        return ED_STATUS_GENERATION_FAILED;
    }
    return ED_STATUS_OK;
}

ed_scheduler_t MiniMaxH3Pipeline::default_scheduler(ed_sampler_t) const {
    return ED_SCHEDULER_DISCRETE;
}

}  // namespace edgedit
