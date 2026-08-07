#include "dit_models/pipelines/minimax_h3_pipeline.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

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

bool h3_profile_enabled();

ed_tiling_params_t h3_vae_tiling(const ModelRuntime& runtime) {
    ed_tiling_params_t tiling = runtime.vae_tiling();
    if (tiling.force_disable) {
        tiling.enabled = false;
    }
    if (h3_profile_enabled()) {
        LOG_INFO("minimax-h3 profile VAE tiling: enabled=%d force-disable=%d rel=%.3f,%.3f",
                 tiling.enabled,
                 tiling.force_disable,
                 tiling.rel_size_x,
                 tiling.rel_size_y);
    }
    return tiling;
}

void h3_resize_for_vision(int source_width, int source_height, int* width, int* height) {
    constexpr int factor = 32;
    constexpr int min_pixels = 3136;
    constexpr int max_pixels = 12845056;
    int resized_width = std::max(factor, static_cast<int>(std::round(static_cast<double>(source_width) / factor)) * factor);
    int resized_height = std::max(factor, static_cast<int>(std::round(static_cast<double>(source_height) / factor)) * factor);
    const double area = static_cast<double>(resized_width) * resized_height;
    if (area > max_pixels) {
        const double scale = std::sqrt(static_cast<double>(source_width) * source_height / max_pixels);
        resized_width = std::max(factor, static_cast<int>(std::floor(source_width / scale / factor)) * factor);
        resized_height = std::max(factor, static_cast<int>(std::floor(source_height / scale / factor)) * factor);
    } else if (area < min_pixels) {
        const double scale = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(source_width) * source_height));
        resized_width = static_cast<int>(std::ceil(source_width * scale / factor)) * factor;
        resized_height = static_cast<int>(std::ceil(source_height * scale / factor)) * factor;
    }
    *width = resized_width;
    *height = resized_height;
}

void h3_reference_video_dimensions(const ed_image_t& image, int* width, int* height) {
    const double ratio = static_cast<double>(image.width) / image.height;
    double nominal_width = ratio >= 1.0 ? 768.0 * ratio : 768.0;
    double nominal_height = ratio >= 1.0 ? 768.0 : 768.0 / ratio;
    if (nominal_width * nominal_height > 768.0 * 1344.0) {
        const double scale = std::sqrt((768.0 * 1344.0) / (nominal_width * nominal_height));
        nominal_width *= scale;
        nominal_height *= scale;
    }
    *width = std::max(32, static_cast<int>(std::round(nominal_width / 32.0)) * 32);
    *height = std::max(32, static_cast<int>(std::round(nominal_height / 32.0)) * 32);
    if (static_cast<int64_t>(image.width) * image.height < static_cast<int64_t>(*width) * *height) {
        *width = std::max(32, static_cast<int>(std::round(static_cast<double>(image.width) / 32.0)) * 32);
        *height = std::max(32, static_cast<int>(std::round(static_cast<double>(image.height) / 32.0)) * 32);
    }
}

float h3_discrete_flow_sigma(int step, int steps, float shift) {
    if (steps <= 1) {
        return 1.0f;
    }
    const float timestep = 999.0f - 999.0f * static_cast<float>(step) / static_cast<float>(steps - 1);
    const float unit_timestep = (timestep + 1.0f) / 1000.0f;
    return shift == 1.0f ? unit_timestep
                         : shift * unit_timestep / (1.0f + (shift - 1.0f) * unit_timestep);
}

bool h3_trace_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_TRACE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool h3_profile_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_PROFILE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool h3_fast_video_postprocess_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool h3_verify_fast_video_postprocess_enabled() {
    const char* value = std::getenv("ED_MINIMAX_H3_VERIFY_FAST_VIDEO_POSTPROCESS");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

int h3_fast_video_postprocess_threads() {
    const char* value = std::getenv("ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char* end = nullptr;
    const long requested = std::strtol(value, &end, 10);
    return end != value && requested > 1 && requested <= std::numeric_limits<int>::max()
               ? static_cast<int>(requested)
               : 0;
}

void h3_trace_tensor(const char* name, const sd::Tensor<float>& tensor) {
    if (!h3_trace_enabled()) {
        return;
    }
    if (tensor.empty()) {
        LOG_INFO("minimax-h3 trace %s: empty", name);
        return;
    }

    uint64_t hash = 1469598103934665603ULL;
    double sum = 0.0;
    double squared_sum = 0.0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (float value : tensor.values()) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
        sum += value;
        squared_sum += static_cast<double>(value) * value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const double count = static_cast<double>(tensor.numel());
    LOG_INFO("minimax-h3 trace %s: shape=%s n=%lld hash=%016llx mean=%.8g rms=%.8g min=%.8g max=%.8g",
             name,
             sd::tensor_shape_to_string(tensor.shape()).c_str(),
             static_cast<long long>(tensor.numel()),
             static_cast<unsigned long long>(hash),
             sum / count,
             std::sqrt(squared_sum / count),
             minimum,
             maximum);
}

sd::Tensor<float> h3_pack_audio_and_video_latents(const sd::Tensor<float>& video,
                                                   const sd::Tensor<float>& audio) {
    if (audio.empty()) {
        return video;
    }
    GGML_ASSERT(video.dim() == 5 && video.shape()[4] == 1);
    GGML_ASSERT(audio.dim() == 4 && audio.shape()[3] == 1);

    const int64_t spatial_size = video.shape()[0] * video.shape()[1] * video.shape()[2];
    const int64_t extra_channels = (audio.numel() + spatial_size - 1) / spatial_size;
    std::vector<int64_t> packed_shape = video.shape();
    packed_shape[3] += extra_channels;
    sd::Tensor<float> packed = sd::zeros<float>(packed_shape);
    std::copy_n(video.data(), video.numel(), packed.data());
    std::copy_n(audio.data(), audio.numel(), packed.data() + video.numel());
    return packed;
}

sd::Tensor<float> h3_image_to_tensor(const ed_image_t& image, int width, int height) {
    if (image.data == nullptr || image.width <= 0 || image.height <= 0 || image.channels <= 0) {
        return {};
    }
    const int source_width = static_cast<int>(image.width);
    const int source_height = static_cast<int>(image.height);
    const int source_channels = static_cast<int>(image.channels);
    sd::Tensor<float> tensor({width, height, 3, 1});
    for (int y = 0; y < height; ++y) {
        const int source_y = std::min(source_height - 1, static_cast<int>((static_cast<int64_t>(y) * source_height) / height));
        for (int x = 0; x < width; ++x) {
            const int source_x = std::min(source_width - 1, static_cast<int>((static_cast<int64_t>(x) * source_width) / width));
            const uint8_t* pixel = image.data +
                                   (static_cast<size_t>(source_y) * source_width + static_cast<size_t>(source_x)) * source_channels;
            for (int channel = 0; channel < 3; ++channel) {
                tensor.index(x, y, channel, 0) = static_cast<float>(pixel[std::min(channel, source_channels - 1)]) / 255.0f;
            }
        }
    }
    return tensor;
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

struct MiniMaxH3Profile {
    int64_t total_ms = 0;
    int64_t cond_context_ms = 0;
    int64_t uncond_context_ms = 0;
    int64_t vision_image_prepare_ms = 0;
    int64_t vision_image_encode_ms = 0;
    int64_t vision_video_prepare_ms = 0;
    int64_t vision_video_encode_ms = 0;
    int64_t text_tokenize_ms = 0;
    int64_t text_encode_ms = 0;
    int64_t keyframe_vae_encode_ms = 0;
    int64_t reference_video_vae_encode_ms = 0;
    int64_t reference_audio_prepare_ms = 0;
    int64_t reference_audio_vae_encode_ms = 0;
    int64_t noise_init_ms = 0;
    int64_t diffusion_cond_ms = 0;
    int64_t diffusion_uncond_ms = 0;
    int64_t cfg_combine_ms = 0;
    int64_t video_vae_decode_ms = 0;
    int64_t video_postprocess_ms = 0;
    int64_t audio_vae_decode_ms = 0;
    int64_t audio_postprocess_ms = 0;
    int diffusion_steps = 0;
    int diffusion_calls = 0;

    void log() const {
        const int64_t diffusion_ms = diffusion_cond_ms + diffusion_uncond_ms + cfg_combine_ms;
        const int64_t conditioning_ms = cond_context_ms + uncond_context_ms + keyframe_vae_encode_ms +
                                        reference_video_vae_encode_ms + reference_audio_prepare_ms +
                                        reference_audio_vae_encode_ms;
        const int64_t decode_ms = video_vae_decode_ms + video_postprocess_ms + audio_vae_decode_ms + audio_postprocess_ms;
        LOG_INFO("minimax-h3 profile: total=%lld ms | conditioning=%lld ms | diffusion=%lld ms (%d steps, %d calls) | decode=%lld ms | noise=%lld ms",
                 static_cast<long long>(total_ms), static_cast<long long>(conditioning_ms),
                 static_cast<long long>(diffusion_ms), diffusion_steps, diffusion_calls,
                 static_cast<long long>(decode_ms), static_cast<long long>(noise_init_ms));
        LOG_INFO("minimax-h3 profile conditioning: context cond=%lld ms uncond=%lld ms | vision image prepare=%lld ms encode=%lld ms | vision video prepare=%lld ms encode=%lld ms | text tokenize=%lld ms encode=%lld ms | keyframe vae=%lld ms | ref video vae=%lld ms | ref audio prepare=%lld ms vae=%lld ms",
                 static_cast<long long>(cond_context_ms), static_cast<long long>(uncond_context_ms),
                 static_cast<long long>(vision_image_prepare_ms), static_cast<long long>(vision_image_encode_ms),
                 static_cast<long long>(vision_video_prepare_ms), static_cast<long long>(vision_video_encode_ms),
                 static_cast<long long>(text_tokenize_ms), static_cast<long long>(text_encode_ms),
                 static_cast<long long>(keyframe_vae_encode_ms), static_cast<long long>(reference_video_vae_encode_ms),
                 static_cast<long long>(reference_audio_prepare_ms), static_cast<long long>(reference_audio_vae_encode_ms));
        LOG_INFO("minimax-h3 profile diffusion: conditional=%lld ms unconditional=%lld ms cfg-combine=%lld ms | decode video-vae=%lld ms video-copy=%lld ms audio-vae=%lld ms audio-copy=%lld ms",
                 static_cast<long long>(diffusion_cond_ms), static_cast<long long>(diffusion_uncond_ms),
                 static_cast<long long>(cfg_combine_ms), static_cast<long long>(video_vae_decode_ms),
                 static_cast<long long>(video_postprocess_ms), static_cast<long long>(audio_vae_decode_ms),
                 static_cast<long long>(audio_postprocess_ms));
    }
};

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
    conditioner_ = std::make_unique<LLM::LLMEmbedder>(LLM::LLMArch::QWEN3_VL,
                                                       runtime.clip_backend(),
                                                       text_offload,
                                                       loader.get_tensor_storage_map(),
                                                       "text_encoders.llm",
                                                       true);
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

    const bool has_audio_vae = std::any_of(loader.get_tensor_storage_map().begin(),
                                           loader.get_tensor_storage_map().end(),
                                           [](const auto& item) { return starts_with(item.first, "audio_vae."); });
    if (has_audio_vae) {
        audio_vae_ = std::make_unique<MiniMaxH3Audio::AudioVAERunner>(runtime.vae_backend(),
                                                                        runtime.vae_offload_params_to_cpu(),
                                                                        loader.get_tensor_storage_map(),
                                                                        "audio_vae");
        audio_vae_->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
        audio_vae_->set_flash_attention_enabled(runtime.flash_attention());
        audio_vae_->alloc_params_buffer();
        audio_vae_->get_param_tensors(registry.tensors(), "audio_vae");
    } else {
        LOG_INFO("MiniMax-H3 audio VAE not provided; generated video will have no decoded audio track");
    }

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
                                           const ed_image_t* ref_images,
                                           int ref_image_count,
                                           const ed_ref_video_t* ref_videos,
                                           int ref_video_count,
                                           int ref_audio_count,
                                           int max_video_frames,
                                           sd::Tensor<float>* context,
                                           sd::Tensor<int32_t>* token_tags,
                                           MiniMaxH3Profile* profile,
                                           std::string* error) {
    if (context == nullptr || token_tags == nullptr || conditioner_ == nullptr || runtime_ == nullptr) {
        return set_minimax_error(error, "MiniMax-H3 text conditioner is not initialized");
    }
    std::string presentations;
    std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
    std::vector<LLM::LLMImageEmbedInfo> image_embed_infos;
    std::vector<std::vector<std::pair<int, sd::Tensor<float>>>> deepstack_image_embeds(3);
    std::vector<std::pair<int, int64_t>> vision_token_ranges;
    const int vision_patch_size = conditioner_->model.params.vision.patch_size;
    for (int image_index = 0; image_index < ref_image_count; ++image_index) {
        const int64_t prepare_begin = profile != nullptr ? ggml_time_ms() : 0;
        const ed_image_t& image = ref_images[image_index];
        int width = 0;
        int height = 0;
        h3_resize_for_vision(image.width, image.height, &width, &height);
        sd::Tensor<float> image_tensor = h3_image_to_tensor(image, width, height);
        if (image_tensor.empty()) {
            return set_minimax_error(error, "MiniMax-H3 Ref2VA image reference is invalid");
        }
        image_tensor = image_tensor * 2.0f - 1.0f;
        if (profile != nullptr) profile->vision_image_prepare_ms += ggml_time_ms() - prepare_begin;
        LOG_DEBUG("MiniMax-H3 Ref2VA vision image=%dx%d tensor=%s",
                  width,
                  height,
                  sd::tensor_shape_to_string(image_tensor.shape()).c_str());
        const int64_t encode_begin = profile != nullptr ? ggml_time_ms() : 0;
        std::vector<sd::Tensor<float>> image_outputs = conditioner_->model.encode_image_outputs(runtime_->n_threads(), image_tensor);
        if (profile != nullptr) profile->vision_image_encode_ms += ggml_time_ms() - encode_begin;
        if (image_outputs.size() != 4 || image_outputs[0].empty()) {
            return set_minimax_error(error, "MiniMax-H3 Ref2VA vision encoder failed");
        }
        sd::Tensor<float> image_embed = std::move(image_outputs[0]);
        const int64_t image_tokens = image_embed.shape()[1];
        const std::string prefix = "<Picture " + std::to_string(image_index + 1) + ">: <|vision_start|>";
        const std::string prefix_text = "<|im_start|>user\n" + presentations + prefix;
        const std::vector<int> prefix_tokens = conditioner_->tokenizer->tokenize(prefix_text, nullptr, true, 0, 4096, false);
        presentations += prefix;
        for (int64_t token = 0; token < image_tokens; ++token) {
            presentations += "<|image_pad|>";
        }
        presentations += "<|vision_end|>\n";
        image_embeds.emplace_back(static_cast<int>(prefix_tokens.size()), std::move(image_embed));
        for (size_t layer = 0; layer < deepstack_image_embeds.size(); ++layer) {
            deepstack_image_embeds[layer].emplace_back(static_cast<int>(prefix_tokens.size()), std::move(image_outputs[layer + 1]));
        }
        const int embed_index = static_cast<int>(prefix_tokens.size());
        image_embed_infos.push_back({embed_index, image_tokens, 1, height / vision_patch_size, width / vision_patch_size});
        vision_token_ranges.emplace_back(embed_index, image_tokens);
    }
    int video_number = 0;
    int audio_number = 0;
    for (int video_index = 0; video_index < ref_video_count; ++video_index) {
        const ed_ref_video_t& reference = ref_videos[video_index];
        if (reference.frames == nullptr || reference.frame_count <= 0) {
            return set_minimax_error(error, "MiniMax-H3 Ref2VA video reference is invalid");
        }
        const int source_fps = reference.fps > 0 ? reference.fps : 24;
        int normalized_frames = static_cast<int>(std::lround(static_cast<double>(reference.frame_count) * 24.0 / source_fps));
        normalized_frames = std::min(normalized_frames, max_video_frames);
        if (normalized_frames < 5) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video reference needs at least 5 frames at 24 fps");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        if (reference.audio.data != nullptr && reference.audio.sample_count > 0) {
            presentations += "<Audio " + std::to_string(++audio_number) + ">: ";
        }
        presentations += "<Video " + std::to_string(++video_number) + ">: ";
        while (normalized_frames % 17 != 5) --normalized_frames;
        std::vector<int> sampled_frames;
        sampled_frames.reserve(static_cast<size_t>((normalized_frames + 11) / 12));
        for (int frame = 0; frame < normalized_frames; frame += 12) {
            sampled_frames.push_back(frame);
        }
        for (size_t sampled_index = 0; sampled_index < sampled_frames.size(); sampled_index += 2) {
            const int64_t prepare_begin = profile != nullptr ? ggml_time_ms() : 0;
            const size_t next_sampled_index = std::min(sampled_index + 1, sampled_frames.size() - 1);
            const int first_frame = sampled_frames[sampled_index];
            const int second_frame = sampled_frames[next_sampled_index];
            const int first_index = std::min(reference.frame_count - 1, static_cast<int>(std::floor(first_frame * source_fps / 24.0)));
            const int second_index = std::min(reference.frame_count - 1, static_cast<int>(std::floor(second_frame * source_fps / 24.0)));
            const ed_image_t& first_image = reference.frames[first_index];
            int width = 0;
            int height = 0;
            h3_resize_for_vision(first_image.width, first_image.height, &width, &height);
            auto first = h3_image_to_tensor(first_image, width, height);
            auto second = h3_image_to_tensor(reference.frames[second_index], width, height);
            if (first.empty() || second.empty()) return set_minimax_error(error, "MiniMax-H3 Ref2VA video frame is invalid");
            sd::Tensor<float> pair({width, height, 2, 3, 1});
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        pair.index(x, y, 0, channel, 0) = first.index(x, y, channel, 0) * 2.f - 1.f;
                        pair.index(x, y, 1, channel, 0) = second.index(x, y, channel, 0) * 2.f - 1.f;
                    }
                }
            }
            if (profile != nullptr) profile->vision_video_prepare_ms += ggml_time_ms() - prepare_begin;
            const int64_t encode_begin = profile != nullptr ? ggml_time_ms() : 0;
            auto outputs = conditioner_->model.encode_video_block_outputs(runtime_->n_threads(), pair);
            if (profile != nullptr) profile->vision_video_encode_ms += ggml_time_ms() - encode_begin;
            if (outputs.size() != 4 || outputs[0].empty()) return set_minimax_error(error, "MiniMax-H3 Ref2VA video vision encoder failed");
            const float first_time = static_cast<float>(first_frame) / 24.0f;
            const float second_time = static_cast<float>(second_frame) / 24.0f;
            std::ostringstream timestamp_stream;
            timestamp_stream << '<' << std::fixed << std::setprecision(1)
                             << (first_time + second_time) * 0.5f << " seconds>";
            const std::string timestamp = timestamp_stream.str();
            const auto prefix_tokens = conditioner_->tokenizer->tokenize("<|im_start|>user\n" + presentations + timestamp + "<|vision_start|>", nullptr, true, 0, 4096, false);
            presentations += timestamp + "<|vision_start|>";
            const int64_t tokens = outputs[0].shape()[1];
            for (int64_t token = 0; token < tokens; ++token) presentations += "<|image_pad|>";
            presentations += "<|vision_end|>";
            const int embed_index = static_cast<int>(prefix_tokens.size());
            image_embeds.emplace_back(embed_index, std::move(outputs[0]));
            for (size_t layer = 0; layer < deepstack_image_embeds.size(); ++layer) deepstack_image_embeds[layer].emplace_back(embed_index, std::move(outputs[layer + 1]));
            image_embed_infos.push_back({embed_index, tokens, 1, height / vision_patch_size, width / vision_patch_size});
            vision_token_ranges.emplace_back(embed_index, tokens);
        }
    }
    for (int audio_index = 0; audio_index < ref_audio_count; ++audio_index) {
        presentations += "<Audio " + std::to_string(++audio_number) + ">: ";
    }
    const std::string text = "<|im_start|>user\n" + presentations +
                             std::string(prompt == nullptr ? "" : prompt) +
                             "<|im_end|>\n<|im_start|>assistant\n";
    const int64_t tokenize_begin = profile != nullptr ? ggml_time_ms() : 0;
    const std::vector<int> tokens = conditioner_->tokenizer->tokenize(text, nullptr, true, 0, 4096, false);
    if (profile != nullptr) profile->text_tokenize_ms += ggml_time_ms() - tokenize_begin;
    if (tokens.empty()) {
        return set_minimax_error(error, "MiniMax-H3 prompt tokenization produced no tokens");
    }
    sd::Tensor<int32_t> ids({static_cast<int64_t>(tokens.size())}, tokens);
    const int64_t text_encode_begin = profile != nullptr ? ggml_time_ms() : 0;
    *context = conditioner_->model.compute(runtime_->n_threads(), ids, {}, image_embeds, {50}, image_embed_infos, "", deepstack_image_embeds);
    if (profile != nullptr) profile->text_encode_ms += ggml_time_ms() - text_encode_begin;
    if (context->empty()) {
        return set_minimax_error(error, "MiniMax-H3 text encoder compute failed");
    }
    std::vector<int32_t> tags(static_cast<size_t>(context->shape()[1]), 1);
    for (const auto& range : vision_token_ranges) {
        const int64_t begin = std::max<int64_t>(0, static_cast<int64_t>(range.first) - 1);
        const int64_t end = std::min<int64_t>(static_cast<int64_t>(tags.size()), static_cast<int64_t>(range.first) + range.second + 1);
        std::fill(tags.begin() + begin, tags.begin() + end, 0);
    }
    const int64_t tag_count = static_cast<int64_t>(tags.size());
    *token_tags = sd::Tensor<int32_t>({tag_count}, std::move(tags));
    h3_trace_tensor("text_context", *context);
    return true;
}

ed_status_t MiniMaxH3Pipeline::decode_video_latent(const sd::Tensor<float>& latent,
                                                    int requested_frames,
                                                    ed_video_t* out,
                                                    MiniMaxH3Profile* profile,
                                                    std::string* error) {
    sd::Tensor<float> vae_latent = vae_->diffusion_to_vae_latents(latent);
    ed_tiling_params_t tiling = h3_vae_tiling(*runtime_);
    const int64_t decode_begin = profile != nullptr ? ggml_time_ms() : 0;
    sd::Tensor<float> video = vae_->decode(runtime_->n_threads(), vae_latent, tiling, true);
    if (profile != nullptr) profile->video_vae_decode_ms += ggml_time_ms() - decode_begin;
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
    const int64_t postprocess_begin = profile != nullptr ? ggml_time_ms() : 0;
    const bool fast_postprocess = h3_fast_video_postprocess_enabled();
    const bool verify_fast_postprocess = fast_postprocess && h3_verify_fast_video_postprocess_enabled();
    const float* video_data     = video.data();
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
    }
    auto convert_frame = [&](int frame) {
        const int source_frame = std::min(frame, decoded_frames - 1);
        if (fast_postprocess) {
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                const size_t x = pixel % width;
                const size_t y = pixel / width;
                for (size_t channel = 0; channel < channels; ++channel) {
                    const size_t offset = x + width * (y + height * (source_frame + decoded_frames * channel));
                    if (verify_fast_postprocess) {
                        GGML_ASSERT(video_data[offset] == video.index(x, y, source_frame, channel, 0));
                    }
                    frames[frame].data[pixel * channels + channel] = h3_to_u8(video_data[offset]);
                }
            }
        } else {
            for (size_t pixel = 0; pixel < pixels; ++pixel) {
                for (size_t channel = 0; channel < channels; ++channel) {
                    frames[frame].data[pixel * channels + channel] =
                        h3_to_u8(video.index(pixel % width, pixel / width, source_frame, channel, 0));
                }
            }
        }
    };
    const int requested_threads = fast_postprocess ? h3_fast_video_postprocess_threads() : 0;
    const int conversion_threads = std::min(frames_count, requested_threads);
    if (conversion_threads > 1) {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(conversion_threads));
        for (int worker = 0; worker < conversion_threads; ++worker) {
            workers.emplace_back([&, worker]() {
                for (int frame = worker; frame < frames_count; frame += conversion_threads) {
                    convert_frame(frame);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    } else {
        for (int frame = 0; frame < frames_count; ++frame) {
            convert_frame(frame);
        }
    }
    out->frames = frames;
    out->frame_count = frames_count;
    if (profile != nullptr) profile->video_postprocess_ms += ggml_time_ms() - postprocess_begin;
    return ED_STATUS_OK;
}

bool MiniMaxH3Pipeline::decode_audio_latent(const sd::Tensor<float>& latent,
                                            ed_video_t* out,
                                            MiniMaxH3Profile* profile,
                                            std::string* error) {
    if (audio_vae_ == nullptr || latent.empty()) {
        return set_minimax_error(error, "MiniMax-H3 audio VAE is not initialized");
    }
    const int64_t decode_begin = profile != nullptr ? ggml_time_ms() : 0;
    sd::Tensor<float> waveform = audio_vae_->decode(runtime_->n_threads(), latent);
    if (profile != nullptr) profile->audio_vae_decode_ms += ggml_time_ms() - decode_begin;
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
    const int64_t postprocess_begin = profile != nullptr ? ggml_time_ms() : 0;
    for (int64_t sample = 0; sample < sample_count; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
            samples[sample * channels + channel] = std::clamp(waveform.index(sample, channel, 0, 0), -1.0f, 1.0f);
        }
    }
    out->audio = samples;
    out->audio_sample_count = static_cast<int>(sample_count);
    out->audio_channels = channels;
    out->audio_sample_rate = 32000;
    if (profile != nullptr) profile->audio_postprocess_ms += ggml_time_ms() - postprocess_begin;
    return true;
}

ed_status_t MiniMaxH3Pipeline::generate_video(const ed_video_generation_params_t* params,
                                              ed_video_t* out,
                                              std::string* error) {
    MiniMaxH3Profile profile;
    MiniMaxH3Profile* profile_ptr = h3_profile_enabled() ? &profile : nullptr;
    const int64_t generation_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
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
    const bool has_keyframes = (params->init_image != nullptr && params->init_image->data != nullptr) ||
                               (params->end_image != nullptr && params->end_image->data != nullptr);
    const bool has_references = params->ref_image_count > 0 || params->ref_video_count > 0 || params->ref_audio_count > 0;
    if (has_keyframes && has_references) {
        set_minimax_error(error, "MiniMax-H3 Ref2VA references cannot be combined with init or end keyframes");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (params->ref_image_count < 0 || params->ref_video_count < 0 || params->ref_audio_count < 0 ||
        (params->ref_image_count > 0 && params->ref_images == nullptr) ||
        (params->ref_video_count > 0 && params->ref_videos == nullptr) ||
        (params->ref_audio_count > 0 && params->ref_audios == nullptr)) {
        set_minimax_error(error, "MiniMax-H3 Ref2VA reference arrays are invalid");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    const int64_t resolved_seed = h3_resolve_seed(params->seed);
    sd::Tensor<float> context;
    sd::Tensor<int32_t> token_tags;
    const int64_t cond_context_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    if (!build_text_context(params->prompt, params->ref_images, params->ref_image_count, params->ref_videos, params->ref_video_count, params->ref_audio_count, frames, &context, &token_tags, profile_ptr, error)) return ED_STATUS_GENERATION_FAILED;
    if (profile_ptr != nullptr) profile.cond_context_ms = ggml_time_ms() - cond_context_begin;
    const float cfg_scale = params->sample.cfg_scale;
    const bool use_cfg = cfg_scale != 1.0f;
    sd::Tensor<float> uncond_context;
    sd::Tensor<int32_t> uncond_token_tags;
    const int64_t uncond_context_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    if (use_cfg && !build_text_context(params->negative_prompt, params->ref_images, params->ref_image_count, params->ref_videos, params->ref_video_count, params->ref_audio_count, frames, &uncond_context, &uncond_token_tags, profile_ptr, error)) {
        return ED_STATUS_GENERATION_FAILED;
    }
    if (profile_ptr != nullptr && use_cfg) profile.uncond_context_ms = ggml_time_ms() - uncond_context_begin;
    const int latent_frames = frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2;
    const int audio_length = std::max(1, static_cast<int>(std::lround(static_cast<double>(frames) * 40.0 / 24.0)));
    const int latent_width = params->width / 16;
    const int latent_height = params->height / 16;
    std::vector<sd::Tensor<float>> keyframe_latents;
    std::vector<int32_t> keyframe_indices;
    auto add_keyframe = [&](const ed_image_t* image, int32_t frame_index, const char* name) -> bool {
        if (image == nullptr || image->data == nullptr) {
            return true;
        }
        sd::Tensor<float> image_tensor = h3_image_to_tensor(*image, params->width, params->height);
        if (image_tensor.empty()) {
            return set_minimax_error(error, "MiniMax-H3 keyframe image is invalid");
        }
        sd::Tensor<float> video_image = image_tensor.reshape({params->width, params->height, 1, 3, 1});
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        sd::Tensor<float> vae_latent = vae_->encode(runtime_->n_threads(), video_image, h3_vae_tiling(*runtime_));
        if (profile_ptr != nullptr) profile.keyframe_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (vae_latent.empty()) {
            return set_minimax_error(error, "MiniMax-H3 keyframe VAE encode failed");
        }
        sd::Tensor<float> latent = vae_->vae_to_diffusion_latents(vae_latent);
        auto condition_rng = std::make_shared<PhiloxRNG>(static_cast<uint64_t>(resolved_seed));
        latent = latent * MiniMaxH3::VISUAL_COND_TIMESTEP +
                 sd::randn_like<float>(latent, condition_rng) * (1.0f - MiniMaxH3::VISUAL_COND_TIMESTEP);
        h3_trace_tensor((std::string(name) + "_keyframe_latent").c_str(), latent);
        keyframe_latents.push_back(std::move(latent));
        keyframe_indices.push_back(frame_index);
        return true;
    };
    if (!add_keyframe(params->init_image, 0, "init") ||
        !add_keyframe(params->end_image, frames - 1, "end")) {
        return ED_STATUS_GENERATION_FAILED;
    }
    std::vector<sd::Tensor<float>> reference_latents;
    std::vector<sd::Tensor<float>> reference_audio_latents;
    std::vector<MiniMaxH3ReferenceBlock> reference_blocks;
    auto encode_reference_audio = [&](const ed_audio_t& source, int32_t* index) -> bool {
        if (audio_vae_ == nullptr || source.data == nullptr || source.sample_count == 0 || source.channels == 0 || source.sample_rate == 0) return false;
        const int64_t prepare_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        const uint64_t samples = std::max<uint64_t>(1, (source.sample_count * 32000ULL + source.sample_rate / 2) / source.sample_rate);
        sd::Tensor<float> waveform({static_cast<int64_t>(((samples + 799) / 800) * 800), 2, 1, 1});
        for (uint64_t sample = 0; sample < samples; ++sample) {
            const double position = static_cast<double>(sample) * source.sample_rate / 32000.0;
            const uint64_t first = std::min<uint64_t>(static_cast<uint64_t>(position), source.sample_count - 1);
            const uint64_t second = std::min<uint64_t>(first + 1, source.sample_count - 1);
            const float fraction = static_cast<float>(position - first);
            for (uint32_t channel = 0; channel < 2; ++channel) {
                const uint32_t source_channel = source.channels == 1 ? 0 : std::min<uint32_t>(channel, source.channels - 1);
                const float a = source.data[first * source.channels + source_channel];
                const float b = source.data[second * source.channels + source_channel];
                waveform.index(sample, channel, 0, 0) = std::clamp(a + (b - a) * fraction, -1.f, 1.f);
            }
        }
        if (profile_ptr != nullptr) profile.reference_audio_prepare_ms += ggml_time_ms() - prepare_begin;
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        auto encoded = audio_vae_->encode(runtime_->n_threads(), waveform);
        if (profile_ptr != nullptr) profile.reference_audio_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (encoded.empty()) return false;
        *index = static_cast<int32_t>(reference_audio_latents.size());
        reference_audio_latents.push_back(std::move(encoded));
        return true;
    };
    for (int reference_index = 0; reference_index < params->ref_image_count; ++reference_index) {
        const ed_image_t& source_image = params->ref_images[reference_index];
        const double source_area = static_cast<double>(source_image.width) * source_image.height;
        const double target_area = static_cast<double>(params->width) * params->height;
        const double scale = std::min(1.0, std::sqrt(target_area / source_area));
        const int reference_width = std::max(32, static_cast<int>(std::round(source_image.width * scale / 32.0)) * 32);
        const int reference_height = std::max(32, static_cast<int>(std::round(source_image.height * scale / 32.0)) * 32);
        sd::Tensor<float> image_tensor = h3_image_to_tensor(source_image, reference_width, reference_height);
        if (image_tensor.empty()) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA image reference is invalid");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        sd::Tensor<float> video_image = image_tensor.reshape({reference_width, reference_height, 1, 3, 1});
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        sd::Tensor<float> vae_latent = vae_->encode(runtime_->n_threads(), video_image, h3_vae_tiling(*runtime_));
        if (profile_ptr != nullptr) profile.reference_video_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (vae_latent.empty()) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA image VAE encode failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        sd::Tensor<float> latent = vae_->vae_to_diffusion_latents(vae_latent);
        auto condition_rng = std::make_shared<PhiloxRNG>(static_cast<uint64_t>(resolved_seed));
        latent = latent * MiniMaxH3::VISUAL_COND_TIMESTEP +
                 sd::randn_like<float>(latent, condition_rng) * (1.0f - MiniMaxH3::VISUAL_COND_TIMESTEP);
        h3_trace_tensor(("ref_image_" + std::to_string(reference_index) + "_latent").c_str(), latent);
        const int32_t encoded_image_index = static_cast<int32_t>(reference_latents.size());
        reference_latents.push_back(std::move(latent));
        reference_blocks.push_back({MiniMaxH3ReferenceKind::IMAGE, encoded_image_index, -1});
    }
    for (int video_index = 0; video_index < params->ref_video_count; ++video_index) {
        const ed_ref_video_t& reference = params->ref_videos[video_index];
        if (reference.frames == nullptr || reference.frame_count <= 0) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video reference is invalid");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        const int source_fps = reference.fps > 0 ? reference.fps : 24;
        int normalized_frames = static_cast<int>(std::lround(static_cast<double>(reference.frame_count) * 24.0 / source_fps));
        normalized_frames = std::min(normalized_frames, frames);
        if (normalized_frames < 5) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video reference needs at least 5 frames at 24 fps");
            return ED_STATUS_INVALID_ARGUMENT;
        }
        while (normalized_frames % 17 != 5) --normalized_frames;
        int reference_width = 0;
        int reference_height = 0;
        h3_reference_video_dimensions(reference.frames[0], &reference_width, &reference_height);
        sd::Tensor<float> video_reference({reference_width, reference_height, normalized_frames, 3, 1});
        for (int frame = 0; frame < normalized_frames; ++frame) {
            const int source_index = std::min(reference.frame_count - 1, static_cast<int>(std::floor(frame * source_fps / 24.0)));
            auto source = h3_image_to_tensor(reference.frames[source_index], reference_width, reference_height);
            if (source.empty()) { set_minimax_error(error, "MiniMax-H3 Ref2VA video frame is invalid"); return ED_STATUS_INVALID_ARGUMENT; }
            for (int channel = 0; channel < 3; ++channel) {
                for (int y = 0; y < reference_height; ++y) {
                    for (int x = 0; x < reference_width; ++x) {
                        video_reference.index(x, y, frame, channel, 0) = source.index(x, y, channel, 0);
                    }
                }
            }
        }
        const int64_t vae_encode_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        auto vae_latent = vae_->encode(runtime_->n_threads(), video_reference, h3_vae_tiling(*runtime_));
        if (profile_ptr != nullptr) profile.reference_video_vae_encode_ms += ggml_time_ms() - vae_encode_begin;
        if (vae_latent.empty()) { set_minimax_error(error, "MiniMax-H3 Ref2VA video VAE encode failed"); return ED_STATUS_GENERATION_FAILED; }
        int32_t audio_index = -1;
        if (reference.audio.data != nullptr && reference.audio.sample_count > 0 && !encode_reference_audio(reference.audio, &audio_index)) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA video audio encode failed"); return ED_STATUS_GENERATION_FAILED;
        }
        auto latent = vae_->vae_to_diffusion_latents(vae_latent);
        auto condition_rng = std::make_shared<PhiloxRNG>(static_cast<uint64_t>(resolved_seed));
        latent = latent * MiniMaxH3::VISUAL_COND_TIMESTEP + sd::randn_like<float>(latent, condition_rng) * (1.0f - MiniMaxH3::VISUAL_COND_TIMESTEP);
        const int32_t encoded_video_index = static_cast<int32_t>(reference_latents.size());
        reference_latents.push_back(std::move(latent));
        reference_blocks.push_back({audio_index >= 0 ? MiniMaxH3ReferenceKind::VIDEO_AUDIO : MiniMaxH3ReferenceKind::VIDEO, encoded_video_index, audio_index});
    }
    for (int audio_index = 0; audio_index < params->ref_audio_count; ++audio_index) {
        int32_t encoded_index = -1;
        if (!encode_reference_audio(params->ref_audios[audio_index], &encoded_index)) {
            set_minimax_error(error, "MiniMax-H3 Ref2VA reference audio is invalid or audio VAE encoding failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        reference_blocks.push_back({MiniMaxH3ReferenceKind::AUDIO, -1, encoded_index});
    }
    sd::Tensor<float> video = sd::zeros<float>({latent_width, latent_height, latent_frames, 24, 1});
    sd::Tensor<float> audio = sd::zeros<float>({audio_length, 2, 32, 1});
    const int64_t noise_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
    sd::Tensor<float> packed = h3_pack_audio_and_video_latents(video, audio);
    auto rng = std::make_shared<PhiloxRNG>(static_cast<uint64_t>(resolved_seed));
    packed = sd::randn_like<float>(packed, rng);
    if (profile_ptr != nullptr) profile.noise_init_ms = ggml_time_ms() - noise_begin;
    h3_trace_tensor("initial_packed_noise", packed);
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    const float video_sigma_shift = params->sample.flow_shift > 0.0f ? params->sample.flow_shift : 12.0f;
    // MiniMax-H3's official scheduler treats `steps` as the number of sigma grid
    // points, including the terminal clean point at sigma=0. The terminal point
    // has no model evaluation, so `steps=20` runs 19 DiT forwards.
    for (int step = 0; step + 1 < steps; ++step) {
        if (profile_ptr != nullptr) ++profile.diffusion_steps;
        const float sigma = h3_discrete_flow_sigma(step, steps, video_sigma_shift);
        const float sigma_next = h3_discrete_flow_sigma(step + 1, steps, video_sigma_shift);
        sd::Tensor<float> timestep({1}, {sigma * 1000.0f});
        DiffusionParams diffusion_params{};
        diffusion_params.x = &packed;
        diffusion_params.timesteps = &timestep;
        diffusion_params.context = &context;
        diffusion_params.minimax_text_token_tags = &token_tags;
        diffusion_params.ref_latents = !reference_latents.empty() ? &reference_latents
                                         : (keyframe_latents.empty() ? nullptr : &keyframe_latents);
        diffusion_params.minimax_reference_blocks = reference_blocks.empty() ? nullptr : &reference_blocks;
        diffusion_params.minimax_reference_audio_latents = reference_audio_latents.empty() ? nullptr : &reference_audio_latents;
        sd::Tensor<int32_t> keyframe_index_tensor;
        if (!keyframe_indices.empty()) {
            keyframe_index_tensor = sd::Tensor<int32_t>({static_cast<int64_t>(keyframe_indices.size())}, keyframe_indices);
            diffusion_params.minimax_keyframe_indices = &keyframe_index_tensor;
        }
        diffusion_params.minimax_audio_length = audio_length;
        diffusion_params.minimax_video_sigma_shift = video_sigma_shift;
        diffusion_params.minimax_audio_sigma_shift = 3.0f;
        const int64_t cond_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
        sd::Tensor<float> velocity = diffusion_->compute(runtime_->n_threads(), diffusion_params);
        if (profile_ptr != nullptr) {
            profile.diffusion_cond_ms += ggml_time_ms() - cond_begin;
            ++profile.diffusion_calls;
        }
        if (velocity.empty()) {
            set_minimax_error(error, "MiniMax-H3 diffusion compute failed");
            return ED_STATUS_GENERATION_FAILED;
        }
        if (use_cfg) {
            diffusion_params.context = &uncond_context;
            diffusion_params.minimax_text_token_tags = &uncond_token_tags;
            const int64_t uncond_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
            sd::Tensor<float> uncond_velocity = diffusion_->compute(runtime_->n_threads(), diffusion_params);
            if (profile_ptr != nullptr) {
                profile.diffusion_uncond_ms += ggml_time_ms() - uncond_begin;
                ++profile.diffusion_calls;
            }
            if (uncond_velocity.empty()) {
                set_minimax_error(error, "MiniMax-H3 unconditional diffusion compute failed");
                return ED_STATUS_GENERATION_FAILED;
            }
            const int64_t cfg_combine_begin = profile_ptr != nullptr ? ggml_time_ms() : 0;
            velocity = uncond_velocity + (velocity - uncond_velocity) * cfg_scale;
            if (profile_ptr != nullptr) profile.cfg_combine_ms += ggml_time_ms() - cfg_combine_begin;
        }
        if (h3_trace_enabled()) {
            LOG_INFO("minimax-h3 trace step=%d sigma=%.8g sigma_next=%.8g", step, sigma, sigma_next);
            h3_trace_tensor(("step_" + std::to_string(step) + "_velocity").c_str(), velocity);
        }
        packed += velocity * (sigma_next - sigma);
        if (h3_trace_enabled()) {
            h3_trace_tensor(("step_" + std::to_string(step) + "_packed").c_str(), packed);
        }
    }
    auto av = diffusion_->split_av_latents(packed, audio_length);
    ed_status_t status = decode_video_latent(av.first, frames, out, profile_ptr, error);
    if (status != ED_STATUS_OK) {
        return status;
    }
    if (audio_vae_ != nullptr && !decode_audio_latent(av.second, out, profile_ptr, error)) {
        ed_free_video(out);
        return ED_STATUS_GENERATION_FAILED;
    }
    if (profile_ptr != nullptr) {
        profile.total_ms = ggml_time_ms() - generation_begin;
        profile.log();
    }
    return ED_STATUS_OK;
}

ed_scheduler_t MiniMaxH3Pipeline::default_scheduler(ed_sampler_t) const {
    return ED_SCHEDULER_DISCRETE;
}

}  // namespace edgedit
