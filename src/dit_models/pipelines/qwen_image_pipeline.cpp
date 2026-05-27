#include "dit_models/pipelines/qwen_image_pipeline.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dit_models/components/autoencoders/vae.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "dit_models/models/qwen_image.hpp"
#include "dit_models/models/wan.hpp"
#include "ggml.h"
#include "utils/util.h"

static constexpr float LD_QWEN_FLOW_SHIFT_DEFAULT = 3.0f;
static constexpr int LD_QWEN_IMAGE_ALIGN = 32;

static ld_status_t ld_tensor_to_image(const sd::Tensor<float>& tensor, ld_image_t* image) {
    if (image == nullptr || tensor.empty()) {
        return LD_STATUS_INVALID_ARGUMENT;
    }
    const auto& shape = tensor.shape();
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
        return LD_STATUS_INVALID_ARGUMENT;
    }

    const size_t width = static_cast<size_t>(shape[0]);
    const size_t height = static_cast<size_t>(shape[1]);
    const size_t channels = static_cast<size_t>(shape[2]);
    const size_t nbytes = width * height * channels;
    auto to_u8 = [](float value) -> uint8_t {
        if (value <= 0.0f) {
            return 0;
        }
        if (value >= 1.0f) {
            return 255;
        }
        return static_cast<uint8_t>(value * 255.0f + 0.5f);
    };
    uint8_t* data = static_cast<uint8_t*>(std::malloc(nbytes));
    if (data == nullptr) {
        return LD_STATUS_OUT_OF_MEMORY;
    }

    const size_t pixels = width * height;
    const float* src = tensor.data();
    if (channels == 3) {
        const float* c0 = src;
        const float* c1 = src + pixels;
        const float* c2 = src + pixels * 2;
        for (size_t i = 0; i < pixels; ++i) {
            data[i * 3 + 0] = to_u8(c0[i]);
            data[i * 3 + 1] = to_u8(c1[i]);
            data[i * 3 + 2] = to_u8(c2[i]);
        }
    } else {
        for (size_t i = 0; i < pixels; ++i) {
            for (size_t c = 0; c < channels; ++c) {
                data[i * channels + c] = to_u8(src[i + pixels * c]);
            }
        }
    }
    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data = data;
    return LD_STATUS_OK;
}

static float ld_qwen_time_snr_shift(float shift, float t) {
    return shift * t / (1.0f + (shift - 1.0f) * t);
}

static float ld_qwen_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return ld_qwen_time_snr_shift(shift, t / 1000.0f);
}

static std::vector<float> ld_qwen_discrete_sigmas(int steps, float shift) {
    std::vector<float> result;
    if (steps <= 0) {
        return result;
    }
    if (steps == 1) {
        result.push_back(ld_qwen_t_to_sigma(999.0f, shift));
        result.push_back(0.0f);
        return result;
    }

    const float step = 999.0f / static_cast<float>(steps - 1);
    result.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        result.push_back(ld_qwen_t_to_sigma(t, shift));
    }
    result.push_back(0.0f);
    return result;
}

namespace lightdit {

QwenImagePipeline::QwenImagePipeline(SDVersion version)
    : version_(version) {
}

QwenImagePipeline::~QwenImagePipeline() {
    reset();
}

void QwenImagePipeline::reset() {
    conditioner_.reset();
    vae_.reset();
    diffusion_.reset();
    runtime_weights_loaded_ = false;
}

bool QwenImagePipeline::has_prefix(const ModelLoader& loader, const std::string& prefix) const {
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (starts_with(item.second.name, prefix)) {
            return true;
        }
    }
    return false;
}

bool QwenImagePipeline::prepare(const ld_context_params_t& params,
                                ModelRuntime& runtime,
                                const ModelLoader& loader,
                                PipelineTensorRegistry& registry,
                                std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    reset();

    if (!ld_version_is_qwen_image(version_)) {
        if (error != nullptr) {
            *error = "QwenImagePipeline got non-Qwen-Image model version";
        }
        return false;
    }
    if (!build_components(params, loader, error)) {
        return false;
    }
    if (!register_tensors(registry, error)) {
        return false;
    }
    return true;
}

bool QwenImagePipeline::build_components(const ld_context_params_t& params,
                                         const ModelLoader& loader,
                                         std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        if (error != nullptr) {
            *error = "QwenImagePipeline requires initialized ModelRuntime backends";
        }
        return false;
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool offload = runtime_->offload_params_to_cpu();
    const bool enable_vision = params.llm_vision_path != nullptr && params.llm_vision_path[0] != '\0';

    conditioner_ = std::make_shared<LLMEmbedder>(runtime_->clip_backend(),
                                                 offload,
                                                 storage,
                                                 version_,
                                                 "",
                                                 enable_vision);

    diffusion_.reset(new Qwen::QwenImageRunner(runtime_->backend(),
                                               offload,
                                               storage,
                                               "model.diffusion_model",
                                               version_,
                                               false));

    vae_ = std::make_shared<WAN::WanVAERunner>(runtime_->vae_backend(),
                                               offload,
                                               storage,
                                               "first_stage_model",
                                               true,
                                               version_);

    return true;
}

bool QwenImagePipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (conditioner_ == nullptr || diffusion_ == nullptr || vae_ == nullptr) {
        if (error != nullptr) {
            *error = "QwenImagePipeline components are not initialized";
        }
        return false;
    }

    registry.clear();

    if (!diffusion_->alloc_params_buffer()) {
        if (error != nullptr) {
            *error = "failed to allocate Qwen-Image transformer parameter buffer";
        }
        return false;
    }
    diffusion_->get_param_tensors(registry.tensors(), "model.diffusion_model");

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());

    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), "first_stage_model");

    registry.ignore_prefix("vae.");
    registry.ignore_prefix("cond_stage_model.");
    registry.ignore_prefix("first_stage_model.encoder");
    registry.ignore_prefix("first_stage_model.conv1.");
    registry.ignore_prefix("text_encoders.llm.lm_head.");
    registry.ignore_prefix("text_encoders.llm.output.weight");
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");

    registry.ignore_prefix("text_encoders.llm.visual.");

    runtime_weights_loaded_ = true;
    return true;
}

void QwenImagePipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             version_ == VERSION_QWEN_IMAGE &&
             runtime_weights_loaded_ &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr;
}

bool QwenImagePipeline::can_generate_image() const {
    return ready_ && runtime_weights_loaded_ && conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

bool QwenImagePipeline::validate_image_params(const ld_image_generation_params_t* params,
                                              std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "image generation params are null";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0) {
        if (error != nullptr) {
            *error = "image width and height must be positive";
        }
        return false;
    }
    if (params->width % LD_QWEN_IMAGE_ALIGN != 0 || params->height % LD_QWEN_IMAGE_ALIGN != 0) {
        if (error != nullptr) {
            *error = sd_format("Qwen-Image width and height must be divisible by %d", LD_QWEN_IMAGE_ALIGN);
        }
        return false;
    }
    if (params->batch_count <= 0) {
        if (error != nullptr) {
            *error = "image batch_count must be positive";
        }
        return false;
    }
    return true;
}

ld_status_t QwenImagePipeline::generate_image(const ld_image_generation_params_t* params,
                                              ld_image_batch_t* out,
                                              std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "QwenImagePipeline is not initialized";
        }
        return LD_STATUS_MODEL_LOAD_FAILED;
    }
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "image output is null";
        }
        return LD_STATUS_INVALID_ARGUMENT;
    }
    out->images = nullptr;
    out->count = 0;

    if (!validate_image_params(params, error)) {
        return LD_STATUS_INVALID_ARGUMENT;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "current Qwen-Image pipeline needs transformer, LLM text encoder, and VAE weights";
        }
        return LD_STATUS_UNSUPPORTED;
    }

    const int count = params->batch_count > 0 ? params->batch_count : 1;
    ld_image_t* images = static_cast<ld_image_t*>(std::calloc(static_cast<size_t>(count), sizeof(ld_image_t)));
    if (images == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate image batch";
        }
        return LD_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < count; ++i) {
        if (!generate_one_image(params, i, runtime_->n_threads(), &images[i], error)) {
            for (int j = 0; j <= i; ++j) {
                std::free(images[j].data);
            }
            std::free(images);
            return LD_STATUS_GENERATION_FAILED;
        }
    }

    out->images = images;
    out->count = count;
    return LD_STATUS_OK;
}

ld_status_t QwenImagePipeline::generate_video(const ld_video_generation_params_t*,
                                              ld_video_t* out,
                                              std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (error != nullptr) {
        *error = "video generation is not supported by QwenImagePipeline";
    }
    return LD_STATUS_UNSUPPORTED;
}

bool QwenImagePipeline::supports_image_generation() const {
    return ready_;
}

bool QwenImagePipeline::supports_video_generation() const {
    return false;
}

ld_sampler_t QwenImagePipeline::default_sample_method() const {
    return LD_SAMPLER_EULER;
}

ld_scheduler_t QwenImagePipeline::default_scheduler(ld_sampler_t method) const {
    if (method == LD_SAMPLER_LCM || method == LD_SAMPLER_TCD) {
        return LD_SCHEDULER_LCM;
    }
    if (method == LD_SAMPLER_DDIM_TRAILING) {
        return LD_SCHEDULER_SIMPLE;
    }
    return LD_SCHEDULER_DISCRETE;
}

bool QwenImagePipeline::generate_one_image(const ld_image_generation_params_t* params,
                                           int batch_index,
                                           int n_threads,
                                           ld_image_t* image,
                                           std::string* error) {
    if (params == nullptr || image == nullptr) {
        if (error != nullptr) {
            *error = "invalid Qwen-Image generation arguments";
        }
        return false;
    }

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    SDCondition condition = conditioner_->get_learned_condition(n_threads, cond_params);
    if (condition.empty() || condition.c_crossattn.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image prompt encoding returned empty condition";
        }
        return false;
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    const int latent_w = params->width / vae_scale_factor;
    const int latent_h = params->height / vae_scale_factor;
    const int patch_size = std::max<int>(1, diffusion_->qwen_image_params.patch_size);
    if (latent_w % patch_size != 0 || latent_h % patch_size != 0) {
        if (error != nullptr) {
            *error = sd_format("Qwen-Image latent size %dx%d must be divisible by patch size %d",
                               latent_w,
                               latent_h,
                               patch_size);
        }
        return false;
    }

    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    float flow_shift = params->sample.flow_shift;
    if (!(flow_shift > 0.0f) || !std::isfinite(flow_shift)) {
        flow_shift = LD_QWEN_FLOW_SHIFT_DEFAULT;
    }

    const int64_t seed = params->seed >= 0 ? params->seed : 42;
    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "QwenImagePipeline has no RNG from ModelRuntime";
        }
        return false;
    }
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    std::vector<float> sigmas = ld_qwen_discrete_sigmas(steps, flow_shift);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create Qwen-Image sigma schedule";
        }
        return false;
    }

    LOG_INFO("qwen-image txt2img: %dx%d latent=%dx%d steps=%d shift=%.2f seed=%" PRId64,
             params->width,
             params->height,
             latent_w,
             latent_h,
             steps,
             flow_shift,
             seed + batch_index);

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    for (int step = 0; step < steps; ++step) {
        const float sigma = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];

        sd::Tensor<float> timesteps({1}, std::vector<float>{sigma * 1000.0f});
        sd::Tensor<float> model_out = diffusion_->compute(n_threads,
                                                          x,
                                                          timesteps,
                                                          condition.c_crossattn,
                                                          {},
                                                          false);
        if (model_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("Qwen-Image transformer compute failed at step %d", step + 1);
            }
            diffusion_->free_compute_buffer();
            return false;
        }

        sd::Tensor<float> denoised = model_out * (-sigma) + x;
        if (sigma == 0.0f) {
            x = denoised;
        } else {
            const sd::Tensor<float> d = (x - denoised) / sigma;
            x += d * (sigma_next - sigma);
        }
        LOG_INFO("qwen-image step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
    }
    diffusion_->free_compute_buffer();

    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    ld_tiling_params_t tiling_params{};
    sd::Tensor<float> decoded = vae_->decode(n_threads,
                                             vae_latents,
                                             tiling_params,
                                             false,
                                             false,
                                             false);
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "Qwen-Image VAE decode failed";
        }
        return false;
    }

    const ld_status_t status = ld_tensor_to_image(decoded, image);
    if (status != LD_STATUS_OK) {
        if (error != nullptr) {
            *error = status == LD_STATUS_OUT_OF_MEMORY
                         ? "failed to allocate decoded Qwen-Image image"
                         : "decoded Qwen-Image tensor has invalid shape";
        }
        return false;
    }
    return true;
}

}  // namespace lightdit
