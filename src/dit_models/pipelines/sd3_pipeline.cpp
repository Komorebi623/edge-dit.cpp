#include "dit_models/pipelines/sd3_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>

#include "core/optimization/cache/cache_runtime.hpp"
#include "dit_models/components/autoencoders/auto_encoder_kl.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "utils/util.h"

namespace lightdit {
namespace {

float sd3_time_snr_shift(float shift, float t) {
    return shift * t / (1.0f + (shift - 1.0f) * t);
}

float sd3_t_to_sigma(float t, float shift) {
    t = t + 1.0f;
    return sd3_time_snr_shift(shift, t / 1000.0f);
}

ld_status_t tensor_to_image(const sd::Tensor<float>& tensor, ld_image_t* image) {
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
    uint8_t* data = static_cast<uint8_t*>(std::malloc(nbytes));
    if (data == nullptr) {
        return LD_STATUS_OUT_OF_MEMORY;
    }

    const size_t pixels = width * height;
    const float* src = tensor.data();
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t c = 0; c < channels; ++c) {
            float value = src[i + pixels * c];
            if (value <= 0.0f) {
                data[i * channels + c] = 0;
            } else if (value >= 1.0f) {
                data[i * channels + c] = 255;
            } else {
                data[i * channels + c] = static_cast<uint8_t>(value * 255.0f + 0.5f);
            }
        }
    }
    image->width = static_cast<uint32_t>(width);
    image->height = static_cast<uint32_t>(height);
    image->channels = static_cast<uint32_t>(channels);
    image->data = data;
    return LD_STATUS_OK;
}

}  // namespace

SD3Pipeline::SD3Pipeline(SDVersion version)
    : version_(version) {
}

bool SD3Pipeline::prepare(const ld_context_params_t&,
                          ModelRuntime& runtime,
                          const ModelLoader& loader,
                          PipelineTensorRegistry& registry,
                          std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    registry.clear();

    if (version_ != VERSION_SD3) {
        if (error != nullptr) {
            *error = "SD3Pipeline got non-SD3 model version";
        }
        return false;
    }
    if (!build_components(loader, error)) {
        return false;
    }
    configure_runtime_flags();
    if (!register_tensors(registry, error)) {
        return false;
    }
    build_ignore_tensors(registry);
    return true;
}

bool SD3Pipeline::build_components(const ModelLoader& loader, std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        if (error != nullptr) {
            *error = "SD3Pipeline requires initialized ModelRuntime backends";
        }
        return false;
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool offload = runtime_->offload_params_to_cpu();
    conditioner_ = std::make_shared<SD3CLIPEmbedder>(runtime_->clip_backend(), offload, storage);
    diffusion_ = std::make_shared<MMDiTModel>(runtime_->backend(), offload, storage);
    vae_ = std::make_shared<AutoEncoderKL>(runtime_->vae_backend(),
                                           offload,
                                           storage,
                                           "first_stage_model",
                                           true,
                                           false,
                                           version_);
    return conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

void SD3Pipeline::configure_runtime_flags() {
    const size_t max_graph_vram = runtime_->max_graph_vram_bytes();
    conditioner_->set_max_graph_vram_bytes(max_graph_vram);
    conditioner_->set_flash_attention_enabled(runtime_->flash_attention());

    diffusion_->set_max_graph_vram_bytes(max_graph_vram);
    diffusion_->set_flash_attention_enabled(runtime_->flash_attention() || runtime_->diffusion_flash_attention());
    diffusion_->set_circular_axes(runtime_->circular_x(), runtime_->circular_y());

    vae_->set_max_graph_vram_bytes(max_graph_vram);
    vae_->set_flash_attention_enabled(runtime_->flash_attention());
}

bool SD3Pipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (!conditioner_ || !diffusion_ || !vae_) {
        if (error != nullptr) {
            *error = "SD3Pipeline components are not initialized";
        }
        return false;
    }

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());

    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors());

    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), "first_stage_model");

    LOG_INFO("sd3 pipeline registered %zu tensors", registry.tensors().size());
    return true;
}

void SD3Pipeline::build_ignore_tensors(PipelineTensorRegistry& registry) const {
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");
    registry.ignore_prefix("first_stage_model.encoder");
    registry.ignore_prefix("first_stage_model.conv1");
    registry.ignore_prefix("first_stage_model.quant");
}

void SD3Pipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr;
}

ld_status_t SD3Pipeline::generate_image(const ld_image_generation_params_t* params,
                                        ld_image_batch_t* out,
                                        std::string* error) {
    if (!ready_ || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "SD3Pipeline is not initialized";
        }
        return LD_STATUS_MODEL_LOAD_FAILED;
    }
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "image output is null";
        }
        return LD_STATUS_INVALID_ARGUMENT;
    }

    if (!validate_image_params(params, error)) {
        return LD_STATUS_INVALID_ARGUMENT;
    }
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "current SD3 pipeline needs transformer, text encoders, and VAE weights";
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
        if (!generate_one_image(params, i, &images[i], error)) {
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

ld_status_t SD3Pipeline::generate_video(const ld_video_generation_params_t*,
                                        ld_video_t* out,
                                        std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (error != nullptr) {
        *error = "video generation is not supported by SD3Pipeline";
    }
    return LD_STATUS_UNSUPPORTED;
}

ld_scheduler_t SD3Pipeline::default_scheduler(ld_sampler_t method) const {
    if (method == LD_SAMPLER_LCM || method == LD_SAMPLER_TCD) {
        return LD_SCHEDULER_LCM;
    }
    if (method == LD_SAMPLER_DDIM_TRAILING) {
        return LD_SCHEDULER_SIMPLE;
    }
    return LD_SCHEDULER_DISCRETE;
}

bool SD3Pipeline::can_generate_image() const {
    return ready_ && runtime_ != nullptr && conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

bool SD3Pipeline::validate_image_params(const ld_image_generation_params_t* params,
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
    if (params->batch_count <= 0) {
        if (error != nullptr) {
            *error = "image batch_count must be positive";
        }
        return false;
    }
    return true;
}

std::vector<float> SD3Pipeline::build_sigmas(int steps, float shift) const {
    std::vector<float> sigmas;
    if (steps <= 0) {
        return sigmas;
    }
    if (steps == 1) {
        sigmas.push_back(sd3_t_to_sigma(999.0f, shift));
        sigmas.push_back(0.0f);
        return sigmas;
    }

    const float step = 999.0f / static_cast<float>(steps - 1);
    sigmas.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i < steps; ++i) {
        const float t = 999.0f - step * static_cast<float>(i);
        sigmas.push_back(sd3_t_to_sigma(t, shift));
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

bool SD3Pipeline::generate_one_image(const ld_image_generation_params_t* params,
                                     int batch_index,
                                     ld_image_t* image,
                                     std::string* error) {
    if (!can_generate_image()) {
        if (error != nullptr) {
            *error = "SD3 pipeline is not ready for image generation";
        }
        return false;
    }

    const int vae_scale_factor = vae_->get_scale_factor();
    if (params->width % vae_scale_factor != 0 || params->height % vae_scale_factor != 0) {
        if (error != nullptr) {
            *error = sd_format("SD3 image size must be divisible by VAE scale factor %d", vae_scale_factor);
        }
        return false;
    }

    ConditionerParams cond_params;
    cond_params.text = params->prompt != nullptr ? params->prompt : "";
    cond_params.clip_skip = -1;
    cond_params.width = params->width;
    cond_params.height = params->height;
    cond_params.adm_in_channels = static_cast<int>(diffusion_->get_adm_in_channels());
    SDCondition cond = conditioner_->get_learned_condition(runtime_->n_threads(), cond_params);
    if (cond.empty()) {
        if (error != nullptr) {
            *error = "SD3 prompt encoding returned empty condition";
        }
        return false;
    }

    SDCondition uncond;
    const float cfg_scale = params->sample.cfg_scale > 0.0f ? params->sample.cfg_scale : 1.0f;
    if (cfg_scale != 1.0f) {
        cond_params.text = params->negative_prompt != nullptr ? params->negative_prompt : "";
        uncond = conditioner_->get_learned_condition(runtime_->n_threads(), cond_params);
        if (uncond.empty()) {
            if (error != nullptr) {
                *error = "SD3 negative prompt encoding returned empty condition";
            }
            return false;
        }
    }

    const int latent_w = params->width / vae_scale_factor;
    const int latent_h = params->height / vae_scale_factor;
    const int steps = params->sample.steps > 0 ? params->sample.steps : 20;
    const ld_sampler_t sampler = params->sample.sampler == LD_SAMPLER_AUTO
                                     ? default_sample_method()
                                     : params->sample.sampler;
    if (sampler != LD_SAMPLER_EULER) {
        if (error != nullptr) {
            *error = "SD3Pipeline currently implements the old default Euler flow path only";
        }
        return false;
    }
    float flow_shift = params->sample.flow_shift;
    if (!(flow_shift > 0.0f) || !std::isfinite(flow_shift)) {
        flow_shift = 3.0f;
    }
    int64_t seed = params->seed;
    if (seed < 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seed = std::rand();
    }
    std::shared_ptr<RNG> rng = runtime_->rng_ptr();
    if (!rng) {
        if (error != nullptr) {
            *error = "SD3Pipeline has no RNG from ModelRuntime";
        }
        return false;
    }
    rng->manual_seed(static_cast<uint64_t>(seed + batch_index));

    sd::Tensor<float> init_latent = sd::zeros<float>({latent_w, latent_h, 16, 1});
    sd::Tensor<float> noise = sd::Tensor<float>::randn(init_latent.shape(), rng);
    std::vector<float> sigmas = build_sigmas(steps, flow_shift);
    if (sigmas.size() < 2) {
        if (error != nullptr) {
            *error = "failed to create SD3 sigma schedule";
        }
        return false;
    }

    LOG_INFO("sd3 txt2img: %dx%d latent=%dx%d steps=%d shift=%.2f cfg=%.2f seed=%" PRId64,
             params->width,
             params->height,
             latent_w,
             latent_h,
             steps,
             flow_shift,
             cfg_scale,
             seed + batch_index);

    sd::Tensor<float> x = init_latent * (1.0f - sigmas[0]) + noise * sigmas[0];
    cache::CacheRuntime cache_runtime;
    const bool cache_enabled = cache_runtime.init(params->sample, version_, sigmas);
    for (int step = 0; step < steps; ++step) {
        const float sigma = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
        sd::Tensor<float> timesteps({1}, std::vector<float>{sigma * 1000.0f});

        cache::CacheStepInfo cache_step;
        cache_step.step_index = step;
        cache_step.num_steps = steps;
        cache_step.sigma = sigma;
        cache_step.sigma_next = sigma_next;
        if (cache_enabled) {
            cache_runtime.begin_step(cache_step);
        }

        DiffusionParams diffusion_params;
        diffusion_params.x = &x;
        diffusion_params.timesteps = &timesteps;
        diffusion_params.context = &cond.c_crossattn;
        diffusion_params.y = &cond.c_vector;
        sd::Tensor<float> cond_out;
        const void* cond_key = static_cast<const void*>(&cond);
        const bool cond_cache_hit = cache_enabled &&
                                    cache_runtime.before_forward(cache::CacheBranch::Cond,
                                                                 cond_key,
                                                                 x,
                                                                 &cond_out);
        if (!cond_cache_hit) {
            cond_out = diffusion_->compute(runtime_->n_threads(), diffusion_params);
            if (!cond_out.empty() && cache_enabled) {
                cache_runtime.after_forward(cache::CacheBranch::Cond,
                                            cond_key,
                                            x,
                                            cond_out);
            }
        }
        if (cond_out.empty()) {
            if (error != nullptr) {
                *error = sd_format("SD3 diffusion compute failed at step %d", step + 1);
            }
            diffusion_->free_compute_buffer();
            return false;
        }

        sd::Tensor<float> model_out = cond_out;
        if (!uncond.empty()) {
            diffusion_params.context = &uncond.c_crossattn;
            diffusion_params.y = &uncond.c_vector;
            sd::Tensor<float> uncond_out;
            const void* uncond_key = static_cast<const void*>(&uncond);
            const bool uncond_cache_hit = cache_enabled &&
                                          cache_runtime.before_forward(cache::CacheBranch::Uncond,
                                                                       uncond_key,
                                                                       x,
                                                                       &uncond_out);
            if (!uncond_cache_hit) {
                uncond_out = diffusion_->compute(runtime_->n_threads(), diffusion_params);
                if (!uncond_out.empty() && cache_enabled) {
                    cache_runtime.after_forward(cache::CacheBranch::Uncond,
                                                uncond_key,
                                                x,
                                                uncond_out);
                }
            }
            if (uncond_out.empty()) {
                if (error != nullptr) {
                    *error = sd_format("SD3 unconditional diffusion compute failed at step %d", step + 1);
                }
                diffusion_->free_compute_buffer();
                return false;
            }
            model_out = uncond_out + cfg_scale * (cond_out - uncond_out);
        }

        sd::Tensor<float> denoised = model_out * (-sigma) + x;
        const sd::Tensor<float> d = (x - denoised) / sigma;
        x += d * (sigma_next - sigma);
        LOG_INFO("sd3 step %d/%d sigma=%.6f next=%.6f", step + 1, steps, sigma, sigma_next);
        if (cache_enabled) {
            cache_runtime.end_step(cache_step);
        }
    }
    if (cache_enabled) {
        cache_runtime.log_summary(static_cast<size_t>(steps));
    }
    diffusion_->free_compute_buffer();

    sd::Tensor<float> vae_latents = vae_->diffusion_to_vae_latents(x);
    ld_tiling_params_t tiling_params{};
    sd::Tensor<float> decoded = vae_->decode(runtime_->n_threads(),
                                             vae_latents,
                                             tiling_params,
                                             false,
                                             runtime_->circular_x(),
                                             runtime_->circular_y());
    if (decoded.empty()) {
        if (error != nullptr) {
            *error = "SD3 VAE decode failed";
        }
        return false;
    }

    const ld_status_t status = tensor_to_image(decoded, image);
    if (status != LD_STATUS_OK) {
        if (error != nullptr) {
            *error = status == LD_STATUS_OUT_OF_MEMORY ? "failed to allocate decoded image" : "decoded SD3 tensor has invalid shape";
        }
        return false;
    }
    return true;
}

}  // namespace lightdit
