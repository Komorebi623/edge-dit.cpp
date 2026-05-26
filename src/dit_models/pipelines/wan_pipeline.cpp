#include "dit_models/pipelines/wan_pipeline.hpp"

#include "dit_models/components/autoencoders/tae.hpp"
#include "dit_models/components/text_encoders/conditioner.hpp"
#include "utils/util.h"

namespace lightdit {

WanPipeline::WanPipeline(SDVersion version)
    : version_(version) {
}

bool WanPipeline::has_prefix(const ModelLoader& loader, const std::string& prefix) {
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (starts_with(item.first, prefix)) {
            return true;
        }
    }
    return false;
}

bool WanPipeline::prepare(const ld_context_params_t&,
                          ModelRuntime& runtime,
                          const ModelLoader& loader,
                          PipelineTensorRegistry& registry,
                          std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    version_ = loader.version();
    registry.clear();

    if (!ld_version_is_wan(version_)) {
        if (error != nullptr) {
            *error = "WanPipeline got non-Wan model version";
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

bool WanPipeline::build_components(const ModelLoader& loader, std::string* error) {
    if (runtime_ == nullptr || runtime_->backend() == nullptr ||
        runtime_->clip_backend() == nullptr || runtime_->vae_backend() == nullptr) {
        if (error != nullptr) {
            *error = "WanPipeline requires initialized ModelRuntime backends";
        }
        return false;
    }

    const auto& storage = loader.get_tensor_storage_map();
    const bool offload = runtime_->offload_params_to_cpu();
    const bool vae_decode_only = true;

    conditioner_ = std::make_shared<T5CLIPEmbedder>(runtime_->clip_backend(),
                                                    offload,
                                                    storage,
                                                    true,
                                                    0,
                                                    true);
    diffusion_ = std::make_shared<WanModel>(runtime_->backend(),
                                            offload,
                                            storage,
                                            "model.diffusion_model",
                                            version_);

    if (has_prefix(loader, "model.high_noise_diffusion_model.")) {
        high_noise_diffusion_ = std::make_shared<WanModel>(runtime_->backend(),
                                                           offload,
                                                           storage,
                                                           "model.high_noise_diffusion_model",
                                                           version_);
    }

    if (diffusion_->get_desc() == "Wan2.1-I2V-14B" ||
        diffusion_->get_desc() == "Wan2.1-FLF2V-14B" ||
        diffusion_->get_desc() == "Wan2.1-I2V-1.3B") {
        clip_vision_ = std::make_shared<FrozenCLIPVisionEmbedder>(runtime_->backend(),
                                                                  offload,
                                                                  storage);
    }

    using_tae_for_main_ = loader.use_tae() && !loader.tae_preview_only();
    if (using_tae_for_main_) {
        vae_ = std::make_shared<TinyVideoAutoEncoder>(runtime_->vae_backend(),
                                                      offload,
                                                      storage,
                                                      "decoder",
                                                      vae_decode_only,
                                                      version_);
    } else {
        vae_ = std::make_shared<WAN::WanVAERunner>(runtime_->vae_backend(),
                                                   offload,
                                                   storage,
                                                   "first_stage_model",
                                                   vae_decode_only,
                                                   version_);
    }

    if (loader.use_tae() && loader.tae_preview_only()) {
        preview_vae_ = std::make_shared<TinyVideoAutoEncoder>(runtime_->vae_backend(),
                                                              offload,
                                                              storage,
                                                              "decoder",
                                                              true,
                                                              version_);
    }

    return conditioner_ != nullptr && diffusion_ != nullptr && vae_ != nullptr;
}

void WanPipeline::configure_runtime_flags() {
    const size_t max_graph_vram = runtime_->max_graph_vram_bytes();
    const bool text_flash = runtime_->flash_attention();
    const bool diffusion_flash = runtime_->flash_attention() || runtime_->diffusion_flash_attention();

    conditioner_->set_max_graph_vram_bytes(max_graph_vram);
    conditioner_->set_flash_attention_enabled(text_flash);

    if (clip_vision_) {
        clip_vision_->set_max_graph_vram_bytes(max_graph_vram);
        clip_vision_->set_flash_attention_enabled(text_flash);
    }

    diffusion_->set_max_graph_vram_bytes(max_graph_vram);
    diffusion_->set_flash_attention_enabled(diffusion_flash);
    diffusion_->set_circular_axes(runtime_->circular_x(), runtime_->circular_y());

    if (high_noise_diffusion_) {
        high_noise_diffusion_->set_max_graph_vram_bytes(max_graph_vram);
        high_noise_diffusion_->set_flash_attention_enabled(diffusion_flash);
        high_noise_diffusion_->set_circular_axes(runtime_->circular_x(), runtime_->circular_y());
    }

    vae_->set_max_graph_vram_bytes(max_graph_vram);
    vae_->set_flash_attention_enabled(text_flash);

    if (preview_vae_) {
        preview_vae_->set_max_graph_vram_bytes(max_graph_vram);
        preview_vae_->set_flash_attention_enabled(text_flash);
    }
}

bool WanPipeline::register_tensors(PipelineTensorRegistry& registry, std::string* error) {
    if (!conditioner_ || !diffusion_ || !vae_) {
        if (error != nullptr) {
            *error = "WanPipeline components are not initialized";
        }
        return false;
    }

    conditioner_->alloc_params_buffer();
    conditioner_->get_param_tensors(registry.tensors());

    if (clip_vision_) {
        clip_vision_->alloc_params_buffer();
        clip_vision_->get_param_tensors(registry.tensors());
    }

    diffusion_->alloc_params_buffer();
    diffusion_->get_param_tensors(registry.tensors());

    if (high_noise_diffusion_) {
        high_noise_diffusion_->alloc_params_buffer();
        high_noise_diffusion_->get_param_tensors(registry.tensors());
    }

    vae_->alloc_params_buffer();
    vae_->get_param_tensors(registry.tensors(), using_tae_for_main_ ? "tae" : "first_stage_model");

    if (preview_vae_) {
        preview_vae_->alloc_params_buffer();
        preview_vae_->get_param_tensors(registry.tensors(), "tae");
    }

    LOG_INFO("wan pipeline registered %zu tensors", registry.tensors().size());
    return true;
}

void WanPipeline::build_ignore_tensors(PipelineTensorRegistry& registry) const {
    registry.ignore_prefix("model.diffusion_model.__x0__");
    registry.ignore_prefix("model.diffusion_model.__32x32__");
    registry.ignore_prefix("model.diffusion_model.__index_timestep_zero__");

    if (using_tae_for_main_) {
        registry.ignore_prefix("first_stage_model.");
    } else {
        registry.ignore_prefix("first_stage_model.encoder");
        registry.ignore_prefix("first_stage_model.conv1");
        registry.ignore_prefix("first_stage_model.quant");
        registry.ignore_prefix("tae.encoder");
    }
}

void WanPipeline::mark_ready() {
    ready_ = runtime_ != nullptr &&
             conditioner_ != nullptr &&
             diffusion_ != nullptr &&
             vae_ != nullptr;
}

ld_status_t WanPipeline::generate_image(const ld_image_generation_params_t*,
                                        ld_image_batch_t* out,
                                        std::string* error) {
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }
    if (error != nullptr) {
        *error = "image generation is not implemented in WanPipeline yet";
    }
    return LD_STATUS_UNSUPPORTED;
}

ld_status_t WanPipeline::generate_video(const ld_video_generation_params_t*,
                                        ld_video_t* out,
                                        std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (error != nullptr) {
        *error = "video generation is not implemented in WanPipeline yet";
    }
    return LD_STATUS_UNSUPPORTED;
}

ld_scheduler_t WanPipeline::default_scheduler(ld_sampler_t method) const {
    if (method == LD_SAMPLER_LCM || method == LD_SAMPLER_TCD) {
        return LD_SCHEDULER_LCM;
    }
    if (method == LD_SAMPLER_DDIM_TRAILING) {
        return LD_SCHEDULER_SIMPLE;
    }
    return LD_SCHEDULER_DISCRETE;
}

}  // namespace lightdit
