#pragma once

#include <cstdint>
#include <vector>

#include "light-dit.h"

namespace lightdit {

enum class DiTPipelineTask {
    ImageGeneration,
    VideoGeneration,
};

// Shared pipeline request surface.
//
// This is intentionally a superset of common DiT generation inputs. Individual
// models decide which fields are meaningful. For example, Flux uses CLIP+T5
// text conditioning and packed image ids; Qwen-Image uses Qwen prompt masks;
// Wan video uses frame controls. The runtime should still submit one common
// request shape.
struct DiTPipelineParams {
    DiTPipelineTask task = DiTPipelineTask::ImageGeneration;

    const char* prompt          = nullptr;
    const char* negative_prompt = nullptr;

    int width       = 512;
    int height      = 512;
    int frames      = 1;
    int batch_count = 1;
    int64_t seed    = -1;
    int clip_skip   = -1;

    float strength         = 0.75f;
    float control_strength = 0.9f;
    float vace_strength    = 1.0f;
    float moe_boundary     = 0.875f;

    bool auto_resize_ref_image = true;
    bool increase_ref_index    = false;

    ld_sample_params_t sample_params            = {};
    ld_sample_params_t high_noise_sample_params = {};
    ld_tiling_params_t vae_tiling_params        = {};
    ld_cache_params_t cache                     = {};
    ld_hires_params_t hires                     = {};
    ld_pm_params_t pm_params                    = {};

    const ld_lora_t* loras = nullptr;
    uint32_t lora_count    = 0;

    ld_image_t init_image    = {};
    ld_image_t end_image     = {};
    ld_image_t mask_image    = {};
    ld_image_t control_image = {};

    const ld_image_t* ref_images = nullptr;
    int ref_image_count          = 0;

    const ld_image_t* control_frames = nullptr;
    int control_frame_count          = 0;

    static DiTPipelineParams from_image_params(const ld_img_gen_params_t& params) {
        DiTPipelineParams out;
        out.task                  = DiTPipelineTask::ImageGeneration;
        out.loras                 = params.loras;
        out.lora_count            = params.lora_count;
        out.prompt                = params.prompt;
        out.negative_prompt       = params.negative_prompt;
        out.clip_skip             = params.clip_skip;
        out.init_image            = params.init_image;
        out.ref_images            = params.ref_images;
        out.ref_image_count       = params.ref_images_count;
        out.auto_resize_ref_image = params.auto_resize_ref_image;
        out.increase_ref_index    = params.increase_ref_index;
        out.mask_image            = params.mask_image;
        out.width                 = params.width;
        out.height                = params.height;
        out.sample_params         = params.sample_params;
        out.strength              = params.strength;
        out.seed                  = params.seed;
        out.batch_count           = params.batch_count;
        out.control_image         = params.control_image;
        out.control_strength      = params.control_strength;
        out.pm_params             = params.pm_params;
        out.vae_tiling_params     = params.vae_tiling_params;
        out.cache                 = params.cache;
        out.hires                 = params.hires;
        return out;
    }

    static DiTPipelineParams from_video_params(const ld_vid_gen_params_t& params) {
        DiTPipelineParams out;
        out.task                     = DiTPipelineTask::VideoGeneration;
        out.loras                    = params.loras;
        out.lora_count               = params.lora_count;
        out.prompt                   = params.prompt;
        out.negative_prompt          = params.negative_prompt;
        out.clip_skip                = params.clip_skip;
        out.init_image               = params.init_image;
        out.end_image                = params.end_image;
        out.control_frames           = params.control_frames;
        out.control_frame_count      = params.control_frames_size;
        out.width                    = params.width;
        out.height                   = params.height;
        out.sample_params            = params.sample_params;
        out.high_noise_sample_params = params.high_noise_sample_params;
        out.moe_boundary             = params.moe_boundary;
        out.strength                 = params.strength;
        out.seed                     = params.seed;
        out.frames                   = params.video_frames;
        out.vace_strength            = params.vace_strength;
        out.vae_tiling_params        = params.vae_tiling_params;
        out.cache                    = params.cache;
        return out;
    }
};

struct DiTPipelineOutput {
    std::vector<ld_image_t> images;

    bool empty() const {
        return images.empty();
    }

    int count() const {
        return static_cast<int>(images.size());
    }
};

// Thin model-pipeline boundary.
//
// A DiT pipeline owns the model-specific generation recipe:
//   - prompt encoding shape and masks
//   - latent packing / unpacking
//   - timestep or sigma schedule quirks
//   - transformer forward arguments
//   - VAE latent scaling / shifting
//
// Runtime classes should provide backend-specific components and submit one
// common DiTPipelineParams request. They should not implement the model recipe
// themselves.
class DiTPipeline {
public:
    virtual ~DiTPipeline() = default;

    virtual const char* name() const = 0;

    virtual bool supports(DiTPipelineTask task) const = 0;

    virtual bool generate(const DiTPipelineParams& params, DiTPipelineOutput* output) = 0;
};

}  // namespace lightdit
