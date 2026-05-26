#include "light-dit.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include "core/runtime/light_dit_engine.hpp"
#include "utils/util.h"

struct ld_context {
    ld_context_params_t params = {};
    std::unique_ptr<lightdit::LightDitEngine> engine;
    std::string last_error;
    bool initialized = false;
};

static void ld_zero(void * ptr, size_t size) {
    if (ptr != nullptr) {
        std::memset(ptr, 0, size);
    }
}

static void ld_set_error(ld_context_t * ctx, const char * message) {
    if (ctx != nullptr) {
        ctx->last_error = message != nullptr ? message : "";
    }
}

void ld_context_params_init(ld_context_params_t * params) {
    ld_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->n_threads = 0;
    params->weight_type = LD_DTYPE_AUTO;
    params->use_mmap = true;
    params->max_vram_gb = 0.0f;
}

void ld_sample_params_init(ld_sample_params_t * params) {
    ld_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->sampler = LD_SAMPLER_AUTO;
    params->scheduler = LD_SCHEDULER_AUTO;
    params->steps = 20;
    params->cfg_scale = 1.0f;
    params->image_cfg_scale = 1.0f;
    params->distilled_guidance = 3.5f;
    params->eta = 0.0f;
    params->flow_shift = 0.0f;
}

void ld_image_generation_params_init(ld_image_generation_params_t * params) {
    ld_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->width = 1024;
    params->height = 1024;
    params->seed = -1;
    params->batch_count = 1;
    params->strength = 0.75f;
    params->control_strength = 1.0f;
    ld_sample_params_init(&params->sample);
}

void ld_video_generation_params_init(ld_video_generation_params_t * params) {
    ld_zero(params, sizeof(*params));
    if (params == nullptr) {
        return;
    }

    params->width = 1024;
    params->height = 1024;
    params->frames = 1;
    params->seed = -1;
    params->strength = 0.75f;
    params->vace_strength = 1.0f;
    params->moe_boundary = 0.5f;
    ld_sample_params_init(&params->sample);
    ld_sample_params_init(&params->high_noise_sample);
    params->high_noise_sample.steps = -1;
}

ld_context_t* ld_create_context(const ld_context_params_t* params) {
    if (params == nullptr) {
        LOG_ERROR("ld_create_context failed: params is null");
        return nullptr;
    }

    std::unique_ptr<ld_context_t> ctx(new (std::nothrow) ld_context_t());
    if (ctx == nullptr) {
        LOG_ERROR("ld_create_context failed: allocate ld_context failed");
        return nullptr;
    }

    ctx->params = *params;

    try {
        ctx->engine = std::make_unique<lightdit::LightDitEngine>();
    } catch (const std::exception& e) {
        ctx->last_error = std::string("failed to allocate LightDitEngine: ") + e.what();
        LOG_ERROR("%s", ctx->last_error.c_str());
        return nullptr;
    }

    if (ctx->engine == nullptr) {
        ctx->last_error = "failed to allocate LightDitEngine";
        LOG_ERROR("%s", ctx->last_error.c_str());
        return nullptr;
    }

    if (!ctx->engine->init(params)) {
        ctx->last_error = ctx->engine->last_error();
        if (ctx->last_error.empty()) {
            ctx->last_error = "failed to initialize LightDitEngine";
        }

        LOG_ERROR("failed to initialize engine: %s", ctx->last_error.c_str());
        return nullptr;
    }

    ctx->initialized = true;

    return ctx.release();
}

void ld_free_context(ld_context_t * ctx) {
    delete ctx;
}

ld_status_t ld_generate_image(
    ld_context_t* ctx,
    const ld_image_generation_params_t* params,
    ld_image_batch_t* out
) {
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }

    if (ctx == nullptr || params == nullptr || out == nullptr) {
        if (ctx != nullptr) {
            ld_set_error(ctx, "invalid argument: ctx, params, or out is null");
        }
        return LD_STATUS_INVALID_ARGUMENT;
    }

    if (!ctx->initialized || ctx->engine == nullptr) {
        ld_set_error(ctx, "engine is not initialized");
        return LD_STATUS_MODEL_LOAD_FAILED;
    }

    ld_image_batch_t tmp = {};
    ld_status_t status = ctx->engine->generate_image(params, &tmp);

    if (status != LD_STATUS_OK) {
        ld_free_image_batch(&tmp);

        std::string err = ctx->engine->last_error();
        if (err.empty()) {
            err = "image generation failed";
        }

        ld_set_error(ctx, err.c_str());
        return status;
    }

    if (tmp.images == nullptr || tmp.count <= 0) {
        ld_free_image_batch(&tmp);
        ld_set_error(ctx, "engine returned empty image batch");
        return LD_STATUS_GENERATION_FAILED;
    }
    *out = tmp;
    ld_set_error(ctx, "");
    return LD_STATUS_OK;
}

ld_status_t ld_generate_video(
    ld_context_t* ctx,
    const ld_video_generation_params_t* params,
    ld_video_t* out
) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }

    if (ctx == nullptr || params == nullptr || out == nullptr) {
        if (ctx != nullptr) {
            ld_set_error(ctx, "invalid argument: ctx, params, or out is null");
        }
        return LD_STATUS_INVALID_ARGUMENT;
    }

    if (!ctx->initialized || ctx->engine == nullptr) {
        ld_set_error(ctx, "engine is not initialized");
        return LD_STATUS_MODEL_LOAD_FAILED;
    }

    ld_video_t tmp = {};
    ld_status_t status = ctx->engine->generate_video(params, &tmp);

    if (status != LD_STATUS_OK) {
        ld_free_video(&tmp);

        std::string err = ctx->engine->last_error();
        if (err.empty()) {
            err = "video generation failed";
        }

        ld_set_error(ctx, err.c_str());
        return status;
    }

    if (tmp.frames == nullptr || tmp.frame_count <= 0) {
        ld_free_video(&tmp);
        ld_set_error(ctx, "engine returned empty video");
        return LD_STATUS_GENERATION_FAILED;
    }

    *out = tmp;
    ld_set_error(ctx, "");
    return LD_STATUS_OK;
}

void ld_free_image(ld_image_t * image) {
    if (image == nullptr) {
        return;
    }

    std::free(image->data);
    image->data = nullptr;
    image->width = 0;
    image->height = 0;
    image->channels = 0;
}

void ld_free_image_batch(ld_image_batch_t* batch) {
    if (batch == nullptr) {
        return;
    }

    if (batch->images != nullptr && batch->count > 0) {
        for (int i = 0; i < batch->count; ++i) {
            ld_free_image(&batch->images[i]);
        }
    }

    std::free(batch->images);
    batch->images = nullptr;
    batch->count = 0;
}

void ld_free_video(ld_video_t* video) {
    if (video == nullptr) {
        return;
    }

    if (video->frames != nullptr && video->frame_count > 0) {
        for (int i = 0; i < video->frame_count; ++i) {
            ld_free_image(&video->frames[i]);
        }
    }

    std::free(video->frames);
    video->frames = nullptr;
    video->frame_count = 0;
}

const char * ld_get_last_error(const ld_context_t * ctx) {
    if (ctx == nullptr || ctx->last_error.empty()) {
        return nullptr;
    }

    return ctx->last_error.c_str();
}
