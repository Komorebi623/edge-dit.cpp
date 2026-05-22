#include "light-dit.h"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>

struct ld_context {
    ld_context_params_t params;
    std::string model_path;
    std::string last_error;
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

static uint32_t ld_hash_string(const char * text) {
    uint32_t hash = 2166136261u;
    if (text == nullptr) {
        return hash;
    }

    while (*text != '\0') {
        hash ^= static_cast<uint8_t>(*text);
        hash *= 16777619u;
        ++text;
    }

    return hash;
}

static uint32_t ld_mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static bool ld_image_byte_size(int width, int height, uint32_t channels, size_t * size) {
    if (size == nullptr || width <= 0 || height <= 0 || channels == 0) {
        return false;
    }

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    const size_t c = static_cast<size_t>(channels);

    if (w > std::numeric_limits<size_t>::max() / h) {
        return false;
    }

    const size_t pixels = w * h;
    if (pixels > std::numeric_limits<size_t>::max() / c) {
        return false;
    }

    *size = pixels * c;
    return true;
}

static ld_status_t ld_make_naive_image(
    const ld_image_generation_params_t * params,
    int batch_index,
    ld_image_t * image
) {
    if (params == nullptr || image == nullptr) {
        return LD_STATUS_INVALID_ARGUMENT;
    }

    size_t byte_size = 0;
    if (!ld_image_byte_size(params->width, params->height, 3, &byte_size)) {
        return LD_STATUS_INVALID_ARGUMENT;
    }

    uint8_t * data = static_cast<uint8_t *>(std::malloc(byte_size));
    if (data == nullptr) {
        return LD_STATUS_OUT_OF_MEMORY;
    }

    const uint32_t prompt_hash = ld_hash_string(params->prompt);
    const uint32_t neg_hash = ld_hash_string(params->negative_prompt);
    const uint32_t seed = params->seed < 0
        ? prompt_hash
        : static_cast<uint32_t>(params->seed) ^ static_cast<uint32_t>(params->seed >> 32);
    const uint32_t base = ld_mix_u32(seed ^ prompt_hash ^ (neg_hash << 1) ^ static_cast<uint32_t>(batch_index));

    for (int y = 0; y < params->height; ++y) {
        for (int x = 0; x < params->width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * params->width + x) * 3;
            const uint32_t noise = ld_mix_u32(base ^ static_cast<uint32_t>(x * 73856093u) ^ static_cast<uint32_t>(y * 19349663u));

            const uint32_t xf = static_cast<uint32_t>((255ull * static_cast<uint64_t>(x)) / static_cast<uint64_t>(params->width));
            const uint32_t yf = static_cast<uint32_t>((255ull * static_cast<uint64_t>(y)) / static_cast<uint64_t>(params->height));

            data[offset + 0] = static_cast<uint8_t>((xf + (noise & 0x3f) + (base & 0x7f)) & 0xff);
            data[offset + 1] = static_cast<uint8_t>((yf + ((noise >> 8) & 0x3f) + ((base >> 8) & 0x7f)) & 0xff);
            data[offset + 2] = static_cast<uint8_t>(((xf / 2) + (yf / 2) + ((noise >> 16) & 0x7f) + ((base >> 16) & 0x3f)) & 0xff);
        }
    }

    image->width = static_cast<uint32_t>(params->width);
    image->height = static_cast<uint32_t>(params->height);
    image->channels = 3;
    image->data = data;
    return LD_STATUS_OK;
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
}

ld_context_t * ld_create_context(const ld_context_params_t * params) {
    if (params == nullptr) {
        return nullptr;
    }

    ld_context_t * ctx = new (std::nothrow) ld_context();
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->params = *params;
    if (params->model_path != nullptr) {
        ctx->model_path = params->model_path;
        ctx->params.model_path = ctx->model_path.c_str();
    }

    ctx->last_error.clear();
    return ctx;
}

void ld_free_context(ld_context_t * ctx) {
    delete ctx;
}

ld_status_t ld_generate_image(
    ld_context_t * ctx,
    const ld_image_generation_params_t * params,
    ld_image_batch_t * out
) {
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }

    if (ctx == nullptr || params == nullptr || out == nullptr) {
        return LD_STATUS_INVALID_ARGUMENT;
    }

    if (params->width <= 0 || params->height <= 0) {
        ld_set_error(ctx, "image width and height must be positive");
        return LD_STATUS_INVALID_ARGUMENT;
    }

    const int count = params->batch_count > 0 ? params->batch_count : 1;
    if (count > 1024) {
        ld_set_error(ctx, "batch count is too large");
        return LD_STATUS_INVALID_ARGUMENT;
    }

    ld_image_t * images = static_cast<ld_image_t *>(std::calloc(static_cast<size_t>(count), sizeof(ld_image_t)));
    if (images == nullptr) {
        ld_set_error(ctx, "failed to allocate image batch");
        return LD_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < count; ++i) {
        const ld_status_t status = ld_make_naive_image(params, i, &images[i]);
        if (status != LD_STATUS_OK) {
            ld_image_batch_t partial;
            partial.images = images;
            partial.count = count;
            ld_free_image_batch(&partial);
            ld_set_error(ctx, status == LD_STATUS_OUT_OF_MEMORY
                ? "failed to allocate image data"
                : "invalid image generation parameters");
            return status;
        }
    }

    out->images = images;
    out->count = count;
    ld_set_error(ctx, "");
    return LD_STATUS_OK;
}

ld_status_t ld_generate_video(
    ld_context_t * ctx,
    const ld_video_generation_params_t * params,
    ld_video_t * out
) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }

    if (ctx == nullptr || params == nullptr || out == nullptr) {
        return LD_STATUS_INVALID_ARGUMENT;
    }

    ld_set_error(ctx, "Flux video generation is not implemented yet");
    return LD_STATUS_UNSUPPORTED;
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

void ld_free_image_batch(ld_image_batch_t * batch) {
    if (batch == nullptr) {
        return;
    }

    for (int i = 0; i < batch->count; ++i) {
        ld_free_image(&batch->images[i]);
    }

    std::free(batch->images);
    batch->images = nullptr;
    batch->count = 0;
}

void ld_free_video(ld_video_t * video) {
    if (video == nullptr) {
        return;
    }

    for (int i = 0; i < video->frame_count; ++i) {
        ld_free_image(&video->frames[i]);
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
