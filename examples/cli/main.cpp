#include "light-dit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> --prompt <text> [options]\n"
        "  %s --diffusion-model <path> --vae <path> --clip_l <path> --t5xxl <path> --prompt <text> [options]\n\n"
        "Options:\n"
        "  --diffusion-model <path>  Standalone Flux transformer weights\n"
        "  --vae <path>              Standalone VAE weights\n"
        "  --clip_l <path>           CLIP-L text encoder weights\n"
        "  --t5xxl <path>            T5XXL text encoder weights\n"
        "  -o, --output <path>       Output image path, default: output.png\n"
        "  -W, --width <int>         Image width, default: 1024\n"
        "  -H, --height <int>        Image height, default: 1024\n"
        "  --steps <int>             Sampling steps, default: 20\n"
        "  -s, --seed <int64>        Seed, default: -1\n"
        "  -t, --threads <int>       Thread count, default: 0\n"
        "  --guidance <float>        Flux distilled guidance, default: 3.5\n"
        "  --backend <name>          Backend: auto, cpu, cuda, gpu. Default: auto\n"
        "  --gpu                     Alias for --backend gpu\n"
        "  --help              Show this help\n",
        prog,
        prog
    );
}

static bool save_png(const char* path, const ld_image_t& image) {
    if (path == nullptr || image.data == nullptr) {
        return false;
    }

    if (image.channels == 0 || image.channels > 4) {
        std::fprintf(stderr, "unsupported channel count: %u\n", image.channels);
        return false;
    }

    return stbi_write_png(
        path,
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        static_cast<int>(image.channels),
        image.data,
        0,
        nullptr
    ) != 0;
}

struct FluxCliArgs {
    const char* model_path = nullptr;
    const char* diffusion_model_path = nullptr;
    const char* vae_path = nullptr;
    const char* clip_l_path = nullptr;
    const char* t5xxl_path = nullptr;
    const char* prompt = nullptr;
    const char* output_path = "output.png";
    const char* backend = nullptr;

    int width = 1024;
    int height = 1024;
    int steps = 20;
    int threads = 0;

    int64_t seed = -1;
    float guidance = 3.5f;
};

static bool parse_args(int argc, char** argv, FluxCliArgs* args) {
    if (args == nullptr) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];

        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--model") == 0) {
            args->model_path = require_value(key);
        } else if (std::strcmp(key, "--diffusion-model") == 0) {
            args->diffusion_model_path = require_value(key);
        } else if (std::strcmp(key, "--vae") == 0) {
            args->vae_path = require_value(key);
        } else if (std::strcmp(key, "--clip_l") == 0) {
            args->clip_l_path = require_value(key);
        } else if (std::strcmp(key, "--t5xxl") == 0) {
            args->t5xxl_path = require_value(key);
        } else if (std::strcmp(key, "--prompt") == 0 || std::strcmp(key, "-p") == 0) {
            args->prompt = require_value(key);
        } else if (std::strcmp(key, "--output") == 0 || std::strcmp(key, "-o") == 0) {
            args->output_path = require_value(key);
        } else if (std::strcmp(key, "--width") == 0 || std::strcmp(key, "-W") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->width = std::atoi(v);
        } else if (std::strcmp(key, "--height") == 0 || std::strcmp(key, "-H") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->height = std::atoi(v);
        } else if (std::strcmp(key, "--steps") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->steps = std::atoi(v);
        } else if (std::strcmp(key, "--threads") == 0 || std::strcmp(key, "-t") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->threads = std::atoi(v);
        } else if (std::strcmp(key, "--seed") == 0 || std::strcmp(key, "-s") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->seed = std::strtoll(v, nullptr, 10);
        } else if (std::strcmp(key, "--guidance") == 0) {
            const char* v = require_value(key);
            if (!v) return false;
            args->guidance = std::strtof(v, nullptr);
        } else if (std::strcmp(key, "--backend") == 0) {
            args->backend = require_value(key);
        } else if (std::strcmp(key, "--gpu") == 0) {
            args->backend = "gpu";
        } else if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            return false;
        }
    }

    const bool has_full_model = args->model_path != nullptr && std::strlen(args->model_path) > 0;
    const bool has_components =
        args->diffusion_model_path != nullptr && std::strlen(args->diffusion_model_path) > 0 &&
        args->vae_path != nullptr && std::strlen(args->vae_path) > 0 &&
        args->clip_l_path != nullptr && std::strlen(args->clip_l_path) > 0 &&
        args->t5xxl_path != nullptr && std::strlen(args->t5xxl_path) > 0;
    if (!has_full_model && !has_components) {
        std::fprintf(stderr, "--model or the full --diffusion-model/--vae/--clip_l/--t5xxl set is required\n");
        return false;
    }

    if (args->prompt == nullptr || std::strlen(args->prompt) == 0) {
        std::fprintf(stderr, "--prompt is required\n");
        return false;
    }

    if (args->width <= 0 || args->height <= 0) {
        std::fprintf(stderr, "width and height must be positive\n");
        return false;
    }

    if (args->steps <= 0) {
        std::fprintf(stderr, "steps must be positive\n");
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    FluxCliArgs args;

    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (args.backend != nullptr && std::strlen(args.backend) > 0) {
        setenv("LDIT_BACKEND", args.backend, 1);
    }

    ld_context_params_t ctx_params;
    ld_context_params_init(&ctx_params);

    ctx_params.model_path = args.model_path;
    ctx_params.diffusion_model_path = args.diffusion_model_path;
    ctx_params.vae_path = args.vae_path;
    ctx_params.clip_l_path = args.clip_l_path;
    ctx_params.t5xxl_path = args.t5xxl_path;

    if (args.threads > 0) {
        ctx_params.n_threads = args.threads;
    }

    /*
     * Flux 测试阶段先让内部自动识别 dtype / sampler / scheduler。
     * 如果你的模型是量化 GGUF，也可以在这里手动指定：
     *   ctx_params.weight_type = LD_DTYPE_Q8_0;
     */
    ctx_params.weight_type = LD_DTYPE_AUTO;

    ld_context_t* ctx = ld_create_context(&ctx_params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create light-dit context\n");
        return 2;
    }

    ld_image_generation_params_t gen_params;
    ld_image_generation_params_init(&gen_params);

    gen_params.prompt = args.prompt;
    gen_params.negative_prompt = "";

    gen_params.width = args.width;
    gen_params.height = args.height;
    gen_params.seed = args.seed;
    gen_params.batch_count = 1;

    /*
     * Flux / DiT 测试建议：
     * - sampler/scheduler 用 AUTO，让内部根据模型选择
     * - cfg_scale 设为 1.0，避免传统 CFG 负条件分支
     * - distilled_guidance 用 Flux 常见值 3.5
     */
    gen_params.sample.sampler = LD_SAMPLER_AUTO;
    gen_params.sample.scheduler = LD_SCHEDULER_AUTO;
    gen_params.sample.steps = args.steps;
    gen_params.sample.cfg_scale = 1.0f;
    gen_params.sample.image_cfg_scale = 1.0f;
    gen_params.sample.distilled_guidance = args.guidance;

    ld_image_batch_t output;
    ld_status_t status = ld_generate_image(ctx, &gen_params, &output);

    if (status != LD_STATUS_OK) {
        std::fprintf(stderr, "ld_generate_image failed, status=%d\n", static_cast<int>(status));

        const char* err = ld_get_last_error(ctx);
        if (err != nullptr && std::strlen(err) > 0) {
            std::fprintf(stderr, "last error: %s\n", err);
        }

        ld_free_context(ctx);
        return 3;
    }

    if (output.count <= 0 || output.images == nullptr) {
        std::fprintf(stderr, "generation succeeded but output is empty\n");
        ld_free_context(ctx);
        return 4;
    }

    if (!save_png(args.output_path, output.images[0])) {
        std::fprintf(stderr, "failed to save output image: %s\n", args.output_path);
        ld_free_image_batch(&output);
        ld_free_context(ctx);
        return 5;
    }

    std::printf("saved image to %s\n", args.output_path);

    ld_free_image_batch(&output);
    ld_free_context(ctx);

    return 0;
}
