#include "stable-diffusion.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string model;
    std::string diffusion_model;
    std::string vae;
    std::string clip_l;
    std::string clip_g;
    std::string t5xxl;
    std::string llm;
    std::string prompt;
    std::string negative_prompt;
    std::string output_dir;
    std::string dtype = "bf16";
    std::string backend;
    std::string params_backend;
    std::string max_vram = "0";
    std::string model_args;
    std::string sample_method;
    std::string scheduler;
    int width = 0;
    int height = 0;
    int steps = 0;
    int64_t seed = 0;
    int warmup_runs = 0;
    int measured_runs = 0;
    int qwen_image_layers = 3;
    float guidance = 3.5f;
    float cfg_scale = 1.0f;
    float flow_shift = std::numeric_limits<float>::infinity();
    bool has_flow_shift = false;
    bool flash_attn = false;
    bool diffusion_flash_attn = false;
    bool vae_tiling = false;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --prompt PROMPT --output-dir DIR --width N --height N "
        << "--steps N --seed N --warmup-runs N --measured-runs N "
        << "[--model PATH | --diffusion-model PATH] [component paths] "
        << "[--cfg-scale N] [--distilled-guidance N]\n";
    std::exit(2);
}

std::string take_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        usage(argv[0]);
    }
    ++index;
    return argv[index];
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--model") {
            args.model = take_value(i, argc, argv);
        } else if (key == "--diffusion-model") {
            args.diffusion_model = take_value(i, argc, argv);
        } else if (key == "--vae") {
            args.vae = take_value(i, argc, argv);
        } else if (key == "--clip-l") {
            args.clip_l = take_value(i, argc, argv);
        } else if (key == "--clip-g") {
            args.clip_g = take_value(i, argc, argv);
        } else if (key == "--t5xxl") {
            args.t5xxl = take_value(i, argc, argv);
        } else if (key == "--llm") {
            args.llm = take_value(i, argc, argv);
        } else if (key == "--prompt") {
            args.prompt = take_value(i, argc, argv);
        } else if (key == "--negative-prompt") {
            args.negative_prompt = take_value(i, argc, argv);
        } else if (key == "--output-dir") {
            args.output_dir = take_value(i, argc, argv);
        } else if (key == "--width") {
            args.width = std::stoi(take_value(i, argc, argv));
        } else if (key == "--height") {
            args.height = std::stoi(take_value(i, argc, argv));
        } else if (key == "--steps") {
            args.steps = std::stoi(take_value(i, argc, argv));
        } else if (key == "--seed") {
            args.seed = std::stoll(take_value(i, argc, argv));
        } else if (key == "--guidance" || key == "--distilled-guidance") {
            args.guidance = std::stof(take_value(i, argc, argv));
        } else if (key == "--cfg-scale") {
            args.cfg_scale = std::stof(take_value(i, argc, argv));
        } else if (key == "--flow-shift") {
            args.flow_shift = std::stof(take_value(i, argc, argv));
            args.has_flow_shift = true;
        } else if (key == "--dtype") {
            args.dtype = take_value(i, argc, argv);
        } else if (key == "--backend") {
            args.backend = take_value(i, argc, argv);
        } else if (key == "--params-backend") {
            args.params_backend = take_value(i, argc, argv);
        } else if (key == "--max-vram") {
            args.max_vram = take_value(i, argc, argv);
        } else if (key == "--model-args") {
            args.model_args = take_value(i, argc, argv);
        } else if (key == "--sample-method") {
            args.sample_method = take_value(i, argc, argv);
        } else if (key == "--scheduler") {
            args.scheduler = take_value(i, argc, argv);
        } else if (key == "--warmup-runs") {
            args.warmup_runs = std::stoi(take_value(i, argc, argv));
        } else if (key == "--measured-runs") {
            args.measured_runs = std::stoi(take_value(i, argc, argv));
        } else if (key == "--flash-attn") {
            args.flash_attn = true;
        } else if (key == "--diffusion-flash-attn" || key == "--diffusion-fa") {
            args.diffusion_flash_attn = true;
        } else if (key == "--vae-tiling") {
            args.vae_tiling = true;
        } else if (key == "--qwen-image-layers") {
            args.qwen_image_layers = std::stoi(take_value(i, argc, argv));
        } else {
            std::cerr << "unknown argument: " << key << "\n";
            usage(argv[0]);
        }
    }
    if (args.prompt.empty() || args.output_dir.empty() || args.width <= 0 || args.height <= 0 ||
        args.steps <= 0 || args.measured_runs < 0 || args.warmup_runs < 0 ||
        (args.model.empty() && args.diffusion_model.empty())) {
        usage(argv[0]);
    }
    return args;
}

void log_callback(sd_log_level_t level, const char* text, void*) {
    if (level == SD_LOG_DEBUG) {
        return;
    }
    std::cerr << text;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out << "\\\\";
                break;
            case '"':
                out << "\\\"";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                out << ch;
                break;
        }
    }
    return out.str();
}

void write_ppm(const sd_image_t& image, const fs::path& path) {
    if (image.data == nullptr || image.width == 0 || image.height == 0 || image.channel < 3) {
        throw std::runtime_error("generated image is empty or not RGB-compatible");
    }
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open sample output: " + path.string());
    }
    out << "P6\n" << image.width << " " << image.height << "\n255\n";
    const size_t pixels = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
    for (size_t i = 0; i < pixels; ++i) {
        const uint8_t* pixel = image.data + i * image.channel;
        out.write(reinterpret_cast<const char*>(pixel), 3);
    }
}

void write_metrics(
    const fs::path& output_dir,
    double load_ms,
    const std::vector<double>& warmup_ms,
    const std::vector<double>& measured_ms,
    int steps,
    const fs::path& sample_dir) {
    fs::create_directories(output_dir);
    std::ofstream out(output_dir / "runner_metrics.json");
    if (!out) {
        throw std::runtime_error("failed to open runner_metrics.json");
    }
    auto write_array = [&](const std::vector<double>& values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << std::fixed << std::setprecision(3) << values[i];
        }
        out << "]";
    };
    double mean = 0.0;
    for (double value : measured_ms) {
        mean += value;
    }
    if (!measured_ms.empty()) {
        mean /= static_cast<double>(measured_ms.size());
    }
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"metric_source\": \"stable_diffusion_cpp_c_api\",\n"
        << "  \"measurement_boundary\": \"load_once_e2e_generation_no_output_encoding\",\n"
        << "  \"load_ms\": " << std::fixed << std::setprecision(3) << load_ms << ",\n"
        << "  \"warmup_ms\": ";
    write_array(warmup_ms);
    out << ",\n  \"measured_ms\": ";
    write_array(measured_ms);
    out << ",\n  \"component_ms\": {\n"
        << "    \"text_encoder\": null,\n"
        << "    \"dit\": null,\n"
        << "    \"vae\": null,\n"
        << "    \"per_step_avg\": ";
    if (!measured_ms.empty() && steps > 0) {
        out << std::fixed << std::setprecision(3) << (mean / static_cast<double>(steps));
    } else {
        out << "null";
    }
    out << "\n  },\n"
        << "  \"sample_output_dir\": \"" << json_escape(sample_dir.string()) << "\"\n"
        << "}\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::path output_dir = fs::absolute(args.output_dir);
        fs::path sample_dir = output_dir / "samples" / "stable-diffusion.cpp";
        fs::create_directories(sample_dir);

        sd_set_log_callback(log_callback, nullptr);

        sd_ctx_params_t ctx_params;
        sd_ctx_params_init(&ctx_params);
        ctx_params.model_path = args.model.empty() ? nullptr : args.model.c_str();
        ctx_params.diffusion_model_path = args.diffusion_model.empty() ? nullptr : args.diffusion_model.c_str();
        ctx_params.vae_path = args.vae.empty() ? nullptr : args.vae.c_str();
        ctx_params.clip_l_path = args.clip_l.empty() ? nullptr : args.clip_l.c_str();
        ctx_params.clip_g_path = args.clip_g.empty() ? nullptr : args.clip_g.c_str();
        ctx_params.t5xxl_path = args.t5xxl.empty() ? nullptr : args.t5xxl.c_str();
        ctx_params.llm_path = args.llm.empty() ? nullptr : args.llm.c_str();
        ctx_params.backend = args.backend.empty() ? nullptr : args.backend.c_str();
        ctx_params.params_backend = args.params_backend.empty() ? nullptr : args.params_backend.c_str();
        ctx_params.max_vram = args.max_vram.empty() ? "0" : args.max_vram.c_str();
        ctx_params.model_args = args.model_args.empty() ? nullptr : args.model_args.c_str();
        ctx_params.wtype = str_to_sd_type(args.dtype.c_str());
        ctx_params.rng_type = CUDA_RNG;
        ctx_params.sampler_rng_type = CUDA_RNG;
        ctx_params.flash_attn = args.flash_attn;
        ctx_params.diffusion_flash_attn = args.diffusion_flash_attn;

        double load_start = now_ms();
        sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
        double load_ms = now_ms() - load_start;
        if (ctx == nullptr) {
            std::cerr << "new_sd_ctx failed\n";
            return 1;
        }

        std::vector<double> warmup_ms;
        std::vector<double> measured_ms;
        const int total_runs = args.warmup_runs + args.measured_runs;
        for (int index = 0; index < total_runs; ++index) {
            const bool is_warmup = index < args.warmup_runs;
            const int phase_index = is_warmup ? index : index - args.warmup_runs;

            sd_img_gen_params_t gen_params;
            sd_img_gen_params_init(&gen_params);
            gen_params.prompt = args.prompt.c_str();
            gen_params.negative_prompt = args.negative_prompt.c_str();
            gen_params.width = args.width;
            gen_params.height = args.height;
            gen_params.seed = args.seed;
            gen_params.batch_count = 1;
            gen_params.qwen_image_layers = args.qwen_image_layers;
            gen_params.sample_params.sample_steps = args.steps;
            gen_params.sample_params.guidance.txt_cfg = args.cfg_scale;
            gen_params.sample_params.guidance.img_cfg = args.cfg_scale;
            gen_params.sample_params.guidance.distilled_guidance = args.guidance;
            if (args.has_flow_shift) {
                gen_params.sample_params.flow_shift = args.flow_shift;
            }
            gen_params.sample_params.sample_method = args.sample_method.empty()
                ? sd_get_default_sample_method(ctx)
                : str_to_sample_method(args.sample_method.c_str());
            gen_params.sample_params.scheduler = args.scheduler.empty()
                ? sd_get_default_scheduler(ctx, gen_params.sample_params.sample_method)
                : str_to_scheduler(args.scheduler.c_str());
            if (args.vae_tiling) {
                gen_params.vae_tiling_params.enabled = true;
            }

            sd_image_t* images = nullptr;
            int image_count = 0;
            double start = now_ms();
            bool ok = generate_image(ctx, &gen_params, &images, &image_count);
            double elapsed = now_ms() - start;
            if (!ok || images == nullptr || image_count <= 0) {
                free_sd_images(images, image_count);
                free_sd_ctx(ctx);
                std::cerr << "generate_image failed\n";
                return 1;
            }
            if (is_warmup) {
                warmup_ms.push_back(elapsed);
            } else {
                measured_ms.push_back(elapsed);
                std::ostringstream name;
                name << "output_" << std::setw(3) << std::setfill('0') << phase_index << ".ppm";
                write_ppm(images[0], sample_dir / name.str());
            }
            free_sd_images(images, image_count);
            std::cout << "[sd-cpp-e2e] " << (is_warmup ? "warmup " : "measured ")
                      << phase_index << " " << std::fixed << std::setprecision(3)
                      << (elapsed / 1000.0) << "s\n";
        }

        free_sd_ctx(ctx);
        write_metrics(output_dir, load_ms, warmup_ms, measured_ms, args.steps, sample_dir);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
