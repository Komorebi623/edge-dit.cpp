#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "httplib.h"
#include "routes.h"
#include "runtime.h"

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s --model <model-or-diffusers-dir> [options]\n"
        "  %s --diffusion-model <path> --vae <path> --clip_l <path> --t5xxl <path> [options]\n\n"
        "Server options:\n"
        "  --host <ip>               Listen host, default: 127.0.0.1\n"
        "  --port <int>              Listen port, default: 8080\n"
        "  --verbose                 Print request logs\n\n"
        "Context options:\n"
        "  --model <path>            Model or Diffusers directory\n"
        "  --diffusion-model <path>  Standalone DiT transformer weights\n"
        "  --vae <path>              Standalone VAE weights\n"
        "  --clip_l <path>           CLIP-L text encoder weights\n"
        "  --clip_g <path>           CLIP-G text encoder weights\n"
        "  --t5xxl <path>            T5XXL text encoder weights\n"
        "  --backend <name>          Backend: auto, cpu, cuda, gpu. Default: auto\n"
        "  --gpu                     Alias for --backend gpu\n"
        "  -t, --threads <int>       Thread count, default: auto\n\n"
        "Default generation options:\n"
        "  -W, --width <int>         Default width, default: 1024\n"
        "  -H, --height <int>        Default height, default: 1024\n"
        "  --steps <int>             Default steps, default: 20\n"
        "  -s, --seed <int64>        Default seed, default: -1\n"
        "  --guidance <float>        Default distilled guidance, default: 3.5\n"
        "  --cfg-scale <float>       Default CFG scale, default: 1.0\n"
        "  --flow-shift <float>      Default flow scheduler shift, default: model default\n"
        "  --cache <mode>            Default cache mode: off, easycache, ucache, dbcache, taylorseer, cache-dit\n"
        "  --help                    Show this help\n",
        prog,
        prog);
}

int parse_int(const char* text, int fallback) {
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    return end != text ? static_cast<int>(value) : fallback;
}

int64_t parse_i64(const char* text, int64_t fallback) {
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    return end != text ? static_cast<int64_t>(value) : fallback;
}

float parse_float(const char* text, float fallback) {
    if (text == nullptr) {
        return fallback;
    }
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    return end != text ? value : fallback;
}

bool has_text(const char* text) {
    return text != nullptr && text[0] != '\0';
}

std::string display_model_path(const ld_context_params_t& params) {
    if (has_text(params.model_path)) {
        return params.model_path;
    }
    if (has_text(params.diffusion_model_path)) {
        return params.diffusion_model_path;
    }
    return "";
}

struct Args {
    LightDitServerParams server;
    ld_context_params_t context = {};
    LightDitDefaultGenerationParams defaults;
    const char* backend = nullptr;
};

bool parse_args(int argc, char** argv, Args* args) {
    if (args == nullptr) {
        return false;
    }

    ld_context_params_init(&args->context);

    for (int i = 1; i < argc; ++i) {
        const char* key = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(key, "--host") == 0 || std::strcmp(key, "--listen") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->server.host = value;
        } else if (std::strcmp(key, "--port") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->server.port = parse_int(value, args->server.port);
        } else if (std::strcmp(key, "--verbose") == 0) {
            args->server.verbose = true;
        } else if (std::strcmp(key, "--model") == 0) {
            args->context.model_path = require_value(key);
        } else if (std::strcmp(key, "--diffusion-model") == 0) {
            args->context.diffusion_model_path = require_value(key);
        } else if (std::strcmp(key, "--vae") == 0) {
            args->context.vae_path = require_value(key);
        } else if (std::strcmp(key, "--clip_l") == 0) {
            args->context.clip_l_path = require_value(key);
        } else if (std::strcmp(key, "--clip_g") == 0) {
            args->context.clip_g_path = require_value(key);
        } else if (std::strcmp(key, "--t5xxl") == 0) {
            args->context.t5xxl_path = require_value(key);
        } else if (std::strcmp(key, "--backend") == 0) {
            args->backend = require_value(key);
        } else if (std::strcmp(key, "--gpu") == 0) {
            args->backend = "gpu";
        } else if (std::strcmp(key, "--threads") == 0 || std::strcmp(key, "-t") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->context.n_threads = parse_int(value, args->context.n_threads);
        } else if (std::strcmp(key, "--width") == 0 || std::strcmp(key, "-W") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.width = parse_int(value, args->defaults.width);
        } else if (std::strcmp(key, "--height") == 0 || std::strcmp(key, "-H") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.height = parse_int(value, args->defaults.height);
        } else if (std::strcmp(key, "--steps") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.steps = parse_int(value, args->defaults.steps);
        } else if (std::strcmp(key, "--seed") == 0 || std::strcmp(key, "-s") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.seed = parse_i64(value, args->defaults.seed);
        } else if (std::strcmp(key, "--guidance") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.distilled_guidance = parse_float(value, args->defaults.distilled_guidance);
        } else if (std::strcmp(key, "--cfg-scale") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.cfg_scale = parse_float(value, args->defaults.cfg_scale);
        } else if (std::strcmp(key, "--flow-shift") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            args->defaults.flow_shift = parse_float(value, args->defaults.flow_shift);
        } else if (std::strcmp(key, "--cache") == 0 || std::strcmp(key, "--cache-mode") == 0) {
            const char* value = require_value(key);
            if (value == nullptr) return false;
            if (!ld_cache_mode_from_string(value, &args->defaults.cache_mode)) {
                std::fprintf(stderr, "unsupported cache mode: %s\n", value);
                return false;
            }
        } else if (std::strcmp(key, "--help") == 0 || std::strcmp(key, "-h") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", key);
            return false;
        }
    }

    const bool has_full_model = has_text(args->context.model_path);
    const bool has_components =
        has_text(args->context.diffusion_model_path) &&
        has_text(args->context.vae_path) &&
        has_text(args->context.clip_l_path) &&
        has_text(args->context.t5xxl_path);

    if (!has_full_model && !has_components) {
        std::fprintf(stderr, "--model or the full --diffusion-model/--vae/--clip_l/--t5xxl set is required\n");
        return false;
    }
    if (args->server.port <= 0 || args->server.port > 65535) {
        std::fprintf(stderr, "port must be in 1..65535\n");
        return false;
    }
    if (args->defaults.width <= 0 || args->defaults.height <= 0) {
        std::fprintf(stderr, "default width and height must be positive\n");
        return false;
    }
    if (args->defaults.steps <= 0) {
        std::fprintf(stderr, "default steps must be positive\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    Args args;
    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

    if (has_text(args.backend)) {
        setenv("LDIT_BACKEND", args.backend, 1);
    }

    std::fprintf(stderr, "loading light-dit model: %s\n", display_model_path(args.context).c_str());
    ld_context_t* ctx = ld_create_context(&args.context);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create light-dit context\n");
        return 2;
    }

    std::mutex ctx_mutex;
    LightDitServerRuntime runtime;
    runtime.ctx = ctx;
    runtime.ctx_mutex = &ctx_mutex;
    runtime.server = &args.server;
    runtime.context = &args.context;
    runtime.defaults = &args.defaults;
    runtime.display_model_path = display_model_path(args.context);

    httplib::Server server;
    server.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    if (args.server.verbose) {
        server.set_logger([](const httplib::Request& req, const httplib::Response& res) {
            std::fprintf(stderr, "%s %s -> %d\n", req.method.c_str(), req.path.c_str(), res.status);
        });
    }

    register_lightdit_routes(server, runtime);

    std::fprintf(stderr, "light-dit server listening on http://%s:%d\n",
                 args.server.host.c_str(),
                 args.server.port);
    const bool ok = server.listen(args.server.host, args.server.port);

    ld_free_context(ctx);
    return ok ? 0 : 3;
}
