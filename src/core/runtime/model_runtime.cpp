#include "runtime/model_runtime.hpp"

#include <algorithm>
#include <thread>

#include "utils/rng_philox.hpp"
#include "utils/util.h"

namespace lightdit {

ModelRuntime::~ModelRuntime() {
    reset();
}

bool ModelRuntime::init(const ld_context_params_t* params, std::string* error) {
    if (params == nullptr) {
        return fail(error, "ModelRuntime::init got null params");
    }
    return init(*params, error);
}

bool ModelRuntime::init(const ld_context_params_t& params, std::string* error) {
    reset();
    ggml_log_set(ggml_log_callback_default, nullptr);

    if (!init_threads(params, error)) {
        return false;
    }
    if (!init_flags(params, error)) {
        return false;
    }
    if (!init_rng(params, error)) {
        return false;
    }
    if (!init_backends(params, error)) {
        return false;
    }

    ready_ = true;
    return true;
}

bool ModelRuntime::init_threads(const ld_context_params_t& params, std::string* error) {
    (void)error;
    n_threads_ = params.n_threads;
    if (n_threads_ <= 0) {
        n_threads_ = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    return true;
}

bool ModelRuntime::init_flags(const ld_context_params_t& params, std::string* error) {
    (void)error;
    use_mmap_ = params.use_mmap;
    offload_params_to_cpu_ = params.offload_params_to_cpu;
    free_params_immediately_ = false;
    max_vram_ = params.max_vram_gb;
    max_graph_vram_bytes_ = max_vram_ <= 0.0f
                                 ? 0
                                 : static_cast<size_t>(static_cast<double>(max_vram_) * 1024.0 * 1024.0 * 1024.0);
    flash_attention_ = params.flash_attention;
    diffusion_flash_attention_ = params.diffusion_flash_attention;
    circular_x_ = false;
    circular_y_ = false;
    return true;
}

bool ModelRuntime::init_rng(const ld_context_params_t& params, std::string* error) {
    (void)params;
    (void)error;
    rng_ = std::make_shared<PhiloxRNG>();
    sampler_rng_ = rng_;
    return true;
}

bool ModelRuntime::init_backends(const ld_context_params_t& params, std::string* error) {
    backends_.backend = init_named_backend();
    if (backends_.backend == nullptr) {
        return fail(error, "failed to initialize default ggml backend");
    }
    LOG_INFO("default backend: %s", ggml_backend_name(backends_.backend));

    backends_.clip_backend = backends_.backend;
    if (params.keep_text_encoder_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.clip_backend = ggml_backend_cpu_init();
        if (backends_.clip_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for text encoder");
        }
        backends_.clip_owns_backend = true;
        LOG_INFO("text encoder backend: CPU");
    }

    backends_.vae_backend = backends_.backend;
    if (params.keep_vae_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.vae_backend = ggml_backend_cpu_init();
        if (backends_.vae_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for VAE");
        }
        backends_.vae_owns_backend = true;
        LOG_INFO("VAE backend: CPU");
    }

    backends_.control_net_backend = backends_.backend;
    if (params.keep_control_net_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.control_net_backend = ggml_backend_cpu_init();
        if (backends_.control_net_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for ControlNet");
        }
        backends_.control_net_owns_backend = true;
        LOG_INFO("ControlNet backend: CPU");
    }

    return true;
}

void ModelRuntime::reset() {
    ready_ = false;
    rng_.reset();
    sampler_rng_.reset();
    release_backends();

    n_threads_ = 0;
    use_mmap_ = false;
    offload_params_to_cpu_ = false;
    free_params_immediately_ = false;
    max_vram_ = 0.0f;
    max_graph_vram_bytes_ = 0;
    flash_attention_ = false;
    diffusion_flash_attention_ = false;
    circular_x_ = false;
    circular_y_ = false;
}

void ModelRuntime::release_backends() {
    if (backends_.control_net_owns_backend && backends_.control_net_backend != nullptr) {
        ggml_backend_free(backends_.control_net_backend);
    }
    if (backends_.vae_owns_backend && backends_.vae_backend != nullptr) {
        ggml_backend_free(backends_.vae_backend);
    }
    if (backends_.clip_owns_backend && backends_.clip_backend != nullptr) {
        ggml_backend_free(backends_.clip_backend);
    }
    if (backends_.backend != nullptr) {
        ggml_backend_free(backends_.backend);
    }
    backends_ = {};
}

bool ModelRuntime::fail(std::string* error, const std::string& msg) {
    if (error != nullptr) {
        *error = msg;
    }
    LOG_ERROR("%s", msg.c_str());
    reset();
    return false;
}

} // namespace lightdit
