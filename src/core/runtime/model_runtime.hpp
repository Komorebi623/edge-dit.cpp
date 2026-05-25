#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "backend/ggml/ggml_extend.hpp"
#include "ggml-backend.h"
#include "light-dit.h"
#include "utils/rng.hpp"

namespace lightdit {

using ld_ctx_params_t = ld_context_params_t;
using sample_method_t = ld_sampler_t;
using scheduler_t = ld_scheduler_t;

constexpr sample_method_t EULER_SAMPLE_METHOD = LD_SAMPLER_EULER;
constexpr sample_method_t EULER_A_SAMPLE_METHOD = LD_SAMPLER_EULER_A;
constexpr sample_method_t LCM_SAMPLE_METHOD = LD_SAMPLER_LCM;
constexpr sample_method_t TCD_SAMPLE_METHOD = LD_SAMPLER_TCD;
constexpr sample_method_t DDIM_TRAILING_SAMPLE_METHOD = LD_SAMPLER_DDIM_TRAILING;

constexpr scheduler_t DISCRETE_SCHEDULER = LD_SCHEDULER_DISCRETE;
constexpr scheduler_t SIMPLE_SCHEDULER = LD_SCHEDULER_SIMPLE;
constexpr scheduler_t LCM_SCHEDULER = LD_SCHEDULER_LCM;

struct RuntimeBackends {
    ggml_backend_t backend = nullptr;
    ggml_backend_t clip_backend = nullptr;
    ggml_backend_t vae_backend = nullptr;
    ggml_backend_t control_net_backend = nullptr;

    bool clip_owns_backend = false;
    bool vae_owns_backend = false;
    bool control_net_owns_backend = false;
};

class ModelRuntime final {
public:
    ModelRuntime() = default;
    ~ModelRuntime();

    ModelRuntime(const ModelRuntime&) = delete;
    ModelRuntime& operator=(const ModelRuntime&) = delete;

    bool init(const ld_context_params_t& params, std::string* error);
    bool init(const ld_context_params_t* params, std::string* error);
    void reset();

    bool ready() const { return ready_; }
    bool is_ready() const { return ready_; }

    int n_threads() const { return n_threads_; }
    bool use_mmap() const { return use_mmap_; }
    bool offload_params_to_cpu() const { return offload_params_to_cpu_; }
    bool free_params_immediately() const { return free_params_immediately_; }
    float max_vram() const { return max_vram_; }
    size_t max_graph_vram_bytes() const { return max_graph_vram_bytes_; }
    bool flash_attention() const { return flash_attention_; }
    bool diffusion_flash_attention() const { return diffusion_flash_attention_; }
    bool circular_x() const { return circular_x_; }
    bool circular_y() const { return circular_y_; }

    ggml_backend_t backend() const { return backends_.backend; }
    ggml_backend_t clip_backend() const { return backends_.clip_backend; }
    ggml_backend_t vae_backend() const { return backends_.vae_backend; }
    ggml_backend_t control_net_backend() const { return backends_.control_net_backend; }

    RNG& rng() { return *rng_; }
    RNG& sampler_rng() { return *sampler_rng_; }
    std::shared_ptr<RNG> rng_ptr() const { return rng_; }
    std::shared_ptr<RNG> sampler_rng_ptr() const { return sampler_rng_; }

private:
    bool ready_ = false;

    int n_threads_ = 0;
    bool use_mmap_ = false;
    bool offload_params_to_cpu_ = false;
    bool free_params_immediately_ = false;

    float max_vram_ = 0.0f;
    size_t max_graph_vram_bytes_ = 0;

    bool flash_attention_ = false;
    bool diffusion_flash_attention_ = false;
    bool circular_x_ = false;
    bool circular_y_ = false;

    RuntimeBackends backends_;

    std::shared_ptr<RNG> rng_;
    std::shared_ptr<RNG> sampler_rng_;

    bool init_threads(const ld_context_params_t& params, std::string* error);
    bool init_flags(const ld_context_params_t& params, std::string* error);
    bool init_backends(const ld_context_params_t& params, std::string* error);
    bool init_rng(const ld_context_params_t& params, std::string* error);

    void release_backends();
    bool fail(std::string* error, const std::string& msg);
};

} // namespace lightdit
