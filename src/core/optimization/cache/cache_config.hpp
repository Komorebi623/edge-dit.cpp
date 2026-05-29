#pragma once

#include <string>
#include <vector>

#include "core/optimization/cache/cache_types.hpp"

namespace edgedit {
namespace cache {

struct DBCacheConfig {
    bool enabled = false;
    int fn_compute_blocks = 8;
    int bn_compute_blocks = 0;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
    float residual_diff_threshold = 0.08f;
    int max_warmup_steps = 8;
    int max_cached_steps = -1;
    int max_continuous_cached_steps = -1;
    float max_accumulated_residual_diff = -1.0f;
    std::vector<int> steps_computation_mask;
    bool scm_policy_dynamic = true;
};

struct TaylorSeerConfig {
    bool enabled = false;
    int n_derivatives = 1;
    int max_warmup_steps = 2;
    int skip_interval_steps = 1;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
};

struct EasyCacheConfig {
    bool enabled = false;
    float reuse_threshold = 0.2f;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
};

struct UCacheConfig {
    bool enabled = false;
    float reuse_threshold = 1.0f;
    float start_percent = 0.15f;
    float end_percent = 0.95f;
    float error_decay_rate = 1.0f;
    bool use_relative_threshold = true;
    bool adaptive_threshold = true;
    float early_step_multiplier = 0.5f;
    float late_step_multiplier = 1.5f;
    float relative_norm_gain = 1.6f;
    bool reset_error_on_compute = true;
};

struct CacheConfig {
    CacheMode mode = CacheMode::Disabled;
    EasyCacheConfig easycache;
    UCacheConfig ucache;
    DBCacheConfig dbcache;
    TaylorSeerConfig taylorseer;
    int double_fn_blocks = -1;
    int double_bn_blocks = -1;
    int single_fn_blocks = -1;
    int single_bn_blocks = -1;
};

std::vector<int> parse_steps_computation_mask(const char* mask);
CacheConfig cache_config_from_sample_params(const ed_sample_params_t& params);

}  // namespace cache
}  // namespace edgedit
