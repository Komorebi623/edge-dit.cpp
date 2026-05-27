#include "core/optimization/cache/cache_runtime.hpp"

#include "utils/util.h"

namespace lightdit {
namespace cache {

bool CacheRuntime::init(const ld_sample_params_t& sample_params,
                        SDVersion version,
                        const std::vector<float>& sigmas) {
    config_ = cache_config_from_sample_params(sample_params);
    model_spec_ = cache_model_spec_for_version(version);
    method_ = create_cache_method(config_.mode);
    if (method_ == nullptr) {
        return false;
    }

    method_->init(config_, model_spec_, sigmas);
    if (!method_->enabled()) {
        method_.reset();
        return false;
    }

    LOG_INFO("cache enabled: mode=%s model=%s regions=%zu",
             cache_mode_name(config_.mode),
             model_spec_.model_name.c_str(),
             model_spec_.regions.size());
    return true;
}

void CacheRuntime::begin_step(const CacheStepInfo& step) {
    current_step_ = step;
    if (method_ != nullptr) {
        method_->begin_step(step);
    }
}

bool CacheRuntime::before_forward(CacheBranch branch,
                                  const void* condition_key,
                                  const sd::Tensor<float>& input,
                                  sd::Tensor<float>* output) {
    if (method_ == nullptr) {
        return false;
    }
    CacheForwardContext frame;
    frame.step = current_step_;
    frame.branch = branch;
    frame.condition_key = condition_key;
    return method_->before_forward(frame, input, output);
}

void CacheRuntime::after_forward(CacheBranch branch,
                                 const void* condition_key,
                                 const sd::Tensor<float>& input,
                                 const sd::Tensor<float>& output) {
    if (method_ == nullptr) {
        return;
    }
    CacheForwardContext frame;
    frame.step = current_step_;
    frame.branch = branch;
    frame.condition_key = condition_key;
    method_->after_forward(frame, input, output);
}

CacheRegionPlan CacheRuntime::plan_region(const CacheRegionFrame& frame) {
    if (method_ == nullptr) {
        return {};
    }
    return method_->plan_region(frame);
}

void CacheRuntime::end_step(const CacheStepInfo& step) {
    if (method_ != nullptr) {
        method_->end_step(step);
    }
}

void CacheRuntime::log_summary(size_t total_steps) const {
    if (method_ != nullptr) {
        method_->log_summary(total_steps);
    }
}

}  // namespace cache
}  // namespace lightdit
