#pragma once

#include <memory>
#include <vector>

#include "core/optimization/cache/cache_method.hpp"
#include "core/optimization/cache/cache_types.hpp"

namespace edgedit {
namespace cache {

class CacheRuntime {
public:
    CacheRuntime() = default;

    bool init(const ed_sample_params_t& sample_params,
              SDVersion version,
              const std::vector<float>& sigmas);

    bool enabled() const {
        return method_ != nullptr && method_->enabled();
    }

    CacheMode mode() const { return config_.mode; }
    const CacheModelSpec& model_spec() const { return model_spec_; }

    void begin_step(const CacheStepInfo& step);
    bool before_forward(CacheBranch branch,
                        const void* condition_key,
                        const sd::Tensor<float>& input,
                        sd::Tensor<float>* output);
    void after_forward(CacheBranch branch,
                       const void* condition_key,
                       const sd::Tensor<float>& input,
                       const sd::Tensor<float>& output);
    CacheRegionPlan plan_region(const CacheRegionFrame& frame);
    void end_step(const CacheStepInfo& step);
    void log_summary(size_t total_steps) const;

private:
    CacheConfig config_;
    CacheModelSpec model_spec_;
    CacheStepInfo current_step_;
    std::unique_ptr<CacheMethod> method_;
};

}  // namespace cache
}  // namespace edgedit
