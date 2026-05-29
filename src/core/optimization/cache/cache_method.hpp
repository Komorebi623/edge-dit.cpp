#pragma once

#include <memory>

#include "core/optimization/cache/cache_config.hpp"
#include "core/optimization/cache/cache_types.hpp"
#include "utils/tensor.hpp"

namespace edgedit {
namespace cache {

struct CacheForwardContext {
    CacheStepInfo step;
    CacheBranch branch = CacheBranch::Main;
    const void* condition_key = nullptr;
};

struct CacheMethod {
    virtual ~CacheMethod() = default;

    virtual const char* name() const = 0;
    virtual bool enabled() const = 0;

    virtual void init(const CacheConfig& config,
                      const CacheModelSpec& model_spec,
                      const std::vector<float>& sigmas) = 0;

    virtual void begin_step(const CacheStepInfo& step) = 0;
    virtual bool before_forward(const CacheForwardContext& frame,
                                const sd::Tensor<float>& input,
                                sd::Tensor<float>* output) = 0;
    virtual void after_forward(const CacheForwardContext& frame,
                               const sd::Tensor<float>& input,
                               const sd::Tensor<float>& output) = 0;
    virtual CacheRegionPlan plan_region(const CacheRegionFrame& frame) = 0;
    virtual void end_step(const CacheStepInfo& step) = 0;
    virtual void log_summary(size_t total_steps) const = 0;
};

std::unique_ptr<CacheMethod> create_cache_method(CacheMode mode);

}  // namespace cache
}  // namespace edgedit
