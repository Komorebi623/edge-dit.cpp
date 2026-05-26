#pragma once

#include <memory>
#include <string>

#include "light-dit.h"
#include "runtime/model_loader.h"
#include "runtime/model_runtime.hpp"

namespace lightdit {

class PipelineTensorRegistry {
public:
    using TensorMap = ModelLoader::TensorMap;
    using IgnoreTensorSet = ModelLoader::IgnoreTensorSet;

    void clear() {
        tensors_.clear();
        ignore_tensors_.clear();
    }

    void add(const std::string& name, ggml_tensor* tensor) {
        tensors_[name] = tensor;
    }

    void ignore_prefix(const std::string& prefix) {
        ignore_tensors_.insert(prefix);
    }

    TensorMap& tensors() { return tensors_; }
    const TensorMap& tensors() const { return tensors_; }

    IgnoreTensorSet& ignore_tensors() { return ignore_tensors_; }
    const IgnoreTensorSet& ignore_tensors() const { return ignore_tensors_; }

private:
    TensorMap tensors_;
    IgnoreTensorSet ignore_tensors_;
};

class DiTPipeline {
public:
    virtual ~DiTPipeline() = default;

    DiTPipeline(const DiTPipeline&) = delete;
    DiTPipeline& operator=(const DiTPipeline&) = delete;

    virtual const char* name() const = 0;

    virtual bool prepare(const ld_context_params_t& params,
                         ModelRuntime& runtime,
                         const ModelLoader& loader,
                         PipelineTensorRegistry& registry,
                         std::string* error) = 0;

    virtual void mark_ready() = 0;

    virtual ld_status_t generate_image(const ld_image_generation_params_t* params,
                                       ld_image_batch_t* out,
                                       std::string* error) = 0;

    virtual ld_status_t generate_video(const ld_video_generation_params_t* params,
                                       ld_video_t* out,
                                       std::string* error) = 0;

    virtual SDVersion version() const = 0;
    virtual bool ready() const = 0;

    virtual bool supports_image_generation() const = 0;
    virtual bool supports_video_generation() const = 0;

    virtual ld_sampler_t default_sample_method() const = 0;
    virtual ld_scheduler_t default_scheduler(ld_sampler_t method) const = 0;

protected:
    DiTPipeline() = default;
};

std::unique_ptr<DiTPipeline> create_dit_pipeline(SDVersion version,
                                                 std::string* error);

}  // namespace lightdit
