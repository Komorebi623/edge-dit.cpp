#pragma once

#include <memory>
#include <string>

#include "core/runtime/model.h"
#include "runtime/model_runtime.hpp"

namespace lightdit {

class DiTPipeline final {
public:
    DiTPipeline() = default;
    ~DiTPipeline() = default;

    DiTPipeline(const DiTPipeline&) = delete;
    DiTPipeline& operator=(const DiTPipeline&) = delete;

    bool prepare(const ld_context_params_t& params,
             ModelRuntime& runtime,
             const ModelLoader& loader,
             ModelLoader::TensorMap* tensors,
             ModelLoader::IgnoreTensorSet* ignore_tensors,
             std::string* error);

    void mark_ready();

    ld_status_t generate_image(const ld_image_generation_params_t* params,
                               ld_image_batch_t* out,
                               std::string* error);

    ld_status_t generate_video(const ld_video_generation_params_t* params,
                               ld_video_t* out,
                               std::string* error);

    SDVersion version() const { return version_; }
    bool ready() const { return ready_; }

    bool supports_image_generation() const;
    bool supports_video_generation() const;

    ld_sampler_t default_sample_method() const;
    ld_scheduler_t default_scheduler(ld_sampler_t method) const;

private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    ModelLoader* loader_ = nullptr;
    SDVersion version_ = VERSION_COUNT;

    // Stage 1 bridge: LDModel currently owns the Flux components and generation
    // implementation. As the old GGML init is migrated, these components should
    // be flattened into DiTPipeline directly.
    std::unique_ptr<LDModel> model_;

    bool build_components(const ld_context_params_t& params, std::string* error);
    bool bind_weights(std::string* error);
    bool validate_image_params(const ld_image_generation_params_t* params, std::string* error) const;
    bool validate_video_params(const ld_video_generation_params_t* params, std::string* error) const;
    bool prepare_weights(const ModelLoader& loader,
                     ModelLoader::TensorMap* tensors,
                     ModelLoader::IgnoreTensorSet* ignore_tensors,
                     std::string* error);
};

} // namespace lightdit
