#pragma once

#include <memory>
#include <string>

#include "edge-dit.h"
#include "dit_models/pipelines/dit_pipeline.hpp"
#include "runtime/model_runtime.hpp"
#include "runtime/model_loader.h"

namespace edgedit {

class EdgeDitEngine final {
public:
    EdgeDitEngine() = default;
    ~EdgeDitEngine() = default;

    EdgeDitEngine(const EdgeDitEngine&) = delete;
    EdgeDitEngine& operator=(const EdgeDitEngine&) = delete;

    bool init(const ed_ctx_params_t* params);

    ed_status_t generate_image(
        const ed_image_generation_params_t* params,
        ed_image_batch_t* out
    );

    ed_status_t generate_video(
        const ed_video_generation_params_t* params,
        ed_video_t* out
    );

    bool supports_image_generation() const;
    bool supports_video_generation() const;

    sample_method_t get_default_sample_method() const;
    scheduler_t get_default_scheduler(sample_method_t method) const;

    SDVersion version() const {
        return dit_pipeline_ ? dit_pipeline_->version() : VERSION_COUNT;
    }

    const std::string& last_error() const {
        return last_error_;
    }

private:
    ed_ctx_params_t ctx_params_ = {};
    std::unique_ptr<ModelRuntime> runtime_;
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<DiTPipeline> dit_pipeline_;

    std::string last_error_;
    void set_error(const std::string& msg);
};

} // namespace edgedit
