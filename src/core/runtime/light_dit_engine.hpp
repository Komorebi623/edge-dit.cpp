#pragma once

#include <memory>
#include <string>

#include "light-dit.h"
#include "runtime/model_runtime.hpp"
#include "runtime/model_loader.h"
#include "runtime/dit_pipeline.hpp"

namespace lightdit {

class LightDitEngine final {
public:
    LightDitEngine() = default;
    ~LightDitEngine() = default;

    LightDitEngine(const LightDitEngine&) = delete;
    LightDitEngine& operator=(const LightDitEngine&) = delete;

    bool init(const ld_ctx_params_t* params);

    ld_status_t generate_image(
        const ld_image_generation_params_t* params,
        ld_image_batch_t* out
    );

    ld_status_t generate_video(
        const ld_video_generation_params_t* params,
        ld_video_t* out
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
    ld_ctx_params_t ctx_params_ = {};
    std::unique_ptr<ModelRuntime> runtime_;
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<DiTPipeline> dit_pipeline_;

    std::string last_error_;
    void reset_components();
    void set_error(const std::string& msg);
};

} // namespace lightdit
