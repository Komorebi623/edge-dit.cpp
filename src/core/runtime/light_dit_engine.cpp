#include "runtime/light_dit_engine.hpp"

#include <exception>

#include "utils/util.h"

namespace lightdit {

bool LightDitEngine::init(const ld_ctx_params_t* params) {
    last_error_.clear();

    if (params == nullptr) {
        set_error("LightDitEngine::init got null params");
        return false;
    }

    ctx_params_ = *params;

    dit_pipeline_.reset();
    model_loader_.reset();
    runtime_.reset();

    auto cleanup = [&]() {
        dit_pipeline_.reset();
        model_loader_.reset();
        runtime_.reset();
    };

    try {
        runtime_      = std::make_unique<ModelRuntime>();
        model_loader_ = std::make_unique<ModelLoader>();
        dit_pipeline_ = std::make_unique<DiTPipeline>();
    } catch (const std::exception& e) {
        set_error(std::string("failed to allocate engine components: ") + e.what());
        cleanup();
        return false;
    }

    if (!runtime_->init(ctx_params_, &last_error_)) {
        set_error(last_error_.empty() ? "ModelRuntime::init failed" : last_error_);
        cleanup();
        return false;
    }

    auto prepare_tensors = [this](const ModelLoader& loader,
                                ModelLoader::TensorMap* tensors,
                                ModelLoader::IgnoreTensorSet* ignore_tensors,
                                std::string* error) -> bool {
        (void)tensors;
        (void)ignore_tensors;

        if (runtime_ == nullptr || dit_pipeline_ == nullptr) {
            if (error != nullptr) {
                *error = "LightDitEngine::init got null runtime or pipeline";
            }
            return false;
        }

        return dit_pipeline_->prepare(ctx_params_,
                              *runtime_,
                              loader,
                              tensors,
                              ignore_tensors,
                              error);
    };

    if (!model_loader_->init(ctx_params_,
                             prepare_tensors,
                             runtime_->n_threads(),
                             runtime_->use_mmap(),
                             &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::init failed" : last_error_);
        cleanup();
        return false;
    }

    LOG_INFO("LightDitEngine initialized successfully, version=%s",
             ld_version_name(dit_pipeline_->version()));

    dit_pipeline_->mark_ready();
    return true;
}

ld_status_t LightDitEngine::generate_image(const ld_image_generation_params_t* params,
                                           ld_image_batch_t* out) {
    last_error_.clear();
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }

    if (params == nullptr || out == nullptr) {
        set_error("LightDitEngine::generate_image got null params or out");
        return LD_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_ == nullptr || !runtime_->ready() || dit_pipeline_ == nullptr || !dit_pipeline_->ready()) {
        set_error("engine is not initialized");
        return LD_STATUS_MODEL_LOAD_FAILED;
    }
    if (!supports_image_generation()) {
        set_error("current model/version does not support image generation");
        return LD_STATUS_UNSUPPORTED;
    }

    ld_status_t status = dit_pipeline_->generate_image(params, out, &last_error_);
    if (status != LD_STATUS_OK) {
        if (last_error_.empty()) {
            set_error("image generation failed");
        } else {
            set_error(last_error_);
        }
        if (out != nullptr) {
            ld_free_image_batch(out);
        }
        return status;
    }

    if (out->images == nullptr || out->count <= 0) {
        ld_free_image_batch(out);
        set_error("DiT pipeline returned empty image batch");
        return LD_STATUS_GENERATION_FAILED;
    }
    return LD_STATUS_OK;
}

ld_status_t LightDitEngine::generate_video(const ld_video_generation_params_t* params,
                                           ld_video_t* out) {
    last_error_.clear();
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }

    if (params == nullptr || out == nullptr) {
        set_error("LightDitEngine::generate_video got null params or out");
        return LD_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_ == nullptr || !runtime_->ready() || dit_pipeline_ == nullptr || !dit_pipeline_->ready()) {
        set_error("engine is not initialized");
        return LD_STATUS_MODEL_LOAD_FAILED;
    }
    if (!supports_video_generation()) {
        set_error("current model/version does not support video generation");
        return LD_STATUS_UNSUPPORTED;
    }

    ld_status_t status = dit_pipeline_->generate_video(params, out, &last_error_);
    if (status != LD_STATUS_OK) {
        if (last_error_.empty()) {
            set_error("video generation failed");
        } else {
            set_error(last_error_);
        }
        if (out != nullptr) {
            ld_free_video(out);
        }
        return status;
    }

    if (out->frames == nullptr || out->frame_count <= 0) {
        ld_free_video(out);
        set_error("DiT pipeline returned empty video");
        return LD_STATUS_GENERATION_FAILED;
    }
    return LD_STATUS_OK;
}

bool LightDitEngine::supports_image_generation() const {
    return dit_pipeline_ != nullptr && dit_pipeline_->supports_image_generation();
}

bool LightDitEngine::supports_video_generation() const {
    return dit_pipeline_ != nullptr && dit_pipeline_->supports_video_generation();
}

sample_method_t LightDitEngine::get_default_sample_method() const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->default_sample_method() : EULER_A_SAMPLE_METHOD;
}

scheduler_t LightDitEngine::get_default_scheduler(sample_method_t method) const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->default_scheduler(method) : DISCRETE_SCHEDULER;
}

void LightDitEngine::set_error(const std::string& msg) {
    last_error_ = msg;
    LOG_ERROR("%s", last_error_.c_str());
}

} // namespace lightdit
