#include "runtime/edge_dit_engine.hpp"

#include <exception>

#include "utils/util.h"

namespace edgedit {

bool EdgeDitEngine::init(const ed_ctx_params_t* params) {
    last_error_.clear();

    if (params == nullptr) {
        set_error("EdgeDitEngine::init got null params");
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

    if (!model_loader_->load_model_files(ctx_params_, &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::load_model_files failed" : last_error_);
        cleanup();
        return false;
    }

    if (!model_loader_->finalize_names_and_version(&last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::finalize_names_and_version failed" : last_error_);
        cleanup();
        return false;
    }

    if (!model_loader_->apply_dtype_policy(ctx_params_, &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::apply_dtype_policy failed" : last_error_);
        cleanup();
        return false;
    }

    dit_pipeline_ = create_dit_pipeline(model_loader_->version(), &last_error_);
    if (dit_pipeline_ == nullptr) {
        set_error(last_error_.empty() ? "failed to create DiT pipeline" : last_error_);
        cleanup();
        return false;
    }

    PipelineTensorRegistry registry;
    if (!dit_pipeline_->prepare(ctx_params_,
                                *runtime_,
                                *model_loader_,
                                registry,
                                &last_error_)) {
        set_error(last_error_.empty() ? "DiT pipeline prepare failed" : last_error_);
        cleanup();
        return false;
    }

    if (!model_loader_->bind_weights(registry.tensors(),
                                     registry.ignore_tensors(),
                                     runtime_->n_threads(),
                                     runtime_->use_mmap(),
                                     &last_error_)) {
        set_error(last_error_.empty() ? "ModelLoader::bind_weights failed" : last_error_);
        cleanup();
        return false;
    }

    model_loader_->log_weight_stats();
    dit_pipeline_->mark_ready();

    LOG_INFO("EdgeDitEngine initialized successfully, version=%s",
             ed_version_name(dit_pipeline_->version()));
    return true;
}

ed_status_t EdgeDitEngine::generate_image(const ed_image_generation_params_t* params,
                                           ed_image_batch_t* out) {
    last_error_.clear();
    if (out != nullptr) {
        out->images = nullptr;
        out->count = 0;
    }

    if (params == nullptr || out == nullptr) {
        set_error("EdgeDitEngine::generate_image got null params or out");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_ == nullptr || !runtime_->ready() || dit_pipeline_ == nullptr || !dit_pipeline_->ready()) {
        set_error("engine is not initialized");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (!supports_image_generation()) {
        set_error("current model/version does not support image generation");
        return ED_STATUS_UNSUPPORTED;
    }

    ed_status_t status = dit_pipeline_->generate_image(params, out, &last_error_);
    if (status != ED_STATUS_OK) {
        if (last_error_.empty()) {
            set_error("image generation failed");
        } else {
            set_error(last_error_);
        }
        if (out != nullptr) {
            ed_free_image_batch(out);
        }
        return status;
    }

    if (out->images == nullptr || out->count <= 0) {
        ed_free_image_batch(out);
        set_error("DiT pipeline returned empty image batch");
        return ED_STATUS_GENERATION_FAILED;
    }
    return ED_STATUS_OK;
}

ed_status_t EdgeDitEngine::generate_video(const ed_video_generation_params_t* params,
                                           ed_video_t* out) {
    last_error_.clear();
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }

    if (params == nullptr || out == nullptr) {
        set_error("EdgeDitEngine::generate_video got null params or out");
        return ED_STATUS_INVALID_ARGUMENT;
    }
    if (runtime_ == nullptr || !runtime_->ready() || dit_pipeline_ == nullptr || !dit_pipeline_->ready()) {
        set_error("engine is not initialized");
        return ED_STATUS_MODEL_LOAD_FAILED;
    }
    if (!supports_video_generation()) {
        set_error("current model/version does not support video generation");
        return ED_STATUS_UNSUPPORTED;
    }

    ed_status_t status = dit_pipeline_->generate_video(params, out, &last_error_);
    if (status != ED_STATUS_OK) {
        if (last_error_.empty()) {
            set_error("video generation failed");
        } else {
            set_error(last_error_);
        }
        if (out != nullptr) {
            ed_free_video(out);
        }
        return status;
    }

    if (out->frames == nullptr || out->frame_count <= 0) {
        ed_free_video(out);
        set_error("DiT pipeline returned empty video");
        return ED_STATUS_GENERATION_FAILED;
    }
    return ED_STATUS_OK;
}

bool EdgeDitEngine::supports_image_generation() const {
    return dit_pipeline_ != nullptr && dit_pipeline_->supports_image_generation();
}

bool EdgeDitEngine::supports_video_generation() const {
    return dit_pipeline_ != nullptr && dit_pipeline_->supports_video_generation();
}

sample_method_t EdgeDitEngine::get_default_sample_method() const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->default_sample_method() : EULER_A_SAMPLE_METHOD;
}

scheduler_t EdgeDitEngine::get_default_scheduler(sample_method_t method) const {
    return dit_pipeline_ != nullptr ? dit_pipeline_->default_scheduler(method) : DISCRETE_SCHEDULER;
}

void EdgeDitEngine::set_error(const std::string& msg) {
    last_error_ = msg;
    LOG_ERROR("%s", last_error_.c_str());
}

} // namespace edgedit
