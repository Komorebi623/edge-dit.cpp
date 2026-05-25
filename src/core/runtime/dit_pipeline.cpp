#include "runtime/dit_pipeline.hpp"

#include <cstdlib>
#include <new>

#include "utils/util.h"

namespace lightdit {

    bool DiTPipeline::prepare(const ld_context_params_t& params,
                          ModelRuntime& runtime,
                          const ModelLoader& loader,
                          ModelLoader::TensorMap* tensors,
                          ModelLoader::IgnoreTensorSet* ignore_tensors,
                          std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    loader_ = nullptr;
    version_ = loader.version();
    model_.reset();

    if (tensors == nullptr || ignore_tensors == nullptr) {
        if (error != nullptr) {
            *error = "DiTPipeline::prepare got null tensor containers";
        }
        return false;
    }

    if (version_ == VERSION_COUNT) {
        if (error != nullptr) {
            *error = "DiTPipeline::prepare got unknown model version";
        }
        return false;
    }

    model_ = std::make_unique<LDModel>(version_);

    model_->build_manifest(loader);

    if (!model_->validate(error)) {
        return false;
    }

    if (!model_->initialize_flux_transformer_spec(loader,
                                                  runtime_->backend(),
                                                  runtime_->offload_params_to_cpu(),
                                                  error)) {
        return false;
    }

    return prepare_weights(loader, tensors, ignore_tensors, error);
}

bool DiTPipeline::prepare_weights(const ModelLoader& loader,
                                  ModelLoader::TensorMap* tensors,
                                  ModelLoader::IgnoreTensorSet* ignore_tensors,
                                  std::string* error) {
    if (model_ == nullptr) {
        if (error != nullptr) {
            *error = "DiTPipeline::prepare_weights called before model is built";
        }
        return false;
    }

    return model_->prepare_flux_runtime_weights(loader,
                                                runtime_->backend(),
                                                runtime_->clip_backend(),
                                                runtime_->vae_backend(),
                                                runtime_->offload_params_to_cpu(),
                                                tensors,
                                                ignore_tensors,
                                                error);
}

void DiTPipeline::mark_ready() {
    const bool ok = (runtime_ != nullptr &&
                     model_ != nullptr &&
                     version_ != VERSION_COUNT);

    if (ok) {
        model_->mark_runtime_weights_loaded();
    }

    ready_ = ok;
}

bool DiTPipeline::build_components(const ld_context_params_t& params, std::string* error) {
    (void)params;
    model_ = std::make_unique<LDModel>(version_);
    model_->build_manifest(*loader_);
    if (!model_->validate(error)) {
        return false;
    }
    if (!model_->initialize_flux_transformer_spec(*loader_,
                                                  runtime_->backend(),
                                                  runtime_->offload_params_to_cpu(),
                                                  error)) {
        return false;
    }
    return true;
}

bool DiTPipeline::bind_weights(std::string* error) {
    if (!model_->load_flux_runtime_weights(*loader_,
                                           runtime_->backend(),
                                           runtime_->clip_backend(),
                                           runtime_->vae_backend(),
                                           runtime_->offload_params_to_cpu(),
                                           runtime_->n_threads(),
                                           runtime_->use_mmap(),
                                           error)) {
        return false;
    }
    return true;
}

bool DiTPipeline::validate_image_params(const ld_image_generation_params_t* params, std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "image generation params are null";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0) {
        if (error != nullptr) {
            *error = "image width and height must be positive";
        }
        return false;
    }
    if (params->batch_count <= 0) {
        if (error != nullptr) {
            *error = "image batch_count must be positive";
        }
        return false;
    }
    return true;
}

bool DiTPipeline::validate_video_params(const ld_video_generation_params_t* params, std::string* error) const {
    if (params == nullptr) {
        if (error != nullptr) {
            *error = "video generation params are null";
        }
        return false;
    }
    if (params->width <= 0 || params->height <= 0 || params->frames <= 0) {
        if (error != nullptr) {
            *error = "video width, height, and frames must be positive";
        }
        return false;
    }
    return true;
}

ld_status_t DiTPipeline::generate_image(const ld_image_generation_params_t* params,
                                        ld_image_batch_t* out,
                                        std::string* error) {
    if (!ready_ || model_ == nullptr || runtime_ == nullptr) {
        if (error != nullptr) {
            *error = "DiTPipeline is not initialized";
        }
        return LD_STATUS_MODEL_LOAD_FAILED;
    }
    if (out == nullptr) {
        if (error != nullptr) {
            *error = "image output is null";
        }
        return LD_STATUS_INVALID_ARGUMENT;
    }
    out->images = nullptr;
    out->count = 0;

    if (!validate_image_params(params, error)) {
        return LD_STATUS_INVALID_ARGUMENT;
    }
    if (!model_->can_generate_flux_image()) {
        if (error != nullptr) {
            *error = "current Stage 1 pipeline can generate only fully loaded Flux images";
        }
        return LD_STATUS_UNSUPPORTED;
    }

    const int count = params->batch_count > 0 ? params->batch_count : 1;
    ld_image_t* images = static_cast<ld_image_t*>(std::calloc(static_cast<size_t>(count), sizeof(ld_image_t)));
    if (images == nullptr) {
        if (error != nullptr) {
            *error = "failed to allocate image batch";
        }
        return LD_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < count; ++i) {
        if (!model_->generate_flux_image(params, i, runtime_->n_threads(), &images[i], error)) {
            for (int j = 0; j <= i; ++j) {
                std::free(images[j].data);
            }
            std::free(images);
            return LD_STATUS_GENERATION_FAILED;
        }
    }

    out->images = images;
    out->count = count;
    return LD_STATUS_OK;
}

ld_status_t DiTPipeline::generate_video(const ld_video_generation_params_t* params,
                                        ld_video_t* out,
                                        std::string* error) {
    if (out != nullptr) {
        out->frames = nullptr;
        out->frame_count = 0;
    }
    if (!validate_video_params(params, error)) {
        return LD_STATUS_INVALID_ARGUMENT;
    }
    if (error != nullptr) {
        *error = "video generation is not implemented in the Stage 1 DiTPipeline";
    }
    return LD_STATUS_UNSUPPORTED;
}

bool DiTPipeline::supports_image_generation() const {
    return ready_ && version_ != VERSION_COUNT && !(version_ == VERSION_SVD || ld_version_is_wan(version_));
}

bool DiTPipeline::supports_video_generation() const {
    return ready_ && (version_ == VERSION_SVD || ld_version_is_wan(version_));
}

ld_sampler_t DiTPipeline::default_sample_method() const {
    if (ld_version_is_dit(version_)) {
        return LD_SAMPLER_EULER;
    }
    return LD_SAMPLER_EULER_A;
}

ld_scheduler_t DiTPipeline::default_scheduler(ld_sampler_t method) const {
    if (method == LD_SAMPLER_LCM || method == LD_SAMPLER_TCD) {
        return LD_SCHEDULER_LCM;
    }
    if (method == LD_SAMPLER_DDIM_TRAILING) {
        return LD_SCHEDULER_SIMPLE;
    }
    return LD_SCHEDULER_DISCRETE;
}

} // namespace lightdit
