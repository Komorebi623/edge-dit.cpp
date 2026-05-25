#include "runtime/model_loader.hpp"

#include <sstream>

#include "utils/util.h"

namespace lightdit {

ModelLoader::ModelLoader()
    : loader_(std::make_unique<::ModelLoader>()) {}

bool ModelLoader::non_empty(const char* path) {
    return path != nullptr && path[0] != '\0';
}

bool ModelLoader::load_optional_file(const char* path,
                                     const std::string& prefix,
                                     const char* label,
                                     bool required,
                                     std::string* error) {
    if (!non_empty(path)) {
        if (required) {
            if (error != nullptr) {
                *error = std::string("missing required ") + label + " path";
            }
            return false;
        }
        return false;
    }

    LOG_INFO("loading %s from '%s'", label, path);
    if (!loader_->init_from_file(path, prefix)) {
        const std::string msg = loader_->get_last_error().empty()
                                    ? std::string("loading ") + label + " from '" + path + "' failed"
                                    : loader_->get_last_error();
        if (required) {
            if (error != nullptr) {
                *error = msg;
            }
            return false;
        }
        LOG_WARN("%s", msg.c_str());
        return false;
    }
    return true;
}

bool ModelLoader::load_files(const ld_context_params_t& params, std::string* error) {
    bool loaded_any = false;

    loaded_any = load_optional_file(params.model_path, "", "model", false, error) || loaded_any;

    const bool is_unet_hint = loader_->get_ld_version() != VERSION_COUNT &&
                              ld_version_is_unet(loader_->get_ld_version());

    loaded_any = load_optional_file(params.diffusion_model_path,
                                    "model.diffusion_model.",
                                    "diffusion model",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.high_noise_diffusion_model_path,
                                    "model.high_noise_diffusion_model.",
                                    "high noise diffusion model",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.clip_l_path,
                                    is_unet_hint ? "cond_stage_model.transformer." : "text_encoders.clip_l.transformer.",
                                    "clip_l",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.clip_g_path,
                                    is_unet_hint ? "cond_stage_model.1.transformer." : "text_encoders.clip_g.transformer.",
                                    "clip_g",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.clip_vision_path,
                                    "cond_stage_model.transformer.",
                                    "clip_vision",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.t5xxl_path,
                                    "text_encoders.t5xxl.transformer.",
                                    "t5xxl",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.llm_path,
                                    "text_encoders.llm.",
                                    "llm",
                                    false,
                                    error) || loaded_any;
    loaded_any = load_optional_file(params.llm_vision_path,
                                    "text_encoders.llm.visual.",
                                    "llm vision",
                                    false,
                                    error) || loaded_any;

    if (non_empty(params.vae_path)) {
        const bool ok = load_optional_file(params.vae_path, "vae.", "vae", false, error);
        external_vae_is_invalid_ = !ok;
        loaded_any = ok || loaded_any;
    }

    if (non_empty(params.taesd_path)) {
        const bool ok = load_optional_file(params.taesd_path, "tae.", "tae", false, error);
        use_tae_ = true;
        loaded_any = ok || loaded_any;
    }

    if (!loaded_any || loader_->get_tensor_storage_map().empty()) {
        if (error != nullptr) {
            *error = "no model tensors were loaded";
        }
        return false;
    }
    return true;
}

bool ModelLoader::finalize_names_and_version(std::string* error) {
    loader_->convert_tensors_name();
    version_ = loader_->get_ld_version();
    if (version_ == VERSION_COUNT) {
        if (error != nullptr) {
            *error = "failed to infer model version from loaded tensors";
        }
        return false;
    }

    LOG_INFO("model loader initialized: version=%s, files=%zu, tensors=%zu",
             ld_version_name(version_),
             loader_->get_file_paths().size(),
             loader_->get_tensor_storage_map().size());
    return true;
}

bool ModelLoader::apply_dtype_policy(const ld_context_params_t& params, std::string* error) {
    (void)error;
    const ggml_type wtype = ld_dtype_to_ggml(params.weight_type);
    if (wtype != GGML_TYPE_COUNT) {
        loader_->set_wtype_override(wtype);
    }
    return true;
}

bool ModelLoader::bind_weights(int n_threads, bool use_mmap, std::string* error) {
    if (!loader_->load_tensors(tensors_, ignore_tensors_, n_threads, use_mmap)) {
        if (error != nullptr) {
            *error = loader_->get_last_error().empty()
                         ? "failed to load tensors"
                         : loader_->get_last_error();
        }
        return false;
    }
    return true;
}

void ModelLoader::register_tensor(const std::string& name, ggml_tensor* tensor) {
    tensors_[name] = tensor;
}

void ModelLoader::register_tensors(const std::map<std::string, ggml_tensor*>& tensors) {
    tensors_.insert(tensors.begin(), tensors.end());
}

ggml_type ModelLoader::ld_dtype_to_ggml(ld_dtype_t dtype) {
    switch (dtype) {
        case LD_DTYPE_F32: return GGML_TYPE_F32;
        case LD_DTYPE_F16: return GGML_TYPE_F16;
        case LD_DTYPE_BF16: return GGML_TYPE_BF16;
        case LD_DTYPE_Q4_0: return GGML_TYPE_Q4_0;
        case LD_DTYPE_Q4_1: return GGML_TYPE_Q4_1;
        case LD_DTYPE_Q5_0: return GGML_TYPE_Q5_0;
        case LD_DTYPE_Q5_1: return GGML_TYPE_Q5_1;
        case LD_DTYPE_Q8_0: return GGML_TYPE_Q8_0;
        case LD_DTYPE_Q2_K: return GGML_TYPE_Q2_K;
        case LD_DTYPE_Q3_K: return GGML_TYPE_Q3_K;
        case LD_DTYPE_Q4_K: return GGML_TYPE_Q4_K;
        case LD_DTYPE_Q5_K: return GGML_TYPE_Q5_K;
        case LD_DTYPE_Q6_K: return GGML_TYPE_Q6_K;
        case LD_DTYPE_AUTO:
        default:
            return GGML_TYPE_COUNT;
    }
}

std::string ModelLoader::wtype_stat_to_str(const std::map<ggml_type, uint32_t>& stat) {
    std::ostringstream ss;
    bool first = true;
    for (const auto& [type, count] : stat) {
        if (!first) {
            ss << "|";
        }
        first = false;
        ss << ggml_type_name(type) << ":" << count;
    }
    return ss.str();
}

void ModelLoader::log_weight_stats() const {
    LOG_INFO("Weight type stat: %s", wtype_stat_to_str(loader_->get_wtype_stat()).c_str());
    LOG_INFO("Conditioner weight type stat: %s", wtype_stat_to_str(loader_->get_conditioner_wtype_stat()).c_str());
    LOG_INFO("Diffusion model weight type stat: %s", wtype_stat_to_str(loader_->get_diffusion_model_wtype_stat()).c_str());
    LOG_INFO("VAE weight type stat: %s", wtype_stat_to_str(loader_->get_vae_wtype_stat()).c_str());
}

} // namespace lightdit
