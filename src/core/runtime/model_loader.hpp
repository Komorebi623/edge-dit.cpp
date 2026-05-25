#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "core/runtime/model_loader.h"
#include "ggml-backend.h"
#include "light-dit.h"

namespace lightdit {

class ModelLoader final {
public:
    ModelLoader();
    ~ModelLoader() = default;

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;

    bool load_files(const ld_context_params_t& params, std::string* error);
    bool finalize_names_and_version(std::string* error);
    bool apply_dtype_policy(const ld_context_params_t& params, std::string* error);
    bool bind_weights(int n_threads, bool use_mmap, std::string* error);

    void register_tensor(const std::string& name, ggml_tensor* tensor);
    void register_tensors(const std::map<std::string, ggml_tensor*>& tensors);

    std::map<std::string, ggml_tensor*>& tensors() { return tensors_; }
    const std::map<std::string, ggml_tensor*>& tensors() const { return tensors_; }

    void add_ignore_prefix(const std::string& prefix) { ignore_tensors_.insert(prefix); }
    void set_ignore_tensors(const std::set<std::string>& ignores) { ignore_tensors_ = ignores; }
    const std::set<std::string>& ignore_tensors() const { return ignore_tensors_; }

    SDVersion version() const { return version_; }
    bool external_vae_is_invalid() const { return external_vae_is_invalid_; }
    bool use_tae() const { return use_tae_; }
    bool tae_preview_only() const { return tae_preview_only_; }
    bool use_pmid() const { return use_pmid_; }

    ::ModelLoader& raw_loader() { return *loader_; }
    const ::ModelLoader& raw_loader() const { return *loader_; }

    String2TensorStorage& tensor_storage_map() { return loader_->get_tensor_storage_map(); }
    const String2TensorStorage& tensor_storage_map() const { return loader_->get_tensor_storage_map(); }

    void log_weight_stats() const;

private:
    std::unique_ptr<::ModelLoader> loader_;
    SDVersion version_ = VERSION_COUNT;
    std::map<std::string, ggml_tensor*> tensors_;
    std::set<std::string> ignore_tensors_;

    bool external_vae_is_invalid_ = false;
    bool use_tae_ = false;
    bool tae_preview_only_ = false;
    bool use_pmid_ = false;

    static bool non_empty(const char* path);
    bool load_optional_file(const char* path,
                            const std::string& prefix,
                            const char* label,
                            bool required,
                            std::string* error);
    static ggml_type ld_dtype_to_ggml(ld_dtype_t dtype);
    static std::string wtype_stat_to_str(const std::map<ggml_type, uint32_t>& stat);
};

} // namespace lightdit
