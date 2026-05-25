#ifndef LD_RUNTIME_MODEL_H
#define LD_RUNTIME_MODEL_H

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "light-dit.h"
#include "core/runtime/model_loader.h"

namespace Flux {
struct FluxRunner;
}
struct Conditioner;
struct VAE;

struct LDModelComponent {
    std::string name;
    size_t tensor_count = 0;
    int64_t bytes = 0;
    std::map<ggml_type, uint32_t> type_counts;
    std::vector<std::string> examples;
};

class LDModel {
public:
    explicit LDModel(SDVersion version = VERSION_COUNT);
    ~LDModel();

    void build_manifest(const ModelLoader& loader);
    bool validate(std::string* error) const;
    bool initialize_flux_transformer_spec(const ModelLoader& loader,
                                          ggml_backend_t backend,
                                          bool offload_params_to_cpu,
                                          std::string* error);
    bool load_flux_runtime_weights(ModelLoader& loader,
                                   ggml_backend_t diffusion_backend,
                                   ggml_backend_t text_backend,
                                   ggml_backend_t vae_backend,
                                   bool offload_params_to_cpu,
                                   int n_threads,
                                   bool use_mmap,
                                   std::string* error);
    bool encode_flux_prompt(const char* prompt,
                            int n_threads,
                            int clip_skip,
                            std::string* error);
    bool generate_flux_image(const ld_image_generation_params_t* params,
                             int batch_index,
                             int n_threads,
                             ld_image_t* image,
                             std::string* error);

    bool smoke_load_small_tensors(ModelLoader& loader,
                                  int max_tensors,
                                  size_t max_tensor_bytes,
                                  int n_threads,
                                  bool use_mmap,
                                  std::string* error);

    const std::vector<LDModelComponent>& components() const { return components_; }
    int smoke_loaded_tensors() const { return smoke_loaded_tensors_; }
    int64_t smoke_loaded_bytes() const { return smoke_loaded_bytes_; }
    int flux_declared_tensors() const { return flux_declared_tensors_; }
    int flux_missing_tensors() const { return static_cast<int>(flux_missing_tensors_.size()); }
    int flux_shape_mismatch_tensors() const { return static_cast<int>(flux_shape_mismatch_tensors_.size()); }
    int flux_unexpected_tensors() const { return static_cast<int>(flux_unexpected_tensors_.size()); }
    bool runtime_weights_loaded() const { return runtime_weights_loaded_; }
    bool can_generate_flux_image() const;

private:
    SDVersion version_ = VERSION_COUNT;
    std::vector<LDModelComponent> components_;
    int smoke_loaded_tensors_ = 0;
    int64_t smoke_loaded_bytes_ = 0;
    std::unique_ptr<Flux::FluxRunner> flux_runner_;
    ggml_backend_t flux_backend_ = nullptr;
    bool owns_flux_backend_ = false;
    std::shared_ptr<Conditioner> conditioner_;
    std::shared_ptr<VAE> vae_;
    ggml_backend_t conditioner_backend_ = nullptr;
    bool owns_conditioner_backend_ = false;
    ggml_backend_t vae_backend_ = nullptr;
    bool owns_vae_backend_ = false;
    int flux_declared_tensors_ = 0;
    bool runtime_weights_loaded_ = false;
    std::vector<std::string> flux_missing_tensors_;
    std::vector<std::string> flux_shape_mismatch_tensors_;
    std::vector<std::string> flux_unexpected_tensors_;

    LDModelComponent* find_or_add_component(const std::string& name);
    bool has_component(const std::string& name) const;
    void reset_flux_runner();
};

#endif
