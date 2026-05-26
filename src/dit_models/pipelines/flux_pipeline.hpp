#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "dit_models/pipelines/dit_pipeline.hpp"

namespace Flux {
struct FluxRunner;
}

struct Conditioner;
struct VAE;

namespace lightdit {

struct PipelineComponent {
    std::string name;
    size_t tensor_count = 0;
    int64_t bytes = 0;
    std::map<ggml_type, uint32_t> type_counts;
    std::vector<std::string> examples;
};

class FluxPipeline final : public DiTPipeline {
public:
    explicit FluxPipeline(SDVersion version = VERSION_FLUX);
    ~FluxPipeline() override;

    const char* name() const override { return "flux"; }

    bool prepare(const ld_context_params_t& params,
                 ModelRuntime& runtime,
                 const ModelLoader& loader,
                 PipelineTensorRegistry& registry,
                 std::string* error) override;

    void mark_ready() override;

    ld_status_t generate_image(const ld_image_generation_params_t* params,
                               ld_image_batch_t* out,
                               std::string* error) override;

    ld_status_t generate_video(const ld_video_generation_params_t* params,
                               ld_video_t* out,
                               std::string* error) override;

    SDVersion version() const override { return version_; }
    bool ready() const override { return ready_; }

    bool supports_image_generation() const override;
    bool supports_video_generation() const override;

    ld_sampler_t default_sample_method() const override;
    ld_scheduler_t default_scheduler(ld_sampler_t method) const override;

private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    SDVersion version_ = VERSION_COUNT;

    std::vector<PipelineComponent> components_;
    std::unique_ptr<Flux::FluxRunner> flux_runner_;
    ggml_backend_t flux_backend_ = nullptr;

    std::shared_ptr<Conditioner> conditioner_;
    std::shared_ptr<VAE> vae_;
    ggml_backend_t conditioner_backend_ = nullptr;
    ggml_backend_t vae_backend_ = nullptr;

    int flux_declared_tensors_ = 0;
    bool runtime_weights_loaded_ = false;
    std::vector<std::string> flux_missing_tensors_;
    std::vector<std::string> flux_shape_mismatch_tensors_;
    std::vector<std::string> flux_unexpected_tensors_;

    PipelineComponent* find_or_add_component(const std::string& name);
    bool has_component(const std::string& name) const;

    void reset_flux_runner();

    void build_manifest(const ModelLoader& loader);
    bool validate(std::string* error) const;
    bool initialize_flux_transformer_spec(const ModelLoader& loader,
                                          ggml_backend_t backend,
                                          bool offload_params_to_cpu,
                                          std::string* error);

    bool prepare_flux_runtime_weights(const ModelLoader& loader,
                                      ggml_backend_t diffusion_backend,
                                      ggml_backend_t text_backend,
                                      ggml_backend_t vae_backend,
                                      bool offload_params_to_cpu,
                                      PipelineTensorRegistry& registry,
                                      std::string* error);

    bool can_generate_image() const;
    bool generate_one_image(const ld_image_generation_params_t* params,
                            int batch_index,
                            int n_threads,
                            ld_image_t* image,
                            std::string* error);

    bool validate_image_params(const ld_image_generation_params_t* params,
                               std::string* error) const;
    bool validate_video_params(const ld_video_generation_params_t* params,
                               std::string* error) const;
};

}  // namespace lightdit
