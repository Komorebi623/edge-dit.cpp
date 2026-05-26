#pragma once

#include <memory>
#include <string>
#include <vector>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/pipelines/dit_pipeline.hpp"

struct Conditioner;
struct VAE;

namespace lightdit {

class SD3Pipeline final : public DiTPipeline {
public:
    explicit SD3Pipeline(SDVersion version = VERSION_SD3);
    ~SD3Pipeline() override = default;

    const char* name() const override { return "sd3"; }

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

    bool supports_image_generation() const override { return ready_; }
    bool supports_video_generation() const override { return false; }

    ld_sampler_t default_sample_method() const override { return LD_SAMPLER_EULER; }
    ld_scheduler_t default_scheduler(ld_sampler_t method) const override;

private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    SDVersion version_ = VERSION_SD3;

    std::shared_ptr<Conditioner> conditioner_;
    std::shared_ptr<DiffusionModel> diffusion_;
    std::shared_ptr<VAE> vae_;

    bool build_components(const ModelLoader& loader, std::string* error);
    bool register_tensors(PipelineTensorRegistry& registry, std::string* error);
    void configure_runtime_flags();
    void build_ignore_tensors(PipelineTensorRegistry& registry) const;

    bool can_generate_image() const;
    bool validate_image_params(const ld_image_generation_params_t* params,
                               std::string* error) const;
    bool generate_one_image(const ld_image_generation_params_t* params,
                            int batch_index,
                            ld_image_t* image,
                            std::string* error);

    std::vector<float> build_sigmas(int steps, float shift) const;
};

}  // namespace lightdit
