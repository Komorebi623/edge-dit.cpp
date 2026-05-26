#pragma once

#include <memory>
#include <string>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/pipelines/dit_pipeline.hpp"

struct Conditioner;
struct FrozenCLIPVisionEmbedder;
struct VAE;

namespace lightdit {

class WanPipeline final : public DiTPipeline {
public:
    explicit WanPipeline(SDVersion version = VERSION_WAN2);
    ~WanPipeline() override = default;

    const char* name() const override { return "wan"; }

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

    bool supports_image_generation() const override { return false; }
    bool supports_video_generation() const override { return false; }

    ld_sampler_t default_sample_method() const override { return LD_SAMPLER_EULER; }
    ld_scheduler_t default_scheduler(ld_sampler_t method) const override;

private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    SDVersion version_ = VERSION_WAN2;

    std::shared_ptr<Conditioner> conditioner_;
    std::shared_ptr<DiffusionModel> diffusion_;
    std::shared_ptr<DiffusionModel> high_noise_diffusion_;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision_;
    std::shared_ptr<VAE> vae_;
    std::shared_ptr<VAE> preview_vae_;

    bool using_tae_for_main_ = false;

    bool build_components(const ModelLoader& loader, std::string* error);
    bool register_tensors(PipelineTensorRegistry& registry, std::string* error);
    void configure_runtime_flags();
    void build_ignore_tensors(PipelineTensorRegistry& registry) const;

    static bool has_prefix(const ModelLoader& loader, const std::string& prefix);
};

}  // namespace lightdit
