#pragma once

#include <string>
#include <memory>

#include "dit_models/pipelines/dit_pipeline.hpp"

namespace MiniMaxH3 {
struct MiniMaxH3Runner;
}
namespace LLM {
struct LLMEmbedder;
}
namespace MiniMaxH3VAE {
struct MiniMaxH3VideoVAERunner;
}
namespace MiniMaxH3Audio {
struct AudioVAERunner;
}

namespace edgedit {

class MiniMaxH3Pipeline final : public DiTPipeline {
public:
    explicit MiniMaxH3Pipeline(SDVersion version = VERSION_MINIMAX_H3);
    ~MiniMaxH3Pipeline() override;

    const char* name() const override { return "minimax-h3"; }

    bool prepare(const ed_context_params_t& params,
                 ModelRuntime& runtime,
                 const ModelLoader& loader,
                 PipelineTensorRegistry& registry,
                 std::string* error) override;

    void mark_ready() override;

    ed_status_t generate_image(const ed_image_generation_params_t* params,
                               ed_image_batch_t* out,
                               std::string* error) override;

    ed_status_t generate_video(const ed_video_generation_params_t* params,
                               ed_video_t* out,
                               std::string* error) override;

    SDVersion version() const override { return version_; }
    bool ready() const override { return ready_; }

    bool supports_image_generation() const override { return false; }
    bool supports_video_generation() const override { return ready_; }

    ed_sampler_t default_sample_method() const override { return ED_SAMPLER_EULER; }
    ed_scheduler_t default_scheduler(ed_sampler_t method) const override;

private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    SDVersion version_ = VERSION_MINIMAX_H3;
    std::unique_ptr<MiniMaxH3::MiniMaxH3Runner> diffusion_;
    std::unique_ptr<LLM::LLMEmbedder> conditioner_;
    std::unique_ptr<MiniMaxH3VAE::MiniMaxH3VideoVAERunner> vae_;
    std::unique_ptr<MiniMaxH3Audio::AudioVAERunner> audio_vae_;
    ggml_context* sentinel_ctx_ = nullptr;
    ggml_tensor* sentinel_tensor_ = nullptr;

    bool build_text_context(const char* prompt,
                            const ed_image_t* ref_images,
                            int ref_image_count,
                            const ed_ref_video_t* ref_videos,
                            int ref_video_count,
                            int ref_audio_count,
                            int max_video_frames,
                            sd::Tensor<float>* context,
                            sd::Tensor<int32_t>* token_tags,
                            std::string* error);
    ed_status_t decode_video_latent(const sd::Tensor<float>& latent,
                                    int requested_frames,
                                    ed_video_t* out,
                                    std::string* error);
    bool decode_audio_latent(const sd::Tensor<float>& latent,
                             ed_video_t* out,
                             std::string* error);
};

}  // namespace edgedit
