#pragma once
#include <cmath>
#include <string>
namespace MiniMaxH3 {
// Published distilled weights are a separate model experiment, never a step-only
// shortcut applied to the locked base model. The offline merger emits this recipe.
struct TurboRecipe {
    int steps = 0;
    float video_shift = 0.f;
    float audio_shift = 0.f;
    float cfg_scale = 1.f;
    std::string sampler;
    std::string scheduler;
    bool matching_merged_weights = false;
    bool valid(std::string* error) const {
        const char* reason = nullptr;
        if (!matching_merged_weights) reason = "Turbo requires an adjacent verified offline-merge manifest and matching shards";
        else if (steps != 4 && steps != 8) reason = "Turbo NFE must match a published 4/8-step recipe";
        else if (!std::isfinite(video_shift) || !std::isfinite(audio_shift) || video_shift <= 0 || audio_shift <= 0)
            reason = "Turbo flow shifts must be finite and positive";
        else if (sampler != "euler" || scheduler != "simple" || cfg_scale != 1.f)
            reason = "Turbo recipe requires Euler/simple/CFG=1";
        if (error) *error = reason ? reason : "";
        return !reason;
    }
};
} // namespace MiniMaxH3
