#pragma once

// H3 dual-block cache policy and graph pass contract. The algorithm is independent
// of CUDA so its decisions can be tested without starting a GPU. Device storage
// belongs to GGMLRunner's existing RunnerCacheDeviceStore, not this policy.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dit_models/models/minimax_h3_cuda_options.hpp"

struct ggml_tensor;
namespace edgedit { namespace cache { class CacheOperatorRegistry; } }

namespace MiniMaxH3 {

class CudaDualBlockCachePolicy {
public:
    struct Branch {
        bool valid = false;
        int last_step = -1;
        int consecutive = 0;
        int full_steps = 0;
        int hit_steps = 0;
        std::string identity;
    };

    void reset() {
        branches_ = {};
        step_ = -1;
        total_ = 0;
        pending_ = false;
    }
    void configure(const CudaOptimizationOptions& options, int64_t blocks) {
        if (options.cache_dit &&
            (options.cache_front_blocks < 1 || options.cache_back_blocks < 0 ||
             static_cast<int64_t>(options.cache_front_blocks) + options.cache_back_blocks >= blocks ||
             options.cache_warmup_steps < 1 || options.cache_max_consecutive < 1 ||
             !std::isfinite(options.cache_threshold) || options.cache_threshold < 0.f ||
             !std::isfinite(options.cache_start_percent) || !std::isfinite(options.cache_end_percent) ||
             options.cache_start_percent < 0.f || options.cache_end_percent > 1.f ||
             options.cache_start_percent >= options.cache_end_percent)) {
            throw std::invalid_argument("MiniMax-H3: invalid dual-block cache partition/window");
        }
        options_ = options;
        reset();
    }
    void begin_step(int step, int total, bool conditional) {
        if (step < 0 || total <= 0 || step >= total) {
            throw std::invalid_argument("MiniMax-H3: invalid CUDA cache step/total_steps");
        }
        if (pending_) {
            throw std::logic_error("MiniMax-H3: CUDA cache step was not consumed");
        }
        branch_ = conditional ? 1 : 0;
        // Request changes are normally explicit reset(). Schedule changes and
        // non-monotonic/repeated branch calls also invalidate the anchor.
        if (total_ != 0 && total_ != total) branches_ = {};
        auto& state = branches_[branch_];
        if (state.last_step >= 0 && step != state.last_step + 1) state = {};
        step_ = step;
        total_ = total;
        pending_ = true;
    }
    void require_step() const {
        if (!pending_) throw std::logic_error("MiniMax-H3: cache requires begin_cuda_step before each CFG branch");
    }
    void bind_identity(const std::string& identity) {
        require_step();
        auto& state = branches_[branch_];
        if (state.identity != identity) {
            state = {};
            state.identity = identity;
        }
    }
    bool eligible() const {
        require_step();
        const auto& state = branches_[branch_];
        const float progress = total_ > 1 ? static_cast<float>(step_) / (total_ - 1) : 0.f;
        return options_.cache_dit && state.valid && step_ >= options_.cache_warmup_steps &&
               state.last_step + 1 == step_ && state.consecutive < options_.cache_max_consecutive &&
               progress >= options_.cache_start_percent && progress <= options_.cache_end_percent;
    }
    bool hit(const std::vector<float>& per_span_relative_l1) const {
        if (!eligible() || per_span_relative_l1.empty()) return false;
        // No modality may be diluted by a longer video stream. Non-finite/zero
        // reference norm reductions fail closed, including NaN from 0/0.
        return std::all_of(per_span_relative_l1.begin(), per_span_relative_l1.end(), [&](float diff) {
            return std::isfinite(diff) && diff >= 0.f && diff < options_.cache_threshold;
        });
    }
    void finish(bool hit, bool stored) {
        require_step();
        auto& state = branches_[branch_];
        state.last_step = step_;
        if (hit) { ++state.consecutive; ++state.hit_steps; }
        else { state.consecutive = 0; ++state.full_steps; state.valid = stored; }
        pending_ = false;
    }
    void abort() {
        branches_[branch_] = {};
        pending_ = false;
    }
    int branch_index() const { return branch_; }
    const Branch& branch(int index) const { return branches_.at(index); }

    // CPU oracle for the exact same graph RelL1 denominator. A zero reference
    // norm is deliberately not treated as a hit (the generic host operator does
    // so, but that would be unsafe for a changing H3 modality).
    static float relative_l1(const float* current, const float* reference, size_t count) {
        if (!current || !reference || count == 0) return std::numeric_limits<float>::infinity();
        double numerator = 0., denominator = 0.;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(current[i]) || !std::isfinite(reference[i]))
                return std::numeric_limits<float>::infinity();
            numerator += std::fabs(static_cast<double>(current[i]) - reference[i]);
            denominator += std::fabs(static_cast<double>(reference[i]));
        }
        return denominator > 0. ? static_cast<float>(numerator / denominator)
                                : std::numeric_limits<float>::infinity();
    }
private:
    CudaOptimizationOptions options_;
    std::array<Branch, 2> branches_{};
    int branch_ = 0;
    int step_ = -1;
    int total_ = 0;
    bool pending_ = false;
};

// Explicit execution seam: Probe evaluates only [0, Fn); Remainder starts from
// the CURRENT front. A miss evaluates [Fn, L-Bn), captures delta, then suffix;
// a hit adds the old middle delta and evaluates [L-Bn, L). Never model outputs.
struct CudaDualBlockGraphPass {
    enum class Phase { Probe, Remainder };
    Phase phase = Phase::Probe;
    int front_blocks = 0;
    int back_blocks = 0;
    bool reuse_middle = false;
    ggml_tensor* front = nullptr;
    ggml_tensor* reference_rf = nullptr;
    ggml_tensor* middle_delta = nullptr;
    ggml_tensor* current_rf = nullptr;
    edgedit::cache::CacheOperatorRegistry* operators = nullptr;
    std::vector<std::string> indicator_names;
};

} // namespace MiniMaxH3
