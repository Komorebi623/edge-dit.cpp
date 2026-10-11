#pragma once

// MiniMax-H3-only opt-in policy. No global environment mutation or shared
// backend switches: other model pipelines never parse these settings.
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>

namespace MiniMaxH3 {

struct CudaOptimizationOptions {
    bool fused_ops = false;
    bool cache_dit = false;
    bool adaln_cache = false;
    bool sage_attn = false;
    bool sparse_attn = false;
    bool vdn = false;
    bool vsa = false;
    bool turbo = false;
    int cache_front_blocks = 2;
    int cache_back_blocks = 2;
    int cache_warmup_steps = 2;
    int cache_max_consecutive = 2;
    float cache_threshold = 0.04f;
    float cache_start_percent = 0.15f;
    float cache_end_percent = 0.85f;
    int adaln_max_mib = 4096;
    int sparse_block_tokens = 64;
    float sparse_keep_ratio = 0.5f;
    int sparse_warmup_steps = 2;

    bool any_enabled() const {
        return fused_ops || cache_dit || adaln_cache || sage_attn || sparse_attn || vdn || vsa || turbo;
    }
};

// Capabilities must describe real implementations, not planned work. Explicit
// requests fail closed; an unsupported method must never report a speedup.
struct CudaOptimizationCapabilities {
    bool fused_ops = false;
    bool cache_dit = false;
    bool adaln_cache = false;
    bool sage_attn = false;
    bool sparse_attn = false;
    bool vdn = false;
    bool vsa = false;
    bool turbo = false;
};

using CudaOptimizationEnvLookup = std::function<std::string(const char*)>;

inline bool cuda_optimization_error(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

inline bool parse_cuda_optimization_options(const CudaOptimizationEnvLookup& lookup,
                                           CudaOptimizationOptions* output,
                                           std::string* error) {
    if (output == nullptr || !lookup) return cuda_optimization_error(error, "MiniMax-H3: missing optimization configuration");
    CudaOptimizationOptions options;
    struct Flag { const char* name; bool CudaOptimizationOptions::*member; };
    const Flag flags[] = {
        {"ED_H3_FUSED_OPS", &CudaOptimizationOptions::fused_ops},
        {"ED_H3_CACHE_DIT", &CudaOptimizationOptions::cache_dit},
        {"ED_H3_ADALN_CACHE", &CudaOptimizationOptions::adaln_cache},
        {"ED_H3_SAGE_ATTN", &CudaOptimizationOptions::sage_attn},
        {"ED_H3_SPARSE_ATTN", &CudaOptimizationOptions::sparse_attn},
        {"ED_H3_VDN", &CudaOptimizationOptions::vdn},
        {"ED_H3_VSA", &CudaOptimizationOptions::vsa},
        {"ED_H3_TURBO", &CudaOptimizationOptions::turbo},
    };
    for (const auto& flag : flags) {
        const auto value = lookup(flag.name);
        if (value.empty() || value == "0") options.*(flag.member) = false;
        else if (value == "1") options.*(flag.member) = true;
        else return cuda_optimization_error(error, std::string("MiniMax-H3: ") + flag.name + " must be 0 or 1");
    }
    // Inactive cache tuning must not change the baseline configuration.
    if (options.cache_dit) {
        struct Integer { const char* name; int CudaOptimizationOptions::*member; int minimum; };
        const Integer integers[] = {
            {"ED_H3_CACHE_FRONT_BLOCKS", &CudaOptimizationOptions::cache_front_blocks, 1},
            {"ED_H3_CACHE_BACK_BLOCKS", &CudaOptimizationOptions::cache_back_blocks, 0},
            {"ED_H3_CACHE_WARMUP_STEPS", &CudaOptimizationOptions::cache_warmup_steps, 1},
            {"ED_H3_CACHE_MAX_CONSECUTIVE", &CudaOptimizationOptions::cache_max_consecutive, 1},
        };
        for (const auto& setting : integers) {
            const auto value = lookup(setting.name);
            if (value.empty()) continue;
            char* end = nullptr;
            errno = 0;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (errno != 0 || end != value.c_str() + value.size() || parsed < setting.minimum || parsed > 10000)
                return cuda_optimization_error(error, std::string("MiniMax-H3: invalid ") + setting.name);
            options.*(setting.member) = static_cast<int>(parsed);
        }
        struct Real { const char* name; float CudaOptimizationOptions::*member; };
        const Real reals[] = {
            {"ED_H3_CACHE_THRESHOLD", &CudaOptimizationOptions::cache_threshold},
            {"ED_H3_CACHE_START_PERCENT", &CudaOptimizationOptions::cache_start_percent},
            {"ED_H3_CACHE_END_PERCENT", &CudaOptimizationOptions::cache_end_percent},
        };
        for (const auto& setting : reals) {
            const auto value = lookup(setting.name);
            if (value.empty()) continue;
            char* end = nullptr;
            errno = 0;
            const float parsed = std::strtof(value.c_str(), &end);
            if (errno != 0 || end != value.c_str() + value.size() || !std::isfinite(parsed) || parsed < 0.f || parsed > 1.f)
                return cuda_optimization_error(error, std::string("MiniMax-H3: invalid ") + setting.name);
            options.*(setting.member) = parsed;
        }
        if (options.cache_start_percent >= options.cache_end_percent)
            return cuda_optimization_error(error, "MiniMax-H3: cache start must precede cache end");
    }
    if (options.sparse_attn) {
        const auto ratio = lookup("ED_H3_SPARSE_KEEP_RATIO");
        if (!ratio.empty()) {
            char* end = nullptr; errno = 0;
            const float parsed = std::strtof(ratio.c_str(), &end);
            if (errno || end != ratio.c_str() + ratio.size() || !std::isfinite(parsed) || parsed <= 0.f || parsed > 1.f)
                return cuda_optimization_error(error, "MiniMax-H3: sparse keep ratio must be in (0,1]");
            options.sparse_keep_ratio = parsed;
        }
    }
    if (options.adaln_cache) {
        const auto budget = lookup("ED_H3_ADALN_MAX_MIB");
        if (!budget.empty()) {
            char* end = nullptr; errno = 0;
            const long parsed = std::strtol(budget.c_str(), &end, 10);
            if (errno || end != budget.c_str() + budget.size() || parsed < 1 || parsed > 16384)
                return cuda_optimization_error(error, "MiniMax-H3: AdaLN host cache budget must be 1..16384 MiB");
            options.adaln_max_mib = static_cast<int>(parsed);
        }
    }
    *output = options;
    if (error != nullptr) error->clear();
    return true;
}

inline bool validate_cuda_optimization_options(const CudaOptimizationOptions& options,
                                               bool is_cuda,
                                               const CudaOptimizationCapabilities& capabilities,
                                               std::string* error) {
    if (options.any_enabled() && !is_cuda)
        return cuda_optimization_error(error, "MiniMax-H3: ED_H3 optimizations require the CUDA backend");
    if (options.sage_attn && options.sparse_attn)
        return cuda_optimization_error(error, "MiniMax-H3: Sage and SubBlock are alternative attention methods; enable one at a time");
    struct Method { const char* name; bool requested; bool supported; const char* reason; };
    const Method methods[] = {
        {"ED_H3_FUSED_OPS", options.fused_ops, capabilities.fused_ops, "requires ED_ENABLE_CUDA_MODULATION=ON"},
        {"ED_H3_CACHE_DIT", options.cache_dit, capabilities.cache_dit, "requires the H3 dual-block cache implementation"},
        {"ED_H3_ADALN_CACHE", options.adaln_cache, capabilities.adaln_cache, "requires the H3 exact projection prefill implementation"},
        {"ED_H3_SAGE_ATTN", options.sage_attn, capabilities.sage_attn, "requires the H3-scoped Sage tag and ED_ENABLE_CUDA_SAGE_ATTN=ON"},
        {"ED_H3_SPARSE_ATTN", options.sparse_attn, capabilities.sparse_attn, "requires the H3 per-head SubBlock graph implementation"},
        {"ED_H3_VDN", options.vdn, capabilities.vdn, "requires matching H3 adapters and hybrid-attention kernels"},
        {"ED_H3_VSA", options.vsa, capabilities.vsa, "CPU oracle only; requires matching FastH3 gates and native CUDA tiling/compression kernels"},
        {"ED_H3_TURBO", options.turbo, capabilities.turbo, "requires matching public distilled weights; reducing steps alone is not distillation"},
    };
    for (const auto& method : methods) {
        if (method.requested && !method.supported)
            return cuda_optimization_error(error, std::string("MiniMax-H3: ") + method.name + " unsupported: " + method.reason);
    }
    if (error != nullptr) error->clear();
    return true;
}

inline CudaOptimizationEnvLookup cuda_optimization_process_environment() {
    return [](const char* name) {
        const char* value = std::getenv(name);
        return value == nullptr ? std::string{} : std::string(value);
    };
}

} // namespace MiniMaxH3
