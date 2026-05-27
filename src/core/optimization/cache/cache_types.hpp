#pragma once

#include <string>
#include <vector>

#include "light-dit.h"
#include "runtime/model_loader.h"

namespace lightdit {
namespace cache {

enum class CacheMode {
    Disabled,
    EasyCache,
    UCache,
    DBCache,
    TaylorSeer,
    CacheDiT,
};

enum class CacheExecType {
    Full,
    Reuse,
    Probe,
    ResumeFull,
    Taylor,
    Disabled,
};

enum class CacheStorageKind {
    None,
    HiddenState,
    Residual,
    BlockResidual,
    Token,
    Attention,
    Custom,
};

enum class CacheSelectorKind {
    All,
    Random,
    Attention,
    Norm,
    Score,
    Custom,
};

enum class CacheBranch {
    Main,
    Cond,
    Uncond,
};

enum class CacheRegionPattern {
    HiddenOnly,
    HiddenContext,
    ImageText,
    PackedImageText,
    Custom,
};

struct CacheRegionSpec {
    std::string id;
    std::string graph_prefix;
    int block_count = 0;
    CacheRegionPattern pattern = CacheRegionPattern::HiddenOnly;
    std::vector<std::string> input_keys;
    std::vector<std::string> output_keys;
    int default_fn_blocks = 0;
    int default_bn_blocks = 0;
};

struct CacheModelSpec {
    std::string model_name;
    SDVersion version = VERSION_COUNT;
    std::vector<CacheRegionSpec> regions;
    bool separate_cfg = false;
};

struct CacheStepInfo {
    int step_index = -1;
    int num_steps = 0;
    float sigma = 0.0f;
    float sigma_next = 0.0f;
};

struct CacheRegionFrame {
    CacheStepInfo step;
    CacheBranch branch = CacheBranch::Main;
    std::string region_id;
    int block_index = -1;
};

struct CacheRegionPlan {
    CacheExecType exec_type = CacheExecType::Full;
    CacheStorageKind storage_kind = CacheStorageKind::Residual;
    CacheSelectorKind selector_kind = CacheSelectorKind::All;
    int fn_blocks = 0;
    int bn_blocks = 0;
    int probe_blocks = 0;
    bool needs_input_snapshot = false;
    bool needs_output_snapshot = true;
    bool can_reuse = false;
};

inline CacheMode cache_mode_from_ld(ld_cache_mode_t mode) {
    switch (mode) {
        case LD_CACHE_EASYCACHE: return CacheMode::EasyCache;
        case LD_CACHE_UCACHE: return CacheMode::UCache;
        case LD_CACHE_DBCACHE: return CacheMode::DBCache;
        case LD_CACHE_TAYLORSEER: return CacheMode::TaylorSeer;
        case LD_CACHE_CACHE_DIT: return CacheMode::CacheDiT;
        case LD_CACHE_DISABLED:
        default: return CacheMode::Disabled;
    }
}

const char* cache_mode_name(CacheMode mode);
CacheModelSpec cache_model_spec_for_version(SDVersion version);

}  // namespace cache
}  // namespace lightdit
