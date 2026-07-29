#include "runtime/model_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <thread>
#include <utility>

#include "utils/rng_philox.hpp"
#include "utils/util.h"
#include "runtime/model_loader.h"

namespace edgedit {
namespace {

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string requested_backend_name() {
    const char* value = std::getenv("ED_BACKEND");
    if (value == nullptr) {
        return "";
    }
    return value;
}

bool is_auto_backend(const std::string& name) {
    return name.empty() || lowercase(name) == "auto" || lowercase(name) == "default";
}

bool is_generic_gpu_request(const std::string& requested) {
    const std::string request = lowercase(requested);
    return request == "gpu" || request == "cuda" || request == "vulkan" || request == "metal";
}

bool device_name_matches(ggml_backend_dev_t dev, const std::string& requested) {
    if (dev == nullptr) {
        return false;
    }

    const std::string request = lowercase(requested);
    
    if (request == "gpu") {
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }

    const char* name_c = ggml_backend_dev_name(dev);
    if (name_c == nullptr) {
        return false;
    }
    const std::string name = lowercase(name_c);
    if (request == "metal") {
        // ggml's Metal device is named "MTL0", not "metal"
        return contains(name, "metal") || contains(name, "mtl");
    }
    return contains(name, request);
}

bool is_gpu_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

int runtime_gpu_device_ordinal(int fallback) {
    const char* value = std::getenv("ED_CLI_SINGLE_VISIBLE_DEVICE");
    if (value != nullptr && value[0] == '1' && value[1] == '\0') {
        return 0;
    }
    return fallback;
}

ggml_backend_t init_explicit_backend(const std::string& requested, int gpu_device_ordinal) {
    const std::string request = lowercase(requested);
    if (request == "cpu") {
        return ggml_backend_cpu_init();
    }

    ggml_backend_load_all_once();
    const size_t device_count = ggml_backend_dev_count();
    const bool use_gpu_ordinal = is_generic_gpu_request(requested) && gpu_device_ordinal >= 0;
    int matched_gpu_index = 0;
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!device_name_matches(dev, requested)) {
            continue;
        }
        if (use_gpu_ordinal && is_gpu_device(dev) && matched_gpu_index++ != gpu_device_ordinal) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            return backend;
        }
    }

    return init_named_backend(requested);
}

std::string available_backend_names() {
    ggml_backend_load_all_once();
    std::string result;
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* name = ggml_backend_dev_name(dev);
        if (name == nullptr) {
            continue;
        }
        if (!result.empty()) {
            result += ", ";
        }
        result += name;
    }
    return result.empty() ? "none" : result;
}

}  // namespace

ModelRuntime::~ModelRuntime() {
    reset();
}

bool ModelRuntime::init(const ed_context_params_t* params, std::string* error) {
    if (params == nullptr) {
        return fail(error, "ModelRuntime::init got null params");
    }
    return init(*params, error);
}

bool ModelRuntime::init(const ed_context_params_t& params, std::string* error) {
    reset();
    ggml_log_set(ggml_log_callback_default, nullptr);

    if (!init_threads(params, error)) {
        return false;
    }
    if (!init_flags(params, error)) {
        return false;
    }
    if (!init_rng(params, error)) {
        return false;
    }
    if (!init_backends(params, error)) {
        return false;
    }

    ready_ = true;
    return true;
}

namespace {
// Auto-allocate: VRAM to reserve beyond a resident component's weights, for its own
// compute buffer + activations + allocator fragmentation. A resident (non-segmented)
// component's compute buffer is NOT covered by graph_cut_segment_vram_bytes (that only
// bounds offloaded/segmented components), so a component may only stay resident if its
// weights PLUS this headroom fit the budget. Measured DiT compute activations at
// 1024²/20steps/double-forward: sd3 ~3.4G, flux ~1.9G; 4 GiB covers the upper end with
// margin. Undersized headroom is what let sd3 8g fully-resident peak at 10.3G > 8G budget.
constexpr size_t kResidentComputeHeadroom = static_cast<size_t>(4) * 1024 * 1024 * 1024;
// Smallest plausible standalone component (VAE ~0.15-0.5G); used by the all-offload
// fallback: if the budget can't even fit one small component + compute headroom,
// offload everything (equivalent to legacy --offload-to-cpu, safest).
constexpr size_t kMinResidentComponentBytes = static_cast<size_t>(512) * 1024 * 1024;
// Fragmentation + large-segment compute slack subtracted when sizing the SEGMENT
// budget for offloaded components. 2 GiB: an offloaded component's segment carries
// several GB of transient compute/activation on top of its weights that
// graph_cut_segment_vram_bytes underestimates. Shrinking the segment budget further
// (tried 3 GiB) does NOT lower qwen-edit 8g's residual 8292 peak: that peak is two
// text-encode partial segments co-resident during staging, not one oversized segment,
// so a smaller split just makes more segments at the same summed footprint. 2 GiB is
// the sweet spot that keeps other models' segments large (fewer staging round-trips).
constexpr size_t kSegmentBudgetSlack = static_cast<size_t>(2) * 1024 * 1024 * 1024;
// Physical core count (excludes SMT/hyperthreads). On this dual-socket Xeon,
// running matmul-heavy graphs on all 192 logical cores is ~2x SLOWER than on the
// running matmul-heavy graphs on all 192 logical cores is ~2x SLOWER than on the
// 96 physical cores: hyperthreads contend for shared AVX-512/AMX vector units and
// extra threads inflate per-node barrier sync. Parse /proc/cpuinfo for distinct
// (physical id, core id) pairs; fall back to hardware_concurrency() if unavailable.
static int detect_physical_cores() {
    std::ifstream f("/proc/cpuinfo");
    if (!f.is_open()) {
        return 0;
    }
    std::set<std::pair<int, int>> cores;
    int phys = -1;
    int core = -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("physical id", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) { phys = std::atoi(line.c_str() + pos + 1); }
        } else if (line.rfind("core id", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) { core = std::atoi(line.c_str() + pos + 1); }
        } else if (line.empty()) {
            if (phys >= 0 && core >= 0) { cores.insert({phys, core}); }
            phys = -1;
            core = -1;
        }
    }
    if (phys >= 0 && core >= 0) { cores.insert({phys, core}); }
    return static_cast<int>(cores.size());
}
}  // namespace

bool ModelRuntime::init_threads(const ed_context_params_t& params, std::string* error) {
    (void)error;
    n_threads_ = params.n_threads;
    if (n_threads_ <= 0) {
        int physical = detect_physical_cores();
        int logical  = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        n_threads_   = physical > 0 ? physical : logical;
        LOG_INFO("auto thread count: %d (physical cores=%d, logical=%d)",
                 n_threads_, physical, logical);
    }
    return true;
}

bool ModelRuntime::init_flags(const ed_context_params_t& params, std::string* error) {
    (void)error;
    use_mmap_ = params.use_mmap;
    offload_params_to_cpu_ = params.offload_params_to_cpu;
    text_encoder_offload_ = params.text_encoder_offload;
    auto_allocate_ = params.auto_allocate;
    free_params_immediately_ = false;
    max_vram_ = params.max_vram_gb;
    max_graph_vram_bytes_ = max_vram_ <= 0.0f
                                 ? 0
                                 : static_cast<size_t>(static_cast<double>(max_vram_) * 1024.0 * 1024.0 * 1024.0);
    flash_attention_ = params.flash_attention;
    circular_x_ = false;
    circular_y_ = false;
    vae_tiling_ = params.vae_tiling;
    return true;
}

bool ModelRuntime::init_rng(const ed_context_params_t& params, std::string* error) {
    (void)params;
    (void)error;
    rng_ = std::make_shared<PhiloxRNG>();
    sampler_rng_ = rng_;
    return true;
}

bool ModelRuntime::init_backends(const ed_context_params_t& params, std::string* error) {
    const std::string requested_backend = requested_backend_name();
    const int gpu_device_ordinal = runtime_gpu_device_ordinal(parallel_enabled() ? parallel_context_->local_rank() : 0);
    if (is_auto_backend(requested_backend)) {
        backends_.backend = init_named_backend();
    } else {
        LOG_INFO("requested backend: %s", requested_backend.c_str());
        backends_.backend = init_explicit_backend(requested_backend, gpu_device_ordinal);
    }

    if (backends_.backend == nullptr) {
        std::string msg = is_auto_backend(requested_backend)
                              ? "failed to initialize default ggml backend"
                              : "failed to initialize requested ggml backend '" + requested_backend +
                                    "'; available backends: " + available_backend_names();
        return fail(error, msg);
    }
    LOG_INFO("default backend: %s", ggml_backend_name(backends_.backend));

    backends_.clip_backend = backends_.backend;
    if (params.keep_text_encoder_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.clip_backend = ggml_backend_cpu_init();
        if (backends_.clip_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for text encoder");
        }
        backends_.clip_owns_backend = true;
        LOG_INFO("text encoder backend: CPU");
    }

    backends_.vae_backend = backends_.backend;
    if (params.keep_vae_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.vae_backend = ggml_backend_cpu_init();
        if (backends_.vae_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for VAE");
        }
        backends_.vae_owns_backend = true;
        LOG_INFO("VAE backend: CPU");
    }

    backends_.control_net_backend = backends_.backend;
    if (params.keep_control_net_on_cpu && !ggml_backend_is_cpu(backends_.backend)) {
        backends_.control_net_backend = ggml_backend_cpu_init();
        if (backends_.control_net_backend == nullptr) {
            return fail(error, "failed to initialize CPU backend for ControlNet");
        }
        backends_.control_net_owns_backend = true;
        LOG_INFO("ControlNet backend: CPU");
    }

    maybe_enable_vae_tiling_for_low_vram();

    // Auto-derive a VRAM budget when the user enabled weight offload but gave no
    // explicit --max-vram. Without a budget, graph-cut segmentation is disabled and
    // offload_all_params() copies every weight back to the GPU at once, which OOMs
    // for large DiTs (e.g. FLUX ~22.7GB on a 24GB card). Segment the compute graph
    // against most of the device's free VRAM instead of failing.
    if ((offload_params_to_cpu_ || text_encoder_offload_) && max_graph_vram_bytes_ == 0 &&
        backends_.backend != nullptr && !ggml_backend_is_cpu(backends_.backend)) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backends_.backend);
        if (dev != nullptr) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            if (free_bytes > 0) {
                // Reserve headroom for fragmentation and non-graph allocations.
                max_graph_vram_bytes_ = static_cast<size_t>(static_cast<double>(free_bytes) * 0.85);
                LOG_INFO("offload enabled without --max-vram; auto graph VRAM budget = %.2f GB "
                         "(0.85 x %.2f GB free) to enable segmented compute",
                         max_graph_vram_bytes_ / (1024.0 * 1024.0 * 1024.0),
                         free_bytes / (1024.0 * 1024.0 * 1024.0));
            }
        }
    }

    return true;
}

// Adaptive offload decision (see header). Called by each pipeline's build_components
// once per major component (DiT / text-encoder / VAE), BEFORE the runner is
// constructed, because a runner's params_backend (GPU-resident vs CPU-staged) is
// fixed at construction. Only active under --auto-allocate; otherwise returns the
// legacy global offload flag so existing behavior is untouched.
//
// The budget is a HARD cap: effective = min(user --max-vram, live free VRAM). Each
// component's quantized weight bytes are compared against a running tally seeded with
// that budget; a resident component debits the tally. Components are decided in
// priority order DiT -> TE -> VAE (the caller passes them in that order), so the
// largest / most-reused weights get first claim on resident VRAM. After all three are
// decided the caller invokes finalize_auto_segment_budget() to set the graph VRAM
// budget for whatever ended up offloaded, using the leftover (budget - resident).
bool ModelRuntime::plan_component_offload(const ::ModelLoader& loader,
                                          const std::string& weight_prefix,
                                          size_t& remaining_free_bytes) {
    // Not in auto-allocate mode: keep legacy behavior (global offload flag).
    if (!auto_allocate_) {
        return offload_params_to_cpu_;
    }

    // If the runtime backend is CPU there is no GPU to fit into; offload is moot and
    // the resident path is correct (weights already live where compute happens).
    if (backends_.backend == nullptr || ggml_backend_is_cpu(backends_.backend)) {
        return offload_params_to_cpu_;
    }

    // Sum the component's weights using the EFFECTIVE (post-quantization) type:
    // set_wtype_override records the target type in expected_type, so nbytes() alone
    // (which uses `type`) would overestimate a q8/q4 component. Mirror the effective
    // -type logic used by collect_wtype_stat (model_loader.cpp).
    size_t comp_bytes = 0;
    for (const auto& item : loader.get_tensor_storage_map()) {
        if (item.first.rfind(weight_prefix, 0) != 0) {
            continue;  // not in this component
        }
        TensorStorage ts = item.second;
        if (ts.expected_type != GGML_TYPE_COUNT && ts.expected_type != ts.type) {
            ts.type = ts.expected_type;  // account for quantization
        }
        comp_bytes += static_cast<size_t>(ts.nbytes());
    }
    if (comp_bytes == 0) {
        // No weights matched this prefix (component absent) -> honor the global flag.
        return offload_params_to_cpu_;
    }

    // All-offload fallback: if the budget can't even fit one small component plus the
    // compute headroom, nothing can stay resident safely -> offload everything (legacy
    // --offload-to-cpu behavior, safest). Prevents a tiny-budget fully-resident from overshooting.
    if (remaining_free_bytes < kMinResidentComponentBytes + kResidentComputeHeadroom) {
        LOG_INFO("auto-allocate: '%s' budget %.2f GB too small for resident+compute headroom "
                 "-> OFFLOAD (all-offload fallback)",
                 weight_prefix.c_str(),
                 remaining_free_bytes / (1024.0 * 1024.0 * 1024.0));
        return true;
    }

    // Resident iff weights PLUS compute headroom fit the remaining budget. The headroom
    // (kResidentComputeHeadroom) covers the resident component's own compute buffer +
    // activations, which are NOT bounded by the segment budget (that only bounds
    // offloaded components). remaining_free_bytes is the running tally = effective_budget
    // minus components already decided resident.
    const bool fits = comp_bytes + kResidentComputeHeadroom <= remaining_free_bytes;

    if (fits) {
        remaining_free_bytes -= comp_bytes;   // this component sits resident on GPU
        resident_bytes_total_ += comp_bytes;  // accumulated for finalize_auto_segment_budget()
        LOG_INFO("auto-allocate: '%s' %.2f GB -> RESIDENT (%.2f GB budget left)",
                 weight_prefix.c_str(),
                 comp_bytes / (1024.0 * 1024.0 * 1024.0),
                 remaining_free_bytes / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    LOG_INFO("auto-allocate: '%s' %.2f GB > %.2f GB budget left -> OFFLOAD+segment",
             weight_prefix.c_str(),
             comp_bytes / (1024.0 * 1024.0 * 1024.0),
             remaining_free_bytes / (1024.0 * 1024.0 * 1024.0));
    return true;
}

// After all components of a pipeline have been decided via plan_component_offload,
// set the graph VRAM budget for the offloaded ones: whatever budget is left after the
// resident components, minus compute headroom. Because graph_cut_segment_vram_bytes
// already counts each segment's compute + weights + IO, this cap keeps
// (resident + max_segment) within the effective budget. No-op outside auto-allocate.
void ModelRuntime::finalize_auto_segment_budget(size_t effective_budget_bytes) {
    if (!auto_allocate_) {
        return;
    }
    size_t leftover = 0;
    if (effective_budget_bytes > resident_bytes_total_ + kSegmentBudgetSlack) {
        leftover = effective_budget_bytes - resident_bytes_total_ - kSegmentBudgetSlack;
    }
    // Floor: if the leftover is tiny (resident nearly filled the budget) an offloaded
    // component still needs *some* budget to segment against; use a 1 GB floor so a
    // single segment can at least stage. Better a large segment than a hard abort.
    const size_t kMinSegmentBudget = static_cast<size_t>(1) * 1024 * 1024 * 1024;
    if (leftover < kMinSegmentBudget) {
        leftover = kMinSegmentBudget;
    }
    max_graph_vram_bytes_ = leftover;
    LOG_INFO("auto-allocate: segment budget = %.2f GB (effective %.2f GB - resident %.2f GB - headroom)",
             max_graph_vram_bytes_ / (1024.0 * 1024.0 * 1024.0),
             effective_budget_bytes / (1024.0 * 1024.0 * 1024.0),
             resident_bytes_total_ / (1024.0 * 1024.0 * 1024.0));
}

size_t ModelRuntime::effective_budget_bytes() const {
    if (backends_.backend == nullptr || ggml_backend_is_cpu(backends_.backend)) {
        return 0;
    }
    size_t live_free = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends_.backend);
    if (dev != nullptr) {
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &live_free, &total_bytes);
    }
    // max_graph_vram_bytes_ holds the user's --max-vram (bytes) here, 0 if unset.
    if (max_graph_vram_bytes_ > 0 && max_graph_vram_bytes_ < live_free) {
        return max_graph_vram_bytes_;  // user budget is the tighter (hard) cap
    }
    return live_free;
}

// so consumer cards stay under their VRAM wall without a manual flag.
void ModelRuntime::maybe_enable_vae_tiling_for_low_vram() {
    if (vae_tiling_.enabled || vae_tiling_.force_disable) {
        return;  // user explicitly set tiling on or off; respect it, skip low-VRAM auto-enable
    }
    if (backends_.vae_backend == nullptr || ggml_backend_is_cpu(backends_.vae_backend)) {
        return;  // VAE runs on CPU, GPU tiling does not apply
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backends_.vae_backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return;
    }
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    const double total_gib = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    constexpr double kLowVramThresholdGiB = 25.0;
    if (total_bytes == 0 || total_gib > kLowVramThresholdGiB) {
        return;  // large GPU: leave VAE untiled for max throughput
    }
    vae_tiling_.enabled        = true;
    vae_tiling_.rel_size_x     = 5.0f;  // ~32x32 latent tile: matches min VAE peak (empirically measured)
    vae_tiling_.rel_size_y     = 5.0f;
    if (vae_tiling_.target_overlap <= 0.0f) {
        vae_tiling_.target_overlap = 0.25f;
    }
    LOG_INFO("auto-enabled VAE tiling (GPU total VRAM %.1f GiB <= %.0f GiB threshold)",
             total_gib, kLowVramThresholdGiB);
}

void ModelRuntime::reset() {
    ready_ = false;
    rng_.reset();
    sampler_rng_.reset();
    release_backends();

    n_threads_ = 0;
    use_mmap_ = false;
    offload_params_to_cpu_ = false;
    text_encoder_offload_ = false;
    free_params_immediately_ = false;
    max_vram_ = 0.0f;
    max_graph_vram_bytes_ = 0;
    flash_attention_ = false;
    circular_x_ = false;
    circular_y_ = false;
}

void ModelRuntime::release_backends() {
    if (backends_.control_net_owns_backend && backends_.control_net_backend != nullptr) {
        ggml_backend_free(backends_.control_net_backend);
    }
    if (backends_.vae_owns_backend && backends_.vae_backend != nullptr) {
        ggml_backend_free(backends_.vae_backend);
    }
    if (backends_.clip_owns_backend && backends_.clip_backend != nullptr) {
        ggml_backend_free(backends_.clip_backend);
    }
    if (backends_.backend != nullptr) {
        ggml_backend_free(backends_.backend);
    }
    backends_ = {};
}

bool ModelRuntime::fail(std::string* error, const std::string& msg) {
    if (error != nullptr) {
        *error = msg;
    }
    LOG_ERROR("%s", msg.c_str());
    reset();
    return false;
}

} // namespace edgedit
