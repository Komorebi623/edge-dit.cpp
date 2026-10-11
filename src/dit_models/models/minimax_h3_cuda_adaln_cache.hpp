#pragma once

// H3-only, opt-in, exact cache of COMPLETE adaln_proj.linear outputs. This
// header does not modify a model, an environment variable, or a shared backend.
// No neighbouring-step approximation, dtype conversion, or weight release.
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace MiniMaxH3 {
namespace ExactAdaLN {

enum class Status {
    Disabled, Hit, Miss, Stored, Duplicate, InvalidContract, InvalidPlan,
    ShapeMismatch, NonFinite, BudgetExceeded, Stale, UnsupportedType,
    BackendRejected, AllocationFailure
};
inline const char* status_name(Status s) {
    switch (s) {
#define H3_ADALN_STATUS(x) case Status::x: return #x
        H3_ADALN_STATUS(Disabled); H3_ADALN_STATUS(Hit); H3_ADALN_STATUS(Miss);
        H3_ADALN_STATUS(Stored); H3_ADALN_STATUS(Duplicate);
        H3_ADALN_STATUS(InvalidContract); H3_ADALN_STATUS(InvalidPlan);
        H3_ADALN_STATUS(ShapeMismatch); H3_ADALN_STATUS(NonFinite);
        H3_ADALN_STATUS(BudgetExceeded); H3_ADALN_STATUS(Stale);
        H3_ADALN_STATUS(UnsupportedType); H3_ADALN_STATUS(BackendRejected);
        H3_ADALN_STATUS(AllocationFailure);
#undef H3_ADALN_STATUS
    }
    return "Unknown";
}

inline uint32_t float_bits(float x) {
    uint32_t bits;
    static_assert(sizeof(x) == sizeof(bits), "H3 requires fp32");
    std::memcpy(&bits, &x, sizeof(bits));
    return bits; // +0/-0 and adjacent representable timesteps stay distinct.
}

using Shape = std::array<int64_t, GGML_MAX_DIMS>;
enum class Role : uint32_t { Text, ConditionVideo, ConditionAudio, TargetAudio, TargetVideo };
enum class CfgRole : uint32_t { Conditional, Unconditional, Joint };

struct Span {
    int64_t begin = 0, end = 0;
    Role role = Role::Text;
    int32_t source_index = -1;
    uint32_t timestep_row = 0, modality = 1; // H3: video=0, text=1, audio=2.
};
struct StepPlan {
    size_t schedule_index = 0;
    float sigma = 0.f;
    CfgRole cfg_role = CfgRole::Conditional;
    std::vector<float> timesteps; // EXACT order from PackedSequenceLayout, not sorted.
    std::vector<Span> spans;      // Include text-tag runs and reference source indices.
    uint32_t final_video_row = 0, final_audio_row = 0; // final has ONE modality.
};
struct Projection {
    std::string name;
    int64_t hidden = 0;
    uint32_t expand = 0, modalities = 0;
};
inline std::vector<Projection> h3_projections(int64_t hidden, size_t layers) {
    std::vector<Projection> result;
    for (size_t i = 0; i < layers; ++i)
        result.push_back({"blocks." + std::to_string(i) + ".adaln_proj.linear", hidden, 6, 3});
    result.push_back({"final_layer.adaln_proj.linear", hidden, 2, 1});
    return result;
}

// Identity is caller-supplied, NEVER inferred from a path, partial AdaLN digest,
// or tensor address. Use a full-checkpoint manifest digest (all shards + config)
// plus a fresh monotonically changing load_generation on every in-place reload.
// adapter_identity must encode every adapter, its content/revision AND scales;
// use explicit "none" for no adapter and bump adapter_generation on mutation.
struct RequestContract {
    std::string full_weights_identity, adapter_identity, cfg_identity, math_identity;
    uint64_t load_generation = 0, adapter_generation = 0;
    // cfg_identity includes batch/branch order, guidance scales and conditioning
    // identity. math_identity includes backend/device, precision/TF32, TP rank,
    // ggml/kernel revision and adapter execution policy. No cross-device reuse.
    std::string schedule_identity; // scheduler/solver/version, even for same sigmas.
    float video_shift = 1.f, audio_shift = 1.f;
    int64_t timestep_input_dim = 256, time_embed_hidden = 5376, time_embed_dim = 2688;
    int64_t curve_grid = 0; // 0: time_embedder + SiLU; >=2: table lerp, NO SiLU.
    std::vector<float> sigmas; // FULL fixed schedule, including terminal endpoint.
    std::vector<StepPlan> plans; // one per evaluation/CFG branch (can repeat sigma).
    std::vector<Projection> projections;
};
// Adapter for the CURRENT H3 PackedSequenceLayout (no dependency on its
// header). Intersect text-tag modulation runs with role/source segments so
// equal timesteps with different semantic layouts cannot alias. Enum values
// are checked against H3's TEXT/CONDITION_VIDEO/CONDITION_AUDIO/TARGET_AUDIO/
// TARGET_VIDEO order. Use the actual sigma after pipeline's *1000 /1000 roundtrip.
template<class Layout>
StepPlan plan_from_layout(size_t schedule_index, float sigma, CfgRole cfg_role,
                          const Layout& layout) {
    StepPlan plan;
    plan.schedule_index = schedule_index; plan.sigma = sigma; plan.cfg_role = cfg_role;
    plan.timesteps = layout.timesteps;
    plan.final_video_row = static_cast<uint32_t>(layout.video_segment.modulation_row);
    plan.final_audio_row = static_cast<uint32_t>(layout.audio_segment.modulation_row);
    for (const auto& seq : layout.sequence_segments) {
        for (const auto& mod : layout.segments) {
            const int64_t begin = std::max(seq.start, mod.start);
            const int64_t end = std::min(seq.end, mod.end);
            if (begin < end) {
                const int64_t row = mod.modulation_row;
                plan.spans.push_back({begin, end, static_cast<Role>(seq.kind), seq.source_index,
                                      static_cast<uint32_t>(row / 3), static_cast<uint32_t>(row % 3)});
            }
        }
    }
    return plan;
}

struct Options {
    bool enabled = false;
    size_t max_bytes = 256u * 1024u * 1024u;
    size_t max_entries = 16384;
};
struct Statistics {
    uint64_t hits = 0, misses = 0, stores = 0, budget_skips = 0;
};

namespace detail {
inline bool mul(size_t a, size_t b, size_t* out) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    *out = a * b; return true;
}
inline bool projection_shape(const Projection& p, size_t rows, Shape* shape, size_t* bytes) {
    size_t width, n;
    if (p.hidden <= 0 || !p.expand || !p.modalities || !rows ||
        !mul(static_cast<size_t>(p.hidden), p.expand, &width) ||
        !mul(width, p.modalities, &width) || width > static_cast<size_t>(INT64_MAX) ||
        rows > static_cast<size_t>(INT64_MAX) || !mul(width, rows, &n) ||
        !mul(n, sizeof(float), bytes)) return false;
    *shape = {static_cast<int64_t>(width), static_cast<int64_t>(rows), 1, 1};
    return true;
}
// Bounded canonical encoding, exact equality, no hash collision assumptions.
// Integer encoding is explicitly little-endian and float encoding is bitwise.
struct Encoder {
    std::string data;
    size_t limit;
    bool ok = true;
    explicit Encoder(size_t n) : limit(n) {}
    void raw(const char* p, size_t n) {
        if (!ok || n > limit - data.size()) { ok = false; return; }
        data.append(p, n);
    }
    void u64(uint64_t v) {
        char b[8]; for (int i = 0; i < 8; ++i) b[i] = static_cast<char>(v >> (8*i));
        raw(b, 8);
    }
    void f32(float v) { u64(float_bits(v)); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void plan(const StepPlan& p) {
        u64(p.schedule_index); f32(p.sigma); u64(static_cast<uint32_t>(p.cfg_role));
        u64(p.timesteps.size()); for (float t : p.timesteps) f32(t);
        u64(p.spans.size());
        for (const auto& s : p.spans) {
            u64(s.begin); u64(s.end); u64(static_cast<uint32_t>(s.role));
            u64(static_cast<uint32_t>(s.source_index)); u64(s.timestep_row); u64(s.modality);
        }
        u64(p.final_video_row); u64(p.final_audio_row);
    }
    void contract(const RequestContract& c) {
        str("H3-exact-AdaLN-v1"); str(c.full_weights_identity); u64(c.load_generation);
        str(c.adapter_identity); u64(c.adapter_generation); str(c.cfg_identity);
        str(c.math_identity); str(c.schedule_identity); f32(c.video_shift); f32(c.audio_shift);
        u64(c.timestep_input_dim); u64(c.time_embed_hidden); u64(c.time_embed_dim); u64(c.curve_grid);
        u64(c.sigmas.size()); for (float s : c.sigmas) f32(s);
        u64(c.plans.size()); for (const auto& p : c.plans) plan(p);
        u64(c.projections.size()); for (const auto& p : c.projections) {
            str(p.name); u64(p.hidden); u64(p.expand); u64(p.modalities);
        }
    }
};
inline bool unit_finite(float x) { return std::isfinite(x) && x >= 0.f && x <= 1.f; }
inline bool valid(const RequestContract& c) {
    if (c.full_weights_identity.empty() || c.adapter_identity.empty() || c.cfg_identity.empty() ||
        c.math_identity.empty() || c.schedule_identity.empty() ||
        !std::isfinite(c.video_shift) || c.video_shift <= 0 ||
        !std::isfinite(c.audio_shift) || c.audio_shift <= 0 ||
        c.timestep_input_dim <= 0 || c.time_embed_hidden <= 0 || c.time_embed_dim <= 0 ||
        c.curve_grid < 0 || c.curve_grid == 1 || c.sigmas.size() < 2 ||
        c.plans.empty() || c.projections.empty()) return false;
    for (float s : c.sigmas) if (!unit_finite(s)) return false;
    std::unordered_set<std::string> names;
    for (const auto& p : c.projections) {
        Shape shape; size_t bytes;
        if (p.name.empty() || !names.insert(p.name).second ||
            !projection_shape(p, 1, &shape, &bytes)) return false;
    }
    for (const auto& p : c.plans) {
        if (p.schedule_index >= c.sigmas.size()-1 || !unit_finite(p.sigma) ||
            float_bits(p.sigma) != float_bits(c.sigmas[p.schedule_index]) ||
            static_cast<uint32_t>(p.cfg_role) > static_cast<uint32_t>(CfgRole::Joint) ||
            p.timesteps.empty() || p.final_video_row >= p.timesteps.size() ||
            p.final_audio_row >= p.timesteps.size()) return false;
        for (float t : p.timesteps) if (!unit_finite(t)) return false;
        int64_t end = 0;
        for (const auto& s : p.spans) {
            if (s.begin < end || s.begin < 0 || s.end <= s.begin || s.source_index < -1 ||
                s.timestep_row >= p.timesteps.size() || s.modality >= 3 ||
                static_cast<uint32_t>(s.role) > static_cast<uint32_t>(Role::TargetVideo)) return false;
            if ((s.role == Role::ConditionVideo || s.role == Role::TargetVideo) && s.modality != 0) return false;
            if ((s.role == Role::ConditionAudio || s.role == Role::TargetAudio) && s.modality != 2) return false;
            end = s.end;
        }
    }
    return true;
}
inline Shape tensor_shape(const ggml_tensor* t) { return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}; }
} // namespace detail

class GraphInputs;
// One worker/request at a time. Do not share across concurrent graph builds.
// A token is valid only until reset()/begin_request(); async readbacks from an
// old request cannot insert into a new request. GraphInputs must outlive graph
// execution and be destroyed BEFORE the Cache (including on exceptions).
class Cache {
public:
    class StepToken {
        friend class Cache;
        const Cache* owner_ = nullptr;
        uint64_t generation_ = 0;
        size_t plan_ = 0;
    public:
        StepToken() = default;
        explicit operator bool() const { return owner_ != nullptr; }
    };
private:
    friend class GraphInputs;
    struct Slot {
        Shape shape{};
        size_t bytes = 0;
        std::unique_ptr<float[]> data;
        ggml_tensor* pending = nullptr; // valid only inside intercepted graph execution.
    };
    Options options_;
    Statistics stats_;
    uint64_t generation_ = 0;
    std::unique_ptr<char[]> identity_;
    size_t identity_bytes_ = 0, slot_count_ = 0, projection_count_ = 0;
    std::unique_ptr<Slot[]> slots_;
    size_t persistent_bytes_ = 0, staging_bytes_ = 0;
    bool reserve(size_t bytes) {
        if (bytes > options_.max_bytes - used_bytes()) { ++stats_.budget_skips; return false; }
        return true;
    }
    Slot* slot(StepToken token, size_t projection) {
        if (token.owner_ != this || token.generation_ != generation_ || !slots_ ||
            projection >= projection_count_ || token.plan_ >= slot_count_/projection_count_) return nullptr;
        return &slots_[token.plan_*projection_count_ + projection];
    }
    Status allocate(Slot& s, std::unique_ptr<float[]>* data) {
        if (!reserve(s.bytes)) return Status::BudgetExceeded;
        try { data->reset(new float[s.bytes / sizeof(float)]); }
        catch (const std::bad_alloc&) { return Status::AllocationFailure; }
        return Status::Stored;
    }
    Status commit(Slot& s, std::unique_ptr<float[]> data) {
        for (size_t i = 0; i < s.bytes/sizeof(float); ++i)
            if (!std::isfinite(data[i])) return Status::NonFinite;
        s.data = std::move(data); persistent_bytes_ += s.bytes; ++stats_.stores;
        return Status::Stored;
    }
public:
    explicit Cache(Options options = {}) : options_(options) {}
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    size_t used_bytes() const { return persistent_bytes_ + staging_bytes_; }
    size_t max_bytes() const { return options_.max_bytes; }
    const Statistics& statistics() const { return stats_; }
    size_t projection_count() const { return projection_count_; }
    void reset() {
        ++generation_; // Never reset to zero: stale graph tokens must stay stale.
        slots_.reset(); identity_.reset(); identity_bytes_ = slot_count_ = projection_count_ = 0;
        persistent_bytes_ = 0; stats_ = {};
        // Existing graph input copies remain live and budgeted until their owner dies.
    }
    Status begin_request(const RequestContract& c) {
        reset(); // Even malformed/disabled requests invalidate the previous request.
        if (!options_.enabled) return Status::Disabled;
        if (!detail::valid(c)) return Status::InvalidContract;
        size_t count, descriptors;
        if (!detail::mul(c.plans.size(), c.projections.size(), &count) ||
            count > options_.max_entries || !detail::mul(count, sizeof(Slot), &descriptors))
            return Status::BudgetExceeded;
        try {
            detail::Encoder enc(options_.max_bytes); enc.contract(c);
            if (!enc.ok || descriptors > options_.max_bytes ||
                enc.data.size() > options_.max_bytes - descriptors ||
                !reserve(descriptors + enc.data.size())) return Status::BudgetExceeded;
            auto slots = std::unique_ptr<Slot[]>(new Slot[count]);
            for (size_t i = 0; i < c.plans.size(); ++i) {
                for (size_t j = 0; j < c.projections.size(); ++j) {
                    auto& s = slots[i*c.projections.size()+j];
                    if (!detail::projection_shape(c.projections[j], c.plans[i].timesteps.size(),
                                                  &s.shape, &s.bytes)) return Status::InvalidContract;
                }
            }
            auto identity = std::unique_ptr<char[]>(new char[enc.data.size()]);
            std::memcpy(identity.get(), enc.data.data(), enc.data.size());
            identity_bytes_ = enc.data.size(); identity_ = std::move(identity);
            slot_count_ = count; projection_count_ = c.projections.size(); slots_ = std::move(slots);
            persistent_bytes_ = descriptors + identity_bytes_;
        } catch (const std::bad_alloc&) { return Status::AllocationFailure; }
        return Status::Stored;
    }
    // Validate ONCE per model evaluation, not once per layer. The caller must
    // rebind after ANY model/adapter/CFG/math/layout mutation, even mid-request.
    Status bind_step(const RequestContract& actual_contract, size_t plan_index,
                     const StepPlan& actual_plan, StepToken* token) {
        if (!token) return Status::InvalidPlan;
        *token = StepToken{};
        if (!options_.enabled) return Status::Disabled;
        if (!identity_) return Status::Miss;
        try {
            detail::Encoder enc(identity_bytes_); enc.contract(actual_contract);
            if (!enc.ok || enc.data.size() != identity_bytes_ ||
                std::memcmp(identity_.get(), enc.data.data(), identity_bytes_) != 0) return Status::Miss;
            if (plan_index >= actual_contract.plans.size()) return Status::InvalidPlan;
            detail::Encoder a(identity_bytes_), b(identity_bytes_);
            a.plan(actual_plan); b.plan(actual_contract.plans[plan_index]);
            if (!a.ok || !b.ok || a.data != b.data) return Status::Miss;
            token->owner_ = this; token->generation_ = generation_; token->plan_ = plan_index;
            return Status::Hit; // Contract hit; individual layer payloads may still miss.
        } catch (const std::bad_alloc&) { return Status::AllocationFailure; }
    }
    Status expected_shape(StepToken token, size_t projection, Shape* out) {
        auto* s = slot(token, projection);
        if (!s || !out) return Status::Stale;
        *out = s->shape; return Status::Hit;
    }
    Status lookup(StepToken token, size_t projection, const Shape& expected, const float** out) {
        if (!out) return Status::InvalidPlan;
        *out = nullptr;
        auto* s = slot(token, projection);
        if (!s) return Status::Stale;
        if (expected != s->shape) return Status::ShapeMismatch;
        if (!s->data) { ++stats_.misses; return Status::Miss; }
        *out = s->data.get(); ++stats_.hits; return Status::Hit;
    }
    Status insert(StepToken token, size_t projection, const Shape& shape,
                  const float* values, size_t elements) {
        auto* s = slot(token, projection);
        if (!s) return Status::Stale;
        if (shape != s->shape || elements != s->bytes/sizeof(float)) return Status::ShapeMismatch;
        if (!values) return Status::InvalidPlan;
        if (s->data) return Status::Duplicate; // First exact result is immutable.
        std::unique_ptr<float[]> data;
        Status status = allocate(*s, &data); if (status != Status::Stored) return status;
        std::memcpy(data.get(), values, s->bytes); return commit(*s, std::move(data));
    }
    // Pin only a cacheable miss. ggml_set_output prevents gallocr reusing its
    // buffer before post_compute readback; do not read an unpinned intermediate.
    Status intercept(StepToken token, size_t projection, ggml_tensor* tensor) {
        auto* s = slot(token, projection);
        if (!s) return Status::Stale;
        if (!tensor || detail::tensor_shape(tensor) != s->shape) return Status::ShapeMismatch;
        if (tensor->type != GGML_TYPE_F32) return Status::UnsupportedType;
        if (!ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != s->bytes) return Status::ShapeMismatch;
        if (s->data) return Status::Duplicate;
        if (s->pending && s->pending != tensor) return Status::InvalidPlan;
        if (!reserve(s->bytes)) return Status::BudgetExceeded;
        char name[80];
        if (projection + 1 == projection_count_) std::snprintf(name, sizeof(name), "h3_exact_adaln.final");
        else std::snprintf(name, sizeof(name), "h3_exact_adaln.block.%zu", projection);
        ggml_set_name(tensor, name);
        ggml_set_output(tensor); s->pending = tensor; return Status::Miss;
    }
    void discard_pending(StepToken token) {
        for (size_t j = 0; j < projection_count_; ++j) {
            auto* s = slot(token, j); if (s) s->pending = nullptr;
        }
    }
    // Invoke immediately after successful graph compute, BEFORE buffer/context
    // free or graph-cut reset. Never after compute failure. Copies backend-native
    // F32 bytes directly into budgeted storage; does not retain a device buffer.
    Status capture_pending(StepToken token, ggml_backend_t backend) {
        if (!slot(token, 0)) return Status::Stale;
        if (!backend) { discard_pending(token); return Status::BackendRejected; }
        ggml_backend_synchronize(backend);
        Status result = Status::Stored;
        for (size_t j = 0; j < projection_count_; ++j) {
            auto* s = slot(token, j);
            auto* tensor = s->pending; s->pending = nullptr;
            if (!tensor || s->data) continue;
            if (!tensor->data || !tensor->buffer) { result = Status::InvalidPlan; continue; }
            std::unique_ptr<float[]> data;
            Status status = allocate(*s, &data);
            if (status == Status::Stored) {
                ggml_backend_tensor_get(tensor, data.get(), 0, s->bytes);
                status = commit(*s, std::move(data));
            }
            if (status != Status::Stored) result = status;
        }
        return result; // Partial coverage is legitimate; unfilled layers remain misses.
    }
    bool complete(StepToken token) {
        if (!slot(token, 0)) return false;
        for (size_t j = 0; j < projection_count_; ++j) if (!slot(token, j)->data) return false;
        return true;
    }
};

inline bool is_cuda_backend(ggml_backend_t backend) {
    if (!backend) return false;
    const char* name = ggml_backend_name(backend);
    return name && std::strncmp(name, "CUDA", 4) == 0 &&
           (name[4] == '\0' || (name[4] >= '0' && name[4] <= '9'));
}
inline Status begin_cuda_request(Cache& cache, const RequestContract& contract, ggml_backend_t backend) {
    if (!is_cuda_backend(backend)) { cache.reset(); return Status::BackendRejected; }
    return cache.begin_request(contract);
}

// Budgeted, graph-owned host staging. Cache eviction/reset cannot invalidate
// bytes already handed to GGMLRunnerContext::bind_backend_tensor_data.
// Counts descriptor and payload allocations together with persistent cache.
class GraphInputs {
    struct Record { std::unique_ptr<float[]> data; };
    Cache& cache_;
    std::unique_ptr<Record[]> records_;
    size_t capacity_ = 0, count_ = 0, bytes_ = 0;
public:
    explicit GraphInputs(Cache& cache, size_t max_inputs) : cache_(cache) {
        size_t bytes;
        if (!detail::mul(max_inputs, sizeof(Record), &bytes) || !cache_.reserve(bytes)) return;
        try { records_.reset(new Record[max_inputs]); }
        catch (const std::bad_alloc&) { return; }
        capacity_ = max_inputs; bytes_ = bytes; cache_.staging_bytes_ += bytes;
    }
    ~GraphInputs() { cache_.staging_bytes_ -= bytes_; }
    GraphInputs(const GraphInputs&) = delete;
    GraphInputs& operator=(const GraphInputs&) = delete;
    // Context can be GGMLRunnerContext; keeping this templated avoids pulling
    // every shared backend and model dependency into an isolated CPU test.
    template<class Context>
    Status import(Context* ctx, Cache::StepToken token, size_t projection,
                  const Shape& expected, ggml_tensor** out) {
        if (!out) return Status::InvalidPlan;
        *out = nullptr;
        if (!ctx || !ctx->ggml_ctx || !ctx->bind_backend_tensor_data) return Status::InvalidPlan;
        const float* values;
        Status status = cache_.lookup(token, projection, expected, &values);
        if (status != Status::Hit) return status;
        auto* s = cache_.slot(token, projection);
        if (count_ == capacity_ || !cache_.reserve(s->bytes)) return Status::BudgetExceeded;
        std::unique_ptr<float[]> copy;
        try { copy.reset(new float[s->bytes/sizeof(float)]); }
        catch (const std::bad_alloc&) { return Status::AllocationFailure; }
        std::memcpy(copy.get(), values, s->bytes);
        auto* tensor = ggml_new_tensor(ctx->ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, expected.data());
        ggml_set_input(tensor);
        auto* stable = copy.get(); records_[count_++].data = std::move(copy);
        bytes_ += s->bytes; cache_.staging_bytes_ += s->bytes;
        ctx->bind_backend_tensor_data(tensor, stable); *out = tensor;
        return Status::Hit;
    }
};

// Minimal model hook: replace adaln->forward(ctx,t_emb) with this call. A hit
// never invokes project(), thus removes SiLU/linear/bias/adapter projection
// nodes from the graph. On any unusable cache path compute the original graph.
template<class Context, class Project>
ggml_tensor* projection_or_compute(Cache& cache, Cache::StepToken token, size_t index,
                                   Context* ctx, GraphInputs& inputs, Project project,
                                   Status* cache_status = nullptr) {
    Shape shape{}; ggml_tensor* result = nullptr;
    Status status = cache.expected_shape(token, index, &shape);
    if (status == Status::Hit) status = inputs.import(ctx, token, index, shape, &result);
    if (status == Status::Hit) {
        if (cache_status) *cache_status = status;
        return result;
    }
    result = project();
    // Do not overwrite a shape/type error or pin a budget-rejected fallback.
    if (status == Status::Miss) status = cache.intercept(token, index, result);
    if (cache_status) *cache_status = status;
    return result;
}

// Build only existing H3 time_embedding + each adaln->forward output. The
// producer supplied by H3 uses its REAL table/time embedder/Linear/adapter,
// not an independently implemented GEMM. No transformer/attention needed.
template<class Produce>
ggml_cgraph* build_projection_graph(ggml_context* ctx, Cache& cache,
                                    Cache::StepToken token, size_t graph_size, Produce produce) {
    auto* graph = ggml_new_graph_custom(ctx, graph_size, false);
    for (size_t i = 0; i < cache.projection_count(); ++i) {
        auto* output = produce(i);
        if (!output) { cache.discard_pending(token); return nullptr; }
        Status status = cache.intercept(token, i, output);
        if (status != Status::Miss && status != Status::Duplicate) {
            cache.discard_pending(token); return nullptr;
        }
        ggml_build_forward_expand(graph, output);
    }
    return graph;
}

// GGMLRunner's params_ctx/prepare_compute_graph/execute_graph are protected.
// Derive H3's runner from RunnerBridge<GGMLRunner> and initialize that base;
// no shared Runner edits, no unsafe downcasts, no exposing all internals.
// The bridge is a dependent template: include ggml_extend.hpp before
// instantiating it, while the standalone cache still needs only ggml headers.
template<class BaseRunner>
class RunnerBridge : public BaseRunner {
public:
    using BaseRunner::BaseRunner;
    using ExactGraphBuilder = std::function<ggml_cgraph*()>;
    bool exact_prepare_graph(ExactGraphBuilder build, ggml_cgraph** graph) {
        return this->prepare_compute_graph(std::move(build), graph);
    }
    ggml_cgraph* exact_empty_graph() { return this->new_graph_custom(16); }
    ggml_context* exact_params_context() { return this->params_ctx; }
    ggml_backend_t exact_runtime_backend() { return this->runtime_backend; }
    bool exact_execute_graph(ggml_cgraph* graph, int threads,
                             const std::vector<ggml_tensor*>& params,
                             std::function<bool(ggml_cgraph*)> post) {
        return this->template execute_graph<float>(graph, threads, true, params,
                                                    false, true, nullptr, std::move(post)).has_value();
    }
};

// Existing GGMLRunner protected params_ctx + execute_graph(post_compute_cb)
// already supply the needed interception point. Collect ONLY reachable params
// for partial offload; don't stage an entire multi-billion-parameter H3 model.
// Instantiate with GGMLRunner after including backend/ggml/ggml_extend.hpp.
template<class Runner, class Build>
Status run_projection_warmup(Runner& runner, Cache& cache, Cache::StepToken token,
                             int n_threads, Build build) {
    struct PendingGuard {
        Cache& cache; Cache::StepToken token;
        ~PendingGuard() { cache.discard_pending(token); }
    } guard{cache, token};
    Shape first{};
    if (cache.expected_shape(token, 0, &first) != Status::Hit || n_threads <= 0) return Status::InvalidPlan;
    ggml_cgraph* graph = nullptr;
    bool built = true;
    // GGMLRunner::get_compute_graph dereferences the callback result before
    // prepare_compute_graph can check null. Supply an empty graph on builder
    // failure, then abort BEFORE execution; never dereference a failed builder.
    auto safe_build = [&]() -> ggml_cgraph* {
        auto* g = build();
        if (!g) { built = false; return runner.exact_empty_graph(); }
        return g;
    };
    if (!runner.exact_prepare_graph(safe_build, &graph) || !graph || !built)
        return Status::InvalidPlan;
    std::unordered_set<ggml_tensor*> params, visited;
    for (auto* p = ggml_get_first_tensor(runner.exact_params_context()); p;
         p = ggml_get_next_tensor(runner.exact_params_context(), p)) params.insert(p);
    std::vector<ggml_tensor*> selected, todo;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) todo.push_back(ggml_graph_node(graph, i));
    while (!todo.empty()) {
        auto* t = todo.back(); todo.pop_back();
        if (!t || !visited.insert(t).second) continue;
        if (params.count(t)) selected.push_back(t);
        for (auto* src : t->src) if (src) todo.push_back(src);
    }
    Status captured = Status::Miss;
    auto post = [&](ggml_cgraph*) {
        captured = cache.capture_pending(token, runner.exact_runtime_backend());
        return captured == Status::Stored;
    };
    // no_return=true: don't also copy the last large projection as a model output.
    const bool result = runner.exact_execute_graph(graph, n_threads, selected, post);
    if (!result && captured == Status::Miss) return Status::InvalidPlan;
    return captured;
}

// Request-start all-sigma prefill, not a lazy first-denoise miss cache. Build
// must return a projection-only graph for contract.plans[plan_index]. The same
// contract is used by bind_step in ordinary compute; a different identity fails
// closed. Keep ordinary compiled graph reuse disabled unless it refreshes these
// imported leaves (constant shape does NOT imply constant sigma/cache values).
template<class Runner, class Build>
Status prefill_request(Runner& runner, Cache& cache, const RequestContract& contract,
                       int n_threads, Build build) {
    for (size_t i = 0; i < contract.plans.size(); ++i) {
        Cache::StepToken token;
        Status status = cache.bind_step(contract, i, contract.plans[i], &token);
        if (status != Status::Hit) return status;
        status = run_projection_warmup(runner, cache, token, n_threads,
                                      [&]() { return build(i, token); });
        if (status != Status::Stored) return status;
        if (!cache.complete(token)) return Status::Miss;
    }
    return Status::Stored;
}

} // namespace ExactAdaLN
} // namespace MiniMaxH3
