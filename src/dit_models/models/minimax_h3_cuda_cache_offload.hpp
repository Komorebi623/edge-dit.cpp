#pragma once

// H3-only, opt-in Cache-DiT serial parameter-offload seam. No global backend or
// env changes. Source contract verified against cache-dit a7898aaf1ba2c2aabdd94a9526fb5fb82ddc900e
// pattern_3_4_5.py / pattern_base.py and SGLang 53295c697255a5e8ddb263f98ab7e047e2700ad2
// cache_dit_integration.py (MiniMaxH3DiTModel -> Pattern_3).
// Same algorithm: RF=front-input, delta=middle-front, current front+anchor delta
// on a hit, ALWAYS execute suffix. H3 differences remain deliberate: per-span
// RelL1 (the same-full-step RF/delta anchor also matches upstream), zero norm
// fails closed, two CFG
// slots, request/schedule/conditioning identity invalidation. No calibrator,
// pruning, precision changes, or upstream Python async group-offload hooks.
// Existing H3 defaults F/B/W/MC=2/2/2/2, threshold=.04, window=.15..85;
// SGLang generic defaults are 1/0/4/3, .24 (NOT a claimed H3 tuned recipe).
//
// Instantiate RunnerBridge<ExactAdaLN::RunnerBridge<GGMLRunner>> only for H3.
// The VRAM accounting covers planner-managed buffers, not a proof of every
// CUDA library/workspace allocation; full-model peak validation is still needed.
// This header itself does not enable production dispatch. Keep the resident
// guard until the caller wires compute(), mode checks and request reset, and
// actual CUDA/Q4/offload validation is complete. The path is serial, not async.
#include "dit_models/models/minimax_h3_cuda_cache.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "backend/ggml/ggml_graph_cut.h"
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <unordered_set>

namespace MiniMaxH3 { namespace CacheOffload {

inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("MiniMax-H3 Cache-DiT offload: ") + message);
}
inline bool is_cuda(ggml_backend_t backend) {
    const char* n = backend ? ggml_backend_name(backend) : nullptr;
    return n && std::strncmp(n, "CUDA", 4) == 0 &&
           (n[4] == '\0' || (n[4] >= '0' && n[4] <= '9'));
}
struct Options {
    bool enabled = false;
    // Only the standalone CPU regression explicitly enables this seam. Public
    // H3 configure/compute MUST still reject CPU as before.
    bool cpu_test_only = false;
    std::function<bool()> cancelled;
    // Optional test/profiling hook; exceptions abort and restore all CPU weights.
    std::function<void(size_t, ggml_cgraph*, const std::vector<ggml_tensor*>&)> after_segment;
};
struct Accounting { size_t live = 0, peak = 0, limit = 0; };
struct Slot {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor* tensor = nullptr;
    ggml_backend_t backend = nullptr; // backend must outlive State AND its imports
    std::shared_ptr<Accounting> accounting;
    size_t bytes = 0;
    ~Slot() {
        if (buffer) { ggml_backend_synchronize(backend); ggml_backend_buffer_free(buffer); }
        if (ctx) ggml_free(ctx);
        if (accounting) accounting->live -= bytes;
    }
};
using Lease = std::shared_ptr<const Slot>;

// Immutable, independent device snapshots. Accounting includes leases retained
// by an already-built graph after reset. Cache reset can NEVER recycle storage
// under an in-flight consumer or reset its allocation budget to zero early.
class Arena {
    ggml_backend_t backend_;
    std::shared_ptr<Accounting> accounting_ = std::make_shared<Accounting>();
public:
    Arena(ggml_backend_t backend, size_t limit) : backend_(backend) {
        require(backend && limit > 0, "backend and nonzero cache byte budget required");
        accounting_->limit = limit;
    }
    ggml_backend_t backend() const { return backend_; }
    size_t live_bytes() const { return accounting_->live; }
    size_t peak_bytes() const { return accounting_->peak; }
    size_t limit_bytes() const { return accounting_->limit; }
    Lease capture(ggml_tensor* source) {
        require(source && source->type == GGML_TYPE_F32 && ggml_is_contiguous(source) &&
                source->buffer && source->data, "capture requires allocated contiguous F32");
        const auto buft = ggml_backend_buffer_get_type(source->buffer);
        require(buft == ggml_backend_get_default_buffer_type(backend_),
                "capture from host/split/foreign buffer rejected");
        const size_t bytes = ggml_nbytes(source);
        require(bytes <= accounting_->limit - accounting_->live, "cache snapshot byte budget exhausted");
        auto slot = std::make_shared<Slot>();
        slot->backend = backend_;
        ggml_init_params p{ggml_tensor_overhead() + 64, nullptr, true};
        slot->ctx = ggml_init(p);
        require(slot->ctx, "snapshot context allocation failed");
        slot->tensor = ggml_new_tensor(slot->ctx, GGML_TYPE_F32, GGML_MAX_DIMS, source->ne);
        slot->buffer = ggml_backend_alloc_ctx_tensors(slot->ctx, backend_);
        require(slot->buffer, "snapshot buffer allocation failed");
        const auto actual = ggml_backend_buffer_get_size(slot->buffer);
        require(actual <= accounting_->limit - accounting_->live, "aligned snapshot exceeds cache budget");
        slot->bytes = actual; slot->accounting = accounting_;
        accounting_->live += actual;
        accounting_->peak = std::max(accounting_->peak, accounting_->live);
        ggml_backend_synchronize(backend_);
        ggml_backend_tensor_copy(source, slot->tensor); // identical F32 layout, same backend
        ggml_backend_synchronize(backend_);
        return slot;
    }
};

class GraphInputs {
    struct Binding { Lease owner; ggml_tensor* alias; };
    std::vector<Lease> owners_;
    std::vector<Binding> bindings_;
public:
    explicit GraphInputs(std::vector<Lease> owners = {}) : owners_(std::move(owners)) {}
    ggml_tensor* import(ggml_context* ctx, ggml_tensor* source) {
        if (!source) return nullptr;
        for (const auto& binding : bindings_) if (binding.owner->tensor == source) return binding.alias;
        for (const auto& owner : owners_) if (owner && owner->tensor == source) {
            auto* alias = ggml_new_tensor(ctx, source->type, GGML_MAX_DIMS, source->ne);
            ggml_set_name(alias, ("h3_cache_owned_input_" + std::to_string(bindings_.size())).c_str());
            bindings_.push_back({owner, alias});
            rebind();
            return alias;
        }
        throw std::runtime_error("MiniMax-H3 Cache-DiT offload: unowned external cache tensor");
    }
    void rebind() const {
        // Graph-cut reset clears EXTERNAL descriptors. Rebind GRAPH-LOCAL
        // aliases, never the persistent Slot metadata. The caller then refreshes
        // views recursively through GGMLRunner::refresh_graph_view_bindings.
        for (const auto& binding : bindings_) {
            auto* a = binding.alias; auto* s = binding.owner->tensor;
            a->buffer = s->buffer; a->data = s->data; a->extra = s->extra;
        }
    }
    // Call only AFTER graph/context teardown. Owners can be kept elsewhere.
    void release_graph() { bindings_.clear(); owners_.clear(); }
};

// A temporary pass rewrite while get_graph() builds: the model code may stay
// unchanged, and all reference/front/delta operands become graph-local leaves.
// On return or build exception, raw pass pointers are restored to owned slots.
struct PassInputs {
    CudaDualBlockGraphPass& pass;
    ggml_tensor *front, *rf, *delta;
    PassInputs(CudaDualBlockGraphPass& p, GraphInputs& inputs, ggml_context* ctx)
        : pass(p), front(p.front), rf(p.reference_rf), delta(p.middle_delta) {
        // Build all aliases BEFORE mutating pass so constructor failure is safe.
        auto* f = inputs.import(ctx, front); auto* r = inputs.import(ctx, rf);
        auto* d = inputs.import(ctx, delta);
        pass.front = f; pass.reference_rf = r; pass.middle_delta = d;
    }
    ~PassInputs() { pass.front = front; pass.reference_rf = rf; pass.middle_delta = delta; }
};

class State {
public:
    struct Anchor { Lease rf, delta; uint64_t serial = 0; };
    Arena arena;
private:
    std::array<Anchor, 2> anchors_{};
    uint64_t generation_ = 1, serial_ = 0;
public:
    State(ggml_backend_t backend, size_t cache_budget) : arena(backend, cache_budget) {}
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    uint64_t generation() const { return generation_; }
    const Anchor& anchor(int branch) const { return anchors_.at(branch); }
    void reset() { ++generation_; anchors_ = {}; }
    void abort(int branch, uint64_t generation) {
        if (generation == generation_) anchors_.at(branch) = {};
    }
    void commit(int branch, uint64_t generation, Lease rf, Lease delta) {
        require(generation == generation_, "request reset before anchor commit");
        require(rf && delta && rf->backend == arena.backend() && delta->backend == arena.backend() &&
                ggml_are_same_shape(rf->tensor, delta->tensor), "RF/delta anchor incompatible");
        anchors_.at(branch) = {std::move(rf), std::move(delta), ++serial_};
    }
};

struct PassResult {
    std::optional<sd::Tensor<float>> output;
    std::map<std::string, Lease> captures;
    std::map<std::string, float> indicators;
    size_t segments = 0, parameter_tensors = 0, parameter_bytes = 0, empty_parameter_segments = 0;
    Lease capture(const char* name) const {
        const auto it = captures.find(name);
        require(it != captures.end(), "required named snapshot missing");
        return it->second;
    }
};

// Metadata-only allocation measurement of an ALREADY rebound executable.
// Do not feed the unbound full graph or the planner's previous-cut ancestry.
inline size_t measure_bound_compute_buffer(ggml_backend_t backend, ggml_cgraph* executable) {
    require(backend && executable, "bound graph measurement needs backend and executable");
    auto* raw = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(raw, "bound graph allocation planner unavailable");
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> planner(raw, ggml_gallocr_free);
    size_t sizes[1] = {0};
    ggml_gallocr_reserve_n_size(raw, executable, nullptr, nullptr, sizes);
    return sizes[0];
}

// A deliberately small H3-only bridge: uses the real planner, segment builder,
// GGMLRunner execution and parameter restore. NOT a second graph engine. Never
// invokes shared compute_with_graph_cuts' async prefetch path or scheduler.
template<class BaseRunner>
class RunnerBridge : public BaseRunner {
    bool executing_ = false;
public:
    using BaseRunner::BaseRunner;
    using Builder = std::function<ggml_cgraph*()>;
    void h3_cache_release_graph() {
        if (this->runtime_backend) ggml_backend_synchronize(this->runtime_backend);
        this->free_compute_buffer();
        this->reset_graph_cut_run_cache();
        // Retained generic cache chunks are also real VRAM; don't hide them
        // outside the graph + owned-slot budget in this dedicated serial path.
        this->free_cache_chunks(false);
        this->free_compute_ctx();
        sd::ggml_graph_cut::clear_comm_marks();
    }
    // Measure one candidate with actual graph-cut boundaries, without computing
    // or staging parameters. Only existing metadata is temporarily rebound. A
    // borrowed source-parameter buffer marks future cut values as externally
    // allocated for gallocr; data stays NULL and that buffer is NEVER read.
    size_t h3_cache_measure_planned_segment(ggml_cgraph* graph,
                                            const typename BaseRunner::GraphCutSegment& segment) {
        namespace Cut = sd::ggml_graph_cut;
        using Index = decltype(this->cache_tensor_index_);
        struct Scope {
            RunnerBridge& runner;
            Index old_index;
            std::unordered_set<ggml_tensor*> seen;
            std::vector<std::pair<ggml_tensor*, ggml_tensor>> saved;
            std::vector<std::unique_ptr<ggml_tensor>> placeholders;
            explicit Scope(RunnerBridge& r) : runner(r), old_index(std::move(r.cache_tensor_index_)) {}
            void save(ggml_tensor* t) { if (t && seen.insert(t).second) saved.emplace_back(t, *t); }
            ~Scope() {
                for (auto& entry : saved) *entry.first = entry.second;
                runner.cache_tensor_index_ = std::move(old_index);
            }
        } scope(*this);
        // Save precisely the tensors bind_segment_cached_inputs and the segment
        // builder may mutate, including secondary views and output flags.
        for (const auto& input : segment.input_refs) scope.save(Cut::input_tensor(graph, input));
        for (int index : segment.internal_node_indices) {
            auto* t = ggml_graph_node(graph, index);
            scope.save(t); scope.save(t->view_src);
            for (auto* source : t->src) { scope.save(source); if (source) scope.save(source->view_src); }
        }
        for (int index : segment.output_node_indices) {
            auto* t = ggml_graph_node(graph, index); scope.save(t); scope.save(t->view_src);
        }
        ggml_backend_buffer_t marker = nullptr;
        for (auto* p = ggml_get_first_tensor(this->params_ctx); p && !marker;
             p = ggml_get_next_tensor(this->params_ctx, p)) marker = p->buffer;
        for (const auto& input : segment.input_refs) {
            if (input.type != Cut::Segment::INPUT_PREVIOUS_CUT) continue;
            require(marker, "boundary planning requires loaded CPU parameter storage");
            auto* t = Cut::input_tensor(graph, input);
            auto* source = t ? Cut::cache_source_tensor(t) : nullptr;
            require(source, "missing previous-cut source during planning");
            auto storage = std::make_unique<ggml_tensor>(*source);
            storage->op = GGML_OP_NONE; storage->view_src = nullptr; storage->view_offs = 0;
            for (auto*& parent : storage->src) parent = nullptr;
            storage->buffer = marker; storage->data = nullptr; storage->extra = nullptr;
            this->cache_tensor_index_[input.display_name] = storage.get();
            scope.placeholders.push_back(std::move(storage));
        }
        require(this->bind_segment_cached_inputs(graph, segment), "temporary plan boundary binding failed");
        ggml_context* ctx = nullptr;
        auto* executable = Cut::build_segment_graph(graph, segment, &ctx);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ctx, ggml_free);
        require(executable, "temporary bound plan graph unavailable");
        return measure_bound_compute_buffer(this->runtime_backend, executable);
    }

    // Keep the shared topological cut discovery and segment builder. The local
    // merge is the same adjacent greedy policy, but its admission sizes are
    // measured with scoped boundaries rather than recursive historical leafs.
    typename BaseRunner::GraphCutPlan h3_cache_plan(ggml_cgraph* graph,
            const std::unordered_set<const ggml_tensor*>& parameters, size_t budget, bool merge,
            const char* method = "Cache-DiT") {
        namespace Cut = sd::ggml_graph_cut;
        using Segment = typename BaseRunner::GraphCutSegment;
        auto base = Cut::build_plan(this->runtime_backend, graph, parameters, "h3-cache-offload");
        require(base.available && base.valid && !base.segments.empty(), "invalid structural graph-cut plan");
        auto plan = base; plan.segments.clear();
        auto candidate = [&](size_t begin, size_t end) {
            Segment out;
            out.group_name = base.segments[begin].group_name;
            if (end != begin) out.group_name += ".." + base.segments[end].group_name;
            std::set<int> nodes, outputs, seeds;
            for (size_t i = begin; i <= end; ++i) {
                const auto& s = base.segments[i];
                require(s.comm_ops.empty(), "parallel cut in serial cache plan");
                nodes.insert(s.internal_node_indices.begin(), s.internal_node_indices.end());
                outputs.insert(s.output_node_indices.begin(), s.output_node_indices.end());
                seeds.insert(s.seed_node_indices.begin(), s.seed_node_indices.end());
            }
            out.internal_node_indices.assign(nodes.begin(), nodes.end());
            out.output_node_indices.assign(outputs.begin(), outputs.end());
            out.seed_node_indices.assign(seeds.begin(), seeds.end());
            std::unordered_set<ggml_tensor*> internal, seen;
            for (int i : nodes) internal.insert(ggml_graph_node(graph, i));
            for (size_t i = begin; i <= end; ++i) for (const auto& input : base.segments[i].input_refs) {
                auto* t = Cut::input_tensor(graph, input);
                require(t, "missing cut input during local merge");
                if (internal.count(t) || !seen.insert(t).second) continue;
                out.input_refs.push_back(input);
                if (input.type == Segment::INPUT_PREVIOUS_CUT) out.input_previous_cut_bytes += Cut::cache_tensor_bytes(t);
                else if (input.type == Segment::INPUT_PARAM) out.input_param_bytes += ggml_nbytes(t);
                else out.input_external_bytes += ggml_nbytes(t);
            }
            std::sort(out.input_refs.begin(), out.input_refs.end(), [](const auto& a, const auto& b) {
                return a.type != b.type ? a.type < b.type : a.display_name < b.display_name;
            });
            for (int i : outputs) out.output_bytes += Cut::cache_tensor_bytes(ggml_graph_node(graph, i));
            out.compute_buffer_size = h3_cache_measure_planned_segment(graph, out);
            return out;
        };
        auto fits = [&](const Segment& s) {
            size_t left = budget;
            for (size_t bytes : {s.compute_buffer_size, s.input_param_bytes, s.input_previous_cut_bytes, s.output_bytes}) {
                if (bytes > left) return false;
                left -= bytes;
            }
            return true;
        };
        size_t begin = 0;
        while (begin < base.segments.size()) {
            size_t end = begin;
            auto best = candidate(begin, end);
            // An indivisible over-budget segment remains in the plan so the
            // actual, post-bind launch guard rejects it; NEVER silently execute.
            while (merge && fits(best) && end + 1 < base.segments.size()) {
                auto next = candidate(begin, end + 1);
                if (!fits(next)) break;
                best = std::move(next); ++end;
            }
            plan.segments.push_back(std::move(best)); begin = end + 1;
        }
        std::unordered_set<std::string> future;
        for (auto i = plan.segments.rbegin(); i != plan.segments.rend(); ++i) {
            i->future_input_names = future;
            for (const auto& input : i->input_refs)
                if (input.type == Segment::INPUT_PREVIOUS_CUT) future.insert(input.display_name);
        }
        LOG_INFO("minimax-h3 %s scoped plan: base_segments=%zu merged_segments=%zu budget=%zu",
                 method, base.segments.size(), plan.segments.size(), budget);
        return plan;
    }
    PassResult h3_cache_run_pass(Builder build, CudaDualBlockGraphPass& pass,
                                 GraphInputs& inputs, Arena& arena, int threads,
                                 bool no_return, const Options& options) {
        require(options.enabled, "explicit opt-in required");
        const bool cpu_test = options.cpu_test_only && ggml_backend_is_cpu(this->runtime_backend);
        require(is_cuda(this->runtime_backend) || cpu_test, "single CUDA backend required");
        require(arena.backend() == this->runtime_backend && threads > 0, "backend/thread mismatch");
        require(!executing_ && !this->weight_adapter &&
                !(this->process_group_ && this->process_group_->enabled()) &&
                !this->params_on_runtime_backend && !this->phase_params_pinned_,
                "concurrency, adapters, parallelism and pinned weights unsupported");
        require(this->params_backend != this->runtime_backend &&
                ggml_backend_is_cpu(this->params_backend), "immutable CPU source weights required");
        require(cpu_test || this->max_graph_vram_bytes > arena.limit_bytes(),
                "explicit graph VRAM budget must include owned-cache reservation");
        executing_ = true;
        struct Guard {
            RunnerBridge& runner; GraphInputs& inputs;
            ~Guard() { runner.h3_cache_release_graph(); inputs.release_graph(); runner.executing_ = false; }
        } guard{*this, inputs};
        h3_cache_release_graph();
        auto cancelled = [&] { require(!options.cancelled || !options.cancelled(), "cancelled"); };
        cancelled();
        ggml_cgraph* graph = nullptr;
        bool built = true;
        auto scoped_build = [&]() {
            PassInputs binding(pass, inputs, this->compute_ctx);
            auto* g = build();
            if (!g) { built = false; return this->new_graph_custom(16); }
            return g;
        };
        require(this->prepare_compute_graph(scoped_build, &graph) && built && graph,
                "graph construction failed");
        size_t attention_nodes = 0, matmul_nodes = 0, max_bytes = 0; ggml_tensor* largest = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            attention_nodes += node->op == GGML_OP_FLASH_ATTN_EXT;
            matmul_nodes += node->op == GGML_OP_MUL_MAT;
            if (ggml_nbytes(node) > max_bytes) { max_bytes = ggml_nbytes(node); largest = node; }
        }
        LOG_INFO("minimax-h3 Cache-DiT graph: phase=%d flash_enabled=%d flash_nodes=%zu matmul_nodes=%zu largest=%s op=%s shape=%lld,%lld,%lld,%lld bytes=%zu",
                 int(pass.phase), this->flash_attn_enabled, attention_nodes, matmul_nodes,
                 largest ? largest->name : "none", largest ? ggml_op_name(largest->op) : "none",
                 largest ? (long long)largest->ne[0] : 0, largest ? (long long)largest->ne[1] : 0,
                 largest ? (long long)largest->ne[2] : 0, largest ? (long long)largest->ne[3] : 0, max_bytes);
        std::unordered_set<const ggml_tensor*> all_params;
        for (auto* t = ggml_get_first_tensor(this->params_ctx); t; t = ggml_get_next_tensor(this->params_ctx, t)) {
            require(t->buffer && t->data && ggml_backend_buffer_is_host(t->buffer), "non-CPU source parameter");
            all_params.insert(t);
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_backend_supports_op(this->runtime_backend, ggml_graph_node(graph, i)),
                    "unsupported graph operation; CPU fallback is forbidden");
        namespace Cut = sd::ggml_graph_cut;
        // GGML's planner treats cut outputs and its final graph node as roots,
        // not every GGML_TENSOR_FLAG_OUTPUT. Aux captures are appended after the
        // model output, so register BOTH output and taps as explicit cut roots.
        // Otherwise it can silently omit the final head (or an earlier RF tap).
        std::map<std::string, ggml_tensor*> requested;
        for (const auto& item : this->cache_tensor_map)
            if (item.first.compare(0, 9, "h3_cache_") == 0 || item.first.compare(0, 16, "cache_ind:h3_rf_") == 0)
                requested.insert(item);
        auto* final_output = ggml_get_tensor(this->compute_ctx, this->final_result_name.c_str());
        require(no_return || final_output, "final output missing");
        if (final_output) Cut::mark_graph_cut(final_output, "h3_cache_model_output", "output");
        size_t capture_index = 0;
        for (const auto& item : requested)
            Cut::mark_graph_cut(item.second, "h3_cache_capture_" + std::to_string(capture_index++), "output");
        // Persistent-slot reservation and the existing 15/16 margin are unchanged.
        const size_t planning_budget = cpu_test ? 0 :
            (this->max_graph_vram_bytes - arena.limit_bytes()) * 15 / 16;
        auto plan = h3_cache_plan(graph, all_params, planning_budget, !cpu_test);
        require(plan.available && plan.valid && !plan.segments.empty(), "invalid graph-cut plan");
        PassResult result;
        std::unordered_set<std::string> captured;
        for (size_t index = 0; index < plan.segments.size(); ++index) {
            cancelled();
            const auto& segment = plan.segments[index];
            require(segment.comm_ops.empty(), "parallel graph-cut communication not allowed");
            this->reset_segment_runtime_tensors(segment, graph);
            inputs.rebind();
            require(this->bind_segment_cached_inputs(graph, segment), "previous-cut input missing");
            this->refresh_graph_view_bindings(graph);
            auto selected = Cut::runtime_param_tensors(graph, segment, "h3-cache-offload");
            for (auto* t : selected) {
                require(all_params.count(t) && t->buffer && t->data && ggml_backend_buffer_is_host(t->buffer),
                        "invalid selected CPU parameter");
                result.parameter_bytes += ggml_nbytes(t);
            }
            result.parameter_tensors += selected.size();
            if (selected.empty()) ++result.empty_parameter_segments;
            std::unordered_set<ggml_tensor*> produced;
            for (int i : segment.internal_node_indices) produced.insert(ggml_graph_node(graph, i));
            this->cache_tensor_map.clear();
            for (size_t i = 0; i < segment.output_node_indices.size(); ++i) {
                auto* t = Cut::output_tensor(graph, segment, i);
                if (t && Cut::is_graph_cut_tensor(t) && segment.future_input_names.count(t->name))
                    this->cache(t->name, t);
            }
            ggml_context* segment_ctx = nullptr;
            auto* executable = Cut::build_segment_graph(graph, segment, &segment_ctx);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(segment_ctx, ggml_free);
            require(executable, "segment graph construction failed");
            // The plan was measured BEFORE previous-cut inputs were rebound.
            // build_segment_graph recursively adds their old ancestors as leafs;
            // gallocr then counts already-executed layers as new allocations.
            // At real H3 shape this made blocks.10 look like 151 GiB even with
            // 48 FLASH_ATTN_EXT nodes and no dense QK matrices. Admission must
            // use the CURRENT executable, after bind_segment_cached_inputs has
            // severed those producer edges. This API measures sizes only: no GPU
            // buffer allocation or compute, and no relaxation of the byte cap.
            const size_t bound_compute_bytes = measure_bound_compute_buffer(this->runtime_backend, executable);
            if (!cpu_test && index < 2)
                LOG_INFO("minimax-h3 Cache-DiT bound budget: segment=%zu planned_compute=%zu bound_compute=%zu nodes=%d leafs=%d",
                         index, segment.compute_buffer_size, bound_compute_bytes,
                         ggml_graph_n_nodes(executable), Cut::leaf_count(executable));
            if (!cpu_test) {
                // Completed previous segments may leave spare buffers in the
                // generic pool. Release only unused pool entries, never live cut
                // inputs; graph_cache_buffer_bytes already INCLUDES pool bytes.
                this->free_cache_chunk_pool();
                size_t remaining = this->max_graph_vram_bytes;
                auto charge = [&](size_t n) {
                    if (n > remaining) throw std::runtime_error(
                        "MiniMax-H3 Cache-DiT offload: segment exceeds reserved VRAM budget: segment=" + std::to_string(index) +
                        " compute=" + std::to_string(bound_compute_bytes) +
                        " planned_compute=" + std::to_string(segment.compute_buffer_size) +
                        " params=" + std::to_string(segment.input_param_bytes) +
                        " outputs=" + std::to_string(segment.output_bytes) +
                        " cut_buffers=" + std::to_string(this->graph_cache_buffer_bytes()) +
                        " slot_limit=" + std::to_string(arena.limit_bytes()));
                    remaining -= n;
                };
                charge(arena.limit_bytes()); charge(bound_compute_bytes);
                charge(segment.input_param_bytes); charge(segment.output_bytes);
                charge(this->graph_cache_buffer_bytes()); // includes pool, never double-count it
            }
            auto audit = [&](ggml_cgraph* g) {
                auto storage = [&](ggml_tensor* t) {
                    if (!t) return;
                    auto* b = Cut::tensor_buffer(t);
                    require(t->data && b && ggml_backend_buffer_get_type(b) ==
                            ggml_backend_get_default_buffer_type(this->runtime_backend),
                            "unbound/host/split/foreign input in executable segment");
                };
                for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                    auto* t = ggml_graph_node(g, i);
                    require(ggml_backend_supports_op(this->runtime_backend, t), "unsupported segment operation");
                    storage(t); for (auto* s : t->src) storage(s);
                }
                for (int i = 0; i < Cut::leaf_count(g); ++i) storage(Cut::leaf_tensor(g, i));
            };
            auto after = [&](ggml_cgraph* g) {
                audit(g);
                for (const auto& capture : requested) if (!captured.count(capture.first) && produced.count(capture.second)) {
                    if (capture.first.compare(0, 10, "cache_ind:") == 0) {
                        require(capture.second->type == GGML_TYPE_F32 && ggml_nelements(capture.second) == 1,
                                "RF indicator is not a F32 scalar");
                        float scalar;
                        ggml_backend_tensor_get(capture.second, &scalar, 0, sizeof(scalar));
                        result.indicators.emplace(capture.first, scalar);
                    } else result.captures.emplace(capture.first, arena.capture(capture.second));
                    captured.insert(capture.first);
                }
                if (!no_return && !result.output && produced.count(final_output))
                    result.output = sd::make_sd_tensor_from_ggml<float>(final_output);
                if (options.after_segment) options.after_segment(index, g, selected);
                cancelled();
                return true;
            };
            auto before = [&](ggml_cgraph* g) {
                inputs.rebind();
                require(this->bind_segment_cached_inputs(graph, segment), "precompute cut rebind failed");
                this->refresh_graph_view_bindings(g);
                audit(g); cancelled(); return true;
            };
            // Critical: an empty selected list otherwise means offload_all_params.
            // Here it means NO weights exist in this executable; assert that fact
            // via planner selection + audits, then bypass staging only for it.
            auto done = this->template execute_graph<float>(executable, threads, true, selected,
                true, true, &segment.future_input_names, after, before, nullptr, selected.empty());
            require(done.has_value(), "segment compute/copy failed");
            ++result.segments;
        }
        require(captured.size() == requested.size(), "one or more captures were not executed");
        require(no_return || result.output.has_value(), "final output was not executed");
        return result;
    }
};

struct Statistics {
    bool hit = false;
    size_t probe_segments = 0, remainder_segments = 0, parameter_bytes = 0, empty_parameter_segments = 0;
    std::vector<float> per_span_relative_l1;
};

// Caller supplies the existing policy identity/step lifecycle, real graph
// callback, and model.cuda_cache_pass setter. Handles pairwise commit and abort.
// The model pass and TapRegistry are restored before anything can outlive scope.
template<class Runner, class Build, class SetPass>
sd::Tensor<float> compute(Runner& runner, State& state, CudaDualBlockCachePolicy& policy,
                           const CudaOptimizationOptions& tuning,
                           edgedit::cache::CacheOperatorRegistry& operators,
                           Build build, SetPass set_pass, int threads, const Options& options,
                           Statistics* statistics = nullptr) {
    require(options.enabled && tuning.cache_dit, "Cache-DiT is disabled");
    policy.require_step();
    const int branch = policy.branch_index();
    const uint64_t generation = state.generation();
    const auto anchor = state.anchor(branch);
    CudaDualBlockGraphPass pass;
    pass.front_blocks = tuning.cache_front_blocks; pass.back_blocks = tuning.cache_back_blocks;
    pass.operators = &operators;
    if (policy.eligible() && anchor.rf && anchor.delta) pass.reference_rf = anchor.rf->tensor;
    edgedit::cache::TapRegistry taps;
    struct Guard {
        Runner& runner; State& state; CudaDualBlockCachePolicy& policy; SetPass& set_pass;
        int branch; uint64_t generation; bool success = false;
        ~Guard() {
            runner.h3_cache_release_graph(); set_pass(static_cast<CudaDualBlockGraphPass*>(nullptr)); runner.set_tap_registry(nullptr);
            if (!success) { state.abort(branch, generation); policy.abort(); }
        }
    } guard{runner, state, policy, set_pass, branch, generation};
    auto checked_options = options;
    checked_options.cancelled = [&] {
        return state.generation() != generation || (options.cancelled && options.cancelled());
    };
    runner.set_tap_registry(&taps); set_pass(&pass);
    GraphInputs probe_inputs({anchor.rf});
    auto probe = runner.h3_cache_run_pass(build, pass, probe_inputs, state.arena, threads, true, checked_options);
    auto front = probe.capture("h3_cache_front");
    auto rf = probe.capture("h3_cache_rf");
    std::vector<float> metrics;
    for (const auto& name : pass.indicator_names) {
        auto it = probe.indicators.find("cache_ind:" + name);
        metrics.push_back(it == probe.indicators.end() ? std::numeric_limits<float>::infinity() : it->second);
    }
    const bool hit = anchor.delta && policy.hit(metrics) && ggml_are_same_shape(front->tensor, anchor.delta->tensor);
    // On hit, never publish the current RF: it MUST stay paired with anchor delta.
    taps.clear(); pass.phase = CudaDualBlockGraphPass::Phase::Remainder;
    pass.front = front->tensor; pass.reference_rf = nullptr; pass.current_rf = nullptr;
    pass.reuse_middle = hit; pass.middle_delta = hit ? anchor.delta->tensor : nullptr;
    GraphInputs remainder_inputs({front, hit ? anchor.delta : Lease{}});
    auto remainder = runner.h3_cache_run_pass(build, pass, remainder_inputs, state.arena, threads, false, checked_options);
    require(state.generation() == generation, "request changed before finishing step");
    if (!hit) state.commit(branch, generation, rf, remainder.capture("h3_cache_middle_delta"));
    policy.finish(hit, true);
    if (statistics) *statistics = {hit, probe.segments, remainder.segments,
        probe.parameter_bytes + remainder.parameter_bytes,
        probe.empty_parameter_segments + remainder.empty_parameter_segments, metrics};
    guard.success = true;
    return std::move(*remainder.output);
}

} } // namespace MiniMaxH3::CacheOffload
