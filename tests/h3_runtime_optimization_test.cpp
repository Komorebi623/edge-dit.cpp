// Integrated tiny H3 regression: real default compute() versus request-local
// ExactAdaLN prefill/import. Register ONLY CPU; never discover/load GPU backends.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include "ggml/src/ggml-backend-impl.h" // CPU-only observation; never replace the numeric implementation.
// ggml_extend_backend.hpp defines discovery wrappers with these public CPU
// names. Bind directly to the CPU library symbols, rather than invoking those
// wrappers (or including ggml-cpu.h, which conflicts with their C++ linkage).
#if defined(__GNUC__) || defined(__clang__)
extern "C" ggml_backend_reg_t h3_test_cpu_reg() __asm__("ggml_backend_cpu_reg");
extern "C" ggml_backend_t h3_test_cpu_init() __asm__("ggml_backend_cpu_init");
extern "C" void h3_test_cpu_threads(ggml_backend_t, int) __asm__("ggml_backend_cpu_set_n_threads");
#else
#error "This standalone CPU regression requires GNU-compatible symbol aliases"
#endif

namespace {
namespace EA = MiniMaxH3::ExactAdaLN;
size_t checks = 0, comparisons = 0, exact_comparisons = 0;
void check(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void expect(EA::Status actual, EA::Status wanted, const char* message) {
    check(actual == wanted, std::string(message) + ": " + EA::status_name(actual));
}
template<class Fn> void rejects(Fn fn, const char* message) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
template<class Fn> void rejects_with(Fn fn, const char* reason, const char* message) {
    bool rejected = false;
    try { fn(); } catch (const std::exception& error) {
        rejected = true;
        check(std::string(error.what()).find(reason) != std::string::npos,
              std::string(message) + ": unexpected error: " + error.what());
    }
    check(rejected, message);
}
void same(const sd::Tensor<float>& a, const sd::Tensor<float>& b, const char* message) {
    check(!a.empty() && a.shape() == b.shape(), message);
    for (int64_t i = 0; i < a.numel(); ++i)
        check(std::isfinite(a[i]) && std::isfinite(b[i]) &&
              std::fabs(a[i] - b[i]) <= 2e-5f + 2e-5f * std::fabs(a[i]), message);
    ++comparisons;
    if (std::memcmp(a.data(), b.data(), static_cast<size_t>(a.numel()) * sizeof(float)) == 0)
        ++exact_comparisons;
}

// Observe the graphs actually dispatched by the production prefill helper.
// The CPU implementation still executes every graph; this scoped seam only
// records shapes or injects a failure after real compute, and restores the hook.
struct CpuGraphAudit {
    using ShapeKey = std::array<int64_t, 3 * GGML_MAX_DIMS>;
    using Shapes = std::map<const ggml_tensor*, std::multiset<ShapeKey>>;
    inline static CpuGraphAudit* active = nullptr;
    ggml_backend_t backend;
    decltype(ggml_backend_i::graph_compute) original;
    const std::set<const ggml_tensor*>& projection_weights;
    const std::set<const ggml_tensor*>& time_weights;
    Shapes projections;
    size_t graphs = 0, time_gemms = 0, curve_reads = 0;
    std::function<void(size_t, ggml_cgraph*)> before, after;
    CpuGraphAudit(ggml_backend_t b, const std::set<const ggml_tensor*>& p,
                  const std::set<const ggml_tensor*>& t)
        : backend(b), original(b->iface.graph_compute), projection_weights(p), time_weights(t) {
        check(ggml_backend_is_cpu(b) && original && !active, "CPU audit cannot nest/use another backend");
        active = this;
        backend->iface.graph_compute = dispatch;
    }
    ~CpuGraphAudit() { backend->iface.graph_compute = original; active = nullptr; }
    CpuGraphAudit(const CpuGraphAudit&) = delete;
    CpuGraphAudit& operator=(const CpuGraphAudit&) = delete;
    static ggml_status dispatch(ggml_backend_t b, ggml_cgraph* graph) {
        auto& audit = *active;
        check(b == audit.backend, "graph dispatched on an unexpected CPU backend");
        const size_t pass = ++audit.graphs;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS) ++audit.curve_reads;
            if (node->op != GGML_OP_MUL_MAT) continue;
            audit.time_gemms += audit.time_weights.count(node->src[0]);
            if (!audit.projection_weights.count(node->src[0])) continue;
            ShapeKey shape{};
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                shape[d] = node->src[0]->ne[d];
                shape[GGML_MAX_DIMS + d] = node->src[1]->ne[d];
                shape[2 * GGML_MAX_DIMS + d] = node->ne[d];
            }
            check(node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32 &&
                  node->type == GGML_TYPE_F32, "shared embedding changed native projection precision");
            audit.projections[node->src[0]].insert(shape);
        }
        if (audit.before) audit.before(pass, graph);
        const auto status = audit.original(b, graph);
        if (status == GGML_STATUS_SUCCESS && audit.after) audit.after(pass, graph);
        return status;
    }
};

// Same architecture detection fixture as h3_cuda_cache_test.cpp, not a mock.
// Every actual params_ctx weight/bias is allocated and seeded below.
String2TensorStorage tiny_weights(bool curves = false) {
    String2TensorStorage weights;
    auto add = [&](const std::string& suffix, std::initializer_list<int64_t> shape) {
        TensorStorage tensor;
        tensor.name = "model.diffusion_model." + suffix;
        tensor.n_dims = static_cast<int>(shape.size());
        int axis = 0;
        for (auto extent : shape) tensor.ne[axis++] = extent;
        weights.emplace(tensor.name, tensor);
    };
    add("video_patch_proj.weight", {4, 8});
    add("audio_patch_proj.weight", {1, 8});
    add("condition_proj.weight", {8, 8});
    if (curves) add("adaln_t_table", {4, 5});
    else {
        add("time_embedder.proj_in.weight", {4, 8});
        add("time_embedder.proj_out.weight", {8, 4});
    }
    add("rope.inv_freq", {1});
    for (int i = 0; i < 4; ++i) {
        const auto prefix = "blocks." + std::to_string(i) + ".";
        add(prefix + "attn.q_norm.weight", {128});
        add(prefix + "attn.qkv_proj.weight", {8, 384});
        add(prefix + "mlp.fc1.weight", {8, 32});
    }
    return weights;
}
struct GraphCounts { int nodes = 0, blocks = 0, projections = 0, time_projections = 0; };
struct TinyRunner : MiniMaxH3::MiniMaxH3Runner {
    using MiniMaxH3Runner::MiniMaxH3Runner;
    std::set<const ggml_tensor*> projection_weights, time_weights;
    void seed() {
        check(alloc_params_buffer(), "tiny parameter buffer allocation failed");
        size_t seeded = 0;
        std::map<std::string, ggml_tensor*> named;
        get_param_tensors(named, "model.diffusion_model");
        std::map<const ggml_tensor*, std::string> names;
        for (const auto& item : named) names.emplace(item.second, item.first);
        // Do not just initialize the shape-detection map: implicit projection
        // weights/biases and all other params_ctx tensors must be deterministic.
        for (auto* tensor = ggml_get_first_tensor(exact_params_context()); tensor;
             tensor = ggml_get_next_tensor(exact_params_context(), tensor)) {
            check(names.count(tensor) == 1, "params_ctx tensor missing semantic name");
            const std::string& name = names.at(tensor);
            check(tensor->type == GGML_TYPE_F32, "tiny weight must be F32: " + name);
            uint32_t hash = 2166136261u;
            for (unsigned char c : name) hash = (hash ^ c) * 16777619u;
            std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
            for (size_t i = 0; i < data.size(); ++i) {
                const float phase = static_cast<float>(hash % 101u) * .01f;
                data[i] = name.find("norm") != std::string::npos ? 1.f
                        : name.find("rope.inv_freq") != std::string::npos ? .1f
                        : .02f + .005f * std::sin(static_cast<float>(i) + phase);
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
            if (name.find("adaln_proj.linear.weight") != std::string::npos)
                projection_weights.insert(tensor);
            if (name.find("time_embedder.proj_") != std::string::npos &&
                name.find("weight") != std::string::npos) time_weights.insert(tensor);
            ++seeded;
        }
        check(seeded > 40 && projection_weights.size() == 5, "incomplete H3 parameter seed");
        check(config.hidden_size == 8 && config.num_layers == 4 &&
              config.attention_head_dim == 128 && config.num_attention_heads == 1,
              "tiny graph must keep H3 head128 and four real blocks");
    }
    void check_projection_graph_released() {
        check(!compute_ctx && backend_tensor_data_map.empty(),
              "prefill retained a graph or binding into destroyed embedding storage");
    }
    void allow_test_projection_transfers() { max_graph_vram_bytes = 8u * 1024u * 1024u; }
    void use_test_host_parameters() {
        // A second CPU backend tests placement guards without GPU enumeration.
        check(params_backend == runtime_backend && !params_buffer, "test placement must precede allocation");
        params_backend = h3_test_cpu_init();
        check(params_backend != nullptr, "test host backend initialization");
    }
    void enable(int mib = 8) {
        MiniMaxH3::CudaOptimizationOptions options;
        options.adaln_cache = true;
        options.adaln_max_mib = mib;
        rejects([&] { configure_cuda_optimizations(options); }, "production configure accepted CPU");
        // This is the explicit test-only bypass; production gate is unchanged.
        model.cuda_optimizations = options;
        reset_cuda_optimization_state();
    }
    ggml_cgraph* graph(const DiffusionParams& p) {
        static const std::vector<sd::Tensor<float>> empty;
        static const std::vector<MiniMaxH3ReferenceBlock> empty_refs;
        static const sd::Tensor<int32_t> empty_int;
        return build_graph(*p.x, *p.timesteps, *p.context,
            p.ref_latents ? *p.ref_latents : empty,
            p.minimax_reference_audio_latents ? *p.minimax_reference_audio_latents : empty,
            p.minimax_text_token_tags ? *p.minimax_text_token_tags : empty_int,
            p.minimax_keyframe_indices ? *p.minimax_keyframe_indices : empty_int,
            p.minimax_reference_blocks ? *p.minimax_reference_blocks : empty_refs,
            p.minimax_audio_length, p.minimax_video_sigma_shift, p.minimax_audio_sigma_shift);
    }
    GraphCounts inspect(const DiffusionParams& p) {
        ggml_cgraph* g = nullptr;
        check(exact_prepare_graph([&] { return graph(p); }, &g) && g, "inspect graph build failed");
        GraphCounts counts;
        counts.nodes = ggml_graph_n_nodes(g);
        for (int i = 0; i < counts.nodes; ++i) {
            const auto* node = ggml_graph_node(g, i);
            const std::string name(node->name);
            const auto separator = name.find('|');
            if (name.find("ggml_runner_cut:minimax_h3.blocks.") == 0 &&
                separator != std::string::npos && name.substr(separator) == "|hidden_states")
                ++counts.blocks; // Exclude downstream inherited "(view)/(cont)" names.
            if (node->op == GGML_OP_MUL_MAT) {
                counts.projections += projection_weights.count(node->src[0]);
                counts.time_projections += time_weights.count(node->src[0]);
            }
            check(name.find("sage") == std::string::npos, "unexpected Sage node in CPU graph");
        }
        return counts;
    }
    void check_prefill(int branch) {
        auto& state = adaln_branches[branch];
        check(state.cache && state.cache->projection_count() == 5, "branch did not prefill all projections");
        check(state.cache->statistics().stores == state.contract.plans.size() * 5,
              "prefill is lazy/incomplete or unexpectedly re-stored");
        check(state.cache->used_bytes() <= state.cache->max_bytes(), "prefill exceeded retained budget");
        for (size_t i = 0; i < state.contract.plans.size(); ++i) {
            EA::Cache::StepToken token;
            expect(state.cache->bind_step(state.contract, i, state.contract.plans[i], &token),
                   EA::Status::Hit, "bind all-sigma prefill");
            check(state.cache->complete(token), "some future sigma was not prefetched");
            for (size_t j = 0; j < 5; ++j) {
                EA::Shape shape;
                expect(state.cache->expected_shape(token, j, &shape), EA::Status::Hit, "prefill shape");
                check(shape[0] == (j == 4 ? 16 : 144) && shape[1] ==
                      static_cast<int64_t>(state.contract.plans[i].timesteps.size()), "wrong projection layout");
                const float* values = nullptr;
                expect(state.cache->lookup(token, j, shape, &values), EA::Status::Hit, "cache native readback");
                for (int64_t n = 0; n < shape[0] * shape[1]; ++n)
                    check(std::isfinite(values[n]), "nonfinite prefetched projection");
            }
        }
    }
    void compare_projection_readback(int branch) {
        auto& state = adaln_branches[branch];
        EA::Cache direct({true, 8u * 1024u * 1024u, 16384});
        expect(direct.begin_request(state.contract), EA::Status::Stored, "direct projection contract");
        CpuGraphAudit::Shapes direct_shapes;
        {
            CpuGraphAudit audit(exact_runtime_backend(), projection_weights, time_weights);
            expect(EA::prefill_request(*this, direct, state.contract, 1,
                [&](size_t i, auto token) {
                    return build_adaln_projection_graph(direct, token, state.contract.plans[i]);
                }), EA::Status::Stored, "independent real projection compute/readback");
            check(audit.graphs == state.contract.plans.size(), "direct prefill graph count changed");
            direct_shapes = audit.projections;
        }
        EA::Cache segmented({true, 8u * 1024u * 1024u, 16384});
        expect(segmented.begin_request(state.contract), EA::Status::Stored, "segmented H3 contract");
        size_t builds = 0;
        expect(EA::prefill_request_segmented(*this, segmented, state.contract, 1,
            [&](size_t i, size_t j, auto token) {
                ++builds;
                return build_adaln_projection_graph(segmented, token, state.contract.plans[i], j, 1);
            }), EA::Status::Stored, "real H3 ranged projection prefill");
        check(builds == state.contract.plans.size() * 5, "real H3 prefill must split every projection");
        EA::Cache layer_major({true, 8u * 1024u * 1024u, 16384});
        expect(layer_major.begin_request(state.contract), EA::Status::Stored, "layer-major H3 contract");
        {
            CpuGraphAudit audit(exact_runtime_backend(), projection_weights, time_weights);
            expect(prefill_adaln_layer_major(layer_major, state.contract, 1), EA::Status::Stored,
                   "real H3 shared-embedding layer-major schedule prefill");
            // Production 50 blocks + final = 51 projections; this bounded fixture
            // has four REAL blocks + final = five, not 51 synthetic layers.
            check(state.contract.projections.size() == 5 && audit.graphs == 6,
                  "shared prefill must execute one embedding graph plus five layer graphs");
            check(audit.time_gemms == (config.uses_adaln_curves() ? 0 : 2 * state.contract.plans.size()),
                  "time embeddings were recomputed for each projection");
            check(audit.curve_reads == (config.uses_adaln_curves() ? 2 * state.contract.plans.size() : 0),
                  "curve embeddings were recomputed for each projection");
            check(audit.projections == direct_shapes,
                  "shared embedding changed per-plan native GEMM shapes or projection execution counts");
        }
        check(layer_major.used_bytes() == direct.used_bytes(), "shared embedding charge leaked after prefill");
        check_projection_graph_released();
        for (size_t i = 0; i < state.contract.plans.size(); ++i) {
            EA::Cache::StepToken a, b, segmented_token;
            expect(segmented.bind_step(state.contract, i, state.contract.plans[i], &segmented_token),
                   EA::Status::Hit, "segmented H3 bind");
            expect(direct.bind_step(state.contract, i, state.contract.plans[i], &a), EA::Status::Hit, "direct bind");
            expect(state.cache->bind_step(state.contract, i, state.contract.plans[i], &b), EA::Status::Hit, "cached bind");
            for (size_t j = 0; j < 5; ++j) {
                EA::Shape shape;
                expect(direct.expected_shape(a, j, &shape), EA::Status::Hit, "direct shape");
                const float *av = nullptr, *bv = nullptr;
                expect(direct.lookup(a, j, shape, &av), EA::Status::Hit, "direct readback");
                expect(state.cache->lookup(b, j, shape, &bv), EA::Status::Hit, "cached readback");
                check(std::memcmp(av, bv, static_cast<size_t>(shape[0] * shape[1]) * sizeof(float)) == 0,
                      "prefilled projection is not bitexact native F32 readback");
                EA::Cache::StepToken layer_token;
                expect(layer_major.bind_step(state.contract, i, state.contract.plans[i], &layer_token),
                       EA::Status::Hit, "layer-major bind");
                const float* layer_values = nullptr;
                expect(layer_major.lookup(layer_token, j, shape, &layer_values), EA::Status::Hit, "layer-major lookup");
                check(std::memcmp(av, layer_values, static_cast<size_t>(shape[0] * shape[1]) * sizeof(float)) == 0,
                      "layer-major schedule graph changed projection math");
                const float* segmented_values = nullptr;
                expect(segmented.lookup(segmented_token, j, shape, &segmented_values), EA::Status::Hit,
                       "segmented H3 native readback");
                check(std::memcmp(av, segmented_values, static_cast<size_t>(shape[0] * shape[1]) * sizeof(float)) == 0,
                      "segmented H3 builder changed the real projection");
            }
        }
    }
};
struct Inputs {
    sd::Tensor<float> packed{{2, 2, 1, 2}, {.2f, -.3f, .1f, .4f, .5f, -.2f, 0.f, 0.f}};
    sd::Tensor<float> context{{8, 3}};
    sd::Tensor<float> timestep{{1}, {800.f}};
    sd::Tensor<int32_t> tags{{3}, {1, 2, 0}};
    sd::Tensor<int32_t> keyframes{{1}, {0}};
    std::vector<sd::Tensor<float>> videos, audios;
    std::vector<MiniMaxH3ReferenceBlock> references;
    Inputs() {
        for (int64_t i = 0; i < context.numel(); ++i) context[i] = .1f + .01f * static_cast<float>(i);
    }
    DiffusionParams params(bool with_refs = false) {
        DiffusionParams p;
        p.x = &packed; p.context = &context; p.timesteps = &timestep;
        p.minimax_audio_length = 1;
        p.minimax_video_sigma_shift = 12.f; p.minimax_audio_sigma_shift = 3.f;
        if (with_refs) {
            p.ref_latents = &videos; p.minimax_reference_audio_latents = &audios;
            p.minimax_reference_blocks = &references;
            p.minimax_text_token_tags = &tags; p.minimax_keyframe_indices = &keyframes;
        }
        return p;
    }
    void add_references() {
        videos.emplace_back(std::vector<int64_t>{2, 2, 1, 1, 1}, std::vector<float>{.3f, .1f, -.2f, .4f});
        videos.emplace_back(std::vector<int64_t>{2, 2, 1, 1, 1}, std::vector<float>{-.1f, .2f, .3f, .1f});
        audios.emplace_back(std::vector<int64_t>{1, 2, 1, 1}, std::vector<float>{.2f, -.1f});
        references = {{MiniMaxH3ReferenceKind::VIDEO_AUDIO, 0, 0}, {MiniMaxH3ReferenceKind::IMAGE, 1, -1}};
    }
};
const std::vector<float> schedule{.8f, .5f, .125f, 0.f};
void schedule_test(ggml_backend_t backend, bool curves) {
    TinyRunner baseline(backend, tiny_weights(curves)), cached(backend, tiny_weights(curves));
    baseline.seed(); cached.seed(); cached.enable();
    check(!baseline.model.cuda_optimizations.adaln_cache && !baseline.model.cuda_optimizations.sage_attn,
          "default path changed optimization flags");
    check(!baseline.get_context().bind_backend_tensor_data && !baseline.model.adaln_cache &&
          !baseline.model.adaln_graph_inputs, "default CPU context/cache binding changed");
    Inputs in;
    auto p = in.params();
    cached.prepare_cuda_schedule(schedule, "tiny-fixed-flow:v1", "deterministic-all-params:v1");
    int full_nodes = 0, hit_nodes = 0;
    sd::Tensor<float> first_sigma_output;
    for (size_t step = 0; step + 1 < schedule.size(); ++step) {
        in.timestep[0] = schedule[step] * 1000.f;
        const auto reference = baseline.compute(1, p);
        if (step == 0) first_sigma_output = reference;
        if (step == schedule.size() - 2)
            check(std::memcmp(first_sigma_output.data(), reference.data(),
                              static_cast<size_t>(reference.numel()) * sizeof(float)) != 0,
                  "fixture outputs insensitive to sigma (vacuous cache regression)");
        cached.begin_cuda_step(static_cast<int>(step), 3, true);
        const auto output = cached.compute(1, p);
        same(reference, output, "actual default H3 versus exact AdaLN fullgraph differs");
        check(output.shape() == in.packed.shape(), "packed output shape changed");
        cached.check_prefill(1);
        const auto full = baseline.inspect(p), hit = cached.inspect(p);
        check(full.blocks == 4 && hit.blocks == 4, "graph did not execute all four H3 blocks: " +
              std::to_string(full.blocks) + "/" + std::to_string(hit.blocks));
        check(full.projections == 5 && hit.projections == 0, "hit graph retained AdaLN projection GEMMs");
        check(full.time_projections == (curves ? 0 : 2) && hit.time_projections == 0,
              "hit graph retained time-embedding projection GEMMs");
        check(hit.nodes < full.nodes, "prefill did not remove graph nodes");
        full_nodes = full.nodes; hit_nodes = hit.nodes;
    }
    cached.compare_projection_readback(1);
    // Disable through the real production setter; only CPU opt-IN is bypassed.
    cached.configure_cuda_optimizations({});
    same(baseline.compute(1, p), cached.compute(1, p), "cache-off request changed baseline output");
    check(!cached.adaln_branches[0].cache && !cached.adaln_branches[1].cache && !cached.model.adaln_cache &&
          !cached.model.adaln_graph_inputs && !cached.get_context().bind_backend_tensor_data,
          "cache-off did not clear imported state or changed shared CPU context");
    const auto off = cached.inspect(p);
    check(off.projections == 5 && off.time_projections == (curves ? 0 : 2),
          "cache-off did not restore original projection graph");
    std::cout << "PASS actual tiny H3 " << (curves ? "curves/no-SiLU" : "TimeEmbedder/SiLU")
              << " all-sigma default/cache; nodes=" << full_nodes << '/' << hit_nodes
              << "; AdaLN GEMMs=5/0; native projection readback bitexact\n";
}
void branch_layout_test(ggml_backend_t backend) {
    TinyRunner baseline(backend, tiny_weights()), cached(backend, tiny_weights());
    baseline.seed(); cached.seed(); cached.enable();
    Inputs conditional, unconditional;
    conditional.add_references(); unconditional.add_references();
    for (int64_t i = 0; i < unconditional.context.numel(); ++i) unconditional.context[i] *= -.5f;
    auto c = conditional.params(true), u = unconditional.params(true);
    cached.prepare_cuda_schedule(schedule, "tiny-cfg:v1", "deterministic-all-params:v1");
    for (int step = 0; step < 3; ++step) {
        conditional.timestep[0] = unconditional.timestep[0] = schedule[step] * 1000.f;
        for (int branch : {1, 0}) {
            auto& p = branch ? c : u;
            cached.begin_cuda_step(step, 3, branch != 0);
            same(baseline.compute(1, p), cached.compute(1, p), "CFG/reference graph differs from default");
            cached.check_prefill(branch);
        }
        check(cached.adaln_branches[0].cache.get() != cached.adaln_branches[1].cache.get(),
              "CFG branches shared a cache object");
        check(cached.adaln_branches[0].contract.cfg_identity != cached.adaln_branches[1].contract.cfg_identity,
              "CFG identity did not distinguish branches");
    }
    const auto other_identity = cached.adaln_branches[0].contract.cfg_identity;
    auto previous_identity = cached.adaln_branches[1].contract.cfg_identity;
    conditional.timestep[0] = schedule[1] * 1000.f;
    auto refresh = [&](const char* message) {
        cached.begin_cuda_step(1, 3, true);
        same(baseline.compute(1, c), cached.compute(1, c), message);
        cached.check_prefill(1);
        check(cached.adaln_branches[1].contract.cfg_identity != previous_identity, "layout mutation not invalidated");
        previous_identity = cached.adaln_branches[1].contract.cfg_identity;
        check(cached.adaln_branches[0].contract.cfg_identity == other_identity, "conditional refresh disturbed unconditional cache");
        check(cached.inspect(c).projections == 0, "refreshed hit retained projections");
    };
    conditional.videos[0][0] += .15f; refresh("in-place video reference mutation stale");
    conditional.audios[0][0] -= .12f; refresh("in-place audio reference mutation stale");
    conditional.context[0] += .2f; refresh("in-place context mutation stale");
    conditional.tags[0] = 2; refresh("text-tag run layout mutation stale");
    conditional.keyframes[0] = 1; refresh("keyframe identity mutation stale");
    c.minimax_audio_sigma_shift = 4.f; refresh("audio flow-shift mutation stale");
    c.minimax_video_sigma_shift = 10.f; refresh("video flow-shift mutation stale");
    std::swap(conditional.references[0], conditional.references[1]); refresh("reference block ordering stale");
    std::swap(conditional.references[0].video_index, conditional.references[1].video_index);
    refresh("reference source indices stale");
    const auto& plan = cached.adaln_branches[1].contract.plans[1];
    std::set<EA::Role> roles;
    std::set<uint32_t> text_modalities;
    for (const auto& span : plan.spans) {
        roles.insert(span.role);
        if (span.role == EA::Role::Text) text_modalities.insert(span.modality);
    }
    check(roles.size() == 5 && text_modalities.count(0) && text_modalities.count(2),
          "references/text tags not represented in integrated cache plan");
    const auto generation = cached.adaln_request_generation;
    cached.reset_cuda_optimization_state();
    check(cached.adaln_request_generation > generation && cached.adaln_sigmas.empty() &&
          !cached.model.adaln_cache && !cached.model.adaln_graph_inputs &&
          !cached.adaln_branches[0].cache && !cached.adaln_branches[1].cache, "request reset leaked state");
    cached.begin_cuda_step(1, 3, true);
    rejects([&] { cached.compute(1, c); }, "reset accepted a stale schedule");
    cached.prepare_cuda_schedule(schedule, "tiny-cfg:v2", "deterministic-all-params:v2");
    cached.begin_cuda_step(1, 3, true);
    same(baseline.compute(1, c), cached.compute(1, c), "new request after reset differs");
    check(cached.adaln_branches[1].contract.full_weights_identity == "deterministic-all-params:v2" &&
          cached.adaln_branches[1].contract.schedule_identity == "tiny-cfg:v2", "new request identity stale");
    std::cout << "PASS CFG isolation, video/audio references, source/order/text-tag mutation, reset/new request\n";
}
// Bound memory and failure tests to the actual five-projection tiny model.
// The second CPU backend exercises real selected-weight copies and restoration;
// it does NOT validate CUDA placement, the new scoped planner, or mixed speed.
void shared_prefill_budget_test(ggml_backend_t backend, bool curves) {
    TinyRunner native(backend, tiny_weights(curves)), streamed(backend, tiny_weights(curves));
    native.seed(); native.enable();
    streamed.use_test_host_parameters(); streamed.allow_test_projection_transfers(); streamed.seed();
    Inputs in; in.add_references(); auto p = in.params(true);
    native.prepare_cuda_schedule({.8f, .5f, .5f, .125f, 0.f}, "shared-budget:v1", "tiny-f32:all-params");
    native.begin_cuda_step(0, 4, true); native.prepare_adaln_branch(1, p);
    const auto contract = native.adaln_branches[1].contract;
    auto& reference = *native.adaln_branches[1].cache;
    EA::Cache metadata({true, 8u << 20, 16384});
    expect(metadata.begin_request(contract), EA::Status::Stored, "shared parent metadata");
    const size_t parent_metadata = metadata.used_bytes();
    std::vector<size_t> layer_bytes(5, 0);
    auto payload_bytes = [](const EA::Shape& shape) {
        size_t bytes = sizeof(float);
        for (auto extent : shape) bytes *= static_cast<size_t>(extent);
        return bytes;
    };
    for (size_t i = 0; i < contract.plans.size(); ++i) {
        EA::Cache::StepToken token;
        expect(reference.bind_step(contract, i, contract.plans[i], &token), EA::Status::Hit, "budget shape bind");
        for (size_t j = 0; j < 5; ++j) {
            EA::Shape shape{};
            expect(reference.expected_shape(token, j, &shape), EA::Status::Hit, "budget shape");
            layer_bytes[j] += payload_bytes(shape);
        }
    }
    auto embedding_contract = contract;
    embedding_contract.projections = {{"h3.shared_time_embedding", native.config.time_embed_dim, 1, 1}};
    EA::Cache embedding_metadata({true, 8u << 20, 16384});
    expect(embedding_metadata.begin_request(embedding_contract), EA::Status::Stored, "embedding metadata");
    size_t embedding_bytes = embedding_metadata.used_bytes();
    for (const auto& plan : contract.plans)
        embedding_bytes += sizeof(float) * static_cast<size_t>(native.config.time_embed_dim) * plan.timesteps.size();
    const size_t retained = reference.used_bytes(), peak = retained + embedding_bytes;
    size_t sum = parent_metadata;
    for (auto bytes : layer_bytes) sum += bytes;
    check(sum == retained, "unexpected parent payload accounting");

    struct WeightSnapshot { ggml_tensor* tensor; ggml_backend_buffer_t buffer; void* data; void* extra; std::vector<char> bytes; };
    std::vector<WeightSnapshot> weights;
    for (auto* t = ggml_get_first_tensor(streamed.exact_params_context()); t;
         t = ggml_get_next_tensor(streamed.exact_params_context(), t)) {
        weights.push_back({t, t->buffer, t->data, t->extra, std::vector<char>(ggml_nbytes(t))});
        ggml_backend_tensor_get(t, weights.back().bytes.data(), 0, weights.back().bytes.size());
    }
    auto restored = [&] {
        streamed.check_projection_graph_released();
        for (const auto& w : weights) {
            check(w.tensor->buffer == w.buffer && w.tensor->data == w.data && w.tensor->extra == w.extra,
                  "shared prefill did not restore a source parameter binding");
            std::vector<char> actual(w.bytes.size());
            ggml_backend_tensor_get(w.tensor, actual.data(), 0, actual.size());
            check(actual == w.bytes, "shared prefill mutated native F32 source weights");
        }
    };
    auto compare = [&](EA::Cache& cache) {
        check(cache.used_bytes() == retained, "temporary embedding budget not released");
        for (size_t i = 0; i < contract.plans.size(); ++i) {
            EA::Cache::StepToken a, b;
            expect(reference.bind_step(contract, i, contract.plans[i], &a), EA::Status::Hit, "native bind");
            expect(cache.bind_step(contract, i, contract.plans[i], &b), EA::Status::Hit, "shared bind");
            for (size_t j = 0; j < 5; ++j) {
                EA::Shape shape{}; const float *av = nullptr, *bv = nullptr;
                expect(reference.expected_shape(a, j, &shape), EA::Status::Hit, "native shape");
                expect(reference.lookup(a, j, shape, &av), EA::Status::Hit, "native payload");
                expect(cache.lookup(b, j, shape, &bv), EA::Status::Hit, "shared payload");
                check(std::memcmp(av, bv, payload_bytes(shape)) == 0, "budgeted shared projection differs from native");
            }
        }
    };
    // Exact peak, one-byte short, and one complete final-layer payload short.
    for (size_t budget : {peak, peak - 1, peak - layer_bytes.back()}) {
        EA::Cache cache({true, budget, 16384});
        expect(cache.begin_request(contract), EA::Status::Stored, "bounded shared begin");
        EA::Cache::StepToken old;
        expect(cache.bind_step(contract, 0, contract.plans[0], &old), EA::Status::Hit, "bounded old token");
        size_t passes = 0;
        EA::Status result;
        {
            CpuGraphAudit audit(backend, streamed.projection_weights, streamed.time_weights);
            audit.before = [&](size_t pass, ggml_cgraph*) {
                passes = pass;
                size_t wanted = parent_metadata;
                if (pass > 1) {
                    wanted += embedding_bytes;
                    for (size_t j = 0; j + 2 < pass; ++j) wanted += layer_bytes[j];
                }
                check(cache.used_bytes() == wanted && wanted <= budget,
                      "shared embedding was not charged alongside previously captured layers");
            };
            result = streamed.prefill_adaln_layer_major(cache, contract, 1);
        }
        // With a whole final layer missing, intercept() rejects its first
        // tensor before execution. One byte short still builds that layer and
        // fails during readback after some final-plan payloads were stored.
        const size_t expected_passes = budget == peak - layer_bytes.back() ? 5 : 6;
        check(passes == expected_passes, "budget rejection occurred at the wrong projection stage");
        if (budget == peak) { expect(result, EA::Status::Stored, "exact shared peak budget"); compare(cache); }
        else {
            check(result != EA::Status::Stored, "insufficient shared budget was accepted");
            check(cache.used_bytes() == 0 && cache.projection_count() == 0 && !cache.complete(old),
                  "budget failure retained partial projections or transient charge");
            EA::Shape shape{};
            expect(cache.expected_shape(old, 0, &shape), EA::Status::Stale, "budget failure token survived");
        }
        restored();
    }
    // Real compute followed by nonfinite readback, exception, or request reset.
    // Later-pass injection happens only AFTER earlier layers have been stored.
    for (int fault = 0; fault < 4; ++fault) {
        EA::Cache cache({true, peak, 16384});
        expect(cache.begin_request(contract), EA::Status::Stored, "fault begin");
        const size_t target = fault == 0 ? 1 : 4;
        bool injected = false, threw = false;
        EA::Status result = EA::Status::Miss;
        {
            CpuGraphAudit audit(backend, streamed.projection_weights, streamed.time_weights);
            audit.after = [&](size_t pass, ggml_cgraph* graph) {
                if (pass != target) return;
                injected = true;
                if (target > 1) check(cache.statistics().stores == 2 * contract.plans.size(),
                                      "late failure did not exercise partially populated parent cache");
                if (fault == 2) throw std::runtime_error("injected shared prefill post-compute error");
                if (fault == 3) { cache.reset(); return; }
                auto* output = ggml_graph_node(graph, -1);
                check(output->type == GGML_TYPE_F32, "nonfinite injection requires native F32");
                const float nan = std::numeric_limits<float>::quiet_NaN();
                ggml_backend_tensor_set(output, &nan, 0, sizeof(nan));
            };
            try { result = streamed.prefill_adaln_layer_major(cache, contract, 1); }
            catch (const std::runtime_error& error) {
                threw = true;
                check(fault == 2 && std::string(error.what()) == "injected shared prefill post-compute error",
                      "unexpected exception during shared prefill");
            }
        }
        check(injected && (threw || result != EA::Status::Stored), "shared prefill swallowed a compute/readback failure");
        check(cache.used_bytes() == 0 && cache.projection_count() == 0,
              "failed shared prefill leaked partial payload or underflowed its transient budget");
        restored();
        expect(cache.begin_request(contract), EA::Status::Stored, "same-runner recovery begin");
        expect(streamed.prefill_adaln_layer_major(cache, contract, 1), EA::Status::Stored, "same-runner recovery prefill");
        compare(cache); restored();
    }
    std::cout << "PASS shared " << (curves ? "curve" : "TimeEmbedder")
              << " exact/short budgets, partial capture, real CPU transfers, rollback and recovery\n";
}

void shared_prefill_cfg_test(ggml_backend_t backend, bool curves) {
    TinyRunner native(backend, tiny_weights(curves)), cached(backend, tiny_weights(curves));
    native.seed(); cached.seed(); cached.enable();
    const std::vector<float> repeated{.8f, .5f, .5f, .125f, 0.f};
    cached.prepare_cuda_schedule(repeated, "shared-repeat-cfg:v1", "tiny-f32:all-params");
    Inputs uncond, cond; cond.add_references(); uncond.context[0] -= .3f;
    auto u = uncond.params(), c = cond.params(true);
    for (int branch = 0; branch < 2; ++branch) {
        cached.begin_cuda_step(0, 4, branch != 0);
        cached.prepare_adaln_branch(1, branch ? c : u);
        // Compare actual native and shared graphs, including every GEMM shape.
        cached.compare_projection_readback(branch);
        auto& state = cached.adaln_branches[branch];
        expect(state.cache->begin_request(state.contract), EA::Status::Stored, "replace native with shared cache");
        expect(cached.prefill_adaln_layer_major(*state.cache, state.contract, 1), EA::Status::Stored, "shared CFG prefill");
        EA::Cache::StepToken a, b;
        expect(state.cache->bind_step(state.contract, 1, state.contract.plans[1], &a), EA::Status::Hit, "first repeated sigma");
        expect(state.cache->bind_step(state.contract, 2, state.contract.plans[2], &b), EA::Status::Hit, "second repeated sigma");
        EA::Cache::StepToken wrong;
        expect(state.cache->bind_step(state.contract, 1, state.contract.plans[2], &wrong), EA::Status::Miss,
               "equal sigmas aliased different schedule indices");
        for (size_t j = 0; j < 5; ++j) {
            EA::Shape shape{}; const float *av = nullptr, *bv = nullptr;
            expect(state.cache->expected_shape(a, j, &shape), EA::Status::Hit, "repeat shape");
            expect(state.cache->lookup(a, j, shape, &av), EA::Status::Hit, "repeat first");
            expect(state.cache->lookup(b, j, shape, &bv), EA::Status::Hit, "repeat second");
            check(av != bv && std::memcmp(av, bv, static_cast<size_t>(shape[0] * shape[1]) * sizeof(float)) == 0,
                  "repeated sigma lost native values or aliased distinct plan storage");
        }
    }
    auto& a = cached.adaln_branches[0]; auto& b = cached.adaln_branches[1];
    EA::Cache::StepToken token;
    expect(a.cache->bind_step(a.contract, 1, a.contract.plans[1], &token), EA::Status::Hit, "unconditional token");
    EA::Shape shape{};
    expect(b.cache->expected_shape(token, 0, &shape), EA::Status::Stale, "CFG accepted another cache's token");
    expect(b.cache->bind_step(a.contract, 1, a.contract.plans[1], &token), EA::Status::Miss, "CFG identity aliased");
    for (size_t step = 0; step + 1 < repeated.size(); ++step) {
        uncond.timestep[0] = cond.timestep[0] = repeated[step] * 1000.f;
        for (int branch = 0; branch < 2; ++branch) {
            auto& p = branch ? c : u;
            cached.begin_cuda_step(static_cast<int>(step), 4, branch != 0);
            same(native.compute(1, p), cached.compute(1, p), "shared-prefilled CFG full H3 graph differs");
            check(cached.inspect(p).projections == 0, "shared-prefilled CFG consumer recomputed projections");
        }
    }
    std::cout << "PASS shared prefill repeated sigma / CFG isolation and eight full H3 consumers; CPU only\n";
}

// Default regression, also runnable alone with --shared-prefill-reentry. A
// complete native cache needs neither additional storage nor additional compute;
// keep the retained-only budget to catch accidental temporary embedding work.
void shared_prefill_reentry_test(ggml_backend_t backend) {
    TinyRunner runner(backend, tiny_weights()); runner.seed(); runner.enable();
    Inputs in; auto p = in.params();
    runner.prepare_cuda_schedule(schedule, "shared-reentry:v1", "tiny-f32:all-params");
    runner.begin_cuda_step(0, 3, true); runner.prepare_adaln_branch(1, p);
    const auto contract = runner.adaln_branches[1].contract;
    EA::Cache cache({true, runner.adaln_branches[1].cache->used_bytes(), 16384});
    expect(cache.begin_request(contract), EA::Status::Stored, "reentry begin");
    expect(EA::prefill_request(runner, cache, contract, 1, [&](size_t i, auto token) {
        return runner.build_adaln_projection_graph(cache, token, contract.plans[i]);
    }), EA::Status::Stored, "reentry complete native cache");
    const size_t before = cache.used_bytes();
    EA::Cache::StepToken old;
    expect(cache.bind_step(contract, 0, contract.plans[0], &old), EA::Status::Hit, "reentry old token");
    check(cache.complete(old), "reentry fixture was not already complete");
    EA::Status result;
    size_t executions = 0;
    {
        CpuGraphAudit audit(backend, runner.projection_weights, runner.time_weights);
        result = runner.prefill_adaln_layer_major(cache, contract, 1);
        executions = audit.graphs;
    }
    std::cout << "shared-reentry: status=" << EA::status_name(result) << " graphs=" << executions
              << " retained-before=" << before << " retained-after=" << cache.used_bytes() << std::endl;
    check(result == EA::Status::Stored && executions == 0 && cache.used_bytes() == before && cache.complete(old),
          "complete cache reentry allocated temporary embeddings or invalidated valid projections");
}

void failure_test(ggml_backend_t backend) {
    TinyRunner runner(backend, tiny_weights()); runner.seed(); runner.enable();
    Inputs in; auto p = in.params();
    rejects([&] { runner.compute(1, p); }, "missing schedule/step accepted");
    rejects([&] { runner.prepare_cuda_schedule({.8f}, "fixed", "weights"); }, "incomplete schedule accepted");
    rejects([&] { runner.prepare_cuda_schedule(schedule, "", "weights"); }, "empty scheduler identity accepted");
    rejects([&] { runner.prepare_cuda_schedule(schedule, "fixed", ""); }, "empty weights identity accepted");
    runner.prepare_cuda_schedule(schedule, "fixed", "weights");
    runner.begin_cuda_step(0, 2, true);
    rejects([&] { runner.compute(1, p); }, "schedule/total mismatch accepted");
    for (int step : {-1, 3, 4}) {
        rejects([&] {
            runner.begin_cuda_step(step, 3, true);
            runner.compute(1, p);
        }, "out-of-range step accepted");
    }
    runner.begin_cuda_step(0, 3, true);
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
        in.timestep[0] = invalid;
        rejects([&] { runner.compute(1, p); }, "nonfinite timestep accepted");
    }
    const sd::Tensor<float> empty_timestep;
    p.timesteps = &empty_timestep;
    rejects([&] { runner.compute(1, p); }, "empty timestep accepted");
    p.timesteps = &in.timestep;
    runner.begin_cuda_step(0, 3, true); in.timestep[0] = 700.f;
    rejects_with([&] { runner.compute(1, p); }, "actual timestep differs",
                 "actual sigma differs from scheduled sigma accepted");
    in.timestep[0] = 800.f; runner.begin_cuda_step(0, 3, true);
    check(!runner.compute(1, p).empty(), "valid run after rejection failed");
    auto& state = runner.adaln_branches[1];
    EA::Cache::StepToken token;
    expect(state.cache->bind_step(state.contract, 0, state.contract.plans[0], &token), EA::Status::Hit, "failure test bind");
    EA::Shape shape;
    expect(state.cache->expected_shape(token, 0, &shape), EA::Status::Hit, "failure test shape");
    auto wrong = shape; ++wrong[0];
    const float* data = nullptr;
    expect(state.cache->lookup(token, 0, wrong, &data), EA::Status::ShapeMismatch, "illegal cached projection shape");
    check(!data, "shape rejection returned payload");
    auto* ctx = ggml_init({1024 * 1024, nullptr, true});
    check(ctx != nullptr, "shape test context allocation");
    auto* malformed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[0] + 1, shape[1]);
    expect(state.cache->intercept(token, 0, malformed), EA::Status::ShapeMismatch, "illegal graph projection shape");
    ggml_free(ctx);
    EA::Cache tiny({true, 1, 16384});
    expect(tiny.begin_request(state.contract), EA::Status::BudgetExceeded, "one-byte integrated contract budget");
    state.cache->reset();
    expect(state.cache->lookup(token, 0, shape, &data), EA::Status::Stale, "reset accepted old projection token");
    // Real runner budget rejection, not just the low-level contract check.
    // Four blocks * many fixed sigmas exceed one MiB of native F32 projections.
    runner.enable(1);
    std::vector<float> large_schedule(600, .8f); large_schedule.push_back(0.f);
    runner.prepare_cuda_schedule(large_schedule, "long-fixed", "weights");
    runner.begin_cuda_step(0, 600, true);
    std::cout << "EXPECT prefill budget rejection (the runner may log post compute callback failed)" << std::endl;
    rejects_with([&] { runner.compute(1, p); }, "prefill failed",
                 "small real runner prefill budget silently succeeded");
    check(runner.adaln_branches[1].cache &&
          runner.adaln_branches[1].cache->max_bytes() == 1024u * 1024u &&
          runner.adaln_branches[1].cache->used_bytes() <= 1024u * 1024u,
          "failed real prefill exceeded its one-MiB retained budget");
    check(!runner.model.adaln_cache && !runner.model.adaln_graph_inputs, "budget failure enabled partial imports");
    runner.reset_cuda_optimization_state();
    runner.prepare_cuda_schedule(schedule, "fixed-recovery", "weights");
    runner.begin_cuda_step(0, 3, true);
    check(!runner.compute(1, p).empty(), "runner did not recover after prefill failure/reset");
    std::cout << "PASS invalid schedule/sigma/shape, stale token, real runner small-budget failure/recovery\n";
}
struct NegativeTestAdapter : WeightAdapter {
    ggml_tensor* patch_weight(ggml_context*, ggml_backend_t, ggml_tensor* w, const std::string&) override { return w; }
    ggml_tensor* forward_with_lora(ggml_context*, ggml_backend_t, ggml_tensor*, ggml_tensor*, ggml_tensor*,
                                    const std::string&, ForwardParams) override { throw std::runtime_error("adapter must not execute"); }
    size_t get_extra_graph_size() override { return 0; }
};
struct NegativeTestGroup : edgedit::parallel::ProcessGroup {
    edgedit::parallel::Backend backend() const override { return edgedit::parallel::Backend::kCpu; }
    int rank() const override { return 0; }
    int size() const override { return 2; }
    int local_rank() const override { return 0; }
    bool enabled() const override { return true; }
    void barrier() override { throw std::runtime_error("group must not execute"); }
    void all_reduce(const edgedit::parallel::Buffer&, const edgedit::parallel::Buffer&, edgedit::parallel::ReduceOp) override {}
    void all_gather(const edgedit::parallel::Buffer&, const edgedit::parallel::Buffer&) override {}
    void all_to_all(const edgedit::parallel::Buffer&, const edgedit::parallel::Buffer&, size_t) override {}
    void broadcast(const edgedit::parallel::Buffer&, int) override {}
};
void runtime_mode_test(ggml_backend_t backend) {
    TinyRunner runner(backend, tiny_weights()), host_placed(backend, tiny_weights());
    host_placed.use_test_host_parameters();
    Inputs inputs;
    auto p = inputs.params();
    const char* previous = std::getenv("ED_MINIMAX_H3_REUSE_GRAPH");
    const bool existed = previous != nullptr;
    const std::string old = previous ? previous : "";
    for (int method = 0; method < 3; ++method) {
        MiniMaxH3::CudaOptimizationOptions options;
        options.adaln_cache = method == 0;
        options.sparse_attn = method == 1;
        options.sage_attn = method == 2;
        unsetenv("ED_MINIMAX_H3_REUSE_GRAPH");
        runner.validate_cuda_research_execution_mode(options);
        host_placed.model.cuda_optimizations = options; // negative placement test only
        rejects_with([&] { host_placed.validate_cuda_research_execution_mode(options); }, "resident immutable",
                     "CPU lifecycle helpers accidentally enabled production offload");
        rejects_with([&] { host_placed.compute(1, p); }, "resident immutable",
                     "offload reached prefill/build despite production guard");
        check(!host_placed.adaln_branches[0].cache && !host_placed.adaln_branches[1].cache &&
              !host_placed.model.adaln_cache && !host_placed.model.adaln_graph_inputs,
              "rejected offload created request cache state");
        runner.model.cuda_optimizations = options; // CPU-only negative test bypass
        setenv("ED_MINIMAX_H3_REUSE_GRAPH", "1", 1);
        rejects_with([&] { runner.compute(1, p); }, "graph reuse disabled",
                     "late reuse mutation reached prefill/build");
        unsetenv("ED_MINIMAX_H3_REUSE_GRAPH");
        runner.set_weight_adapter(std::make_shared<NegativeTestAdapter>());
        rejects_with([&] { runner.compute(1, p); }, "immutable single-device",
                     "late adapter mutation reached prefill/build");
        runner.set_weight_adapter(nullptr);
        runner.set_process_group(std::make_shared<NegativeTestGroup>());
        rejects_with([&] { runner.compute(1, p); }, "single-device",
                     "late process-group mutation reached prefill/build");
        runner.set_process_group(nullptr);
    }
    if (existed) setenv("ED_MINIMAX_H3_REUSE_GRAPH", old.c_str(), 1);
    else unsetenv("ED_MINIMAX_H3_REUSE_GRAPH");
    runner.model.cuda_optimizations = {};
    runner.model.cuda_optimizations.sparse_attn = true;
    runner.config.num_attention_heads = 56;
    runner.config.num_layers = 50;
    sd::Tensor<float> primary_geometry({32, 32, 37, 2});
    p.x = &primary_geometry;
    rejects_with([&] { runner.compute(1, p); }, "bounded execution required",
                 "primary SubBlock request reached graph construction");
    runner.model.cuda_optimizations = {};
    setenv("ED_MINIMAX_H3_REUSE_GRAPH", "1", 1);
    runner.validate_cuda_research_execution_mode({}); // baseline unchanged
    host_placed.validate_cuda_research_execution_mode({}); // baseline placement unchanged
    if (existed) setenv("ED_MINIMAX_H3_REUSE_GRAPH", old.c_str(), 1);
    else unsetenv("ED_MINIMAX_H3_REUSE_GRAPH");
    std::cout << "PASS late AdaLN/SubBlock/Sage reuse/adapter/process-group mutations rejected before prefill; baseline unchanged\n";
}
} // namespace

int main(int argc, char** argv) {
    ggml_backend_t backend = nullptr;
    try {
        // Do not use init_by_type/load_all/dev_count: those enumerate GPUs.
        ggml_backend_register(h3_test_cpu_reg());
        backend = h3_test_cpu_init();
        check(backend != nullptr && ggml_backend_is_cpu(backend), "independent CPU backend unavailable");
        h3_test_cpu_threads(backend, 1);
        if (argc == 2 && std::string(argv[1]) == "--shared-prefill-reentry") {
            shared_prefill_reentry_test(backend);
            ggml_backend_free(backend);
            return 0;
        }
        check(argc == 1, "unknown runtime regression argument");
        schedule_test(backend, false);
        schedule_test(backend, true);
        branch_layout_test(backend);
        shared_prefill_budget_test(backend, false);
        shared_prefill_budget_test(backend, true);
        shared_prefill_cfg_test(backend, false);
        shared_prefill_cfg_test(backend, true);
        shared_prefill_reentry_test(backend);
        failure_test(backend);
        runtime_mode_test(backend);
        ggml_backend_free(backend);
        std::cout << "H3 integrated runtime optimization tests passed (" << checks << " checks; "
                  << comparisons << " actual fullgraph comparisons, " << exact_comparisons
                  << " bitexact; CPU only, no GPU enumeration/Sage execution)\n";
        return 0;
    } catch (const std::exception& error) {
        if (backend) ggml_backend_free(backend);
        std::cerr << "H3 integrated runtime optimization regression FAILED: " << error.what() << '\n';
        return 1;
    }
}
