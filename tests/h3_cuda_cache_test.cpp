// CPU-only regression: never enumerates or initializes a CUDA device. Exercises
// the actual H3 graph and runner two-pass path using the existing CPU backend.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
MiniMaxH3::CudaOptimizationOptions cache_options() {
    MiniMaxH3::CudaOptimizationOptions options;
    options.cache_dit = true;
    options.cache_front_blocks = 1;
    options.cache_back_blocks = 1;
    options.cache_warmup_steps = 1;
    options.cache_max_consecutive = 2;
    options.cache_start_percent = 0.f;
    options.cache_end_percent = 1.f;
    return options;
}
void policy_test() {
    using Policy = MiniMaxH3::CudaDualBlockCachePolicy;
    Policy policy;
    auto options = cache_options();
    policy.configure(options, 4);
    bool threw = false;
    try { policy.require_step(); } catch (const std::logic_error&) { threw = true; }
    check(threw, "step lifecycle missing-call not rejected");
    policy.begin_step(0, 10, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f, 0.f, 0.f}), "cold/warmup hit");
    policy.finish(false, true);
    policy.begin_step(1, 10, false);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f, 0.f, 0.f}), "CFG branches shared anchors");
    policy.finish(false, true);
    policy.begin_step(1, 10, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f, .05f, 0.f}), "audio guard diluted by video");
    check(!policy.hit({.05f, 0.f, 0.f}), "text guard not checked");
    check(!policy.hit({0.f, 0.f, .05f}), "video guard not checked");
    check(!policy.hit({0.f, NAN, 0.f}), "NaN accepted");
    check(!policy.hit({0.f, INFINITY, 0.f}), "zero norm/inf accepted");
    check(policy.hit({.01f, .02f, .03f}), "safe per-modality hit rejected");
    policy.finish(true, true);
    policy.begin_step(2, 10, true);
    policy.bind_identity("request-A");
    check(policy.hit({0.f, 0.f, 0.f}), "second consecutive hit rejected");
    policy.finish(true, true);
    policy.begin_step(3, 10, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f, 0.f, 0.f}), "max consecutive exceeded");
    policy.finish(false, true);
    policy.begin_step(4, 10, true);
    policy.bind_identity("changed-reference-or-layout");
    check(!policy.hit({0.f, 0.f, 0.f}), "conditioning change accepted");
    policy.finish(false, true);
    policy.reset();
    policy.begin_step(5, 10, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f, 0.f, 0.f}), "request reset leaked anchor");
    policy.finish(false, true);
    policy.begin_step(7, 10, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f}), "step gap retained branch anchor");
    policy.finish(false, true);
    policy.begin_step(8, 12, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f}), "changed total_steps retained branch anchor");
    policy.finish(false, true);
    policy.begin_step(8, 12, true);
    policy.bind_identity("request-A");
    check(!policy.hit({0.f}), "repeated step retained branch anchor");
    policy.finish(false, true);
    options.cache_start_percent = .3f;
    options.cache_end_percent = .6f;
    policy.configure(options, 4);
    for (int i = 0; i < 10; ++i) {
        policy.begin_step(i, 10, true);
        policy.bind_identity("window");
        const bool expected = i >= 3 && i <= 5;
        check(policy.hit({0.f, 0.f, 0.f}) == expected, "cache window wrong");
        policy.finish(false, true);
    }
    float a[] = {1.f, -2.f}, b[] = {1.01f, -2.02f}, zeros[] = {0.f, 0.f};
    check(std::fabs(Policy::relative_l1(b, a, 2) - .01f) < 1e-6f, "RF relative L1 math wrong");
    check(!std::isfinite(Policy::relative_l1(a, zeros, 2)), "zero reference norm accepted");
    options.cache_front_blocks = 3;
    threw = false;
    try { policy.configure(options, 4); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "empty middle partition accepted");
}

String2TensorStorage tiny_weights() {
    String2TensorStorage weights;
    auto add = [&](const std::string& suffix, std::initializer_list<int64_t> shape) {
        TensorStorage tensor;
        tensor.name = "model.diffusion_model." + suffix;
        tensor.n_dims = static_cast<int>(shape.size());
        int i = 0;
        for (auto extent : shape) tensor.ne[i++] = extent;
        weights.emplace(tensor.name, tensor);
    };
    add("video_patch_proj.weight", {4, 8});
    add("audio_patch_proj.weight", {1, 8});
    add("condition_proj.weight", {8, 8});
    add("time_embedder.proj_in.weight", {4, 8});
    add("time_embedder.proj_out.weight", {8, 4});
    add("rope.inv_freq", {1});
    for (int i = 0; i < 4; ++i) {
        const auto prefix = "blocks." + std::to_string(i) + ".";
        add(prefix + "attn.q_norm.weight", {128});
        add(prefix + "attn.qkv_proj.weight", {8, 384});
        add(prefix + "mlp.fc1.weight", {8, 32});
    }
    return weights;
}

// CPU execution is deliberately available only by explicitly calling the
// algorithm helper in this test. Public configure/compute still reject CPU.
struct CpuCacheTestRunner : MiniMaxH3::MiniMaxH3Runner {
    std::vector<int> block_nodes;
    std::vector<int> pass_nodes;
    using MiniMaxH3Runner::MiniMaxH3Runner;
    ggml_cgraph* test_graph(const DiffusionParams& params) {
        static const std::vector<sd::Tensor<float>> empty_conditions;
        static const std::vector<MiniMaxH3ReferenceBlock> empty_refs;
        static const sd::Tensor<int32_t> empty_int;
        auto* graph = build_graph(*params.x, *params.timesteps, *params.context,
                                 empty_conditions, empty_conditions, empty_int, empty_int,
                                 empty_refs, params.minimax_audio_length,
                                 params.minimax_video_sigma_shift, params.minimax_audio_sigma_shift);
        int count = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const char* name = ggml_graph_node(graph, i)->name;
            const std::string node_name(name);
            const auto begin = node_name.find("ggml_runner_cut:minimax_h3.blocks.");
            const auto separator = node_name.find('|');
            if (begin == 0 && separator != std::string::npos &&
                node_name.substr(separator) == "|hidden_states" &&
                node_name.substr(34, separator - 34).find('.') == std::string::npos) ++count;
        }
        block_nodes.push_back(count);
        pass_nodes.push_back(ggml_graph_n_nodes(graph));
        return graph;
    }
    sd::Tensor<float> full(const DiffusionParams& params) {
        model.cuda_cache_pass = nullptr;
        auto get_graph = [&] { return test_graph(params); };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(get_graph, 1, false, false), params.x->dim());
    }
    sd::Tensor<float> cached(const DiffusionParams& params) {
        auto get_graph = [&] { return test_graph(params); };
        return compute_cuda_dual_block_cache(1, params, get_graph);
    }
    void enable_test_cache() {
        model.cuda_optimizations = cache_options();
        cuda_cache_policy.configure(model.cuda_optimizations, config.num_layers);
        reset_cuda_optimization_state();
    }
    void seed_weights() {
        check(alloc_params_buffer(), "tiny weights allocation failed");
        std::map<std::string, ggml_tensor*> tensors;
        get_param_tensors(tensors, "model.diffusion_model");
        for (const auto& [name, tensor] : tensors) {
            std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = name.find("norm") != std::string::npos ? 1.f
                         : name.find("rope.inv_freq") != std::string::npos ? .1f
                         : .02f + .005f * std::sin(static_cast<float>(i));
            }
            check(tensor->type == GGML_TYPE_F32, "unexpected test weight type");
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
        }
    }
};
void same_tensor(const sd::Tensor<float>& a, const sd::Tensor<float>& b, const char* message) {
    check(a.shape() == b.shape() && !a.empty(), message);
    for (int64_t i = 0; i < a.numel(); ++i)
        check(std::isfinite(a[i]) && std::isfinite(b[i]) && std::fabs(a[i] - b[i]) < 2e-5f, message);
}
void graph_test() {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(backend != nullptr, "CPU backend unavailable");
    {
        auto weights = tiny_weights();
        CpuCacheTestRunner runner(backend, weights);
        check(runner.config.attention_head_dim == 128 && runner.config.num_layers == 4,
              "test does not use H3 real head_dim");
        runner.seed_weights();
        sd::Tensor<float> x({2, 2, 1, 2}, {.2f, -.3f, .1f, .4f, .5f, -.2f, 0.f, 0.f});
        sd::Tensor<float> context({8, 2});
        for (int64_t i = 0; i < context.numel(); ++i) context[i] = .1f + .01f * static_cast<float>(i);
        sd::Tensor<float> timestep({1}, {800.f});
        DiffusionParams params;
        params.x = &x;
        params.context = &context;
        params.timesteps = &timestep;
        params.minimax_audio_length = 1;
        params.minimax_video_sigma_shift = 12.f;
        params.minimax_audio_sigma_shift = 3.f;
        const auto baseline = runner.full(params);
        bool threw = false;
        try { runner.configure_cuda_optimizations(cache_options()); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "public CUDA gate accepted CPU cache");
        runner.enable_test_cache();
        runner.begin_cuda_step(0, 8, true);
        const auto miss = runner.cached(params);
        same_tensor(baseline, miss, "split miss differs from uncached full H3 graph");
        std::cout << "block nodes full/probe/miss: " << runner.block_nodes[0] << '/' << runner.block_nodes[1] << '/' << runner.block_nodes[2] << '\n';
        check(runner.block_nodes[1] == 1 && runner.block_nodes[2] == 3,
              "miss did not execute real front/middle/suffix partition");
        runner.begin_cuda_step(1, 8, true);
        const auto hit = runner.cached(params);
        same_tensor(miss, hit, "exact-input hit differs from real miss");
        check(runner.cuda_cache_policy.branch(1).hit_steps == 1, "exact RF probe did not hit");
        check(runner.block_nodes[3] == 1 && runner.block_nodes[4] == 1,
              "hit recomputed middle or omitted real suffix");
        runner.begin_cuda_step(2, 8, true);
        runner.cached(params);
        runner.begin_cuda_step(3, 8, true);
        runner.cached(params);
        check(runner.cuda_cache_policy.branch(1).full_steps == 2, "forced middle refresh did not run");
        runner.begin_cuda_step(3, 8, false);
        same_tensor(baseline, runner.cached(params), "CFG cold branch differs from baseline");
        check(runner.cuda_cache_policy.branch(0).hit_steps == 0, "CFG device anchor shared");
        context[0] += .1f;
        runner.begin_cuda_step(4, 8, true);
        const auto changed = runner.cached(params);
        check(runner.cuda_cache_policy.branch(1).hit_steps == 0, "changed conditioning did not invalidate");
        runner.begin_cuda_step(6, 10, true);
        same_tensor(changed, runner.cached(params), "total_steps change altered cold graph output");
        check(runner.cuda_cache_policy.branch(1).hit_steps == 0, "total_steps change retained device anchor");
        runner.begin_cuda_step(8, 10, true);
        same_tensor(changed, runner.cached(params), "step gap altered cold graph output");
        check(runner.cuda_cache_policy.branch(1).hit_steps == 0, "step gap retained device anchor");
        runner.reset_cuda_optimization_state();
        runner.begin_cuda_step(5, 8, true);
        same_tensor(changed, runner.cached(params), "request reset changes cold result");
        check(runner.cuda_cache_policy.branch(1).hit_steps == 0, "request reset leaked device cache");
        runner.reset_cuda_optimization_state();
        runner.model.cuda_optimizations = {};
        same_tensor(changed, runner.full(params), "cache off path changed after reset");
        const auto capability = MiniMaxH3::MiniMaxH3Runner::cuda_optimization_capabilities();
        check(capability.cache_dit && capability.adaln_cache && capability.sparse_attn,
              "capability does not describe implemented operations");
        std::cout << "actual H3 CPU graph: full, split miss, RF hit, suffix, CFG, reset passed\n";
    }
    ggml_backend_free(backend);
}
}
int main() {
    try {
        policy_test();
        graph_test();
        std::cout << "H3 dual-block cache CPU tests passed (no CUDA initialized)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
