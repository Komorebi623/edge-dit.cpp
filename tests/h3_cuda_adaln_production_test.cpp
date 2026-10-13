// Opt-in test of the real production configure/schedule/compute dispatch.
// Shared fixture/header tests retain their own serial-cuts coverage; here no
// injection of model.adaln_cache, no custom segment loop and no bypassed gate.
#define main h3_offload_helper_test_main
#include "h3_cuda_adaln_offload_test.cpp"
#undef main

namespace {
void production_matrix(ggml_backend_t cuda, ggml_backend_t cpu, bool curves, bool mixed) {
    Harness baseline(cuda, curves, true), candidate(cuda, curves, true);
    // Enough for each tiny layer, but below the whole model: real budget merge
    // and production compute_with_graph_cuts (not the test's run_cuts loop).
    baseline.set_max_graph_vram_bytes(64u * 1024u);
    candidate.set_max_graph_vram_bytes(64u * 1024u);
    MiniMaxH3::CudaOptimizationOptions options; options.adaln_cache = true;
    candidate.configure_cuda_optimizations(options);
    Inputs inputs; if (mixed) inputs.add_references(); auto p = inputs.params(mixed);
    const auto original_context = inputs.context;
    for (int request = 0; request < 2; ++request) {
        candidate.prepare_cuda_schedule(schedule, "production-tiny-fixed", candidate.weights_identity);
        for (size_t step = 0; step + 1 < schedule.size(); ++step) {
            inputs.timestep[0] = schedule[step] * 1000.f;
            for (int branch = 0; branch < 2; ++branch) {
                inputs.context = original_context;
                if (!branch) for (int64_t n = 0; n < inputs.context.numel(); ++n) inputs.context[n] *= -.5f;
                candidate.begin_cuda_step(int(step), int(schedule.size()-1), branch != 0);
                const auto before = baseline.MiniMaxH3Runner::compute(1, p);
                const auto after = candidate.MiniMaxH3Runner::compute(1, p);
                compare(before, after, "PRODUCTION/curve="+std::to_string(curves)+"/mixed="+std::to_string(mixed)+
                        "/request="+std::to_string(request)+"/step="+std::to_string(step)+"/branch="+std::to_string(branch),
                        2e-5f, 2e-5f);
                candidate.restored(); baseline.restored();
            }
        }
        candidate.reset_cuda_optimization_state();
        candidate.configure_cuda_optimizations(options);
    }
    setenv("ED_ASYNC_OFFLOAD", "1", 1);
    bool rejected = false;
    try { candidate.MiniMaxH3Runner::compute(1, p); } catch (const std::invalid_argument&) { rejected = true; }
    unsetenv("ED_ASYNC_OFFLOAD"); require(rejected, "unsupported async offload did not fail closed");
    candidate.reset_cuda_optimization_state();
    require(!candidate.adaln_branches[0].cache && !candidate.adaln_branches[1].cache,
            "request reset left cache alive");
}
}
int main(int argc, char** argv) {
    try {
        require_opt_in(argc, argv);
        for (const char* key : {"ED_ASYNC_OFFLOAD", "ED_MINIMAX_H3_REUSE_GRAPH"})
            require(env(key).empty() || env(key)=="0", std::string("must be disabled: ")+key, 2);
        ggml_backend_register(h3_offload_cpu_reg());
        auto reg = ggml_backend_cuda_reg();
        require(reg && ggml_backend_reg_dev_count(reg)==1 && ggml_backend_cuda_get_device_count()==1,
                "exactly one CUDA device required", 3);
        ggml_backend_register(reg);
        Backend cuda(ggml_backend_cuda_init(0),ggml_backend_free),cpu(h3_offload_cpu_init(),ggml_backend_free);
        require(cuda && cpu && ggml_backend_is_cuda(cuda.get()), "CUDA backend required", 3);
        for (bool curves : {false,true}) for (bool mixed : {false,true})
            production_matrix(cuda.get(), cpu.get(), curves, mixed);
        require(comparisons==48, "incomplete real production dispatch matrix");
        std::cout << "PASS production CUDA AdaLN offload comparisons=" << comparisons
                  << "; layer-major prefill, real budget planner/compute, two requests and CFG branches; not full Q4 proof\n";
        return 0;
    } catch(const Failure& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return e.code; }
      catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
