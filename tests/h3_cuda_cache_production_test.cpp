// Production Cache-DiT dispatch verification, not a test-only pass invocation.
#define H3_CACHE_OFFLOAD_CUDA_TEST
#define H3_CACHE_OFFLOAD_NO_MAIN
#include "h3_cuda_cache_offload_test.cpp"
int main(int argc,char** argv) {
    try {
        try { require_gpu_permission(argc,argv); }
        catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
        auto* cuda=ggml_backend_cuda_init(0);demand(cuda,"CUDA required");
        {
            OffloadHarness candidate(cuda,true),reference(cuda,true);
            candidate.set_max_graph_vram_bytes(64u<<20);reference.set_max_graph_vram_bytes(64u<<20);
            auto options=cache_options();options.cache_threshold=100.f;
            candidate.configure_cuda_optimizations(options);reference.configure_cuda_optimizations({});
            sd::Tensor<float> x({2,2,1,2},{.2f,-.3f,.1f,.4f,.5f,-.2f,0.f,0.f});
            sd::Tensor<float> context({8,3}); for(int64_t i=0;i<context.numel();++i)context[i]=.03f*float(i+1);
            sd::Tensor<float> time({1},{700.f});DiffusionParams p{};p.x=&x;p.context=&context;p.timesteps=&time;p.minimax_audio_length=2;
            size_t comparisons=0,hits=0;
            for(int request=0;request<2;++request) {
                for(int step=0;step<6;++step) for(int branch=0;branch<2;++branch) {
                    // Identical input/timestep permits exact full/miss/hit checks.
                    candidate.begin_cuda_step(step,6,branch!=0);
                    auto a=reference.MiniMaxH3Runner::compute(1,p);
                    auto b=candidate.MiniMaxH3Runner::compute(1,p);
                    demand(a.shape()==b.shape(),"production output shape differs");
                    for(int64_t n=0;n<a.numel();++n)
                        demand(std::isfinite(a[n]) && std::isfinite(b[n]) && std::abs(a[n]-b[n])<3e-5f,
                               "production cache changed identical-step output");
                    ++comparisons;
                }
                hits+=candidate.cuda_cache_policy.branch(0).hit_steps+candidate.cuda_cache_policy.branch(1).hit_steps;
                candidate.reset_cuda_optimization_state();
                demand(!candidate.model.cuda_cache_pass && !candidate.tap_registry(),"production reset left taps/pass");
            }
            demand(comparisons==24 && hits>0,"production cache did not exercise hits");
            std::cout<<"PASS production CUDA Cache-DiT comparisons="<<comparisons<<" hits="<<hits<<"; two requests and CFG roles\n";
        }
        ggml_backend_free(cuda);return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
