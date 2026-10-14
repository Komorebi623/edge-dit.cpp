// Dedicated, OPT-IN H3F8 executable. No GPU execution during build.
// Build with the SAME candidate macros/includes/libraries as the F16 sidecar,
// replacing its source path with this file. Requires Sage CU compiled by CUDA
// >=12.8, SM89. Launch only via scripts/test_h3_gpu_parity.sh.
// Upstream d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:
// core.py::sageattn_qk_int8_pv_fp8_cuda(qk_quant_gran="per_warp",
// pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False), V scale_max=2.25.
// This is an explicit official configuration, NOT auto's per_thread QK default.
// Same predeclared tolerances and 126-case matrix; no fallback or threshold change.
// Reuse the production-cut fixture to avoid a second drifting test implementation.
#define main h3_sage_fp16_fixture_unused_main
#include "h3_cuda_sage_offload_test.cpp"
#undef main

int main(int argc,char** argv) {
    try {
        require_opt_in(argc,argv); // BEFORE any CUDA discovery/registration
        Env h3fp8("ED_H3_SAGE_PV_FP8","1"),gate("ED_SAGE_ATTN","0"),fp8("ED_SAGE_PV_FP8","0"),fallback("ED_SAGE_FALLBACK","0"),
            pool("ED_GRAPH_CUT_CACHE_POOL","1"),compact("ED_GRAPH_CUT_CACHE_COMPACT","0"),
            no_capture("GGML_CUDA_DISABLE_GRAPHS","1"); // count submissions, not graph replays; NOT a perf test
        ggml_backend_register(h3_sage_cpu_reg());auto* reg=ggml_backend_cuda_reg();
        require(reg && ggml_backend_reg_dev_count(reg)==1 && ggml_backend_cuda_get_device_count()==1,"exactly one actual CUDA device",3);
        ggml_backend_register(reg);Backend cuda(ggml_backend_cuda_init(0),ggml_backend_free);
        require(cuda && ggml_backend_is_cuda(cuda.get()),"missing CUDA backend",3);
        const auto dispatch_before=ed_cuda_sage_attn_h3_fp8_dispatch_count();
        // H3F8 is explicitly opt-in in addition to the H3 method's master gate.
        // The included F16-only negative matrix is deliberately not reused here.
        require(ed_cuda_h3_sage_fp8_arch_supported(8,9),"SM89 arch policy");
        for(auto arch: {std::pair<int,int>{8,0},{8,6},{9,0},{12,0}})
            require(!ed_cuda_h3_sage_fp8_arch_supported(arch.first,arch.second),"unsupported arch policy");
        require(ed_cuda_sage_attn_h3_fp8_dispatch_count()==dispatch_before,"rejected ops counted as dispatches");
        std::cout<<"TEST-ONLY H3F8 per-warp INT8-QK/E4M3-PV/FP32+FP16-accum V-scale-max=2.25 smooth-K no dense fallback; device="<<env("CUDA_VISIBLE_DEVICES")<<'\n';
        struct Case {int l,h,b;bool half,packed,strided;int pattern;};
        const Case cases[]={{7,2,1,false,false,false,0},{63,2,1,false,false,true,1},
            {64,2,1,true,true,true,0},{65,2,2,false,false,true,0},{127,2,1,true,false,true,2},
            {128,2,1,false,false,false,1},{129,2,1,false,true,true,0},
            {257,2,2,true,false,true,0},{513,2,1,false,false,true,2}};
        size_t empty=0;
        for(const auto& c:cases) {
            Harness h(cuda.get(),c.l,c.h,c.b,c.half,c.packed,c.strided);
            std::vector<sd::Tensor<float>> prior;
            const auto prefix="L"+std::to_string(c.l)+"/B"+std::to_string(c.b)+"/F16"+std::to_string(c.half);
            for(int seed:{1,2,1}) {
                h.fill(seed,c.pattern);const auto ref=h.reference();
                const auto dense=h.evaluate(false,false),dense_cut=h.evaluate(false,true);
                const auto sage=h.evaluate(true,false),sage_cut=h.evaluate(true,true);
                compare(ref,dense,prefix+"/CUDA-dense-reference");compare(dense,dense_cut,prefix+"/dense-cuts");
                compare(dense,sage,prefix+"/CUDA-dense-vs-Sage",true);compare(sage,sage_cut,prefix+"/Sage-cuts");
                if(prior.size()==2)compare(prior.front(),sage_cut,prefix+"/repeat-request");
                prior.push_back(sage_cut);
            }
            double changed=0;for(int64_t i=0;i<prior[0].numel();++i)changed+=std::fabs(prior[0][i]-prior[1][i]);
            require(changed>1e-3,"input change did not affect output");
            bool failed=false;try{h.evaluate(true,true,1);}catch(const Failure& e){failed=std::string(e.what())=="injected post-compute failure";}
            require(failed,"failure injection not reached");h.restored();
            compare(prior.front(),h.evaluate(true,true),prefix+"/post-error-recovery");empty+=h.empty_param_segments;
        }
        require(comparisons==126 && segments>0 && dispatches>0 && reused_qkv>0,"incomplete/reuse matrix");
        const auto actual_dispatches=ed_cuda_sage_attn_h3_fp8_dispatch_count()-dispatch_before;
        require(actual_dispatches==dispatches,"actual CUDA H3SA submission count differs from audited graphs");
        std::cout<<"PASS comparisons="<<comparisons<<" segments="<<segments<<" H3SA_dispatches="<<dispatches
                 <<" actual_CUDA_H3F8_submissions="<<actual_dispatches
                 <<" audited_nodes="<<audited_nodes<<" reused_QK_cut_addresses="<<reused_qkv
                 <<" empty-param-whole-set-stagings="<<empty<<'\n';
        std::cout<<"NOT a full Q4 model, async lifecycle, max-VRAM, or performance validation.\n";return 0;
    } catch(const Failure& e){std::cerr<<"FAIL/NOT RUN: "<<e.what()<<'\n';return e.code;}
      catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
