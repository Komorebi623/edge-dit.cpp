// CPU-only by default: two REAL GGML CPU backend instances exercise staging,
// graph cuts, persistent device-slot copies and restoration. No CUDA discovery.
// Optional CUDA compilation uses -DH3_CACHE_OFFLOAD_CUDA_TEST and must be run by
// the primary agent through test_h3_gpu_parity.sh. This worker never runs GPU.
#define H3_CUDA_CACHE_TEST_NO_MAIN
#include "h3_cuda_cache_test.cpp"
#include "dit_models/models/minimax_h3_cuda_cache_offload.hpp"
#include <cstring>
#include <set>
#include <ctime>
#ifdef H3_CACHE_OFFLOAD_CUDA_TEST
#include "ggml-cuda.h"
#if defined(GGML_BACKEND_DL) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#error "Requires directly linked NVIDIA CUDA and CPU backends"
#endif
static_assert(GGML_MAX_NAME == 128, "ggml ABI mismatch");
#endif
namespace {
namespace CO = MiniMaxH3::CacheOffload;
size_t offload_checks = 0, executed_segments = 0, compared_outputs = 0;
void demand(bool ok, const std::string& message) {
    ++offload_checks;
    if (!ok) throw std::runtime_error(message);
}
struct OffloadHarness : CO::RunnerBridge<MiniMaxH3::MiniMaxH3Runner> {
    using Base = CO::RunnerBridge<MiniMaxH3::MiniMaxH3Runner>;
    struct Saved { ggml_tensor* tensor; ggml_backend_buffer_t buffer; void* data; void* extra; std::vector<float> values; };
    std::vector<Saved> saved;
    std::vector<std::set<std::string>> transferred;
    bool middle_computed = false;
    CO::State state;
    CO::Options execution;
    bool fail_next_build = false;
    int fail_after_segment = -1;
    size_t step_segments = 0;
    bool gpu;
    OffloadHarness(ggml_backend_t runtime, bool cuda = false)
        : Base(runtime, tiny_weights(), "model.diffusion_model", cuda), state(runtime, 16u*1024*1024), gpu(cuda) {
        if (!cuda) {
            demand(params_backend == runtime_backend, "unexpected initial CPU backend");
            params_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            ggml_init_params staging{size_t(MAX_PARAMS_TENSOR_NUM)*ggml_tensor_overhead(),nullptr,true};
            offload_ctx = ggml_init(staging);
        }
        demand(params_backend && params_backend != runtime_backend, "need independent backend instances");
        set_flash_attention_enabled(false);
        demand(alloc_params_buffer(), "source parameters allocation failed");
        std::map<std::string, ggml_tensor*> named;
        get_param_tensors(named, "model.diffusion_model");
        for (const auto& item : named) {
            auto* t = item.second;
            ggml_set_name(t, item.first.c_str());
            demand(t->type == GGML_TYPE_F32, "test requires F32 source weights");
            std::vector<float> data(static_cast<size_t>(ggml_nelements(t)));
            uint32_t hash = 2166136261u; for (unsigned char c : item.first) hash = (hash ^ c) * 16777619u;
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = item.first.find("norm") != std::string::npos ? 1.f :
                    item.first.find("rope.inv_freq") != std::string::npos ? .1f :
                    .02f + .005f * std::sin(float(i) + float(hash%101u)*.01f);
            ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(float));
            saved.push_back({t,t->buffer,t->data,t->extra,std::move(data)});
        }
        execution.enabled = true; execution.cpu_test_only = !cuda;
        // Tiny CUDA still uses a real bounded planner, not an unbounded override.
        max_graph_vram_bytes = 32u*1024*1024;
        execution.after_segment = [&](size_t, ggml_cgraph* graph, const std::vector<ggml_tensor*>& selected) {
            for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
                const std::string name=ggml_graph_node(graph,i)->name;
                if (name.find("ggml_runner_cut:minimax_h3.blocks.1.")!=std::string::npos ||
                    name.find("ggml_runner_cut:minimax_h3.blocks.2.")!=std::string::npos ||
                    name.find("ggml_runner_cut:minimax_h3.blocks.1|")!=std::string::npos ||
                    name.find("ggml_runner_cut:minimax_h3.blocks.2|")!=std::string::npos) middle_computed=true;
            }
            std::set<std::string> names;
            for (auto* t : selected) {
                names.insert(t->name);
                for (const auto& s : saved) if (s.tensor == t) {
                    demand(t->buffer != s.buffer && t->data != s.data, "parameter was not really staged");
                    std::vector<float> data(s.values.size());
                    ggml_backend_tensor_get(t, data.data(), 0, data.size()*sizeof(float));
                    demand(data == s.values, "parameter staging changed values");
                }
            }
            transferred.push_back(std::move(names));
            ++executed_segments;
            if (int(step_segments++) == fail_after_segment)
                throw std::runtime_error("injected segment exception");
        };
        model.cuda_optimizations = cache_options();
        cuda_cache_policy.configure(model.cuda_optimizations, config.num_layers);
    }
    ~OffloadHarness() { h3_cache_release_graph(); model.cuda_cache_pass = nullptr; set_tap_registry(nullptr); }
    void restored() {
        demand(partial_offload_pairs.empty() && !partial_runtime_params_buffer && !params_on_runtime_backend,
               "parameter staging leaked beyond execution");
        for (const auto& s : saved) {
            demand(s.tensor->buffer == s.buffer && s.tensor->data == s.data && s.tensor->extra == s.extra,
                   "CPU parameter pointer restoration failed");
            demand(std::memcmp(s.tensor->data,s.values.data(),s.values.size()*sizeof(float)) == 0,
                   "source parameter values changed");
        }
        demand(!model.cuda_cache_pass && !tap_registry(), "pass or taps outlived request");
    }
    ggml_cgraph* test_graph(const DiffusionParams& p) {
        if (fail_next_build) { fail_next_build = false; throw std::runtime_error("injected build exception"); }
        static const std::vector<sd::Tensor<float>> empty;
        static const std::vector<MiniMaxH3ReferenceBlock> refs;
        static const sd::Tensor<int32_t> ints;
        return build_graph(*p.x, *p.timesteps, *p.context,
            p.ref_latents ? *p.ref_latents : empty,
            p.minimax_reference_audio_latents ? *p.minimax_reference_audio_latents : empty,
            p.minimax_text_token_tags ? *p.minimax_text_token_tags : ints,
            p.minimax_keyframe_indices ? *p.minimax_keyframe_indices : ints,
            p.minimax_reference_blocks ? *p.minimax_reference_blocks : refs,
            p.minimax_audio_length, p.minimax_video_sigma_shift, p.minimax_audio_sigma_shift);
    }
    sd::Tensor<float> cached(const DiffusionParams& p, int step, bool cond, CO::Statistics* stats=nullptr) {
        transferred.clear(); step_segments=0; middle_computed=false;
        cuda_cache_policy.begin_step(step, 10, cond);
        cuda_cache_policy.bind_identity(cuda_cache_identity(p));
        auto out = CO::compute(*this, state, cuda_cache_policy, model.cuda_optimizations, cuda_cache_operators,
            [&] { return test_graph(p); }, [&](auto* pass) { model.cuda_cache_pass = pass; }, 1, execution, stats);
        restored();
        return restore_trailing_singleton_dims(std::move(out), p.x->dim());
    }
    sd::Tensor<float> baseline(const DiffusionParams& p) {
        model.cuda_cache_pass=nullptr;
        auto output=GGMLRunner::compute<float>([&] { return test_graph(p); },1,true,false);
        demand(output.has_value(),"full tiny H3 baseline failed");
        h3_cache_release_graph(); restored();
        return restore_trailing_singleton_dims(std::move(*output), p.x->dim());
    }
    void reset() { state.reset(); cuda_cache_policy.reset(); }
    void weightless() {
        sd::Tensor<float> data({4},{1.f,2.f,3.f,4.f});
        CO::GraphInputs inputs;
        MiniMaxH3::CudaDualBlockGraphPass pass;
        auto result=h3_cache_run_pass([&] {
            auto* source=make_input(data);
            auto* output=ggml_scale(compute_ctx,source,2.f);
            auto* graph=new_graph_custom(32);ggml_build_forward_expand(graph,output);return graph;
        },pass,inputs,state.arena,1,false,execution);
        demand(result.empty_parameter_segments>0 && result.parameter_tensors==0 && result.parameter_bytes==0,
               "weightless graph staged model weights");
        demand(result.output && result.output->numel()==4,"weightless graph missing output");
        for(size_t i=0;i<4;++i)demand((*result.output)[i]==data[i]*2.f,"weightless graph result mismatch");
        restored();
    }
    bool middle_transferred() const {
        for (const auto& segment : transferred) for (const auto& name : segment)
            if (name.find(".blocks.1.") != std::string::npos || name.find(".blocks.2.") != std::string::npos) return true;
        return false;
    }
};

void lease_test(ggml_backend_t backend) {
    CO::Arena arena(backend, 1024*1024);
    ggml_init_params p{ggml_tensor_overhead()*8+1024,nullptr,true};
    auto* ctx=ggml_init(p);
    auto* t=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,3);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    std::vector<float> values(12,.25f);ggml_backend_tensor_set(t,values.data(),0,48);
    auto captured=arena.capture(t);
    const auto bytes=arena.live_bytes();
    demand(bytes>0,"capture budget not charged");
    std::weak_ptr<const CO::Slot> weak=captured;
    CO::GraphInputs inputs({captured});
    auto* graph_ctx=ggml_init(p);
    auto* alias=inputs.import(graph_ctx,captured->tensor);
    demand(alias!=captured->tensor,"graph input aliases persistent descriptor");
    alias->data=nullptr;alias->buffer=nullptr;alias->extra=nullptr;
    inputs.rebind(); demand(alias->data==captured->tensor->data,"reset leaf was not rebound");
    captured.reset();
    ggml_backend_buffer_free(buffer);ggml_free(ctx); // producer context REALLY gone
    demand(!weak.expired() && arena.live_bytes()==bytes,"producer lifetime freed captured input");
    std::vector<float> actual(12);ggml_backend_tensor_get(alias,actual.data(),0,48);
    demand(actual==values,"independent capture lost source values");
    ggml_free(graph_ctx); inputs.release_graph();
    demand(weak.expired() && arena.live_bytes()==0,"graph-held lease leaked");
}
void compare_outputs(const sd::Tensor<float>& expected,const sd::Tensor<float>& actual,bool cuda,const char* label) {
    ++compared_outputs;
    demand(expected.shape()==actual.shape() && !actual.empty(),label);
    const float abs_tol=cuda?3e-4f:2e-5f,rel_tol=cuda?3e-3f:2e-5f;
    for(int64_t i=0;i<actual.numel();++i)
        demand(std::isfinite(expected[i]) && std::isfinite(actual[i]) &&
               std::fabs(expected[i]-actual[i])<=abs_tol+rel_tol*std::fabs(expected[i]),label);
}
void run_matrix(ggml_backend_t backend,bool cuda) {
    lease_test(backend);
    OffloadHarness runner(backend,cuda);
    sd::Tensor<float> x({2,2,1,2},{.2f,-.3f,.1f,.4f,.5f,-.2f,0.f,0.f});
    sd::Tensor<float> context({8,2});
    for(int64_t i=0;i<context.numel();++i)context[i]=.1f+.01f*float(i);
    sd::Tensor<float> timestep({1},{800.f});
    DiffusionParams p;p.x=&x;p.context=&context;p.timesteps=&timestep;
    p.minimax_audio_length=1;p.minimax_video_sigma_shift=12.f;p.minimax_audio_sigma_shift=3.f;
    const auto base=runner.baseline(p);
    CO::Statistics miss_stats,hit_stats;
    auto miss=runner.cached(p,0,true,&miss_stats);
    compare_outputs(base,miss,cuda,"split miss vs full H3 mismatch");
    demand(!miss_stats.hit && runner.middle_transferred() && runner.middle_computed,"miss skipped real middle parameters/compute");
    const auto cold_transfer=miss_stats.parameter_bytes;
    const auto anchor=runner.state.anchor(1);
    auto hit=runner.cached(p,1,true,&hit_stats);
    compare_outputs(miss,hit,cuda,"exact-input hit vs miss mismatch");
    demand(hit_stats.hit,"identical front did not hit");
    demand(!runner.middle_transferred() && !runner.middle_computed,"hit transferred or computed middle block");
    demand(hit_stats.parameter_bytes<miss_stats.parameter_bytes,"hit did not reduce parameter transfer");
    demand(runner.state.anchor(1).serial==anchor.serial && runner.state.anchor(1).rf==anchor.rf &&
           runner.state.anchor(1).delta==anchor.delta,"hit advanced RF independently of delta");
    auto cfg=runner.cached(p,1,false,&miss_stats);
    compare_outputs(base,cfg,cuda,"cold CFG branch mismatch");
    demand(!miss_stats.hit,"CFG branch shared anchor");
    runner.cached(p,2,true,&hit_stats);demand(hit_stats.hit,"second hit rejected");
    runner.cached(p,3,true,&miss_stats);demand(!miss_stats.hit && runner.middle_transferred(),"max-consecutive not enforced");
    context[0]+=.01f;
    auto changed=runner.cached(p,4,true,&miss_stats);
    demand(!miss_stats.hit,"conditioning identity failed to invalidate");
    compare_outputs(runner.baseline(p),changed,cuda,"changed conditioning cold result mismatch");
    runner.cached(p,6,true,&miss_stats);demand(!miss_stats.hit,"step gap failed to invalidate");
    runner.reset(); runner.cached(p,0,true);
    runner.fail_after_segment=int(hit_stats.probe_segments); // first actual remainder segment
    bool threw=false;
    try{runner.cached(p,1,true);}catch(const std::exception& e){threw=std::string(e.what())=="injected segment exception";}
    demand(threw,"segment failure not injected");runner.restored();
    demand(!runner.state.anchor(1).rf && !runner.cuda_cache_policy.branch(1).valid,"exception retained anchor");
    runner.fail_after_segment=-1;runner.cached(p,2,true,&miss_stats);demand(!miss_stats.hit,"failure recovery reused anchor");
    runner.fail_next_build=true;threw=false;
    try{runner.cached(p,3,true);}catch(const std::exception& e){threw=std::string(e.what())=="injected build exception";}
    demand(threw,"build exception not injected");runner.restored();
    runner.cached(p,4,true,&miss_stats);demand(!miss_stats.hit,"build failure recovery reused anchor");
    size_t polls=0;runner.execution.cancelled=[&]{return ++polls>=4;};threw=false;
    try{runner.cached(p,5,true);}catch(const std::exception& e){threw=std::string(e.what()).find("cancelled")!=std::string::npos;}
    demand(threw,"cancellation not observed");runner.restored();
    runner.execution.cancelled={};runner.cached(p,6,true,&miss_stats);demand(!miss_stats.hit,"cancel recovery reused anchor");
    // Reset while a real segment has just executed: old graph leases must remain
    // valid, but the old request must not commit any RF/delta into new state.
    const auto audit=runner.execution.after_segment;bool reset_once=false;
    runner.execution.after_segment=[&](size_t i,ggml_cgraph* g,const std::vector<ggml_tensor*>& s){audit(i,g,s);if(!reset_once){reset_once=true;runner.state.reset();}};
    threw=false;try{runner.cached(p,7,true);}catch(const std::exception& e){threw=std::string(e.what()).find("cancelled")!=std::string::npos;}
    demand(threw && reset_once,"in-flight state reset not rejected");runner.restored();
    runner.execution.after_segment=audit;runner.reset();runner.cached(p,0,true,&miss_stats);demand(!miss_stats.hit,"reset request had a hit");
    const auto budget_peak=runner.state.arena.peak_bytes();
    runner.reset();
    // A local anchor copy deliberately retains its two slots until this scope
    // ends; accounting must continue to include those live leases.
    demand(runner.state.arena.live_bytes()>0 && budget_peak>=runner.state.arena.live_bytes(),"lease accounting lost reset-held buffers");
    runner.weightless();
    std::cout<<"Cache-DiT "<<(cuda?"CUDA":"CPU-to-CPU")<<" full/miss/hit/CFG/restore/reset passed; hit transfer "
             <<cold_transfer<<" -> "<<hit_stats.parameter_bytes<<" bytes, probe/remainder="<<hit_stats.probe_segments<<"/"<<hit_stats.remainder_segments<<'\n';
}
void mixed_matrix(ggml_backend_t backend,bool cuda) {
    OffloadHarness runner(backend,cuda);
    sd::Tensor<float> packed({2,2,1,2},{.2f,-.3f,.1f,.4f,.5f,-.2f,0.f,0.f});
    sd::Tensor<float> context({8,3});
    for(int64_t i=0;i<context.numel();++i)context[i]=.1f+.01f*float(i);
    sd::Tensor<float> timestep({1},{800.f});
    sd::Tensor<int32_t> tags({3},{1,2,0}), keyframes({1},{0});
    std::vector<sd::Tensor<float>> videos, audios;
    videos.emplace_back(std::vector<int64_t>{2,2,1,1,1},std::vector<float>{.3f,.1f,-.2f,.4f});
    videos.emplace_back(std::vector<int64_t>{2,2,1,1,1},std::vector<float>{-.1f,.2f,.3f,.1f});
    audios.emplace_back(std::vector<int64_t>{1,2,1,1},std::vector<float>{.2f,-.1f});
    std::vector<MiniMaxH3ReferenceBlock> refs={{MiniMaxH3ReferenceKind::VIDEO_AUDIO,0,0},{MiniMaxH3ReferenceKind::IMAGE,1,-1}};
    DiffusionParams p;p.x=&packed;p.timesteps=&timestep;p.context=&context;p.minimax_audio_length=1;
    p.ref_latents=&videos;p.minimax_reference_audio_latents=&audios;p.minimax_reference_blocks=&refs;
    p.minimax_text_token_tags=&tags;p.minimax_keyframe_indices=&keyframes;
    const auto original=context;
    for(int repeat=0;repeat<2;++repeat)for(int branch=0;branch<2;++branch) {
        context=original;if(!branch)for(int64_t i=0;i<context.numel();++i)context[i]*=-.5f;
        for(float sigma:{800.f,500.f,200.f}) {
            timestep[0]=sigma;runner.reset();
            const auto full=runner.baseline(p);
            CO::Statistics miss,hit;
            const auto a=runner.cached(p,0,branch!=0,&miss);
            compare_outputs(full,a,cuda,"mixed/sigma/CFG miss mismatch");
            const auto b=runner.cached(p,1,branch!=0,&hit);
            compare_outputs(a,b,cuda,"mixed/sigma/CFG exact hit mismatch");
            demand(hit.hit && !runner.middle_transferred() && !runner.middle_computed,"mixed hit recomputed or transferred middle");
            demand(hit.per_span_relative_l1.size()>=5,"mixed per-reference/modality RF spans missing");
            for(float metric:hit.per_span_relative_l1)demand(std::isfinite(metric) && metric<.04f,"mixed RF scalar invalid");
        }
    }
    // Change ONLY an audio reference value. Full identity must invalidate an
    // otherwise identical CFG/time/layout anchor, not just its long video span.
    audios[0][0]+=.05f;CO::Statistics changed;
    auto output=runner.cached(p,2,true,&changed);
    demand(!changed.hit,"in-place audio reference change retained anchor");
    compare_outputs(runner.baseline(p),output,cuda,"audio reference mutation miss mismatch");
    runner.reset();
    demand(runner.state.arena.live_bytes()==0,"completed request leaked snapshots");
}
void negative_matrix(ggml_backend_t backend) {
    CO::State state(backend,1024*1024);
    ggml_init_params p{ggml_tensor_overhead()*16+1024,nullptr,true};
    auto* ctx=ggml_init(p);
    auto* f=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,3);
    auto* h=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,4);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    const float values[12]={1,2,3,4,5,6,7,8,9,10,11,12};
    ggml_backend_tensor_set(f,values,0,sizeof(values));
    auto lease=state.arena.capture(f);const auto generation=state.generation();
    state.commit(1,generation,lease,lease);
    auto* view=ggml_view_2d(ctx,f,2,3,f->nb[1],0);
    bool threw=false;try{state.arena.capture(view);}catch(const std::exception&){threw=true;}
    demand(threw,"non-contiguous capture accepted");
    threw=false;try{state.arena.capture(h);}catch(const std::exception&){threw=true;}
    demand(threw,"F16 capture silently changed precision");
    CO::Arena tiny(backend,1);threw=false;
    try{tiny.capture(f);}catch(const std::exception&){threw=true;}
    demand(threw && tiny.live_bytes()==0,"over-budget allocation accepted/leaked");
    state.reset();threw=false;
    try{state.commit(1,generation,lease,lease);}catch(const std::exception&){threw=true;}
    demand(threw && !state.anchor(1).rf,"stale request committed old anchor");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
#ifdef H3_CACHE_OFFLOAD_CUDA_TEST
void require_gpu_permission(int argc,char** argv) {
    auto env=[](const char* name){auto* s=std::getenv(name);return std::string(s?s:"");};
    demand(argc==2 && std::string(argv[1])=="--run" && env("H3_GPU_PARITY_ALLOW")=="1","NOT RUN: explicit GPU permission required");
    auto uuid=env("CUDA_VISIBLE_DEVICES");
    demand(uuid.size()==40 && uuid.rfind("GPU-",0)==0 && uuid==env("H3_GPU_PARITY_PREFLIGHT_DEVICE"),"NOT RUN: single GPU UUID/preflight required");
    for(size_t i=4;i<uuid.size();++i)
        demand((i==12 || i==17 || i==22 || i==27)?uuid[i]=='-':
               std::string("0123456789abcdefABCDEF").find(uuid[i])!=std::string::npos,
               "NOT RUN: malformed full GPU UUID");
    const auto stamp=env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    demand(!stamp.empty() && stamp.find_first_not_of("0123456789")==std::string::npos,"NOT RUN: preflight epoch missing");
    const double age=std::difftime(std::time(nullptr),0)-std::strtod(stamp.c_str(),nullptr);
    demand(age>=0 && age<=30,"NOT RUN: stale preflight");
}
#endif
}

#ifndef H3_CACHE_OFFLOAD_NO_MAIN
int main(int argc,char** argv) {
    try {
#ifdef H3_CACHE_OFFLOAD_CUDA_TEST
        try { require_gpu_permission(argc,argv); }
        catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
        auto* cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        demand(cpu,"CPU reference unavailable");ggml_backend_free(cpu);
        auto* backend=ggml_backend_cuda_init(0);
        demand(backend,"CUDA backend unavailable");
        {run_matrix(backend,true);mixed_matrix(backend,true);negative_matrix(backend);}ggml_backend_free(backend);
#else
        (void)argc;(void)argv;
        auto* backend=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        demand(backend,"CPU backend unavailable");
        {run_matrix(backend,false);mixed_matrix(backend,false);negative_matrix(backend);}ggml_backend_free(backend);
#endif
        std::cout<<"PASS checks="<<offload_checks<<" real segments="<<executed_segments<<" full output comparisons="<<compared_outputs<<'\n';
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}

#endif // H3_CACHE_OFFLOAD_NO_MAIN
