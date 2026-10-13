#include "dit_models/models/minimax_h3_cuda_adaln_cache.hpp"
#include "ggml-alloc.h"
#ifdef H3_ADALN_TEST_REAL_RUNNER
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
// Bypass discovery wrappers: link the CPU symbols directly, even in a future
// multi-backend test build. No load_all/device enumeration in this regression.
#if defined(__GNUC__) || defined(__clang__)
extern "C" ggml_backend_reg_t h3_adaln_cpu_reg() __asm__("ggml_backend_cpu_reg");
extern "C" ggml_backend_t h3_adaln_cpu_init() __asm__("ggml_backend_cpu_init");
extern "C" void h3_adaln_cpu_threads(ggml_backend_t, int) __asm__("ggml_backend_cpu_set_n_threads");
#else
#error "This CPU regression requires GNU-compatible symbol aliases"
#endif
#else
#include "ggml-cpu.h"
#define h3_adaln_cpu_init ggml_backend_cpu_init
#define h3_adaln_cpu_threads ggml_backend_cpu_set_n_threads
#endif
#include <iostream>
#include <map>
#include <stdexcept>

using namespace MiniMaxH3::ExactAdaLN;
static size_t checks = 0;
static void check(bool ok, const char* text) {
    ++checks;
    if (!ok) throw std::runtime_error(text);
}
static void expect(Status actual, Status expected, const char* text) {
    ++checks;
    if (actual != expected) throw std::runtime_error(std::string(text) + ": " + status_name(actual));
}
static RequestContract contract() {
    RequestContract c;
    c.full_weights_identity = "all-shards-and-config:sha256:synthetic-v1";
    c.load_generation = 1; c.adapter_identity = "none";
    c.cfg_identity = "cond+uncond:guidance=5:prompts=synthetic";
    c.math_identity = "ggml:CPU:f32:tp1:tiny-synthetic";
    c.schedule_identity = "fixed-flow:v1";
    c.video_shift = 12.f; c.audio_shift = 3.f;
    c.timestep_input_dim = 4; c.time_embed_hidden = 6; c.time_embed_dim = 4;
    c.sigmas = {0.75f, 0.5f, 0.f};
    StepPlan p;
    p.sigma = 0.75f; p.timesteps = {0.25f, 0.55f, 0.999f, 1.f};
    p.final_video_row = 0; p.final_audio_row = 1;
    p.spans = {{0,2,Role::Text,-1,0,1}, {2,4,Role::ConditionVideo,0,2,0},
               {4,6,Role::ConditionAudio,0,3,2}, {6,8,Role::TargetAudio,-1,1,2},
               {8,10,Role::TargetVideo,-1,0,0}};
    c.plans.push_back(p);
    p.schedule_index = 1; p.sigma = 0.5f; p.timesteps[0] = 0.5f; p.timesteps[1] = 0.8f;
    c.plans.push_back(p);
    c.projections = h3_projections(4, 2);
    return c;
}
static Cache::StepToken bind(Cache& cache, const RequestContract& c, size_t i = 0) {
    Cache::StepToken t;
    expect(cache.bind_step(c, i, c.plans[i], &t), Status::Hit, "bind exact request");
    return t;
}
static std::vector<float> payload(const Shape& s) {
    std::vector<float> v(static_cast<size_t>(s[0]*s[1]));
    for (size_t i = 0; i < v.size(); ++i) v[i] = (static_cast<int>(i%11)-5)*0.125f;
    return v;
}
static void identity_tests() {
    auto c = contract();
    Cache off;
    expect(off.begin_request(c), Status::Disabled, "off by default");
    Cache cache({true, 1u<<20, 100});
    expect(cache.begin_request(c), Status::Stored, "begin");
    auto t = bind(cache, c);
    Shape shape{}; expect(cache.expected_shape(t,0,&shape), Status::Hit, "shape");
    check(shape == Shape{72,4,1,1}, "entire block width hidden*6*3");
    Shape final{}; cache.expected_shape(t,2,&final);
    check(final == Shape{8,4,1,1}, "entire final width hidden*2");
    auto v = payload(shape); const float* result = nullptr;
    expect(cache.lookup(t,0,shape,&result), Status::Miss, "empty miss");
    expect(cache.insert(t,0,shape,v.data(),v.size()), Status::Stored, "store entire projection");
    t = bind(cache,c);
    expect(cache.lookup(t,0,shape,&result), Status::Hit, "same sigma/layout hit");
    check(std::memcmp(result,v.data(),v.size()*sizeof(float)) == 0, "byte-identical projection");
    expect(cache.insert(t,0,shape,v.data(),v.size()), Status::Duplicate, "immutable first capture");
    expect(cache.lookup(bind(cache,c,1),0,shape,&result), Status::Miss, "different exact step miss");
    using Mutation = std::function<void(RequestContract&)>;
    const std::vector<std::pair<const char*,Mutation>> mutations = {
        {"role",[](auto& a){a.plans[0].spans[1].role=Role::TargetVideo;}},
        {"source",[](auto& a){a.plans[0].spans[1].source_index=2;}},
        {"modality",[](auto& a){a.plans[0].spans[0].modality=2;}},
        {"row",[](auto& a){a.plans[0].spans[1].timestep_row=0;}},
        {"span length",[](auto& a){a.plans[0].spans[0].end=1;}},
        {"row order",[](auto& a){std::swap(a.plans[0].timesteps[0],a.plans[0].timesteps[1]);}},
        {"final row",[](auto& a){a.plans[0].final_audio_row=0;}},
        {"nextafter sigma",[](auto& a){a.sigmas[0]=std::nextafter(a.sigmas[0],1.f);a.plans[0].sigma=a.sigmas[0];}},
        {"nextafter timestep",[](auto& a){a.plans[0].timesteps[0]=std::nextafter(a.plans[0].timesteps[0],1.f);}},
        {"other schedule step",[](auto& a){a.sigmas[1]=0.4f;a.plans[1].sigma=0.4f;}},
        {"schedule id",[](auto& a){a.schedule_identity="fixed-flow:v2";}},
        {"full weight identity",[](auto& a){a.full_weights_identity += ":other-shard";}},
        {"in-place weight reload",[](auto& a){++a.load_generation;}},
        {"adapter identity",[](auto& a){a.adapter_identity="lora-sha256:scale=0.5";}},
        {"adapter update",[](auto& a){++a.adapter_generation;}},
        {"video shift",[](auto& a){a.video_shift=11.f;}},
        {"audio shift",[](auto& a){a.audio_shift=4.f;}},
        {"CFG identity",[](auto& a){a.cfg_identity="guidance=6";}},
        {"CFG role",[](auto& a){a.plans[0].cfg_role=CfgRole::Unconditional;}},
        {"math identity",[](auto& a){a.math_identity="CUDA0:tf32";}},
        {"curves",[](auto& a){a.curve_grid=4;}},
        {"time embedding shape",[](auto& a){a.time_embed_dim=8;}},
        {"projection identity",[](auto& a){a.projections[0].name += ".other";}}
    };
    for (const auto& mutation : mutations) {
        auto other = c; mutation.second(other); Cache::StepToken miss;
        expect(cache.bind_step(other,0,other.plans[0],&miss), Status::Miss, mutation.first);
        check(!miss,"mismatch leaves no valid token");
    }
    auto actual = c.plans[0]; actual.cfg_role=CfgRole::Unconditional; Cache::StepToken miss;
    expect(cache.bind_step(c,0,actual,&miss), Status::Miss,"actual plan mismatch");
    Shape bad=shape; ++bad[0];
    expect(cache.lookup(t,0,bad,&result), Status::ShapeMismatch,"lookup shape mismatch");
    expect(cache.insert(t,1,bad,v.data(),v.size()),Status::ShapeMismatch,"insert shape mismatch");
    expect(cache.insert(t,1,shape,v.data(),v.size()-1),Status::ShapeMismatch,"element count mismatch");
    v[0]=std::numeric_limits<float>::quiet_NaN();
    size_t used=cache.used_bytes();
    expect(cache.insert(t,1,shape,v.data(),v.size()),Status::NonFinite,"NaN payload rejected");
    check(cache.used_bytes()==used,"failed insert consumes no bytes");
    v[0]=std::numeric_limits<float>::infinity();
    expect(cache.insert(t,1,shape,v.data(),v.size()),Status::NonFinite,"Inf payload rejected");
    expect(cache.begin_request(c),Status::Stored,"same request reset");
    expect(cache.lookup(t,0,shape,&result),Status::Stale,"old token invalidated");
    expect(cache.lookup(bind(cache,c),0,shape,&result),Status::Miss,"request reset removes payload");
    c.video_shift=0;
    expect(cache.begin_request(c),Status::InvalidContract,"invalid request rejects and resets");
    std::cout << "PASS identity: same exact hit; 23 identity/layout/schedule misses; reset; shape; finite\n";
}
static void validation_budget_tests() {
    auto c=contract();
    const std::vector<std::function<void(RequestContract&)>> bad = {
        [](auto& a){a.full_weights_identity.clear();}, [](auto& a){a.adapter_identity.clear();},
        [](auto& a){a.video_shift=std::numeric_limits<float>::infinity();},
        [](auto& a){a.audio_shift=std::numeric_limits<float>::quiet_NaN();},
        [](auto& a){a.sigmas[0]=std::numeric_limits<float>::quiet_NaN();},
        [](auto& a){a.plans[0].timesteps[0]=std::numeric_limits<float>::infinity();},
        [](auto& a){a.plans[0].timesteps[0]=1.1f;}, [](auto& a){a.plans[0].spans[0].timestep_row=9;},
        [](auto& a){a.plans[0].spans[0].modality=3;}, [](auto& a){a.plans[0].spans[0].end=0;},
        [](auto& a){a.plans[0].sigma=0.2f;}, [](auto& a){a.curve_grid=1;},
        [](auto& a){a.projections[1].name=a.projections[0].name;},
        [](auto& a){a.projections[0].hidden=INT64_MAX;}
    };
    for (const auto& fn : bad) { Cache cache({true,1u<<20,100}); auto other=c;fn(other);
        expect(cache.begin_request(other),Status::InvalidContract,"invalid contract");check(cache.used_bytes()==0,"no retained allocation"); }
    Cache size_probe({true,1u<<20,100}); size_probe.begin_request(c);
    const size_t metadata=size_probe.used_bytes(); Shape shape{};
    size_probe.expected_shape(bind(size_probe,c),0,&shape); auto v=payload(shape);
    const size_t bytes=v.size()*sizeof(float);
    Cache cache({true,metadata+bytes,100});
    expect(cache.begin_request(c),Status::Stored,"metadata fits"); auto token=bind(cache,c);
    expect(cache.insert(token,0,shape,v.data(),v.size()),Status::Stored,"exact budget fits");
    check(cache.used_bytes()==cache.max_bytes(),"exact budget ceiling");
    expect(cache.insert(token,1,shape,v.data(),v.size()),Status::BudgetExceeded,"over budget rejected");
    check(cache.used_bytes()<=cache.max_bytes(),"retained allocations below cap");
    Cache small({true,metadata-1,100});expect(small.begin_request(c),Status::BudgetExceeded,"metadata budget");
    Cache entry_limit({true,1u<<20,1});expect(entry_limit.begin_request(c),Status::BudgetExceeded,"entry ceiling");
    auto overflow=c; overflow.projections[0].expand=UINT32_MAX; overflow.projections[0].hidden=INT64_MAX;
    Cache overflow_cache({true,1u<<20,100});expect(overflow_cache.begin_request(overflow),Status::InvalidContract,"overflow safe");
    std::cout << "PASS validation/budget: finite plan; missing identity; overflow; exact ceiling; metadata/entry cap\n";
}
static void layout_and_staging_tests() {
    enum class Kind {TEXT,CONDITION_VIDEO,CONDITION_AUDIO,TARGET_AUDIO,TARGET_VIDEO};
    struct Seq {int64_t start,end;Kind kind;int32_t source_index=-1;};
    struct Mod {int64_t start,end,modulation_row;};
    struct Layout {std::vector<float> timesteps;std::vector<Seq> sequence_segments;
        std::vector<Mod> segments;Mod video_segment{},audio_segment{};};
    auto c=contract();auto p=c.plans[0];
    Layout layout;layout.timesteps=p.timesteps;
    layout.video_segment={8,10,0};layout.audio_segment={6,8,1};
    layout.sequence_segments={{0,2,Kind::TEXT},{2,4,Kind::CONDITION_VIDEO,0},
        {4,6,Kind::CONDITION_AUDIO,0},{6,8,Kind::TARGET_AUDIO},{8,10,Kind::TARGET_VIDEO}};
    layout.segments={{0,1,1},{1,2,2},{2,4,6},{4,6,11},{6,8,5},{8,10,0}};
    auto actual=plan_from_layout(0,0.75f,CfgRole::Conditional,layout);
    check(actual.spans.size()==6,"text tag runs retained");
    check(actual.spans[0].role==Role::Text&&actual.spans[1].modality==2,"text-tag modality mapping");
    check(actual.spans[2].source_index==0&&actual.spans[2].timestep_row==2,"reference source/time mapping");
    check(actual.final_audio_row==1&&actual.final_video_row==0,"final timestep row mapping");
    c.plans[0]=actual;Cache probe({true,1u<<20,100});probe.begin_request(c);
    size_t metadata=probe.used_bytes();Shape shape;probe.expected_shape(bind(probe,c),0,&shape);auto v=payload(shape);
    Cache cache({true,metadata+v.size()*4,100});cache.begin_request(c);auto t=bind(cache,c);
    cache.insert(t,0,shape,v.data(),v.size());
    {GraphInputs inputs(cache,3);check(cache.used_bytes()==cache.max_bytes(),"staging descriptor pressure doesn't exceed budget");}
    check(cache.used_bytes()==cache.max_bytes(),"rejected staging descriptor retained no bytes");
    cache.reset();check(cache.used_bytes()==0,"budget reset returns zero");
    // FP32 sign is part of exact identity, even though ordinary float == treats both as equal.
    c.plans[0].timesteps[0]=0.f;probe.begin_request(c);auto different=c;different.plans[0].timesteps[0]=-0.f;
    Cache::StepToken rejected;expect(probe.bind_step(different,0,different.plans[0],&rejected),Status::Miss,"signed zero exact mismatch");
    std::cout<<"PASS layout adapter: text tags/reference roles/final rows; staging budget; signed-zero bits\n";
}
struct TinyContext {
    ggml_context* ggml_ctx=nullptr;
    std::function<void(ggml_tensor*,const void*)> bind_backend_tensor_data;
};
static void import_budget_test() {
    auto c=contract();Cache probe({true,1u<<20,100});probe.begin_request(c);
    const size_t metadata=probe.used_bytes();Shape shape;probe.expected_shape(bind(probe,c),0,&shape);auto v=payload(shape);
    // Room for cache and one staging descriptor, but not the staging payload.
    Cache cache({true,metadata+v.size()*4+sizeof(std::unique_ptr<float[]>),100});
    cache.begin_request(c);auto token=bind(cache,c);cache.insert(token,0,shape,v.data(),v.size());
    auto* ctx=ggml_init({1u<<20,nullptr,true});
    TinyContext tiny{ctx,[](ggml_tensor*,const void*){throw std::runtime_error("budget rejection must not bind");}};
    {GraphInputs inputs(cache,1);ggml_tensor* output=nullptr;
        expect(inputs.import(&tiny,token,0,shape,&output),Status::BudgetExceeded,"staging payload ceiling");
        check(!output&&cache.used_bytes()<=cache.max_bytes(),"budget failed import leaves no graph tensor");}
    ggml_free(ctx);
    std::cout<<"PASS import budget: graph staging payload ceiling fails closed\n";
}
struct CpuFixture {
    ggml_backend_t backend=h3_adaln_cpu_init();
    ggml_context* params=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    CpuFixture() { check(backend,"CPU backend init");h3_adaln_cpu_threads(backend,1); }
    ~CpuFixture() {if(buffer)ggml_backend_buffer_free(buffer);if(params)ggml_free(params);ggml_backend_free(backend);}
};
static ggml_context* context() { return ggml_init({2u*1024u*1024u,nullptr,true}); }
static void graph_tests(bool curves) {
    auto c=contract();if(curves)c.curve_grid=5;
    Cache cache({true,1u<<20,100});expect(cache.begin_request(c),Status::Stored,"graph begin");
    auto token=bind(cache,c);
    CpuFixture f; f.params=context();
    std::vector<ggml_tensor*> weights,biases;
    for(const auto& p:c.projections){auto width=p.hidden*p.expand*p.modalities;
        weights.push_back(ggml_new_tensor_2d(f.params,GGML_TYPE_F32,c.time_embed_dim,width));
        biases.push_back(ggml_new_tensor_1d(f.params,GGML_TYPE_F32,width));}
    auto* table=ggml_new_tensor_2d(f.params,GGML_TYPE_F32,4,5);
    f.buffer=ggml_backend_alloc_ctx_tensors(f.params,f.backend);check(f.buffer,"CPU params allocation");
    for(auto* tensor:weights){auto v=payload({tensor->ne[0],tensor->ne[1],1,1});ggml_backend_tensor_set(tensor,v.data(),0,v.size()*4);}
    for(auto* tensor:biases){auto v=payload({tensor->ne[0],1,1,1});ggml_backend_tensor_set(tensor,v.data(),0,v.size()*4);}
    auto tv=payload({4,5,1,1});ggml_backend_tensor_set(table,tv.data(),0,tv.size()*4);
    auto* ctx=context(); TinyContext tiny{ctx,{}};
    std::map<ggml_tensor*,const void*> bindings;
    tiny.bind_backend_tensor_data=[&](auto* t,const void* v){bindings[t]=v;};
    ggml_tensor* embedding=nullptr;
    std::vector<float> features=payload({4,4,1,1});
    std::vector<int32_t> low(4),high(4);std::vector<float> frac(4);
    if(curves){for(size_t i=0;i<4;++i){float position=c.plans[0].timesteps[i]*4;int j=std::min(static_cast<int>(std::floor(position)),3);low[i]=j;high[i]=j+1;frac[i]=position-j;}
        auto* li=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,4);auto* hi=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,4);auto* fr=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1,4);
        bindings[li]=low.data();bindings[hi]=high.data();bindings[fr]=frac.data();
        auto* lower=ggml_get_rows(ctx,table,li);auto* upper=ggml_get_rows(ctx,table,hi);
        embedding=ggml_add(ctx,lower,ggml_mul(ctx,ggml_sub(ctx,upper,lower),fr));
    }else{embedding=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,4);bindings[embedding]=features.data();}
    std::vector<ggml_tensor*> outputs;
    auto* graph=build_projection_graph(ctx,cache,token,1024,[&](size_t i){
        auto* input=curves?embedding:ggml_silu(ctx,embedding);
        auto* output=ggml_add(ctx,ggml_mul_mat(ctx,weights[i],input),biases[i]);outputs.push_back(output);return output;});
    check(graph,"projection-only graph built");
    check(std::string(outputs[0]->name)=="h3_exact_adaln.block.0","block readback name");
    check(std::string(outputs[2]->name)=="h3_exact_adaln.final","final readback name");
    check(outputs[0]->flags&GGML_TENSOR_FLAG_OUTPUT,"capture output pinned");
    auto* allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(f.backend));
    check(ggml_gallocr_alloc_graph(allocator,graph),"CPU graph allocation");
    for(const auto& b:bindings)ggml_backend_tensor_set(b.first,b.second,0,ggml_nbytes(b.first));
    expect(ggml_backend_graph_compute(f.backend,graph)==GGML_STATUS_SUCCESS?Status::Stored:Status::Miss,Status::Stored,"CPU projection compute");
    std::vector<std::vector<float>> baseline;
    for(auto* tensor:outputs){std::vector<float> v(ggml_nelements(tensor));ggml_backend_tensor_get(tensor,v.data(),0,v.size()*4);baseline.push_back(std::move(v));}
    expect(cache.capture_pending(token,f.backend),Status::Stored,"interception readback");
    check(cache.complete(token),"all block/final projections captured");
    ggml_gallocr_free(allocator);ggml_free(ctx); // CACHE must survive scratch destruction.
    ctx=context();tiny.ggml_ctx=ctx;bindings.clear();
    {
        GraphInputs inputs(cache,c.projections.size());
        auto* hitgraph=ggml_new_graph_custom(ctx,1024,false);int calls=0;
        std::vector<ggml_tensor*> imported;
        for(size_t i=0;i<c.projections.size();++i){Status status;
            auto* t=projection_or_compute(cache,token,i,&tiny,inputs,[&]()->ggml_tensor*{++calls;return nullptr;},&status);
            expect(status,Status::Hit,"graph projection cache hit");check(t&&t->op==GGML_OP_NONE,"hit leaf, no GEMM");
            auto* copied=ggml_dup(ctx,t);ggml_set_output(copied);imported.push_back(copied);ggml_build_forward_expand(hitgraph,copied);}
        check(calls==0,"cache hit skips actual projection callback");
        for(int i=0;i<ggml_graph_n_nodes(hitgraph);++i)check(ggml_graph_node(hitgraph,i)->op!=GGML_OP_MUL_MAT,"no projection matmul on hit");
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(f.backend));
        check(ggml_gallocr_alloc_graph(allocator,hitgraph),"hit graph allocation");
        cache.reset(); // graph-owned copies stay valid even when request is reset.
        check(cache.used_bytes()>0&&cache.used_bytes()<=cache.max_bytes(),"reset retains live staging budget");
        for(const auto& b:bindings)ggml_backend_tensor_set(b.first,b.second,0,ggml_nbytes(b.first));
        check(ggml_backend_graph_compute(f.backend,hitgraph)==GGML_STATUS_SUCCESS,"hit graph compute");
        for(size_t i=0;i<imported.size();++i){std::vector<float> v(baseline[i].size());ggml_backend_tensor_get(imported[i],v.data(),0,v.size()*4);
            check(std::memcmp(v.data(),baseline[i].data(),v.size()*4)==0,"imported projection bit exact");}
        ggml_gallocr_free(allocator);
    }
    check(cache.used_bytes()==0,"staging freed and accounting zero");ggml_free(ctx);
    expect(begin_cuda_request(cache,c,f.backend),Status::BackendRejected,"production CUDA-only gate");
    std::cout << "PASS CPU tiny " << (curves?"curves(no SiLU)":"native(SiLU)") << ": full 2-layer+final GEMM/readback/import; bit exact; zero hit GEMMs; reset-safe staging\n";
}
static void tensor_rejection_tests() {
    auto c=contract();Cache cache({true,1u<<20,100});cache.begin_request(c);auto t=bind(cache,c);
    auto* ctx=context();auto* f16=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,72,4);
    expect(cache.intercept(t,0,f16),Status::UnsupportedType,"F16 fails closed, no conversion");
    auto* bad=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,71,4);
    expect(cache.intercept(t,0,bad),Status::ShapeMismatch,"tensor shape mismatch");
    auto* source=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,72);
    auto* noncontiguous=ggml_transpose(ctx,source);
    expect(cache.intercept(t,0,noncontiguous),Status::ShapeMismatch,"strided tensor rejected");
    auto* out=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,72,4);
    expect(cache.intercept(t,0,out),Status::Miss,"valid tensor intercepted");
    expect(cache.capture_pending(t,nullptr),Status::BackendRejected,"null backend readback rejected");
    cache.reset();expect(cache.capture_pending(t,nullptr),Status::Stale,"stale asynchronous readback rejected");
    ggml_free(ctx);
    std::cout << "PASS tensor rejection: dtype/stride/shape/null backend/stale readback\n";
}
#ifdef H3_ADALN_TEST_REAL_RUNNER
struct TinyH3Runner : RunnerBridge<GGMLRunner> {
    MiniMaxH3::TimeEmbedder time;
    std::vector<std::shared_ptr<MiniMaxH3::AdaLayerNormModulation>> projections;
    sd::Tensor<float> features;
    bool curves;
    ggml_tensor* table=nullptr;
    sd::Tensor<int32_t> lower,upper;
    sd::Tensor<float> fraction;
    struct ParamBinding {
        ggml_tensor* tensor;
        ggml_backend_buffer_t buffer;
        void* data;
        void* extra;
        std::vector<float> values;
    };
    std::vector<ParamBinding> original_params;
    size_t executions = 0, transferred = 0, fail_on = 0;
    enum class Failure { None, BeforeCompute, AfterReadback, ThrowAfterReadback, NonFinite };
    Failure failure = Failure::None;
    ggml_tensor* projection_output = nullptr;
    explicit TinyH3Runner(ggml_backend_t backend,bool use_curves=false,bool transfer=false)
        : RunnerBridge<GGMLRunner>(backend), time(4,6,4), curves(use_curves) {
        // Test-only: distinct CPU handles exercise the REAL partial-offload
        // copy/swap/restore path, with no CUDA backend/device enumeration.
        if (transfer) { params_backend = h3_adaln_cpu_init(); check(params_backend, "host backend"); }
        time.init(params_ctx,{},"time_embedder");
        for(int i=0;i<3;++i){auto p=std::make_shared<MiniMaxH3::AdaLayerNormModulation>(4,4,i==2?2:6,i==2?1:3,true,false);
            p->apply_silu=!curves;
            p->init(params_ctx,{},i==2?"final_layer.adaln_proj":"blocks."+std::to_string(i)+".adaln_proj");projections.push_back(p);}
        if(curves)table=ggml_new_tensor_2d(params_ctx,GGML_TYPE_F32,4,5);
        check(alloc_params_buffer(),"real H3 runner params allocation");
        for(auto* t=ggml_get_first_tensor(params_ctx);t;t=ggml_get_next_tensor(params_ctx,t)){
            auto v=payload({ggml_nelements(t),1,1,1});ggml_backend_tensor_set(t,v.data(),0,ggml_nbytes(t));
            original_params.push_back({t,t->buffer,t->data,t->extra,std::move(v)});}
    }
    std::string get_desc() override {return "H3-AdaLN-CPU-tiny";}
    void check_restored() {
        check(!partial_runtime_params_buffer && !partial_offload_ctx && partial_offload_pairs.empty(),
              "partial parameter staging survived a segment");
        check(!compute_ctx && !compute_allocr && backend_tensor_data_map.empty(),
              "projection graph/bindings survived execution teardown");
        check_host_params_restored();
    }
    void check_host_params_restored() {
        for (const auto& p : original_params) {
            check(p.tensor->buffer == p.buffer && p.tensor->data == p.data && p.tensor->extra == p.extra,
                  "CPU parameter binding was not restored");
            std::vector<float> values(p.values.size());
            ggml_backend_tensor_get(p.tensor,values.data(),0,values.size()*sizeof(float));
            check(values == p.values, "CPU source weights mutated");
        }
    }
    bool exact_execute_graph(ggml_cgraph* graph,int threads,const std::vector<ggml_tensor*>& params,
                             std::function<bool(ggml_cgraph*)> post) {
        ++executions;
        const auto injected = executions == fail_on ? failure : Failure::None;
        auto verify_transfer = [&]() {
            if (params_backend == runtime_backend) return;
            size_t changed = 0;
            for (const auto& p : original_params) {
                const bool selected = std::find(params.begin(),params.end(),p.tensor) != params.end();
                check((p.tensor->data != p.data) == selected && (p.tensor->buffer != p.buffer) == selected,
                      "offload did not copy exactly the reachable parameters");
                if (selected) ++changed;
            }
            check(changed == (curves ? 3u : 6u), "segment contains more than one real projection's weights");
            transferred += changed;
        };
        if (injected == Failure::BeforeCompute) {
            check(offload_partial_params(params), "test pre-compute staging");
            verify_transfer();
            return false; // run_projection_warmup must restore these live swaps
        }
        const bool result = RunnerBridge<GGMLRunner>::exact_execute_graph(graph,threads,params,[&](auto* g) {
            verify_transfer();
            if (injected == Failure::NonFinite) {
                const float bad = std::numeric_limits<float>::quiet_NaN();
                ggml_backend_tensor_set(projection_output,&bad,0,sizeof(bad));
            }
            const bool captured = post(g);
            if (injected == Failure::ThrowAfterReadback) throw std::runtime_error("injected post-readback exception");
            return captured;
        });
        return injected == Failure::AfterReadback ? false : result;
    }
    std::optional<sd::Tensor<float>> consume_cuts(ggml_cgraph* graph, const sd::ggml_graph_cut::Plan& plan) {
        return compute_with_graph_cuts<float>(graph,plan,1,true);
    }
    std::vector<sd::Tensor<float>> layer_features, layer_fractions;
    std::vector<sd::Tensor<int32_t>> layer_lower, layer_upper;
    ggml_cgraph* build_layer(Cache& cache, const RequestContract& contract, size_t projection,
                            const std::vector<size_t>& plans, const std::vector<Cache::StepToken>& tokens) {
        layer_features.clear(); layer_fractions.clear(); layer_lower.clear(); layer_upper.clear();
        auto* merged = new_graph_custom(4096);
        for (auto i : plans) {
            auto* g = build(cache, tokens[i], contract.plans[i], projection, 1);
            ggml_build_forward_expand(merged, ggml_graph_node(g, -1));
            layer_features.push_back(std::move(features)); layer_fractions.push_back(std::move(fraction));
            layer_lower.push_back(std::move(lower)); layer_upper.push_back(std::move(upper));
        }
        return merged;
    }
    ggml_cgraph* parameterless_graph(Cache& cache,Cache::StepToken token,size_t projection) {
        Shape shape;cache.expected_shape(token,projection,&shape);
        return build_projection_graph(compute_ctx,cache,token,64,[&](size_t) {
            return ggml_new_tensor(compute_ctx,GGML_TYPE_F32,GGML_MAX_DIMS,shape.data());
        },projection,1);
    }
    ggml_cgraph* build(Cache& cache,Cache::StepToken token,const StepPlan& p,
                       size_t first=0,size_t count=std::numeric_limits<size_t>::max()){
        features=sd::Tensor<float>({4,static_cast<int64_t>(p.timesteps.size())},
                                 timestep_embedding(p.timesteps,4,10000,true,1.f));
        auto ctx=get_context();ggml_tensor* emb=nullptr;
        if(curves){std::vector<int32_t> lo(p.timesteps.size()),hi(p.timesteps.size());std::vector<float> f(p.timesteps.size());
            for(size_t i=0;i<p.timesteps.size();++i){float position=std::clamp(p.timesteps[i],0.f,1.f)*4;
                int index=std::min(static_cast<int>(std::floor(position)),3);lo[i]=index;hi[i]=index+1;f[i]=position-index;}
            lower=sd::Tensor<int32_t>({static_cast<int64_t>(lo.size())},lo);
            upper=sd::Tensor<int32_t>({static_cast<int64_t>(hi.size())},hi);
            fraction=sd::Tensor<float>({1,static_cast<int64_t>(f.size())},f);
            auto* li=make_input(lower);auto* ui=make_input(upper);auto* fi=make_input(fraction);
            auto* l=ggml_get_rows(compute_ctx,table,li);auto* u=ggml_get_rows(compute_ctx,table,ui);
            emb=ggml_add(compute_ctx,l,ggml_mul(compute_ctx,ggml_sub(compute_ctx,u,l),fi));
        }else{auto* input=make_input(features);emb=time.forward(&ctx,input);}
        return build_projection_graph(compute_ctx,cache,token,4096,[&](size_t i){
            projection_output=projections[i]->forward(&ctx,emb);return projection_output;
        },first,count);
    }
};
static void real_runner_test(bool curves){
    auto c=contract();if(curves)c.curve_grid=5;
    CpuFixture f;Cache cache({true,1u<<20,100});cache.begin_request(c);
    TinyH3Runner runner(f.backend,curves);size_t built=0;
    expect(prefill_request(runner,cache,c,1,[&](size_t i,auto t){++built;return runner.build(cache,t,c.plans[i]);}),Status::Stored,"real runner all-schedule prefill");
    check(built==2,"all sigmas prefilled before denoising");
    for(size_t i=0;i<c.plans.size();++i){auto t=bind(cache,c,i);check(cache.complete(t),"real H3 every layer filled");
        for(size_t j=0;j<3;++j){Shape shape;cache.expected_shape(t,j,&shape);const float* data;
            expect(cache.lookup(t,j,shape,&data),Status::Hit,"real H3 cache projection hit");
            for(size_t k=0;k<static_cast<size_t>(shape[0]*shape[1]);++k)check(std::isfinite(data[k]),"real H3 finite projection");}}
    // Verify a builder failure neither executes a null graph nor inserts stale pointers.
    auto stale=bind(cache,c);
    expect(run_projection_warmup(runner,cache,stale,1,[]()->ggml_cgraph*{return nullptr;}),
           Status::InvalidPlan,"real runner null builder safe");
    check(cache.used_bytes()==0,"failed standalone warmup discards partial request");
    runner.check_restored();
    expect(run_projection_warmup(runner,cache,stale,1,[]()->ggml_cgraph*{return nullptr;}),Status::InvalidPlan,"stale warmup fails closed");
    std::cout << "PASS REAL H3 " << (curves?"curve table/no SiLU":"TimeEmbedder/SiLU")
              << " AdaLayerNormModulation + GGMLRunner all-sigma prefill (CPU tiny)\n";
}

// This consumer runs the existing graph-cut executor, not a mock graph. It is
// deliberately resident CPU: it validates imported leaf lifetime across cuts,
// NOT CUDA planner budgets, asynchronous prefetch, or production offload support.
static void graph_cut_consumer(TinyH3Runner& runner, Cache& cache, const RequestContract& c,
                                size_t projection, bool reset_after_build) {
    const auto token = bind(cache,c);
    Shape shape; expect(cache.expected_shape(token,projection,&shape),Status::Hit,"consumer shape");
    const float* values=nullptr;
    expect(cache.lookup(token,projection,shape,&values),Status::Hit,"consumer projection");
    std::vector<float> expected(values,values+shape[0]*shape[1]);
    for (auto& v : expected) v = v*2.f+v;
    const size_t initial_bytes=cache.used_bytes();
    {
        GraphInputs inputs(cache,1);
        ggml_cgraph* graph=nullptr;
        check(runner.exact_prepare_graph([&]() {
            auto ctx=runner.get_context();
            ctx.bind_backend_tensor_data=[&](auto* t,const void* data){runner.set_backend_tensor_data(t,data);};
            ggml_tensor* imported=nullptr;
            expect(inputs.import(&ctx,token,projection,shape,&imported),Status::Hit,"cut import");
            ggml_set_name(imported,"h3_adaln_test.imported");
            auto* first=ggml_scale(ctx.ggml_ctx,imported,2.f);
            sd::ggml_graph_cut::mark_graph_cut(first,"h3_adaln_test.first","activation");
            auto* output=ggml_add(ctx.ggml_ctx,first,imported); // reuse same leaf after the cut
            auto* g=ggml_new_graph_custom(ctx.ggml_ctx,128,false);
            ggml_build_forward_expand(g,output);
            return g;
        },&graph),"consumer graph preparation");
        const auto plan=sd::ggml_graph_cut::build_plan(runner.exact_runtime_backend(),graph,{},"H3-AdaLN-cut-tiny");
        check(plan.valid && plan.available && plan.segments.size()>1,"consumer must really split into multiple cuts");
        if(reset_after_build) cache.reset(); // imported storage must be independent
        const auto output=runner.consume_cuts(graph,plan);
        check(output.has_value() && output->numel()==static_cast<int64_t>(expected.size()),"cut consumer compute");
        check(std::memcmp(output->data(),expected.data(),expected.size()*sizeof(float))==0,
              "cut consumer differs from cached native projection");
        runner.exact_release_projection_graph();
    }
    check(cache.used_bytes()==(reset_after_build?0:initial_bytes),"consumer staging accounting leak");
}

static void segmented_runner_test(bool curves) {
    auto c=contract(); if(curves)c.curve_grid=5;
    // Include a repeated sigma with a distinct CFG role and timestep row order.
    auto repeat=c.plans[0]; repeat.cfg_role=CfgRole::Unconditional;
    std::swap(repeat.timesteps[0],repeat.timesteps[1]); c.plans.push_back(repeat);
    CpuFixture f;
    TinyH3Runner direct(f.backend,curves), staged(f.backend,curves,true);
    Cache reference({true,1u<<20,100}), cache({true,1u<<20,100});
    expect(reference.begin_request(c),Status::Stored,"reference begin");
    expect(prefill_request(direct,reference,c,1,[&](size_t i,auto t){return direct.build(reference,t,c.plans[i]);}),
           Status::Stored,"resident real H3 projection reference");
    auto build=[&](size_t i,size_t j,auto t){
        staged.check_host_params_restored();
        return staged.build(cache,t,c.plans[i],j,1);
    };
    for(int request=0;request<2;++request) {
        expect(cache.begin_request(c),Status::Stored,"segmented begin");
        const size_t before=staged.executions;
        expect(prefill_request_segmented(staged,cache,c,1,build),Status::Stored,"segmented real H3 prefill");
        check(staged.executions-before==c.plans.size()*c.projections.size(),"one synchronous execution per projection/sigma/CFG");
        staged.check_restored();
        for(size_t i=0;i<c.plans.size();++i) {
            auto a=bind(reference,c,i),b=bind(cache,c,i);
            check(cache.complete(b),"all real projections present");
            for(size_t j=0;j<c.projections.size();++j) {
                Shape shape;cache.expected_shape(b,j,&shape);const float *av=nullptr,*bv=nullptr;
                expect(reference.lookup(a,j,shape,&av),Status::Hit,"resident result");
                expect(cache.lookup(b,j,shape,&bv),Status::Hit,"segmented result");
                check(std::memcmp(av,bv,static_cast<size_t>(shape[0]*shape[1])*sizeof(float))==0,
                      "segmented transferred real H3 projection not bitexact");
            }
        }
        const size_t executions=staged.executions,used=cache.used_bytes(),copies=staged.transferred;
        expect(prefill_request_segmented(staged,cache,c,1,build),Status::Stored,"repeat prefill");
        check(staged.executions==executions && staged.transferred==copies && cache.used_bytes()==used,
              "repeat prefill recomputed/transferred/allocated");
        graph_cut_consumer(direct,cache,c,0,false);
        graph_cut_consumer(direct,cache,c,0,false); // graph destruction/rebuild
        graph_cut_consumer(direct,cache,c,2,true); // final projection + reset-safe copies
    }
    check(staged.transferred>0,"test never actually transferred weights");
    std::cout << "PASS segmented real H3 " << (curves?"curve table":"time embedder")
              << ": actual CPU parameter copy/swap/restore, resident bitexact, repeated sigma/CFG/request, real consumer graph cuts\n";
}

static void segmented_failure_test() {
    auto c=contract(); CpuFixture f; TinyH3Runner runner(f.backend,false,true);
    Cache cache({true,1u<<20,100});
    auto build=[&](size_t i,size_t j,auto t){return runner.build(cache,t,c.plans[i],j,1);};
    for(auto failure : {TinyH3Runner::Failure::BeforeCompute,TinyH3Runner::Failure::AfterReadback,
                       TinyH3Runner::Failure::ThrowAfterReadback,TinyH3Runner::Failure::NonFinite}) {
        expect(cache.begin_request(c),Status::Stored,"failure begin");const auto stale=bind(cache,c);
        runner.failure=failure;runner.fail_on=runner.executions+3; // fail AFTER two good segments
        bool threw=false; Status status=Status::Miss;
        try { status=prefill_request_segmented(runner,cache,c,1,build); }
        catch(const std::runtime_error& e) {
            check(std::string(e.what())=="injected post-readback exception","unexpected exception");threw=true;
        }
        if(failure==TinyH3Runner::Failure::ThrowAfterReadback) check(threw,"exception swallowed");
        else expect(status,failure==TinyH3Runner::Failure::NonFinite?Status::NonFinite:Status::InvalidPlan,"failure propagated");
        check(cache.used_bytes()==0 && !cache.complete(stale),"failure retained earlier/captured segments");
        runner.check_restored();
        expect(cache.capture_pending(stale,f.backend),Status::Stale,"failure pending pointer survived");
        runner.failure=TinyH3Runner::Failure::None;
        expect(cache.begin_request(c),Status::Stored,"failure recovery begin");
        expect(prefill_request_segmented(runner,cache,c,1,build),Status::Stored,"same runner failure recovery");
        runner.check_restored();
    }
    // Throw after intercept but before execution: no pending tensor may outlive its graph.
    cache.begin_request(c); const size_t executions=runner.executions;
    bool threw=false;
    try { prefill_request_segmented(runner,cache,c,1,[&](size_t i,size_t j,auto t)->ggml_cgraph* {
        build(i,j,t); throw std::runtime_error("injected builder exception");
    }); } catch(const std::runtime_error& e) {threw=std::string(e.what())=="injected builder exception";}
    check(threw && cache.used_bytes()==0 && runner.executions==executions,"builder exception cleanup");
    runner.check_restored();
    for(int invalid=0;invalid<3;++invalid) {
        cache.begin_request(c);
        expect(prefill_request_segmented(runner,cache,c,1,[&](size_t i,size_t j,auto t)->ggml_cgraph* {
            if(invalid==0) {build(i,j,t);return nullptr;}
            if(invalid==1) return runner.build(cache,t,c.plans[i],(j+1)%3,1);
            return runner.parameterless_graph(cache,t,j);
        }),Status::InvalidPlan,"null/wrong/parameterless segment rejected before transfer");
        check(cache.used_bytes()==0 && runner.executions==executions,"invalid graph staged weights or retained cache");
        runner.check_restored();
    }
    cache.begin_request(c);
    auto changed=c;changed.schedule_identity+="-changed";
    expect(prefill_request_segmented(runner,cache,changed,1,build),Status::Miss,"schedule mutation fails closed");
    check(cache.used_bytes()==0,"identity mismatch retains request");
    cache.begin_request(c); const auto old=bind(cache,c);cache.begin_request(c);
    const size_t new_bytes=cache.used_bytes();cache.abort(old);
    check(cache.used_bytes()==new_bytes,"late failure aborted a newer request");
    // Force a late payload-budget failure, not just descriptor allocation failure.
    Shape shape;cache.expected_shape(bind(cache,c),0,&shape);
    Cache limited({true,new_bytes+2*static_cast<size_t>(shape[0]*shape[1])*sizeof(float)-1,100});
    expect(limited.begin_request(c),Status::Stored,"limited begin");
    expect(prefill_request_segmented(runner,limited,c,1,[&](size_t i,size_t j,auto t){
        return runner.build(limited,t,c.plans[i],j,1);
    }),Status::InvalidPlan,"late projection budget rejected");
    check(limited.used_bytes()==0,"late budget failure retained earlier segment");runner.check_restored();
    std::cout << "PASS segmented failures: staged/builder/readback/late-return/nonfinite/budget, atomic reset and same-runner recovery\n";
}

static void layer_major_runner_test(bool curves) {
    auto c=contract(); if(curves)c.curve_grid=5;
    CpuFixture f; TinyH3Runner streamed(f.backend,curves,true), resident(f.backend,curves);
    Cache cache({true,1u<<20,100}), reference({true,1u<<20,100});
    cache.begin_request(c); reference.begin_request(c);
    expect(prefill_request(resident,reference,c,1,[&](size_t i,auto t){return resident.build(reference,t,c.plans[i]);}),
           Status::Stored,"layer-major reference");
    auto build=[&](size_t j,const auto& plans,const auto& tokens){return streamed.build_layer(cache,c,j,plans,tokens);};
    expect(prefill_request_layer_major(streamed,cache,c,1,build),Status::Stored,"layer-major streamed request");
    check(streamed.executions==3,"layer-major must execute/transfer each projection once, not per plan");
    check(streamed.transferred==3*(curves?3u:6u),"layer-major transferred repeated timestep weights");
    streamed.check_restored();
    for(size_t i=0;i<c.plans.size();++i)for(size_t j=0;j<c.projections.size();++j) {
        auto a=bind(reference,c,i),b=bind(cache,c,i);Shape shape;cache.expected_shape(b,j,&shape);
        const float *av=nullptr,*bv=nullptr;reference.lookup(a,j,shape,&av);cache.lookup(b,j,shape,&bv);
        check(std::memcmp(av,bv,static_cast<size_t>(shape[0]*shape[1])*sizeof(float))==0,"layer-major math differs");
    }
    const size_t calls=streamed.executions;
    expect(prefill_request_layer_major(streamed,cache,c,1,build),Status::Stored,"layer-major repeat");
    check(streamed.executions==calls,"repeat layer-major executed");
    for(auto failure:{TinyH3Runner::Failure::BeforeCompute,TinyH3Runner::Failure::AfterReadback,
                      TinyH3Runner::Failure::ThrowAfterReadback,TinyH3Runner::Failure::NonFinite}) {
        cache.begin_request(c);streamed.failure=failure;streamed.fail_on=streamed.executions+2;
        bool rejected=false;
        try { rejected=prefill_request_layer_major(streamed,cache,c,1,build)!=Status::Stored; }
        catch(const std::exception&) { rejected=true; }
        check(rejected && cache.used_bytes()==0,"failed layer-major retained earlier layer/plan");
        streamed.check_restored();streamed.failure=TinyH3Runner::Failure::None;
        cache.begin_request(c);
        expect(prefill_request_layer_major(streamed,cache,c,1,build),Status::Stored,"layer-major same-runner recovery");
    }
    std::cout<<"PASS real layer-major once-per-projection transfer, exact math and atomic failures/recovery curve="<<curves<<"\n";
}

#endif
int main(){try{
#ifdef H3_ADALN_TEST_REAL_RUNNER
    ggml_backend_register(h3_adaln_cpu_reg());
#endif
    identity_tests();validation_budget_tests();layout_and_staging_tests();import_budget_test();tensor_rejection_tests();graph_tests(false);graph_tests(true);
#ifdef H3_ADALN_TEST_REAL_RUNNER
    real_runner_test(false);real_runner_test(true);
    segmented_runner_test(false);segmented_runner_test(true);segmented_failure_test();
    layer_major_runner_test(false);layer_major_runner_test(true);
#endif
    std::cout<<"H3 exact AdaLN tests passed ("<<checks<<" checks; CPU only, no weights downloaded)\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
