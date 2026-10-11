#include "dit_models/models/minimax_h3_cuda_adaln_cache.hpp"
#include "ggml-alloc.h"
#ifdef H3_ADALN_TEST_REAL_RUNNER
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#else
#include "ggml-cpu.h"
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
    ggml_backend_t backend=ggml_backend_cpu_init();
    ggml_context* params=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    CpuFixture() { check(backend,"CPU backend init");ggml_backend_cpu_set_n_threads(backend,1); }
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
    explicit TinyH3Runner(ggml_backend_t backend,bool use_curves=false)
        : RunnerBridge<GGMLRunner>(backend), time(4,6,4), curves(use_curves) {
        time.init(params_ctx,{},"time_embedder");
        for(int i=0;i<3;++i){auto p=std::make_shared<MiniMaxH3::AdaLayerNormModulation>(4,4,i==2?2:6,i==2?1:3,true,false);
            p->apply_silu=!curves;
            p->init(params_ctx,{},i==2?"final_layer.adaln_proj":"blocks."+std::to_string(i)+".adaln_proj");projections.push_back(p);}
        if(curves)table=ggml_new_tensor_2d(params_ctx,GGML_TYPE_F32,4,5);
        check(alloc_params_buffer(),"real H3 runner params allocation");
        for(auto* t=ggml_get_first_tensor(params_ctx);t;t=ggml_get_next_tensor(params_ctx,t)){
            auto v=payload({ggml_nelements(t),1,1,1});ggml_backend_tensor_set(t,v.data(),0,ggml_nbytes(t));}
    }
    std::string get_desc() override {return "H3-AdaLN-CPU-tiny";}
    ggml_cgraph* build(Cache& cache,Cache::StepToken token,const StepPlan& p){
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
        return build_projection_graph(compute_ctx,cache,token,4096,[&](size_t i){return projections[i]->forward(&ctx,emb);});
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
    expect(run_projection_warmup(runner,cache,bind(cache,c),1,[]()->ggml_cgraph*{return nullptr;}),
           Status::InvalidPlan,"real runner null builder safe");
    auto stale=bind(cache,c);cache.reset();
    expect(run_projection_warmup(runner,cache,stale,1,[]()->ggml_cgraph*{return nullptr;}),Status::InvalidPlan,"stale warmup fails closed");
    std::cout << "PASS REAL H3 " << (curves?"curve table/no SiLU":"TimeEmbedder/SiLU")
              << " AdaLayerNormModulation + GGMLRunner all-sigma prefill (CPU tiny)\n";
}
#endif
int main(){try{
    identity_tests();validation_budget_tests();layout_and_staging_tests();import_budget_test();tensor_rejection_tests();graph_tests(false);graph_tests(true);
#ifdef H3_ADALN_TEST_REAL_RUNNER
    real_runner_test(false);real_runner_test(true);
#endif
    std::cout<<"H3 exact AdaLN tests passed ("<<checks<<" checks; CPU only, no weights downloaded)\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
