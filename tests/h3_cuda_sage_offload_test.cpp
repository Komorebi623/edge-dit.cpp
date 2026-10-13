// OPT-IN, standalone CUDA-only numerical + serial graph-cut lifecycle test.
// Build with scripts/build_h3_sage_offload_test.sh; run ONLY via the GPU guard.
// Synthetic attention, NOT full H3/Q4 quality, async offload, max-VRAM, or speed.
// Upstream mapping: SageAttention d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:
// core.py::sageattn_qk_int8_pv_fp16_cuda(per_warp, fp32, smooth_k=True),
// csrc/qattn/qk_int_sv_f16_cuda_sm80.cu; SM89 auto instead selects FP8-PV.
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_cuda_sage.hpp"
#include "backend/ggml/cuda/ed_cuda_sage_attn.h"
#include "ggml-cuda.h"
#include <array>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_set>
#if defined(GGML_BACKEND_DL) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#error "Requires directly linked NVIDIA CUDA and CPU backends"
#endif
static_assert(GGML_MAX_NAME == 128, "candidate ggml ABI mismatch");
extern "C" ggml_backend_reg_t h3_sage_cpu_reg() __asm__("ggml_backend_cpu_reg");
namespace {
namespace Cut = sd::ggml_graph_cut;
using namespace MiniMaxH3;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
struct Failure : std::runtime_error {
    int code;
    Failure(const std::string& s, int c=1) : std::runtime_error(s), code(c) {}
};
void require(bool ok, const std::string& s, int code=1) { if (!ok) throw Failure(s,code); }
std::string env(const char* k) { const char* v=std::getenv(k);return v?v:""; }
bool gpu_uuid(const std::string& v) {
    if(v.size()!=40 || v.compare(0,4,"GPU-")!=0)return false;
    for(size_t i=4;i<v.size();++i) {
        if(i==12 || i==17 || i==22 || i==27) { if(v[i]!='-')return false; }
        else if(std::string("0123456789abcdefABCDEF").find(v[i])==std::string::npos)return false;
    }return true;
}
void require_opt_in(int argc,char** argv) {
    require(argc==2 && std::string(argv[1])=="--run" && env("H3_GPU_PARITY_ALLOW")=="1",
            "NOT RUN: use scripts/test_h3_gpu_parity.sh --run --device GPU-... --binary ...",2);
    require(gpu_uuid(env("CUDA_VISIBLE_DEVICES")) && env("CUDA_VISIBLE_DEVICES")==env("H3_GPU_PARITY_PREFLIGHT_DEVICE"),
            "NOT RUN: single full UUID and matching preflight required",2);
    const auto stamp=env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    require(!stamp.empty() && stamp.find_first_not_of("0123456789")==std::string::npos,"NOT RUN: missing preflight",2);
    const double age=std::difftime(std::time(nullptr),0)-std::strtod(stamp.c_str(),nullptr);
    require(std::isfinite(age) && age>=0 && age<=30,"NOT RUN: stale preflight",2);
}
struct Env {
    const char* key; std::string old;bool existed;
    Env(const char* k,const char* v):key(k),old(env(k)),existed(std::getenv(k)!=nullptr) {
        require(setenv(k,v,1)==0,"set test environment");
    }
    ~Env(){if(existed)setenv(key,old.c_str(),1);else unsetenv(key);}
};
size_t comparisons=0,segments=0,dispatches=0,audited_nodes=0,reused_qkv=0;
struct Params {ggml_custom_op_t fun;int n_tasks;void* userdata;};
H3SageRequest request() {
    H3SageRequest r;r.enabled=true;r.flash_attn=true;r.scope=H3SageScope::DiT;
    r.layer_idx=1;r.total_layers=4;return r;
}
void compare(const sd::Tensor<float>& reference,const sd::Tensor<float>& actual,
             const std::string& label,bool approximate=false) {
    require(!reference.empty() && reference.shape()==actual.shape(),label+": shape");
    double sse=0,ref2=0,act2=0,dot=0;float worst=0,peak=0;
    for(int64_t i=0;i<reference.numel();++i) {
        const double a=reference[i],b=actual[i];
        require(std::isfinite(a)&&std::isfinite(b),label+": nonfinite");
        sse+=(a-b)*(a-b);ref2+=a*a;act2+=b*b;dot+=a*b;
        worst=std::max(worst,float(std::fabs(a-b)));peak=std::max(peak,float(std::fabs(a)));
        if(!approximate)require(std::fabs(a-b)<=2e-4+2e-4*std::fabs(a),label+": exact path mismatch");
    }
    const double rel=std::sqrt(sse/std::max(ref2,1e-30));
    const double cosine=dot/std::sqrt(std::max(ref2*act2,1e-30));
    // Predeclared approximation criteria. Fail closed; no automatic relaxation,
    // no fallback overwrite; log the real raw variant's error, including tails.
    if(approximate)require(rel<=0.03 && cosine>=0.999 && worst<=0.025+0.05*peak,label+": INT8 attention accuracy");
    ++comparisons;
    std::cout<<"PARITY "<<label<<" n="<<actual.numel()<<" max_abs="<<worst
             <<" rel_L2="<<rel<<" cosine="<<cosine<<" approximate="<<approximate<<'\n';
}
struct Harness : GGMLRunner {
    int length,heads,batch;bool half,packed,strided;
    std::array<ggml_tensor*,3> source{};
    struct Saved {ggml_backend_buffer_t buffer;void* data;void* extra;std::vector<uint8_t> bytes;};
    std::array<Saved,3> saved{};
    std::array<std::vector<float>,3> values;
    std::unordered_set<void*> prior_qkv;
    size_t empty_param_segments=0;
    Harness(ggml_backend_t cuda,int l,int h,int b,bool f16,bool p,bool stride)
        :GGMLRunner(cuda,true),length(l),heads(h),batch(b),half(f16),packed(p),strided(stride) {
        require(params_backend!=runtime_backend,"must actually stage CPU parameters");
        require(!packed || batch==1,"packed fixture batch");
        const auto type=half?GGML_TYPE_F16:GGML_TYPE_F32;
        const int full=length+(strided?3:0);
        source[0]=ggml_new_tensor_3d(params_ctx,type,128,full,heads*batch);
        source[1]=ggml_new_tensor_3d(params_ctx,type,128,full,heads*batch);
        source[2]=packed?ggml_new_tensor_3d(params_ctx,type,128,full,heads):
            ggml_new_tensor_4d(params_ctx,type,128,heads,full,batch);
        for(int i=0;i<3;++i)ggml_format_name(source[i],"sage_test_cpu_source_%d",i);
        require(alloc_params_buffer(),"allocate CPU sources");
        for(int i=0;i<3;++i) {
            auto* t=source[i];require(t->buffer && ggml_backend_buffer_is_host(t->buffer),"source not CPU");
            saved[i]={t->buffer,t->data,t->extra,{}};
        }
    }
    std::string get_desc() override {return "h3-sage-attention-offload-test";}
    ~Harness(){cleanup();}
    void cleanup() {
        ggml_backend_synchronize(runtime_backend);
        free_compute_buffer();reset_graph_cut_run_cache();free_compute_ctx();
    }
    void restored() {
        require(!params_on_runtime_backend && partial_offload_pairs.empty() && !partial_runtime_params_buffer,"staging survived cleanup");
        for(int i=0;i<3;++i){auto* t=source[i];const auto& s=saved[i];
            require(t->buffer==s.buffer && t->data==s.data && t->extra==s.extra,"source binding not restored");
            require(s.bytes.empty() || std::memcmp(t->data,s.bytes.data(),s.bytes.size())==0,"source mutated");}
    }
    void fill(int seed,int pattern) {
        restored();const int full=length+(strided?3:0),offset=strided?1:0;
        for(int which=0;which<3;++which) {
            auto* t=source[which];std::vector<float> host(size_t(ggml_nelements(t)),std::numeric_limits<float>::quiet_NaN());
            values[which].resize(size_t(128)*length*heads*batch);
            for(int b=0;b<batch;++b)for(int h=0;h<heads;++h)for(int j=0;j<length;++j)for(int d=0;d<128;++d) {
                // Head, batch, request and channel all differ; catches transposes,
                // lost batches, stale buffer reuse and wrong 1/128 logit scaling.
                uint32_t x=uint32_t(1+d+131*j+8191*h+65521*b+977*seed+524287*which);
                x^=x<<13;x^=x>>17;x^=x<<5;
                float v=(float(x%2001)-1000.f)/1000.f;
                if(which<2) {
                    if(pattern==1)v*=0.001f; // near-uniform attention, not a dense fallback
                    else if(pattern==2)v+=(d%23==0?2.5f:0.f); // DC/outlier channels
                } else v+=.05f*float(h+3*b)+.02f*float(seed);
                if(half)v=ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
                values[which][size_t(d)+128*(j+length*(h+heads*b))]=v;
                const size_t index=(which<2 || packed)?size_t(d)+128*(j+offset+full*(h+heads*b)):
                    size_t(d)+128*(h+heads*(j+offset+full*b));
                host[index]=v;
            }
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            if(half)for(size_t j=0;j<host.size();++j){const auto v=ggml_fp32_to_fp16(host[j]);std::memcpy(bytes.data()+2*j,&v,2);}
            else std::memcpy(bytes.data(),host.data(),bytes.size());
            ggml_backend_tensor_set(t,bytes.data(),0,bytes.size());saved[which].bytes=std::move(bytes);
        }
    }
    sd::Tensor<float> reference() const {
        std::vector<int64_t> shape{128*heads,length};if(batch>1)shape.push_back(batch);
        sd::Tensor<float> out(shape);std::vector<double> scores(length);
        for(int b=0;b<batch;++b)for(int h=0;h<heads;++h)for(int i=0;i<length;++i) {
            double maximum=-std::numeric_limits<double>::infinity();
            for(int j=0;j<length;++j){double s=0;for(int d=0;d<128;++d)
                s+=double(values[0][d+128*(i+length*(h+heads*b))])*values[1][d+128*(j+length*(h+heads*b))];
                scores[j]=s/std::sqrt(128.);maximum=std::max(maximum,scores[j]);}
            double sum=0;for(auto& s:scores){s=std::exp(s-maximum);sum+=s;}
            for(int d=0;d<128;++d){double v=0;for(int j=0;j<length;++j)v+=scores[j]*values[2][d+128*(j+length*(h+heads*b))];
                out[d+128*(h+heads*(i+length*b))]=float(v/sum);}
        }return out;
    }
    ggml_tensor* input(int which) {
        auto* t=source[which];auto* c=compute_ctx;
        if(strided) {
            if(which<2 || packed)t=ggml_view_3d(c,t,128,length,heads*batch,t->nb[1],t->nb[2],t->nb[1]);
            else t=ggml_view_4d(c,t,128,heads,length,batch,t->nb[1],t->nb[2],t->nb[3],t->nb[2]);
        }
        return t;
    }
    ggml_cgraph* graph(bool sage,bool cuts,ggml_tensor** raw_out) {
        auto* g=new_graph_custom(512);auto* c=compute_ctx;
        std::array<ggml_tensor*,3> qkv;
        for(int i=0;i<3;++i) {
            // Producer's actual CUDA CONT/CAST reads offset/stride and CPU staged
            // sources. Materialized outputs are cached and rebound as cut leaves.
            auto* t=ggml_cont(c,h3_sage_cont_f32(c,input(i)));
            require(t->nb[2]==t->nb[1]*t->ne[1] && t->nb[3]==t->nb[2]*t->ne[2],
                    "test producer scale requires padded input "+std::to_string(i));
            t=ggml_scale(c,t,1.f);
            if(i==2 && packed)t=ggml_cont(c,ggml_permute(c,t,0,2,1,3));
            qkv[i]=t;if(cuts)Cut::mark_graph_cut(t,"sage-qkv-producer",std::to_string(i));
        }
        ggml_tensor* out=nullptr;*raw_out=nullptr;
        if(sage) {
            auto result=h3_sage_try_build(c,runtime_backend,qkv[0],qkv[1],qkv[2],heads,request());
            require(result.route==H3SageRoute::Sage && result.output,"Sage not built: "+result.reason);out=result.output;
            ggml_build_forward_expand(g,out);
            for(int i=0;i<ggml_graph_n_nodes(g);++i)if(ggml_graph_node(g,i)->op==GGML_OP_CUSTOM){
                require(*raw_out==nullptr,"more than one custom attention");*raw_out=ggml_graph_node(g,i);}
            require(*raw_out,"missing H3SA CUSTOM");
        } else {
            std::cerr<<"DENSE TEST GRAPH L="<<length<<" B="<<batch<<" half="<<half<<" strided="<<strided<<" packed="<<packed<<" cuts="<<cuts<<'\n';
            out=ggml_ext_attention_ext(c,runtime_backend,qkv[0],qkv[1],qkv[2],heads,nullptr,true,false,1.f,false);
        }
        // Cut after the complete attention, never through the atomic CUSTOM op.
        if(cuts)Cut::mark_graph_cut(out,"sage-attention-consumer","out");
        out=ggml_scale(c,ggml_cont(c,out),1.f);ggml_build_forward_expand(g,out);return g;
    }
    void audit(ggml_cgraph* g,ggml_tensor* raw,const Params& tag,bool is_cut,bool poison) {
        auto storage=[&](ggml_tensor* t){
            require(t && t->data,"unallocated executed tensor");auto* buffer=Cut::tensor_buffer(t);
            require(buffer && !ggml_backend_buffer_is_host(buffer),"CPU buffer in CUDA graph");
            auto* type=ggml_backend_buffer_get_type(buffer);
            require(type==ggml_backend_cuda_buffer_type(0) && ggml_backend_buft_get_device(type)==ggml_backend_get_device(runtime_backend),
                    "split/foreign-device buffer");
        };
        for(int i=0;i<ggml_graph_n_nodes(g);++i) {
            auto* t=ggml_graph_node(g,i);++audited_nodes;require(ggml_backend_supports_op(runtime_backend,t),"unsupported CUDA node");
            storage(t);for(auto* s:t->src)if(s)storage(s);
            if(t->op==GGML_OP_CUSTOM) {
                require(t==raw,"unexpected custom/fallback op");Params actual{};std::memcpy(&actual,t->op_params,sizeof(actual));
                require(actual.fun==tag.fun && actual.n_tasks==tag.n_tasks && actual.userdata==tag.userdata,"tag changed across graph cuts");
                require(h3_sage_params_valid(edgedit::ggml_ext::sage_attn_params_from_userdata(actual.userdata)),"lost H3SA tag");
                if(poison) {
                    ++dispatches;
                    for(int j=0;j<3;++j) {
                        require(t->src[j]->type==GGML_TYPE_F32,"mixed dtype bypassed normalization");
                        if(is_cut && j<2) {
                            require(t->src[j]->op==GGML_OP_NONE,"Q/K were recomputed instead of bound cut leaves");
                            if(prior_qkv.count(t->src[j]->data))++reused_qkv;
                            prior_qkv.insert(t->src[j]->data);
                        }
                    }
                    std::vector<float> nan(size_t(ggml_nelements(t)),std::numeric_limits<float>::quiet_NaN());
                    ggml_backend_tensor_set(t,nan.data(),0,nan.size()*sizeof(float));
                }
            }
        }
    }
    sd::Tensor<float> evaluate(bool sage,bool cuts,int fail=-1) {
        cleanup();restored();struct Guard{Harness& h;~Guard(){h.cleanup();}}guard{*this};
        ggml_cgraph* g=nullptr;ggml_tensor* raw=nullptr;
        require(prepare_compute_graph([&]{return graph(sage,cuts,&raw);},&g),"graph construction");
        Params tag{};if(raw)std::memcpy(&tag,raw->op_params,sizeof(tag));
        for(int i=0;i<ggml_graph_n_nodes(g);++i)require(ggml_backend_supports_op(runtime_backend,ggml_graph_node(g,i)),"unsupported before plan");
        auto execute=[&](ggml_cgraph* part,const std::vector<ggml_tensor*>& params,bool last,int index,const std::unordered_set<std::string>* keep){
            auto out=execute_graph<float>(part,1,true,params,true,!last,keep,
                [&](auto* x){audit(x,raw,tag,cuts,false);if(index==fail)throw Failure("injected post-compute failure");return true;},
                [&](auto* x){audit(x,raw,tag,cuts,true);return true;});
            require(out.has_value(),"CUDA graph execution failed (no fallback)");restored();++segments;return *out;
        };
        if(!cuts)return execute(g,{source[0],source[1],source[2]},true,0,nullptr);
        std::unordered_set<const ggml_tensor*> params{source[0],source[1],source[2]};
        const auto plan=Cut::build_plan(runtime_backend,g,params,"h3-sage-test");
        require(plan.available && plan.valid && plan.has_cuts && plan.segments.size()==3,"exact three real producer/attention/output cuts required");
        size_t owner=0;for(const auto& s:plan.segments)for(int idx:s.internal_node_indices)if(ggml_graph_node(g,idx)==raw)++owner;
        require(owner==(sage?1u:0u),"H3SA CUSTOM must belong to exactly one CUDA segment");
        sd::Tensor<float> out;
        for(size_t i=0;i<plan.segments.size();++i) {
            const auto& s=plan.segments[i];require(s.comm_ops.empty(),"single device only");
            reset_segment_runtime_tensors(s,g);require(bind_segment_cached_inputs(g,s),"bind real QKV cut cache");
            const bool last=i+1==plan.segments.size();
            if(!last)for(size_t j=0;j<s.output_node_indices.size();++j){auto* t=Cut::output_tensor(g,s,j);
                if(t && Cut::is_graph_cut_tensor(t) && s.future_input_names.count(t->name))cache(t->name,t);}
            ggml_context* context=nullptr;auto* part=Cut::build_segment_graph(g,s,&context);
            std::unique_ptr<ggml_context,decltype(&ggml_free)> owned(context,ggml_free);require(part,"segment build");
            auto selected=Cut::runtime_param_tensors(g,s,"h3-sage-test");if(selected.empty())++empty_param_segments;
            out=execute(part,selected,last,int(i),&s.future_input_names);
        }return out;
    }
};
void negative_policy(ggml_backend_t cuda) {
    auto* ctx=ggml_init({4*1024*1024,nullptr,true});require(ctx,"negative context");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> own(ctx,ggml_free);
    auto* q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,7,2);
    auto* k=ggml_dup_tensor(ctx,q);auto* v=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,2,7);
    auto r=request();
    auto check=[&](H3SageRequest x,H3SageRoute route,const ggml_tensor* mask=nullptr){
        require(h3_sage_try_build(ctx,cuda,q,k,v,2,x,mask).route==route,"negative route");};
    r.layer_idx=0;check(r,H3SageRoute::Dense);r.layer_idx=3;check(r,H3SageRoute::Dense);
    r.scope=H3SageScope::TokenRefiner;check(r,H3SageRoute::Dense);
    r=request();r.flash_attn=false;check(r,H3SageRoute::Reject);
    r=request();auto* mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,7,7);check(r,H3SageRoute::Reject,mask);
    k->ne[1]=6;check(r,H3SageRoute::Reject);k->ne[1]=7;check(r,H3SageRoute::Sage);
    {Env e("ED_SAGE_FALLBACK","1");check(r,H3SageRoute::Reject);}
    {Env e("ED_SAGE_PV_FP8","1");check(r,H3SageRoute::Reject);}
    check(r,H3SageRoute::Sage); // same process/request recovery
    auto* raw=h3_sage_custom_graph(ctx,q,k,v,2);
    ed_cuda_sage_attn_scratch_sizes sizes;
    require(ed_cuda_sage_attn_scratch_sizes_for(raw,&sizes),"H3 scratch size query");
    require(sizes.q_int8_bytes==128u*7u*2u && sizes.v_f16_bytes==2u*sizes.q_int8_bytes &&
            sizes.qm_bytes==1 && sizes.qk_bias_bytes==1 && sizes.v_fp8_bytes==1 && sizes.v_scale_bytes==1 &&
            sizes.uniform_flag_bytes==1 && sizes.head_entropy_bytes==1 && sizes.detect_acc_bytes==1,
            "not the H3 smooth-K-only variant / stale linked CUDA sidecar");
    ed_cuda_sage_attn_scratch empty;
    require(!ed_cuda_sage_attn_custom_compute(raw,&empty,nullptr),"unbound tensor accepted by CUDA dispatch");

}
} // namespace
int main(int argc,char** argv) {
    try {
        require_opt_in(argc,argv); // BEFORE any CUDA discovery/registration
        Env gate("ED_SAGE_ATTN","0"),fp8("ED_SAGE_PV_FP8","0"),fallback("ED_SAGE_FALLBACK","0"),
            pool("ED_GRAPH_CUT_CACHE_POOL","1"),compact("ED_GRAPH_CUT_CACHE_COMPACT","0"),
            no_capture("GGML_CUDA_DISABLE_GRAPHS","1"); // count submissions, not graph replays; NOT a perf test
        ggml_backend_register(h3_sage_cpu_reg());auto* reg=ggml_backend_cuda_reg();
        require(reg && ggml_backend_reg_dev_count(reg)==1 && ggml_backend_cuda_get_device_count()==1,"exactly one actual CUDA device",3);
        ggml_backend_register(reg);Backend cuda(ggml_backend_cuda_init(0),ggml_backend_free);
        require(cuda && ggml_backend_is_cuda(cuda.get()),"missing CUDA backend",3);
        const auto dispatch_before=ed_cuda_sage_attn_h3_dispatch_count();
        negative_policy(cuda.get());
        require(ed_cuda_sage_attn_h3_dispatch_count()==dispatch_before,"rejected ops counted as dispatches");
        std::cout<<"TEST-ONLY Sage per-warp INT8-QK/F16-PV/FP32-accum smooth-K no dense fallback; device="<<env("CUDA_VISIBLE_DEVICES")<<'\n';
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
        const auto actual_dispatches=ed_cuda_sage_attn_h3_dispatch_count()-dispatch_before;
        require(actual_dispatches==dispatches,"actual CUDA H3SA submission count differs from audited graphs");
        std::cout<<"PASS comparisons="<<comparisons<<" segments="<<segments<<" H3SA_dispatches="<<dispatches
                 <<" actual_CUDA_H3SA_submissions="<<actual_dispatches
                 <<" audited_nodes="<<audited_nodes<<" reused_QK_cut_addresses="<<reused_qkv
                 <<" empty-param-whole-set-stagings="<<empty<<'\n';
        std::cout<<"NOT a full Q4 model, async lifecycle, max-VRAM, or performance validation.\n";return 0;
    } catch(const Failure& e){std::cerr<<"FAIL/NOT RUN: "<<e.what()<<'\n';return e.code;}
      catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
