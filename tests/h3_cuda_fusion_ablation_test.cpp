// Standalone H3/CUDA full-subgraph ablation. Never changes process environment.
// NOT end-to-end attribution, a model benchmark, or newly earned SwiGLU/QKV/RoPE credit.
// Build against an immutable matching snapshot; run ONLY through GPU parity wrapper.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include <chrono>
#include <cctype>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>

#if !defined(GGML_USE_CUDA) || defined(GGML_BACKEND_DL) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#error "Requires directly linked NVIDIA CUDA; CPU fallback is forbidden"
#endif
static_assert(GGML_MAX_NAME == 128, "GGML ABI mismatch");
namespace {
struct Failure : std::runtime_error { int code; Failure(std::string s, int c=1):std::runtime_error(s),code(c){} };
void need(bool b, const std::string& s, int c=1) { if (!b) throw Failure(s,c); }
std::string env(const char* n) { const char* p=std::getenv(n); return p?p:""; }
void guard(int argc, char** argv) {
    need(argc==2 && std::string(argv[1])=="--run" && env("H3_GPU_PARITY_ALLOW")=="1",
         "NOT RUN: use test_h3_gpu_parity.sh --run --device FULL_UUID --binary THIS_BINARY",2);
    auto u=env("CUDA_VISIBLE_DEVICES");
    need(u.size()==40 && u.substr(0,4)=="GPU-" && u==env("H3_GPU_PARITY_PREFLIGHT_DEVICE"),"single UUID/preflight required",2);
    for(size_t i=4;i<u.size();++i) need((i==12||i==17||i==22||i==27)?u[i]=='-':std::isxdigit(static_cast<unsigned char>(u[i])),"invalid GPU UUID",2);
    auto stamp=env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    need(!stamp.empty() && stamp.find_first_not_of("0123456789")==std::string::npos,"missing preflight epoch",2);
    double age=double(std::time(nullptr))-std::strtod(stamp.c_str(),nullptr);
    need(std::isfinite(age)&&age>=0&&age<=30,"preflight older than 30s",2);
    // Reject contamination instead of setting global flags (other work stays untouched).
    for(auto n:{"ED_H3_FUSED_OPS", "ED_MINIMAX_H3_ROPE_TAIL_VIEW", "ED_MINIMAX_H3_DISABLE_ROPE_SLICE_VIEW",
                "ED_DISABLE_CUDA_ROPE", "ED_DISABLE_CUDA_CONCAT_VEC4", "ED_DISABLE_CUDA_MUL_ADD_FUSION",
                "GGML_CUDA_DISABLE_FUSION", "ED_CUDA_FUSED_MODULATION_DISABLE_VEC4"})
        need(env(n).empty()||env(n)=="0",std::string("confounding environment: ")+n,2);
}
// Equivalent structural portion of ggml_can_fuse for this fixed 2-op graph.
// Avoid ggml-impl.h: it conflicts with the project's logger declaration.
// Actual dispatcher also checks alias ranges; inspect with a profiler for launch proof.
bool native_pair(ggml_cgraph* g,int i) {
    auto m=ggml_graph_node(g,i),a=ggml_graph_node(g,i+1);
    if(m->op!=GGML_OP_MUL||a->op!=GGML_OP_ADD||m->view_src||
       (m->flags&GGML_TENSOR_FLAG_OUTPUT)||!ggml_are_same_shape(m,a)||
       !(m->flags&GGML_TENSOR_FLAG_COMPUTE)||!(a->flags&GGML_TENSOR_FLAG_COMPUTE)||
       (a->src[0]!=m&&a->src[1]!=m)||!ggml_is_contiguous(a))return false;
    int uses=0;
    for(int j=0;j<ggml_graph_n_nodes(g);++j)for(auto src:ggml_graph_node(g,j)->src)if(src==m)++uses;
    return uses==1;
}
constexpr size_t CHUNK=1<<18;
constexpr int REPEATS=20, WARMUP=5;
constexpr double ATOL=2e-6, RTOL=2e-5;
float val(size_t i,int stream,int replay) { return float(int((i*17+size_t(stream)*71+size_t(replay)*113)%1021)-510)/512.f; }
using Ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
using Alloc=std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)>;
using Backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>;
struct Graph {
    ggml_backend_t b;
    Ctx c{ggml_init({16*1024*1024,nullptr,true}),ggml_free};
    using InputBuffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    std::vector<InputBuffer> input_buffers;
    Alloc alloc{nullptr,ggml_gallocr_free};
    ggml_cgraph* g=nullptr;
    std::vector<ggml_tensor*> inputs,outputs;
    std::vector<float> pe;
    int64_t n=0,rot=0,hidden=5376;
    bool custom=false,rope=false,gate=false;
    int native_pairs=0;
    Graph(ggml_backend_t backend):b(backend){ need(bool(c),"context allocation failed"); }
    ggml_tensor* input(int64_t a,int64_t b=1,int64_t d=1,int64_t e=1) {
        auto t=ggml_new_tensor_4d(c.get(),GGML_TYPE_F32,a,b,d,e); ggml_set_input(t); inputs.push_back(t); return t;
    }
    void storage(ggml_tensor* t) {
        while(t->view_src)t=t->view_src;
        need(t->buffer && !ggml_backend_buffer_is_host(t->buffer) &&
             ggml_backend_buffer_get_type(t->buffer)==ggml_backend_cuda_buffer_type(0),"non-local CUDA storage / CPU fallback");
        need(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer))==ggml_backend_get_device(b),"wrong device");
    }
    void finish() {
        g=ggml_new_graph_custom(c.get(),512,false);
        for(auto t:outputs){need(t,"CUSTOM not constructed");ggml_set_output(t);ggml_build_forward_expand(g,t);}
        // Keep input allocations external to gallocr: its legal lifetime reuse
        // would otherwise let output NaN-poisoning overwrite a dead graph input
        // before the next replay, creating a test artifact rather than a kernel bug.
        for(auto* tensor:inputs) {
            auto* type=ggml_backend_cuda_buffer_type(0);
            InputBuffer buffer(ggml_backend_buft_alloc_buffer(type,ggml_backend_buft_get_alloc_size(type,tensor)),ggml_backend_buffer_free);
            need(bool(buffer),"external input allocation failed");
            need(ggml_backend_tensor_alloc(buffer.get(),tensor,ggml_backend_buffer_get_base(buffer.get()))==GGML_STATUS_SUCCESS,
                 "external input tensor binding failed");
            input_buffers.push_back(std::move(buffer));
        }
        alloc.reset(ggml_gallocr_new(ggml_backend_cuda_buffer_type(0)));need(bool(alloc),"allocator missing");
        for(int i=0;i<ggml_graph_n_nodes(g);++i) need(ggml_backend_supports_op(b,ggml_graph_node(g,i)),"unsupported CUDA op before allocation");
        need(ggml_gallocr_alloc_graph(alloc.get(),g),"CUDA graph allocation failed");
        for(int i=0;i<ggml_graph_n_nodes(g);++i){auto t=ggml_graph_node(g,i);storage(t);need(ggml_backend_supports_op(b,t),"unsupported allocated CUDA op");for(auto s:t->src)if(s)storage(s);
            if(i+1<ggml_graph_n_nodes(g) && native_pair(g,i))++native_pairs;
        }
        for(auto t:inputs)storage(t);
        if(!rope)need(custom ? native_pairs==0 : native_pairs==1,"unexpected native MUL+ADD fusion eligibility");
        // Unchanged CUDA dispatcher chooses contiguous/vec4 vs non-contiguous path.
        // This is a source-predicate audit, NOT a runtime kernel-launch counter.
        for(auto t:outputs)if(rope){
            need(t->op==GGML_OP_CONCAT,"partial RoPE concat absent");
            bool cont=ggml_is_contiguous(t->src[0])&&ggml_is_contiguous(t->src[1]);
            bool aligned=(uintptr_t(t->src[0]->data)%16==0)&&(uintptr_t(t->src[1]->data)%16==0)&&(uintptr_t(t->data)%16==0);
            bool vec4=cont&&aligned&&t->src[0]->ne[0]%4==0&&t->ne[0]%4==0;
            need(cont!=custom,"tail layout did not change as requested");
            need(vec4==(!custom && rot%4==0),"concat vec4 predicate mismatch");
            std::cout<<"PATH predicted_concat="<<(cont?(vec4?"contiguous_vec4":"contiguous_scalar"):"non_contiguous")
                     <<" rot="<<rot<<" tail="<<128-rot<<" strides="<<t->src[1]->nb[0]<<','<<t->src[1]->nb[1]<<','<<t->src[1]->nb[2]<<'\n';
        }
        std::cout<<"GRAPH nodes="<<ggml_graph_n_nodes(g)<<" allocator_bytes="<<ggml_gallocr_get_buffer_size(alloc.get(),0)
                 <<" native_mul_add_eligible="<<native_pairs<<" allocation=real_gallocr no_scheduler=1\n";
    }
    void fill(int replay){
        std::vector<float> v(CHUNK);
        for(size_t r=0;r<inputs.size();++r){auto t=inputs[r]; size_t count=ggml_nelements(t);
            if(rope&&r==2){ggml_backend_tensor_set(t,pe.data(),0,pe.size()*sizeof(float));continue;}
            for(size_t off=0;off<count;off+=CHUNK){auto len=std::min(CHUNK,count-off);for(size_t j=0;j<len;++j)v[j]=val(off+j,int(r),replay);ggml_backend_tensor_set(t,v.data(),off*4,len*4);}
        }
        poison();
    }
    void poison(){
        std::vector<float> v(CHUNK,std::numeric_limits<float>::quiet_NaN());
        for(auto t:outputs){size_t count=ggml_nelements(t);for(size_t off=0;off<count;off+=CHUNK)ggml_backend_tensor_set(t,v.data(),off*4,std::min(CHUNK,count-off)*4);}
    }
    double compute(){
        ggml_backend_synchronize(b);
        auto begin=std::chrono::steady_clock::now();
        need(ggml_backend_graph_compute(b,g)==GGML_STATUS_SUCCESS,"CUDA full-graph compute failed");
        ggml_backend_synchronize(b);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
    float oracle(size_t i,size_t out,int replay) const {
        if(!rope){float x=val(i,0,replay),a=val(i%hidden,gate?2:1,replay);
            if(gate)return val(i,1,replay)+x*a;
            return (x+x*val(i%hidden,2,replay))+a;
        }
        size_t d=i%128, token=(i/128)%n,head=i/(128*n);
        size_t off=token*(128*56)+head*128;
        if(d>=size_t(rot))return val(off+d,int(out),replay);
        size_t pair=d%(rot/2),pos=4*(pair+(rot/2)*token);
        float x=val(off+pair,int(out),replay),y=val(off+pair+rot/2,int(out),replay);
        return d<size_t(rot/2)?x*pe[pos]+y*pe[pos+1]:x*pe[pos+2]+y*pe[pos+3];
    }
};
void rope_graph(Graph& t,int64_t n,int64_t rot,bool custom){
    t.rope=true;t.n=n;t.rot=rot;t.custom=custom;
    // Production normalizes Q/K before partial RoPE: both normalization
    // outputs are contiguous [d,heads,N,1], NOT the preceding packed QKV views.
    auto q=t.input(128,56,n);auto k=t.input(128,56,n);
    auto pe=t.input(2,2,rot/2,n);
    t.pe.resize(size_t(4*(rot/2)*n));
    // Same [[cos,-sin],[sin,cos]] layout as H3 build_rope, three axes x16
    // in the real rot96 case. Other rot widths are explicitly synthetic controls.
    for(int64_t token=0;token<n;++token)for(int64_t p=0;p<rot/2;++p){
        float position=float((token/(p<16?1458:p<32?27:1))%(p<16?16:p<32?54:27));
        float angle=position*std::pow(10000.f,-float(p%16)/16.f);
        size_t off=4*(p+(rot/2)*token);t.pe[off]=std::cos(angle);t.pe[off+1]=-std::sin(angle);t.pe[off+2]=std::sin(angle);t.pe[off+3]=std::cos(angle);
    }
    MiniMaxH3::CudaOptimizationOptions options;options.fused_ops=custom;
    for(auto input:{q,k})t.outputs.push_back(MiniMaxH3::apply_partial_rope(t.c.get(),input,pe,&options,t.b));
    t.finish();
}
void mod_graph(Graph& t,int64_t n,bool gate,bool custom){
    t.n=n;t.gate=gate;t.custom=custom;
    auto c=t.c.get();auto x=t.input(t.hidden,n);
    auto a=t.input(t.hidden,gate?n:1);auto scale=t.input(t.hidden);
    ggml_tensor* out;
    if(gate)out=custom?edgedit::ggml_ext::fused_residual_gate_custom(c,a,x,scale):ggml_add(c,a,ggml_mul(c,x,scale));
    else out=custom?edgedit::ggml_ext::fused_modulate_custom(c,x,a,scale):ggml_add(c,ggml_add(c,x,ggml_mul(c,x,scale)),a);
    need(out && out->op==(custom?GGML_OP_CUSTOM:GGML_OP_ADD),"unexpected modulation/gate graph route");
    t.outputs.push_back(out);t.finish();
}
struct Error {double maxabs=0,sumsq=0,refsum=0,worst=0;size_t count=0,bits=0;
    void add(float a,float b){need(std::isfinite(a)&&std::isfinite(b),"non-finite numerical output");double diff=double(a)-b,abs=std::abs(diff);maxabs=std::max(maxabs,abs);sumsq+=diff*diff;refsum+=double(b)*b;++count;uint32_t aa,bb;std::memcpy(&aa,&a,4);std::memcpy(&bb,&b,4);bits+=aa!=bb;worst=std::max(worst,abs/(ATOL+RTOL*std::abs(double(b))));}
    void report(const char* name){std::cout<<"QUALITY "<<name<<" count="<<count<<" maxabs="<<maxabs<<" rmse="<<std::sqrt(sumsq/count)<<" relL2="<<std::sqrt(sumsq/std::max(refsum,1e-30))<<" bit_differences="<<bits<<" worst_scaled="<<worst<<'\n';need(worst<=1,"fixed numerical tolerance exceeded");}
};
void compare(Graph& a,Graph& b,int replay){
    Error ab,ao,bo;std::vector<float> av(CHUNK),bv(CHUNK);
    for(size_t o=0;o<a.outputs.size();++o){size_t count=ggml_nelements(a.outputs[o]);need(count==size_t(ggml_nelements(b.outputs[o])),"shape mismatch");
        for(size_t off=0;off<count;off+=CHUNK){size_t len=std::min(CHUNK,count-off);ggml_backend_tensor_get(a.outputs[o],av.data(),off*4,len*4);ggml_backend_tensor_get(b.outputs[o],bv.data(),off*4,len*4);
            for(size_t j=0;j<len;++j){float ref=a.oracle(off+j,o,replay);ab.add(bv[j],av[j]);ao.add(av[j],ref);bo.add(bv[j],ref);}}
    }
    ab.report("candidate_vs_original");ao.report("original_vs_independent_scalar");bo.report("candidate_vs_independent_scalar");
}
double stats(const char* label,const std::vector<double>& v){
    auto sorted=v;std::sort(sorted.begin(),sorted.end());double mean=std::accumulate(v.begin(),v.end(),0.)/v.size(),sq=0;for(auto x:v)sq+=(x-mean)*(x-mean);
    std::cout<<"TIMING "<<label<<" repeats="<<v.size()<<" mean_ms="<<mean<<" median_ms="<<(sorted[9]+sorted[10])/2<<" stddev_ms="<<std::sqrt(sq/v.size())<<" min_ms="<<sorted.front()<<" max_ms="<<sorted.back()<<" samples_ms=";
    for(auto x:v)std::cout<<x<<',';std::cout<<'\n';return mean;
}
void run_pair(Graph& a,Graph& b,const std::string& name){
    std::cout<<"CASE "<<name<<" n="<<a.n<<" dtype=F32 warmup="<<WARMUP<<" repeats="<<REPEATS<<" order=alternating_AB_BA sync_each_graph=1 full_graph=1 transfers_build_alloc_excluded=1\n";
    a.fill(0);b.fill(0);a.compute();b.compute();compare(a,b,0);
    a.fill(1);b.fill(1);
    for(int i=0;i<WARMUP;++i){a.compute();b.compute();}
    std::vector<double> at,bt;
    for(int i=0;i<REPEATS;++i){if(i%2){bt.push_back(b.compute());at.push_back(a.compute());}else{at.push_back(a.compute());bt.push_back(b.compute());}}
    compare(a,b,1);
    // Re-poison after graph/capture reuse to catch a replay that failed to write.
    a.poison();b.poison();a.compute();b.compute();compare(a,b,1);
    double am=stats("original",at),bm=stats("candidate",bt);
    std::cout<<"RESULT "<<name<<" n="<<a.n<<" original_ms="<<am<<" candidate_ms="<<bm<<" candidate_over_original="<<bm/am<<" change_pct="<<(bm/am-1)*100<<" end_to_end_causality=unproven\n";
}
} // namespace
int main(int argc,char** argv){try{
    guard(argc,argv);std::cout<<std::setprecision(9)<<std::unitbuf;
    auto reg=ggml_backend_cuda_reg();need(reg&&ggml_backend_reg_dev_count(reg)==1,"exactly one visible CUDA device required",3);
    Backend b{ggml_backend_cuda_init(0),ggml_backend_free};need(b&&ggml_backend_is_cuda(b.get()),"CUDA initialization failed",3);
    need(ggml_backend_get_device(b.get())==ggml_backend_reg_dev_get(reg,0),"CUDA device mismatch",3);
    MiniMaxH3::Config cfg;
    need(cfg.attention_head_dim==128&&cfg.num_attention_heads==56&&cfg.hidden_size==5376&&cfg.rope_inv_freq_len==16,"H3 default dimensions drifted");
    std::cout<<"H3 FUSION ABLATION UUID="<<env("CUDA_VISIBLE_DEVICES")<<" d=128 heads=56 hidden=5376 axes=3 inv_freq=16 actual_rot=96 actual_tail=32 atol="<<ATOL<<" rtol="<<RTOL<<" GGML_CUDA_DISABLE_GRAPHS="<<env("GGML_CUDA_DISABLE_GRAPHS")<<'\n';
    std::cout<<"SCOPE no_weight_load no_offload no_attention no_end_to_end_claim; QKV/projection/normalization/SwiGLU excluded_common_work; post_norm_QK_layout=contiguous; RoPE_custom_shared; native_MUL_ADD already existed; predicted paths require external profiler for launch proof; modulation_boundary=single_span_without_norm_or_projection\n";
    int cases=0;
    for(auto nr:std::vector<std::pair<int64_t,int64_t>>{{15465,96},{15466,96},{15467,96},{15466,64},{15466,126}}){
        Graph a(b.get()),c(b.get());rope_graph(a,nr.first,nr.second,false);rope_graph(c,nr.first,nr.second,true);
        run_pair(a,c,"rope_qk_rot"+std::to_string(nr.second)+(nr.second==96?"_H3":"_synthetic_control"));++cases;
    }
    for(auto n:{15465,15466,15467})for(bool gate:{false,true}){Graph a(b.get()),c(b.get());mod_graph(a,n,gate,false);mod_graph(c,n,gate,true);run_pair(a,c,gate?"gate_native_mul_add_vs_custom":"modulation_native_mul_add_add_vs_custom");++cases;}
    need(cases==11,"incomplete matrix");std::cout<<"PASS cases=11 A_B_full_graph_measurements=440 quality_scans=99 GPU_complete=1 NOT_FULL_MODEL_RESULT\n";return 0;
}catch(const Failure& e){std::cerr<<"FAIL / NOT RUN: "<<e.what()<<'\n';return e.code;}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
