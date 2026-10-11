// Standalone CPU-only real ggml graph regression. Links ONLY ggml-cpu + ggml-base.
// Deterministic tiny activations/schema fixtures are NOT trained model weights.
#include "dit_models/models/minimax_h3_vdn_reference.hpp"
#include "dit_models/models/minimax_h3_vdn_graph.hpp"
#include "ggml-cpu.h"
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>

namespace G=MiniMaxH3::VDNGraph;
namespace R=MiniMaxH3::VDNReference;
static int assertions=0, graphs=0, nodes=0;
static float max_error=0;
static void check(bool ok,const char* msg) {
    ++assertions;
    if(!ok) throw std::runtime_error(msg);
}
template<class Fn> static void rejects(Fn fn,const char* message) {
    bool failed=false;
    try {fn();} catch(const std::invalid_argument&) {failed=true;}
    check(failed,message);
}
struct Context {
    ggml_context* ctx;
    Context():ctx(ggml_init({64*1024*1024,nullptr,false})) {check(ctx,"ggml context");}
    ~Context(){ggml_free(ctx);}
};
static void put(ggml_tensor* t,int64_t a,int64_t b,int64_t h,int64_t f,float v) {
    auto* dst=static_cast<char*>(t->data)+a*t->nb[0]+b*t->nb[1]+h*t->nb[2]+f*t->nb[3];
    std::memcpy(dst,&v,sizeof(v));
}
static float get(ggml_tensor* t,int64_t a,int64_t b,int64_t h=0,int64_t f=0) {
    float v;
    auto* src=static_cast<const char*>(t->data)+a*t->nb[0]+b*t->nb[1]+h*t->nb[2]+f*t->nb[3];
    std::memcpy(&v,src,sizeof(v)); return v;
}
static void close(float got,float expected,const char* message) {
    check(std::isfinite(got),"graph result nonfinite");
    float err=std::abs(got-expected); max_error=std::max(max_error,err);
    ++assertions;
    if(err>3e-5f+3e-5f*std::abs(expected)) {
        std::ostringstream ss;ss<<message<<": got "<<got<<" expected "<<expected<<" error "<<err;
        throw std::runtime_error(ss.str());
    }
}
static void compute(Context& c,const std::vector<ggml_tensor*>& outputs,
                    std::initializer_list<ggml_op> required={}) {
    auto* graph=ggml_new_graph_custom(c.ctx,8192,false);
    for(auto* t:outputs) ggml_build_forward_expand(graph,t);
    std::set<ggml_op> ops;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto op=ggml_graph_node(graph,i)->op;
        check(op!=GGML_OP_CUSTOM,"test graph must not call a custom solve or host reference");
        ops.insert(op);
    }
    for(auto op:required) check(ops.count(op)!=0,"expected native math op missing");
    check(ggml_graph_compute_with_ctx(c.ctx,graph,1)==GGML_STATUS_SUCCESS,"CPU actual graph failed");
    ++graphs; nodes+=ggml_graph_n_nodes(graph);
}
static ggml_tensor* tensor(Context& c,int64_t a,int64_t b=1,int64_t h=1,int64_t f=1) {
    return ggml_new_tensor_4d(c.ctx,GGML_TYPE_F32,a,b,h,f);
}
static ggml_tensor* activation(Context& c,int64_t d,int64_t s,int64_t h,int64_t f,bool strided) {
    auto* t=strided ? tensor(c,d,h,s,f) : tensor(c,d,s,h,f);
    return strided ? ggml_permute(c.ctx,t,0,2,1,3) : t;
}
static void matrix_equal(ggml_tensor* t,const R::Matrix& m,int64_t dk,int64_t dv,int64_t h,int64_t f,const char* msg) {
    for(int64_t i=0;i<dv;++i)for(int64_t j=0;j<dk;++j) close(get(t,j,i,h,f),m[size_t(i*dk+j)],msg);
}

static void frame_and_scan_case(bool strided,bool bridge,int64_t radius,bool text) {
    Context c;
    G::LayoutContract l{3,3,2,2,5};
    auto* k=activation(c,l.key_dim,l.tokens_per_frame,l.heads,l.frames,strided);
    auto* v=activation(c,l.value_dim,l.tokens_per_frame,l.heads,l.frames,strided);
    auto* beta=activation(c,1,l.tokens_per_frame,l.heads,l.frames,strided);
    auto* alpha=tensor(c,l.key_dim,1,l.heads,l.frames);
    auto* q=activation(c,l.key_dim,l.tokens_per_frame,l.heads,l.frames,strided);
    auto* text_state=text ? tensor(c,l.key_dim,l.value_dim,l.heads,1) : nullptr;
    std::vector<std::vector<R::Factor>> rf(size_t(l.heads));
    std::vector<std::vector<R::Matrix>> ra(size_t(l.heads)),rb(size_t(l.heads));
    for(int64_t h=0;h<l.heads;++h) {
        if(text)for(int64_t i=0;i<l.value_dim;++i)for(int64_t j=0;j<l.key_dim;++j)
            put(text_state,j,i,h,0,.2f+.03f*float(i)-.04f*float(j)+.02f*float(h));
        for(int64_t f=0;f<l.frames;++f) {
            std::vector<float> rk,rv,rbet,ralt;
            for(int64_t s=0;s<l.tokens_per_frame;++s) {
                float be=.2f+.1f*float(s)+.02f*float(h+f);
                put(beta,0,s,h,f,be); rbet.push_back(be);
                for(int64_t d=0;d<l.key_dim;++d) {
                    float ke=.13f*float(d+1)-.08f*float(s)+.017f*float(f+2*h);
                    float ve=-.19f*float(d)+.21f*float(s+1)+.025f*float(f-h);
                    put(k,d,s,h,f,ke);put(v,d,s,h,f,ve);
                    put(q,d,s,h,f,.27f*float(d+1)-.09f*float(s+f)+.02f*float(h));
                    rk.push_back(ke);rv.push_back(ve);
                }
            }
            for(int64_t d=0;d<l.key_dim;++d) {
                float al=.61f+.07f*float(d)+.014f*float(f+h);
                put(alpha,d,0,h,f,al);ralt.push_back(al);
            }
            auto stats=R::frame_statistics(rk,rv,rbet,size_t(l.key_dim));
            ra[h].push_back(stats.first);rb[h].push_back(stats.second);
            rf[h].push_back(R::factor(stats.first,stats.second,ralt,size_t(l.tokens_per_frame),R::DeltaRule::SanaScaled));
        }
    }
    auto stats=G::frame_statistics(c.ctx,l,k,v,beta);
    G::GraphRequest request;request.requireweights=false;request.checkpoint.delta_rule="sana_scaled";
    auto factors=G::build_frame_factors(c.ctx,l,k,v,beta,alpha,request);
    auto windows=G::window_bounds(l.frames,radius);
    auto scan=G::outside_window_states(c.ctx,l,factors,windows,bridge,text_state);
    bool full=radius>=l.frames-1;
    check(scan.linear_active==!full,"full-cover branch guard");
    if(full)check(scan.prefix.empty()&&scan.suffix.empty(),"full cover must not build scans");
    std::vector<ggml_tensor*> reads,roots{stats.A,stats.B,factors.transition,factors.injection};
    for(int64_t f=0;f<l.frames;++f) {
        reads.push_back(G::readout(c.ctx,scan.outside[f],G::frame(c.ctx,q,f)));
        roots.push_back(scan.outside[f]);roots.push_back(reads.back());
        if(!full){roots.push_back(scan.prefix[f]);roots.push_back(scan.suffix[f]);}
    }
    compute(c,roots,{GGML_OP_MUL_MAT,GGML_OP_MUL,GGML_OP_DIAG,GGML_OP_SCALE});
    std::vector<R::Window> rw;for(auto w:windows)rw.push_back({size_t(w.lo),size_t(w.hi)});
    for(int64_t h=0;h<l.heads;++h) {
        std::vector<R::Matrix> expected;
        if(!text)expected=R::outside_window_states(rf[h],rw,bridge);
        else {
            R::Matrix start(size_t(l.key_dim*l.value_dim));
            for(int64_t i=0;i<l.value_dim;++i)for(int64_t j=0;j<l.key_dim;++j)
                start[size_t(i*l.key_dim+j)]=.5f*get(text_state,j,i,h,0);
            std::vector<R::Matrix> pre(size_t(l.frames)),suf(size_t(l.frames));auto state=start;
            for(int64_t f=0;f<l.frames;++f){state=R::apply_state(state,rf[h][f]);pre[f]=state;}
            state=start;
            for(int64_t p=l.frames;p>0;--p){state=R::apply_state(state,rf[h][p-1]);suf[p-1]=state;}
            for(int64_t f=0;f<l.frames;++f) {
                auto w=windows[f];auto a=w.lo ? pre[w.lo-1] : start;
                auto b=w.hi+1<l.frames ? suf[w.hi+1] : start;
                for(int64_t j=0;j<l.key_dim;++j) {
                    float da=1,db=1;
                    if(bridge){for(int64_t p=w.lo;p<=f;++p)da*=rf[h][p].alpha[j];for(int64_t p=f;p<=w.hi;++p)db*=rf[h][p].alpha[j];}
                    for(int64_t i=0;i<l.value_dim;++i){size_t ix=size_t(i*l.key_dim+j);a[ix]=a[ix]*da+b[ix]*db;}
                }
                if(full)std::fill(a.begin(),a.end(),0.f);
                expected.push_back(a);
            }
        }
        for(int64_t f=0;f<l.frames;++f) {
            matrix_equal(stats.A,ra[h][f],l.key_dim,l.key_dim,h,f,"A frame statistics");
            matrix_equal(stats.B,rb[h][f],l.key_dim,l.value_dim,h,f,"B frame statistics");
            matrix_equal(factors.transition,rf[h][f].transition,l.key_dim,l.key_dim,h,f,"Sana transition row alpha");
            matrix_equal(factors.injection,rf[h][f].injection,l.key_dim,l.value_dim,h,f,"Sana cB");
            matrix_equal(scan.outside[f],expected[f],l.key_dim,l.value_dim,h,0,"outside window scan");
            std::vector<float> rq;
            for(int64_t s=0;s<l.tokens_per_frame;++s)for(int64_t d=0;d<l.key_dim;++d)rq.push_back(get(q,d,s,h,f));
            auto ro=R::readout(expected[f],rq,size_t(l.key_dim));
            for(int64_t s=0;s<l.tokens_per_frame;++s)for(int64_t d=0;d<l.value_dim;++d)
                close(get(reads[f],d,s,h),ro[size_t(s*l.value_dim+d)],"linear readout");
            if(!full) {
                R::Matrix state(size_t(l.key_dim*l.value_dim),0.f);
                if(text)for(int64_t i=0;i<l.value_dim;++i)for(int64_t j=0;j<l.key_dim;++j)
                    state[size_t(i*l.key_dim+j)]=.5f*get(text_state,j,i,h,0);
                auto initial=state;
                for(int64_t p=0;p<=f;++p)state=R::apply_state(state,rf[h][p]);
                matrix_equal(scan.prefix[f],state,l.key_dim,l.value_dim,h,0,"forward prefix");
                state=initial;for(int64_t p=l.frames;p>f;--p)state=R::apply_state(state,rf[h][p-1]);
                matrix_equal(scan.suffix[f],state,l.key_dim,l.value_dim,h,0,"reverse suffix");
            }
        }
    }
}

static void rectangular_state_case() {
    Context c;G::LayoutContract l{2,3,3,2,2};
    auto* k=tensor(c,2,3,2,2);auto* v=tensor(c,3,3,2,2);auto* beta=tensor(c,1,3,2,2);auto* alpha=tensor(c,2,1,2,2);
    for(int h=0;h<2;++h)for(int f=0;f<2;++f) {
        for(int s=0;s<3;++s) {
            put(beta,0,s,h,f,.3f+.1f*s);
            for(int d=0;d<2;++d)put(k,d,s,h,f,.1f*(d+1+s+h+f));
            for(int d=0;d<3;++d)put(v,d,s,h,f,.2f*(d+1)-.13f*s+.01f*(h+f));
        }
        for(int d=0;d<2;++d)put(alpha,d,0,h,f,.7f+.1f*d);
    }
    auto stats=G::frame_statistics(c.ctx,l,k,v,beta);
    auto fac=G::factor(c.ctx,l,stats,alpha,G::DeltaRule::SanaScaled);
    auto scan=G::outside_window_states(c.ctx,l,fac,G::window_bounds(2,0),false);
    auto* read=G::readout(c.ctx,scan.outside[0],G::frame(c.ctx,k,0));
    compute(c,{stats.A,stats.B,fac.transition,fac.injection,scan.outside[0],scan.outside[1],read},{GGML_OP_MUL_MAT});
    for(int h=0;h<2;++h)for(int f=0;f<2;++f)for(int i=0;i<3;++i)for(int j=0;j<2;++j) {
        float b=0;for(int s=0;s<3;++s)b+=get(v,i,s,h,f)*get(beta,0,s,h,f)*get(k,j,s,h,f);
        close(get(stats.B,j,i,h,f),b,"rectangular B orientation");
        close(get(scan.outside[1-f],j,i,h),b/std::sqrt(3.f),"rectangular outside state");
    }
    for(int h=0;h<2;++h)for(int s=0;s<3;++s)for(int i=0;i<3;++i) {
        float e=0;for(int j=0;j<2;++j)e+=get(scan.outside[0],j,i,h)*get(k,j,s,h,0);
        close(get(read,i,s,h),e,"rectangular state readout");
    }
}

static void softmax_case(int64_t prefix,int64_t suffix,int64_t radius) {
    Context c;G::ProtectedLayout l{prefix,4,2,suffix};
    int64_t total=l.total();auto windows=G::window_bounds(4,radius);
    auto* q=activation(c,3,total,2,1,true);auto* k=activation(c,3,total,2,1,true);auto* v=activation(c,2,total,2,1,true);
    for(int64_t h=0;h<2;++h)for(int64_t t=0;t<total;++t) {
        for(int d=0;d<3;++d) {
            put(q,d,t,h,0,.11f*float(1+d+t)-.035f*float(h));
            put(k,d,t,h,0,.07f*float(1+2*d)-.045f*float(t)+.03f*float(h));
        }
        for(int d=0;d<2;++d)put(v,d,t,h,0,.13f*float(t)-.12f*float(d)+.023f*float(h));
    }
    float scale=1.f/std::sqrt(3.f);
    auto* out=G::local_softmax(c.ctx,l,windows,q,k,v,scale);
    compute(c,{out},{GGML_OP_MUL_MAT,GGML_OP_SOFT_MAX});
    int64_t video_end=prefix+8;
    for(int64_t h=0;h<2;++h)for(int64_t t=0;t<total;++t) {
        std::vector<int64_t> visible;
        for(int64_t s=0;s<total;++s) {
            bool global_q=t<prefix || t>=video_end;
            bool global_k=s<prefix || s>=video_end;
            bool seen=global_q || global_k;
            if(!seen) {auto w=windows[(t-prefix)/2];auto f=(s-prefix)/2;seen=w.lo<=f&&f<=w.hi;}
            if(seen)visible.push_back(s);
        }
        std::vector<float> logits;float max=-std::numeric_limits<float>::infinity();
        for(auto s:visible) {
            float logit=0;for(int d=0;d<3;++d)logit+=get(q,d,t,h)*get(k,d,s,h);
            logit*=scale;logits.push_back(logit);max=std::max(max,logit);
        }
        float sum=0;for(float& e:logits){e=std::exp(e-max);sum+=e;}
        for(int d=0;d<2;++d) {
            float value=0;for(size_t i=0;i<visible.size();++i)value+=logits[i]/sum*get(v,d,visible[i],h);
            close(get(out,d,t,h),value,"protected global/local exact softmax");
        }
    }
}

static void features_epilogue_case() {
    Context c;
    auto* raw=activation(c,3,3,2,2,true);
    auto* norm=tensor(c,3);auto* gate=activation(c,3,3,2,2,true);
    for(int d=0;d<3;++d)put(norm,d,0,0,0,.8f+.1f*d);
    for(int f=0;f<2;++f)for(int h=0;h<2;++h)for(int t=0;t<3;++t)for(int d=0;d<3;++d) {
        // Exact-zero, sub-epsilon and regular signed rows check normalize clamp.
        float x=t==0 ? 0.f : (t==1 ? 1e-9f*float(d+1) : -.9f+.4f*float(d)+.2f*float(h+f));
        put(raw,d,t,h,f,x);put(gate,d,t,h,f,.3f+.09f*float(d)+.02f*float(h));
    }
    auto* q=G::linear_features(c.ctx,raw,true);
    auto* v=G::linear_features(c.ctx,raw,false);
    float eps=1e-5f;
    auto* out=G::norm_gate(c.ctx,v,norm,gate,eps);
    compute(c,{q,v,out},{GGML_OP_L2_NORM,GGML_OP_RMS_NORM,GGML_OP_MUL});
    for(int f=0;f<2;++f)for(int h=0;h<2;++h)for(int t=0;t<3;++t) {
        float silu[3],squares=0;
        for(int d=0;d<3;++d){float x=get(raw,d,t,h,f);silu[d]=x/(1.f+std::exp(-x));squares+=silu[d]*silu[d];}
        for(int d=0;d<3;++d) {
            close(get(v,d,t,h,f),silu[d],"value SiLU feature");
            close(get(q,d,t,h,f),silu[d]/std::max(std::sqrt(squares),1e-6f),"NoPE SiLU L2 feature");
            close(get(out,d,t,h,f),silu[d]/std::sqrt(squares/3.f+eps)*get(norm,d,0)*get(gate,d,t,h,f),"weighted RMS output gate");
        }
    }
    rejects([&]{G::linear_features(c.ctx,raw,true,true);},"cannot omit checkpoint short_conv");
    rejects([&]{G::norm_gate(c.ctx,v,norm,gate,0.f);},"norm epsilon cannot be guessed");
}

static void uneven_window_and_clamp_case() {
    Context c;G::LayoutContract l{1,1,1,1,5};
    auto* tr=tensor(c,1,1,1,5);auto* in=tensor(c,1,1,1,5);auto* al=tensor(c,1,1,1,5);
    std::vector<R::Factor> factors;
    for(int f=0;f<5;++f) {
        float alpha=f==2 ? 1e-20f : .5f+.08f*f;
        put(tr,0,0,0,f,.7f+.03f*f);put(in,0,0,0,f,.2f*(f+1));put(al,0,0,0,f,alpha);
        // Oracle product equivalent to the official bridge's clamp_min(1e-12).
        factors.push_back({{.7f+.03f*f},{.2f*(f+1)},{std::max(alpha,1e-12f)}});
    }
    std::vector<G::Window> windows{{0,1},{0,4},{1,3},{2,4},{3,4}};
    std::vector<R::Window> rw;for(auto w:windows)rw.push_back({size_t(w.lo),size_t(w.hi)});
    auto scan=G::outside_window_states(c.ctx,l,{tr,in,al},windows,true);
    compute(c,scan.outside,{GGML_OP_CLAMP,GGML_OP_LOG});
    auto expected=R::outside_window_states(factors,rw,true);
    for(int f=0;f<5;++f)close(get(scan.outside[f],0,0),expected[f][0],"uneven window/clamped alpha bridge");
    close(get(scan.outside[1],0,0),0.f,"one full-cover query has no video complement");
    check(scan.linear_active,"one full-cover query does not disable other query complements");
}

static G::AdapterSpec spec() {
    G::AdapterSpec s;s.attention_prefix="transformer.blocks.0.attn";s.delta_rule="sana_scaled";
    s.hidden=6;s.heads=2;s.head_dim=3;s.gate_rank=3;s.lora_rank=2;
    s.lora_targets={"base_model.model.transformer.blocks.0.attn.orig.to_q","base_model.model.transformer.blocks.0.attn.orig.to_out.0"};
    s.lora_adapters={"default","turbo"};return s;
}
static G::PublicVdnPayloadAdapter payload(Context& c,const G::AdapterSpec& s) {
    G::PublicVdnPayloadAdapter p;
    auto add=[&](const std::string& key,int64_t a,int64_t b=1){p.branchweights.emplace(key,tensor(c,a,b));};
    auto pre=s.attention_prefix+".linear_attention.";int64_t w=s.heads*s.head_dim;
    add(pre+"alpha.down.weight",s.hidden,s.head_dim);add(pre+"alpha.up.weight",s.head_dim,w);
    add(pre+"alpha.A_log",s.heads);add(pre+"alpha.dt_bias",w);add(pre+"beta_proj.weight",s.hidden,s.heads);
    add(pre+"output_gate.down.weight",s.hidden,s.gate_rank);add(pre+"output_gate.up.weight",s.gate_rank,w);
    add(pre+"output_gate.up.bias",w);add(pre+"norm.weight",s.head_dim);add(s.attention_prefix+".to_out_linear.weight",w,s.hidden);
    if(s.softmax_gate){add(s.attention_prefix+".softmax_gate.up.weight",s.hidden,s.heads);add(s.attention_prefix+".softmax_gate.up.bias",s.heads);}
    for(auto target:s.lora_targets)for(auto a:s.lora_adapters) {
        p.lora.emplace(target+".lora_A."+a+".weight",tensor(c,s.hidden,s.lora_rank));
        p.lora.emplace(target+".lora_B."+a+".weight",tensor(c,s.lora_rank,s.hidden));
    }
    return p; // uncomputed, shape-only fixture; NEVER execute as model weights
}
static void adapter_and_rejection_cases() {
    Context c;auto s=spec();auto p=payload(c,s);p.validate_schema(s);
    check(std::string(p.name()).find("public-named-payload")!=std::string::npos,"public adapter name");
    G::GraphRequest request;request.checkpoint=s;
    rejects([&]{G::validate_request(request);},"requireweights default must reject missing payload");
    request.payload=&p;
    rejects([&]{G::validate_request(request);},"must not treat schema as merged LoRA");
    request.lora_merged=true;G::validate_request(request); // preflight ONLY, no graph/model inference
    for(const std::string rule:{"vdn_solve","vdn_scaled","bogus",""})for(bool need:{false,true}) {
        request.checkpoint.delta_rule=rule;request.requireweights=need;
        rejects([&]{G::validate_request(request);},"original solve/unknown rules cannot fall back to Sana");
    }
    for(auto it=p.branchweights.begin();it!=p.branchweights.end();++it) {
        auto bad=p;bad.branchweights.erase(it->first);
        rejects([&]{bad.validate_schema(s);},"every branch/gate/norm/projection key required");
        bad=p;bad.branchweights[it->first]=tensor(c,it->second->ne[0]+1,it->second->ne[1]);
        rejects([&]{bad.validate_schema(s);},"every named branch tensor shape checked");
    }
    for(auto it=p.lora.begin();it!=p.lora.end();++it) {
        auto bad=p;bad.lora.erase(it->first);
        rejects([&]{bad.validate_schema(s);},"every LoRA pair required");
    }
    auto bad=p;bad.branchweights["unexpected.weight"]=tensor(c,1);
    rejects([&]{bad.validate_schema(s);},"extra branch key rejected");
    bad=p;bad.lora["unexpected.lora_A.default.weight"]=tensor(c,1);
    rejects([&]{bad.validate_schema(s);},"extra LoRA key rejected");
    auto altered=s;altered.short_conv=true;
    rejects([&]{p.validate_schema(altered);},"must not silently discard short_conv");
    altered=s;altered.lora_adapters.push_back("default");
    rejects([&]{p.validate_schema(altered);},"duplicate configured adapters rejected");
    altered=s;altered.lora_targets={"base_model.model.ffn.fc1"};
    rejects([&]{p.validate_schema(altered);},"unsupported LoRA target namespace rejected");
    altered=s;altered.attention_prefix="wrong";
    rejects([&]{p.validate_schema(altered);},"requires exact attn namespace");
    auto no_gate=s;no_gate.softmax_gate=false;
    auto pg=payload(c,no_gate);pg.validate_schema(no_gate);
    G::LayoutContract l{2,2,1,1,1};auto* k=tensor(c,2);auto* v=tensor(c,2);auto* beta=tensor(c,1);auto* alpha=tensor(c,2);
    request.requireweights=false;request.checkpoint.delta_rule="sana_scaled";
    rejects([&]{G::build_frame_factors(c.ctx,l,k,v,tensor(c,2),alpha,request);},"wrong beta layout rejected");
    auto* half=ggml_new_tensor_4d(c.ctx,GGML_TYPE_F16,2,1,1,1);
    rejects([&]{G::frame_statistics(c.ctx,l,half,v,beta);},"native stats must not silently downcast F32 A");
    request.checkpoint.delta_rule="vdn_solve";
    rejects([&]{G::build_frame_factors(c.ctx,l,k,v,beta,alpha,request);},"actual frame builder refuses solve");
    auto* A=tensor(c,2,2);auto* B=tensor(c,2,2);
    rejects([&]{G::factor(c.ctx,l,{A,B},alpha,G::DeltaRule::VdnSolve);},"factor cannot bypass backend refusal");
    auto windows=G::window_bounds(1,0);check(G::full_window_cover(1,windows),"single frame full cover");
    rejects([&]{G::validate_windows(2,{{0,0},{0,2}});},"out-of-range window rejected");
    rejects([&]{G::validate_windows(2,{{1,1},{0,1}});},"window must include query");
    rejects([&]{G::window_bounds(0,0);},"zero frames rejected");
    auto huge=G::window_bounds(3,std::numeric_limits<int64_t>::max());
    check(G::full_window_cover(3,huge),"window radius saturates without overflow");
    rejects([&]{G::ProtectedLayout{std::numeric_limits<int64_t>::max(),1,1,1}.total();},"protected layout length overflow rejected");
    auto chunks=G::window_bounds(12,1,5);
    check(chunks[0].lo==0&&chunks[0].hi==9&&chunks[5].lo==0&&chunks[5].hi==11&&chunks[11].lo==5&&chunks[11].hi==11,
          "official c1 chunk bounds and short last chunk");
}

// Changing a query frame's injection must NOT alter that same frame's
// outside-window state (catch scan-bank off-by-one/local double counting).
static void no_doublecount_case() {
    Context c;G::LayoutContract l{1,1,1,1,3};
    auto* tr=tensor(c,1,1,1,3);auto* in=tensor(c,1,1,1,3);auto* al=tensor(c,1,1,1,3);
    for(int f=0;f<3;++f){put(tr,0,0,0,f,1.f);put(al,0,0,0,f,1.f);put(in,0,0,0,f,float(f+1));}
    auto scan=G::outside_window_states(c.ctx,l,{tr,in,al},G::window_bounds(3,0),true);
    compute(c,scan.outside,{GGML_OP_MUL_MAT,GGML_OP_LOG,GGML_OP_UNARY});
    close(get(scan.outside[0],0,0),5,"outside left end");close(get(scan.outside[1],0,0),4,"outside middle");close(get(scan.outside[2],0,0),3,"outside right end");
    put(in,0,0,0,1,999.f);compute(c,scan.outside);
    close(get(scan.outside[1],0,0),4,"local injection cannot leak into own readout");
    auto full=G::outside_window_states(c.ctx,l,{tr,in,al},G::window_bounds(3,99),true,tensor(c,1));
    compute(c,full.outside,{GGML_OP_FILL});
    check(!full.linear_active&&full.prefix.empty(),"full window disables text-conditioned branch too");
    for(auto* out:full.outside)close(get(out,0,0),0,"full window exact zero");
}
int main() {
    try {
        adapter_and_rejection_cases();
        for(bool strided:{false,true})for(bool bridge:{false,true})for(int64_t radius:{0,1,99})
            frame_and_scan_case(strided,bridge,radius,false);
        for(bool bridge:{false,true})for(int64_t radius:{0,1,99})frame_and_scan_case(true,bridge,radius,true);
        rectangular_state_case();no_doublecount_case();features_epilogue_case();uneven_window_and_clamp_case();
        for(int64_t prefix:{0,2})for(int64_t suffix:{0,1})for(int64_t radius:{0,1,99})softmax_case(prefix,suffix,radius);
        std::cout<<"PASS h3_vdn_graph: "<<graphs<<" CPU actual graphs, "<<nodes<<" native nodes, "
                 <<assertions<<" checks; max_abs_error="<<max_error<<"\n"
                 <<"Math/schema components only; VdnSolve requires cuSOLVER; NOT completed deployment.\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL h3_vdn_graph: "<<e.what()<<"\n";return 1;}
}
