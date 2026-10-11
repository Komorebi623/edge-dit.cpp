#pragma once
// Native ggml CPU-testable primitives, NOT a deployed VDN-H3 model.
// Source contract: OpenVDN/vdn-minimax-h3@e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b
// scan.py / delta_rule.py / branch.py / hybrid_attention.py. No reference math here.
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace MiniMaxH3 { namespace VDNGraph {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
// ggml axes (fastest first). Already prepared linear Q/K features: SiLU + L2,
// NoPE; V has SiLU only. Passing raw RoPE Q/K violates this caller contract.
// beta must be sigmoid-derived in [0,1], alpha finite in (0,1]; runtime values
// remain graph tensors and are NOT scanned/copied to host for validation.
// K/Q=[dk,S,H,F], V=[dv,S,H,F], beta=[1,S,H,F], alpha=[dk,1,H,F].
// A/transition=[dk,dk,H,F], B/injection/state=[dk,dv,H,F]. In mathematical
// notation state[v,k]; a tensor's first axis is the KEY channel, not VALUE.
struct LayoutContract {
    int64_t key_dim = 0, value_dim = 0, tokens_per_frame = 0, heads = 0, frames = 0;
    void validate() const {
        require(key_dim > 0 && value_dim > 0 && tokens_per_frame > 0 && heads > 0 && frames > 0,
                "VDN layout dimensions must be positive");
        require(frames <= std::numeric_limits<int64_t>::max() / tokens_per_frame,
                "VDN video length overflow");
    }
};
inline void shape(const ggml_tensor* t, int64_t a, int64_t b, int64_t c, int64_t d) {
    require(t && t->type == GGML_TYPE_F32, "VDN native math requires F32 tensors");
    require(t->ne[0] == a && t->ne[1] == b && t->ne[2] == c && t->ne[3] == d,
            "VDN layout contract mismatch");
    require(t->nb[0] == sizeof(float), "VDN channel axis must be contiguous");
}
inline ggml_tensor* constant(ggml_context* ctx, int64_t a, int64_t b,
                             int64_t c, int64_t d, float value) {
    require(ctx != nullptr, "VDN null ggml context");
    return ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a,b,c,d), value);
}
inline ggml_tensor* transpose(ggml_context* ctx, ggml_tensor* a) {
    return ggml_cont(ctx, ggml_permute(ctx, a, 1,0,2,3));
}
inline ggml_tensor* matmul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    // Conventional a @ b with row-major mathematical matrices / ggml col,row axes.
    auto* out = ggml_mul_mat(ctx, transpose(ctx,b), a);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}
struct Statistics { ggml_tensor* A; ggml_tensor* B; };
inline Statistics frame_statistics(ggml_context* ctx, const LayoutContract& l,
                                   ggml_tensor* keys, ggml_tensor* values, ggml_tensor* beta) {
    l.validate();
    shape(keys,l.key_dim,l.tokens_per_frame,l.heads,l.frames);
    shape(values,l.value_dim,l.tokens_per_frame,l.heads,l.frames);
    shape(beta,1,l.tokens_per_frame,l.heads,l.frames);
    // Repack the S contraction axis, including non-contiguous frame/head views.
    auto* kt = transpose(ctx,keys);
    auto* kbt = transpose(ctx,ggml_mul(ctx,keys,beta));
    auto* vbt = transpose(ctx,ggml_mul(ctx,values,beta));
    auto* A = ggml_mul_mat(ctx,kt,kbt);
    auto* B = ggml_mul_mat(ctx,kt,vbt);
    ggml_mul_mat_set_prec(A,GGML_PREC_F32);
    ggml_mul_mat_set_prec(B,GGML_PREC_F32);
    return {A,B};
}

enum class DeltaRule { SanaScaled, VdnSolve, VdnScaled };
inline DeltaRule parse_rule(const std::string& name) {
    if (name == "sana_scaled") return DeltaRule::SanaScaled;
    if (name == "vdn_solve") return DeltaRule::VdnSolve;
    if (name == "vdn_scaled") return DeltaRule::VdnScaled;
    throw std::invalid_argument("VDN unknown checkpoint delta_rule");
}
inline void require_native_backend(DeltaRule rule) {
    // There is no Cholesky / triangular-solve op in this ggml build. No capability
    // bool can turn it on. A real cuSOLVER-backed graph implementation is required.
    if (rule != DeltaRule::SanaScaled)
        throw std::invalid_argument("VDN vdn_solve/vdn_scaled needs matching exact Cholesky + triangular-solve backend (cuSOLVER); native ggml backend unavailable; refusing Sana substitution");
}
struct Factors { ggml_tensor* transition; ggml_tensor* injection; ggml_tensor* alpha; };
inline Factors factor(ggml_context* ctx, const LayoutContract& l, Statistics stats,
                      ggml_tensor* alpha, DeltaRule rule) {
    l.validate();
    require_native_backend(rule); // reject before building anything, even for zero A
    shape(stats.A,l.key_dim,l.key_dim,l.heads,l.frames);
    shape(stats.B,l.key_dim,l.value_dim,l.heads,l.frames);
    shape(alpha,l.key_dim,1,l.heads,l.frames);
    auto* identity = ggml_diag(ctx,constant(ctx,l.key_dim,1,l.heads,l.frames,1.f));
    auto* residual = ggml_sub(ctx,identity,ggml_scale(ctx,stats.A,1.f/float(l.tokens_per_frame)));
    // alpha scales ROWS of transition, while bridge scales COLUMNS of state.
    auto* transition = ggml_mul(ctx,residual,transpose(ctx,alpha));
    auto* injection = ggml_scale(ctx,stats.B,1.f/std::sqrt(float(l.tokens_per_frame)));
    return {transition,injection,alpha};
}
inline ggml_tensor* frame(ggml_context* ctx, ggml_tensor* a, int64_t f) {
    require(a && f >= 0 && f < a->ne[3], "VDN frame index out of range");
    return ggml_view_4d(ctx,a,a->ne[0],a->ne[1],a->ne[2],1,a->nb[1],a->nb[2],a->nb[3],size_t(f)*a->nb[3]);
}
inline ggml_tensor* apply(ggml_context* ctx, ggml_tensor* state,
                         ggml_tensor* transition, ggml_tensor* injection) {
    require(state && transition && injection, "VDN null recurrence tensor");
    shape(state,transition->ne[0],injection->ne[1],transition->ne[2],1);
    shape(transition,state->ne[0],state->ne[0],state->ne[2],1);
    shape(injection,state->ne[0],state->ne[1],state->ne[2],1);
    return ggml_add(ctx,matmul(ctx,state,transition),injection);
}
struct Window { int64_t lo, hi; }; // inclusive; already clamped to [0,F-1]
inline void validate_windows(int64_t frames, const std::vector<Window>& windows) {
    require(frames > 0 && int64_t(windows.size()) == frames, "VDN windows length mismatch");
    for (int64_t f=0; f<frames; ++f)
        require(windows[f].lo >= 0 && windows[f].lo <= f && f <= windows[f].hi && windows[f].hi < frames,
                "VDN invalid inclusive window");
}
inline std::vector<Window> window_bounds(int64_t frames, int64_t radius, int64_t chunk=0) {
    require(frames > 0 && radius >= 0 && chunk >= 0, "VDN invalid window configuration");
    // Saturated arithmetic; chunk=5 expresses official c1 chunk-aligned windows.
    std::vector<Window> out;
    for (int64_t f=0; f<frames; ++f) {
        if (!chunk) out.push_back({f-std::min(radius,f),f+std::min(radius,frames-1-f)});
        else {
            int64_t c=f/chunk, last=(frames-1)/chunk;
            int64_t low=c-std::min(radius,c), high=c+std::min(radius,last-c);
            int64_t end=high == last ? frames-1 : (high+1)*chunk-1;
            out.push_back({low*chunk,end});
        }
    }
    return out;
}
inline bool full_window_cover(int64_t frames, const std::vector<Window>& windows) {
    validate_windows(frames,windows);
    return std::all_of(windows.begin(),windows.end(),[frames](Window w){return w.lo==0 && w.hi==frames-1;});
}
struct ScanResult {
    std::vector<ggml_tensor*> prefix, suffix, outside;
    bool linear_active = false;
};
inline ScanResult outside_window_states(ggml_context* ctx, const LayoutContract& l,
                                        Factors factors, const std::vector<Window>& windows,
                                        bool alpha_bridge=true, ggml_tensor* full_text_injection=nullptr) {
    l.validate(); validate_windows(l.frames,windows);
    shape(factors.transition,l.key_dim,l.key_dim,l.heads,l.frames);
    shape(factors.injection,l.key_dim,l.value_dim,l.heads,l.frames);
    shape(factors.alpha,l.key_dim,1,l.heads,l.frames);
    if (full_text_injection) shape(full_text_injection,l.key_dim,l.value_dim,l.heads,1);
    auto* zero=constant(ctx,l.key_dim,l.value_dim,l.heads,1,0.f);
    ScanResult out;
    out.outside.resize(size_t(l.frames),zero);
    // Official hybrid full-cover guard: even text-conditioned branch is disabled.
    if (full_window_cover(l.frames,windows)) return out;
    out.linear_active=true;
    out.prefix.resize(size_t(l.frames)); out.suffix.resize(size_t(l.frames));
    // full_text_injection matches official _text_chunk_state, NOT _text_state
    // (which has already been halved). Apply TEXT_STATE_SCALE exactly once.
    auto* start=full_text_injection ? ggml_scale(ctx,full_text_injection,.5f) : zero;
    auto* state=start;
    for(int64_t f=0; f<l.frames; ++f) {
        state=::MiniMaxH3::VDNGraph::apply(ctx,state,frame(ctx,factors.transition,f),frame(ctx,factors.injection,f));
        out.prefix[f]=state;
    }
    state=start;
    for(int64_t p=l.frames; p>0; --p) {
        int64_t f=p-1;
        state=::MiniMaxH3::VDNGraph::apply(ctx,state,frame(ctx,factors.transition,f),frame(ctx,factors.injection,f));
        out.suffix[f]=state;
    }
    // Clamp matches official log(alpha.clamp_min(1e-12)). Log-prefix differences
    // keep the gather O(F), rather than building O(F*radius) product nodes.
    std::vector<ggml_tensor*> logs(size_t(l.frames)+1);
    logs[0]=constant(ctx,l.key_dim,1,l.heads,1,0.f);
    if(alpha_bridge) for(int64_t f=0; f<l.frames; ++f)
        logs[f+1]=ggml_add(ctx,logs[f],ggml_log(ctx,ggml_clamp(ctx,frame(ctx,factors.alpha,f),1e-12f,1.f)));
    for(int64_t f=0; f<l.frames; ++f) {
        auto w=windows[f];
        auto* left=w.lo ? out.prefix[w.lo-1] : start;
        auto* right=w.hi+1<l.frames ? out.suffix[w.hi+1] : start;
        if(alpha_bridge) {
            left=ggml_mul(ctx,left,ggml_exp(ctx,ggml_sub(ctx,logs[f+1],logs[w.lo])));
            right=ggml_mul(ctx,right,ggml_exp(ctx,ggml_sub(ctx,logs[w.hi+1],logs[f])));
        }
        out.outside[f]=ggml_add(ctx,left,right);
    }
    return out;
}
inline ggml_tensor* readout(ggml_context* ctx, ggml_tensor* state, ggml_tensor* queries) {
    require(state && queries, "VDN null readout tensor");
    shape(state,queries->ne[0],state->ne[1],queries->ne[2],1);
    shape(queries,state->ne[0],queries->ne[1],state->ne[2],1);
    auto* out=ggml_mul_mat(ctx,ggml_cont(ctx,state),ggml_cont(ctx,queries));
    ggml_mul_mat_set_prec(out,GGML_PREC_F32);
    return out; // [dv,S,H,1]
}

// Conv-less features only. Caller supplies PRE-RoPE raw Q/K/V. Released short_conv
// cannot be silently skipped; its spatial+temporal weights require separate ops.
inline ggml_tensor* linear_features(ggml_context* ctx, ggml_tensor* raw,
                                    bool normalize_qk, bool checkpoint_short_conv=false) {
    require(raw && raw->type==GGML_TYPE_F32 && raw->nb[0]==sizeof(float),
            "VDN features require F32 contiguous channels");
    require(!checkpoint_short_conv,"VDN short_conv feature graph not implemented");
    auto* activated=ggml_silu(ctx,ggml_cont(ctx,raw));
    return normalize_qk ? ggml_l2_norm(ctx,activated,1e-6f) : activated;
}
inline ggml_tensor* norm_gate(ggml_context* ctx, ggml_tensor* read,
                              ggml_tensor* norm_weight, ggml_tensor* gate,
                              float checkpoint_rms_epsilon) {
    require(read && gate && std::isfinite(checkpoint_rms_epsilon) && checkpoint_rms_epsilon>0,
            "VDN norm epsilon must be explicit from checkpoint norm contract");
    shape(read,read->ne[0],read->ne[1],read->ne[2],read->ne[3]);
    shape(norm_weight,read->ne[0],1,1,1); // one norm shared across heads
    shape(gate,read->ne[0],read->ne[1],read->ne[2],read->ne[3]);
    return ggml_mul(ctx,ggml_mul(ctx,ggml_rms_norm(ctx,ggml_cont(ctx,read),checkpoint_rms_epsilon),norm_weight),gate);
}

// Native exact local softmax, deliberately no custom kernels. Flattened tensor
// axes [d,total,H,1]. Protected text/audio GLOBAL queries are dense; GLOBAL keys
// remain visible to all video rows. This component supports prefix and suffix.
// Anchor mode is NONE only: anchor rows/columns/both must not use this contract.
struct ProtectedLayout {
    int64_t prefix_tokens=0, frames=0, tokens_per_frame=0, suffix_tokens=0;
    int64_t total() const {
        require(prefix_tokens>=0 && suffix_tokens>=0 && frames>0 && tokens_per_frame>0,
                "VDN invalid protected layout");
        require(prefix_tokens<=std::numeric_limits<int64_t>::max()-suffix_tokens,
                "VDN protected global length overflow");
        require(frames <= (std::numeric_limits<int64_t>::max()-prefix_tokens-suffix_tokens)/tokens_per_frame,
                "VDN protected length overflow");
        return prefix_tokens+frames*tokens_per_frame+suffix_tokens;
    }
};
inline ggml_tensor* rows(ggml_context* ctx, ggml_tensor* a, int64_t begin, int64_t count) {
    require(a && begin>=0 && count>0 && begin<=a->ne[1]-count, "VDN rows outside tensor");
    return ggml_view_4d(ctx,a,a->ne[0],count,a->ne[2],1,a->nb[1],a->nb[2],a->nb[3],size_t(begin)*a->nb[1]);
}
inline ggml_tensor* local_softmax(ggml_context* ctx, const ProtectedLayout& l,
                                 const std::vector<Window>& windows,
                                 ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, float scale) {
    const int64_t total=l.total(); validate_windows(l.frames,windows);
    require(q && k && v && std::isfinite(scale) && scale>0, "VDN invalid softmax inputs/scale");
    shape(q,q->ne[0],total,q->ne[2],1); shape(k,q->ne[0],total,q->ne[2],1);
    shape(v,v->ne[0],total,q->ne[2],1);
    auto attend=[&](ggml_tensor* queries,ggml_tensor* keys,ggml_tensor* values) {
        auto* logits=ggml_mul_mat(ctx,ggml_cont(ctx,keys),ggml_cont(ctx,queries));
        ggml_mul_mat_set_prec(logits,GGML_PREC_F32);
        auto* p=ggml_soft_max(ctx,ggml_scale(ctx,logits,scale));
        auto* out=ggml_mul_mat(ctx,transpose(ctx,values),p);
        ggml_mul_mat_set_prec(out,GGML_PREC_F32);
        return out;
    };
    // Preserve dense behavior on full coverage; no gathering / duplicated keys.
    if(full_window_cover(l.frames,windows)) return attend(q,k,v);
    ggml_tensor* output=nullptr;
    auto append=[&](ggml_tensor* piece) { output=output ? ggml_concat(ctx,output,piece,1) : piece; };
    if(l.prefix_tokens) append(attend(rows(ctx,q,0,l.prefix_tokens),k,v));
    const int64_t video_end=l.prefix_tokens+l.frames*l.tokens_per_frame;
    for(int64_t f=0;f<l.frames;++f) {
        auto w=windows[f];
        int64_t begin=l.prefix_tokens+w.lo*l.tokens_per_frame;
        int64_t count=(w.hi-w.lo+1)*l.tokens_per_frame;
        auto* keys=rows(ctx,k,begin,count); auto* values=rows(ctx,v,begin,count);
        if(l.prefix_tokens) {
            keys=ggml_concat(ctx,rows(ctx,k,0,l.prefix_tokens),keys,1);
            values=ggml_concat(ctx,rows(ctx,v,0,l.prefix_tokens),values,1);
        }
        if(l.suffix_tokens) {
            keys=ggml_concat(ctx,keys,rows(ctx,k,video_end,l.suffix_tokens),1);
            values=ggml_concat(ctx,values,rows(ctx,v,video_end,l.suffix_tokens),1);
        }
        append(attend(rows(ctx,q,l.prefix_tokens+f*l.tokens_per_frame,l.tokens_per_frame),keys,values));
    }
    if(l.suffix_tokens) append(attend(rows(ctx,q,video_end,l.suffix_tokens),k,v));
    return output;
}

// Public named-payload adapter: checks the released branch/gate/norm/projection
// and PEFT LoRA schema only. No file loading, no random initialization, no claim
// that a schema-valid tensor was trained, merged, or verified against a digest.
struct AdapterSpec {
    std::string attention_prefix; // exact checkpoint module path ending in .attn
    std::string delta_rule; // MUST come from ModelSpec, never default to Sana
    int64_t hidden=0, heads=0, head_dim=0, gate_rank=0;
    bool softmax_gate=true;
    bool short_conv=false; // schema unsupported here => reject, never ignore
    std::vector<std::string> lora_targets; // full PEFT module names from config
    std::vector<std::string> lora_adapters; // e.g. default, turbo
    int64_t lora_rank=0;
};
struct PublicVdnPayloadAdapter {
    using NamedTensors=std::map<std::string,ggml_tensor*>;
    NamedTensors branchweights, lora;
    static const char* name() { return "OpenVDN.vdn-minimax-h3.public-named-payload.v1"; }
    static const char* branch_artifact() { return "linear_branch/"; }
    static const char* lora_artifact() { return "adapters/{default,turbo}/"; }
    void validate_schema(const AdapterSpec& s) const {
        require(s.hidden>0 && s.heads>0 && s.head_dim>0 && s.gate_rank>0,
                "VDN adapter invalid model dimensions");
        require(s.heads<=std::numeric_limits<int64_t>::max()/s.head_dim, "VDN adapter width overflow");
        require(s.attention_prefix.size()>=5 && s.attention_prefix.compare(s.attention_prefix.size()-5,5,".attn")==0,
                "VDN adapter requires exact .attn checkpoint prefix");
        parse_rule(s.delta_rule);
        require(!s.short_conv,"VDN short_conv checkpoint schema unsupported; do not discard its weights");
        NamedTensors expected;
        auto check=[&](const std::string& key,int64_t a,int64_t b) {
            auto it=branchweights.find(key);
            require(it!=branchweights.end() && it->second, "VDN missing trained branch/gate/outputnorm tensor");
            auto* t=it->second;
            require((t->type==GGML_TYPE_F32 || t->type==GGML_TYPE_F16 || t->type==GGML_TYPE_BF16) &&
                    t->ne[0]==a && t->ne[1]==b && t->ne[2]==1 && t->ne[3]==1,
                    "VDN public payload tensor shape/type mismatch (ggml reverses PyTorch axes)");
            expected.emplace(key,t);
        };
        const auto p=s.attention_prefix+".linear_attention.";
        const int64_t width=s.heads*s.head_dim;
        check(p+"alpha.down.weight",s.hidden,s.head_dim);
        check(p+"alpha.up.weight",s.head_dim,width);
        check(p+"alpha.A_log",s.heads,1);
        check(p+"alpha.dt_bias",width,1);
        check(p+"beta_proj.weight",s.hidden,s.heads);
        check(p+"output_gate.down.weight",s.hidden,s.gate_rank);
        check(p+"output_gate.up.weight",s.gate_rank,width);
        check(p+"output_gate.up.bias",width,1);
        check(p+"norm.weight",s.head_dim,1);
        check(s.attention_prefix+".to_out_linear.weight",width,s.hidden);
        if(s.softmax_gate) {
            check(s.attention_prefix+".softmax_gate.up.weight",s.hidden,s.heads);
            check(s.attention_prefix+".softmax_gate.up.bias",s.heads,1);
        }
        require(expected.size()==branchweights.size(),"VDN unexpected branch payload keys; refuse ignored weights");
        require(!s.lora_targets.empty() && !s.lora_adapters.empty() && s.lora_rank>0,
                "VDN released payload requires configured LoRA targets/adapters/rank");
        NamedTensors seen;
        for(const auto& target:s.lora_targets) for(const auto& adapter:s.lora_adapters) {
            require(!target.empty() && !adapter.empty(),"VDN empty LoRA target/adapter");
            auto ends=[](const std::string& text,const char* suffix) {
                std::string tail(suffix);
                return text.size()>=tail.size() && text.compare(text.size()-tail.size(),tail.size(),tail)==0;
            };
            require((target.find(".attn.orig.")!=std::string::npos ||
                     (target.find("token_refiner.")!=std::string::npos && target.find(".attn.")!=std::string::npos)) &&
                    (ends(target,".to_q") || ends(target,".to_k") || ends(target,".to_v") || ends(target,".to_out.0")),
                    "VDN LoRA target must name a configured backbone/token-refiner Q/K/V/O projection");
            auto a=lora.find(target+".lora_A."+adapter+".weight");
            auto b=lora.find(target+".lora_B."+adapter+".weight");
            require(a!=lora.end() && b!=lora.end() && a->second && b->second,"VDN missing paired public LoRA tensors");
            auto valid=[](ggml_tensor* t) {return (t->type==GGML_TYPE_F32 || t->type==GGML_TYPE_F16 || t->type==GGML_TYPE_BF16) && t->ne[2]==1 && t->ne[3]==1;};
            require(valid(a->second) && valid(b->second) && a->second->ne[0]==s.hidden && a->second->ne[1]==s.lora_rank &&
                    b->second->ne[0]==s.lora_rank && b->second->ne[1]==s.hidden,
                    "VDN LoRA shape mismatch; only square backbone Q/K/V/O targets supported");
            require(seen.emplace(a->first,a->second).second && seen.emplace(b->first,b->second).second,
                    "VDN duplicate configured LoRA target/adapter");
        }
        require(seen.size()==lora.size(),"VDN unexpected LoRA payload keys");
    }
};
struct GraphRequest {
    bool requireweights=true; // false is explicitly math-test-only, NOT inference
    AdapterSpec checkpoint;
    const PublicVdnPayloadAdapter* payload=nullptr;
    bool lora_merged=false; // caller must actually merge/apply LoRA; schema is not merging
};
inline void validate_request(const GraphRequest& r) {
    DeltaRule rule=parse_rule(r.checkpoint.delta_rule);
    require_native_backend(rule); // original vdn_solve checkpoint ALWAYS rejected here
    if(r.requireweights) {
        require(r.payload,"VDN inference requires trained named payload; random weights forbidden");
        r.payload->validate_schema(r.checkpoint);
        require(r.lora_merged,"VDN payload LoRA not applied; schema validation is not inference");
        // No inference entrypoint exists. This is an eligibility preflight only.
    }
}
inline Factors build_frame_factors(ggml_context* ctx,const LayoutContract& l,
                                   ggml_tensor* k,ggml_tensor* v,ggml_tensor* beta,
                                   ggml_tensor* alpha,const GraphRequest& request) {
    validate_request(request); l.validate();
    if(request.requireweights)
        require(l.heads==request.checkpoint.heads && l.key_dim==request.checkpoint.head_dim &&
                l.value_dim==request.checkpoint.head_dim,"VDN checkpoint/layout mismatch");
    return factor(ctx,l,frame_statistics(ctx,l,k,v,beta),alpha,parse_rule(request.checkpoint.delta_rule));
}
} } // namespace MiniMaxH3::VDNGraph
