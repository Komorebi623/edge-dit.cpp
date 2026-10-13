// Standalone (no torch / no libggml): c++ -std=c++17 -O2 -DMINIMAX_H3_SPARSE_REFERENCE_ONLY -Isrc tests/h3_cuda_sparse_test.cpp -o /tmp/h3-sparse-test
// Optional graph tests: compile WITHOUT that macro and link existing CPU ggml.
#include "dit_models/models/minimax_h3_cuda_sparse.hpp"
#include <iostream>
#include <stdexcept>
#ifndef MINIMAX_H3_SPARSE_REFERENCE_ONLY
#include "../third_party/ggml/src/ggml-backend-impl.h" // test-only capability rejection
#endif
#ifndef MINIMAX_H3_SPARSE_REFERENCE_ONLY
// Alias avoids colliding with the shared backend's C++ convenience wrapper.
// Register/init ONLY the directly linked CPU backend: never load GPU plugins.
#ifndef GGML_BACKEND_DL
extern "C" ggml_backend_reg_t h3_test_cpu_reg() asm("ggml_backend_cpu_reg");
#else
static ggml_backend_reg_t h3_test_cpu_module = nullptr;
static ggml_backend_reg_t h3_test_cpu_reg() { return h3_test_cpu_module; }
#endif
#endif
using namespace MiniMaxH3;
namespace {
constexpr int D = 128;
constexpr float KV = 1.0f / 128.0f;
void check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct Input {
    int64_t length, heads;
    std::vector<float> q, k, v;
    Input(int64_t l, int64_t h) : length(l), heads(h), q(size_t(D*l*h)), k(q.size()), v(q.size()) {
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = 0.35f * std::sin(float(i % 7919) * 0.093f);
            k[i] = 0.4f * std::cos(float(i % 8191) * 0.067f);
            v[i] = 0.7f * std::sin(float(i % 7877) * 0.019f) + 0.1f;
        }
    }
    size_t qi(int d, int64_t t, int64_t h) const { return size_t(d + D*(t + length*h)); }
    size_t vi(int d, int64_t t, int64_t h) const { return size_t(d + D*(h + heads*t)); }
};
// Deliberately independent dense oracle; F32 math, NOT flash's KV F16 rounding.
std::vector<float> dense(const Input& x) {
    std::vector<float> output(x.q.size()), weights(size_t(x.length));
    for (int64_t h = 0; h < x.heads; ++h) {
        for (int64_t t = 0; t < x.length; ++t) {
            float max_logit = -INFINITY;
            for (int64_t j = 0; j < x.length; ++j) {
                float dot = 0;
                for (int d = 0; d < D; ++d) dot += x.q[x.qi(d,t,h)] * x.k[x.qi(d,j,h)];
                weights[size_t(j)] = dot / std::sqrt(float(D));
                max_logit = std::max(max_logit, weights[size_t(j)]);
            }
            float sum = 0;
            for (float& w : weights) { w = std::exp(w - max_logit); sum += w; }
            for (int d = 0; d < D; ++d) {
                float value = 0;
                for (int64_t j = 0; j < x.length; ++j) value += weights[size_t(j)] * x.v[x.vi(d,j,h)];
                output[x.vi(d,t,h)] = value / sum;
            }
        }
    }
    return output;
}
float max_error(const std::vector<float>& a, const std::vector<float>& b) {
    check(a.size() == b.size(), "wrong output size");
    float error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite output");
        error = std::max(error, std::fabs(a[i]-b[i]));
    }
    return error;
}
SparseAttentionReference reference(const Input& x, const SparseAttentionConfig& config,
                                    const SparseAttentionLayout& layout) {
    SparseAttentionReference out;
    std::string error;
    check(h3_subblock_attention_reference(x.q.data(), x.k.data(), x.v.data(), x.length,
        x.heads, D, KV, config, layout, &out, &error), error.c_str());
    return out;
}
SparseAttentionLayout mixed_layout() {
    return {{{0,3,SparseTokenKind::Text}, {3,68,SparseTokenKind::TargetVideo},
             {68,72,SparseTokenKind::Audio}, {72,77,SparseTokenKind::Reference},
             {77,141,SparseTokenKind::TargetVideo}, {141,143,SparseTokenKind::Query},
             {143,146,SparseTokenKind::Audio}}};
}
void test_reference() {
    SparseAttentionConfig config;
    check(config.sparse_block_tokens == 64 && config.sparse_keep_ratio == .5f &&
          config.sparse_warmup_steps == 2, "option defaults changed");
    std::string budget_reason;
    check(!h3_subblock_graph_budget({{{0, 9472, SparseTokenKind::TargetVideo}}},
                                  56, 50, 131072, &budget_reason), "production expansion accepted");
    check(budget_reason.find("bounded execution") != std::string::npos, "capacity error lost");
    check(h3_subblock_graph_budget({{{0, 146, SparseTokenKind::TargetVideo}}},
                                 2, 1, 131072), "tiny expansion incorrectly rejected");
    check(!h3_subblock_graph_budget({{{0, std::numeric_limits<int64_t>::max(), SparseTokenKind::TargetVideo}}},
                                  std::numeric_limits<int64_t>::max(), 50, 131072), "overflow budget accepted");
    Input x(146,2);
    auto layout = mixed_layout();
    config.sparse_keep_ratio = 1;
    auto all = reference(x, config, layout);
    const auto baseline = dense(x);
    check(all.video_blocks == 3 && all.topk == 3, "span-local nondivisible block count");
    const float all_error = max_error(all.output, baseline);
    check(all_error < 2e-6f, "all-selected must approximate dense including real tail");
    config.sparse_keep_ratio = .5f;
    auto sparse = reference(x, config, layout);
    check(sparse.topk == 2, "ceil keep ratio budget");
    for (const auto& span : layout.spans) {
        if (span.kind == SparseTokenKind::TargetVideo) continue;
        for (int64_t t = span.begin; t < span.end; ++t)
            for (int64_t h = 0; h < x.heads; ++h)
                for (int d = 0; d < D; ++d)
                    check(std::fabs(sparse.output[x.vi(d,t,h)]-baseline[x.vi(d,t,h)]) < 1e-7f,
                          "text/audio/reference/query output was skipped or sparsified");
    }
    // Per-head route disagrees by construction; selected blocks must stay head-local.
    Input opposing(128,2);
    std::fill(opposing.q.begin(), opposing.q.end(), 1.f);
    for (int64_t h = 0; h < 2; ++h)
        for (int64_t t = 0; t < 128; ++t)
            for (int d = 0; d < D; ++d) {
                opposing.k[opposing.qi(d,t,h)] = ((t < 64) == (h == 0)) ? 1.f : -1.f;
                opposing.v[opposing.vi(d,t,h)] = t < 64 ? 2.f : 7.f;
            }
    SparseAttentionLayout video{{{0,128,SparseTokenKind::TargetVideo}}};
    config.sparse_topk = 1;
    auto per_head = reference(opposing, config, video);
    check(per_head.selected_blocks == std::vector<int32_t>({0,0,1,1}), "router collapsed/averaged heads");
    check(std::fabs(per_head.output[0]-2.f) < 1e-5f && std::fabs(per_head.output[128]-7.f) < 1e-5f,
          "head dimension/output packing wrong");
    // True SubBlock LSE differs from whole-block means (mean of +4/-4 is 0).
    Input lse(128,1);
    std::fill(lse.q.begin(), lse.q.end(), 1.f);
    for (int64_t t = 0; t < 128; ++t)
        for (int d = 0; d < D; ++d)
            lse.k[lse.qi(d,t,0)] = t < 64 ? ((t/16)%2 == 0 ? 4.f : -4.f) : .25f;
    auto subblock = reference(lse, config, video);
    check(subblock.selected_blocks == std::vector<int32_t>({0,0}), "router is plain whole-block pooling, not SubBlock");
    // K-tail has one real, negative-score cell; including three exp(0) padded
    // cells would incorrectly outrank the four low-scoring REAL cells of block0.
    Input tail(65,1);
    std::fill(tail.q.begin(), tail.q.end(), 1.f);
    for (int64_t t = 0; t < 65; ++t)
        for (int d = 0; d < D; ++d) tail.k[tail.qi(d,t,0)] = t < 64 ? -2.f : -3.f;
    auto tail_route = reference(tail, config, {{{0,65,SparseTokenKind::TargetVideo}}});
    check(tail_route.selected_blocks == std::vector<int32_t>({0,0}), "padded subcells contributed exp(0) to LSE");
    // Forced KV includes late audio, reference video and query, even when their
    // tokens are far outside the selected video block. Zero V video isolates it.
    Input protected_x(135,1);
    std::fill(protected_x.q.begin(), protected_x.q.end(), 0);
    std::fill(protected_x.k.begin(), protected_x.k.end(), 0);
    std::fill(protected_x.v.begin(), protected_x.v.end(), 0);
    for (int64_t t = 128; t < 135; ++t)
        for (int d = 0; d < D; ++d) protected_x.v[protected_x.vi(d,t,0)] = float(t-127);
    SparseAttentionLayout protected_layout{{{0,128,SparseTokenKind::TargetVideo},
        {128,131,SparseTokenKind::Audio}, {131,133,SparseTokenKind::Reference}, {133,135,SparseTokenKind::Query}}};
    auto forced = reference(protected_x, config, protected_layout);
    check(std::fabs(forced.output[0]-28.f/71.f) < 1e-6f, "protected KV not forced into sparse video query");
    check(std::fabs(forced.output[protected_x.vi(0,134,0)]-28.f/135.f) < 1e-6f,
          "last nonvideo Q not updated densely");
    // Numerical scale oracle: logits are [1/sqrt(128),0], NOT [1/128,0].
    Input scale_x(2,1);
    std::fill(scale_x.q.begin(), scale_x.q.end(), 0);
    std::fill(scale_x.k.begin(), scale_x.k.end(), 0);
    std::fill(scale_x.v.begin(), scale_x.v.end(), 0);
    scale_x.q[0] = scale_x.k[0] = 1;
    scale_x.v[0] = 1;
    auto scaled = reference(scale_x, config, {{{0,2,SparseTokenKind::TargetVideo}}});
    const float expected = 1.f/(1.f+std::exp(-1.f/std::sqrt(128.f)));
    check(std::fabs(scaled.output[0]-expected) < 1e-7f, "kv_scale confused with softmax scale");
    check(std::fabs(scaled.output[0]-1.f/(1.f+std::exp(-KV))) > .01f, "wrong scale would pass");
    // Stable softmax and LSE: unshifted exp(12800/sqrt(128)) overflows.
    for (float& f : opposing.q) f *= 100.f;
    auto large = reference(opposing, config, video);
    for (float f : large.output) check(std::isfinite(f), "large finite logits need stable softmax");
    // Default ceil on a single partial block is valid; no fake no-video success.
    SparseAttentionConfig invalid;
    auto reject = [&](const SparseAttentionConfig& opts, const SparseAttentionLayout& l,
                      int64_t dim = D, float kv = KV, int64_t heads = 2) {
        SparseAttentionReference out; std::string error;
        check(!h3_subblock_attention_reference(x.q.data(),x.k.data(),x.v.data(),x.length,
            heads,dim,kv,opts,l,&out,&error) && !error.empty(), "invalid contract accepted");
    };
    invalid.sparse_topk=0; reject(invalid,layout);
    invalid.sparse_topk=4; reject(invalid,layout);
    invalid.sparse_topk=-2; reject(invalid,layout);
    invalid=SparseAttentionConfig{}; invalid.sparse_keep_ratio=0; reject(invalid,layout);
    invalid.sparse_keep_ratio=1.01f; reject(invalid,layout);
    invalid.sparse_keep_ratio=std::numeric_limits<float>::quiet_NaN(); reject(invalid,layout);
    invalid=SparseAttentionConfig{}; invalid.sparse_block_tokens=32; reject(invalid,layout);
    invalid=SparseAttentionConfig{}; reject(invalid,layout,64); reject(invalid,layout,D,.125f); reject(invalid,layout,D,KV,0);
    reject(invalid,{{{0,146,SparseTokenKind::Audio}}});
    reject(invalid,{{{1,146,SparseTokenKind::TargetVideo}}});
    reject(invalid,{{{0,65,SparseTokenKind::TargetVideo},{64,146,SparseTokenKind::Audio}}});
    reject(invalid,{{{0,145,SparseTokenKind::TargetVideo}}});
    reject(invalid,{{{0,146,static_cast<SparseTokenKind>(99)}}});
    x.q[0]=std::numeric_limits<float>::infinity(); reject(invalid,layout);
    std::cout << "reference: all-selected max_abs=" << all_error
              << "; perhead/subblock/tails/protected/dense-query/scale/stability/invalid passed\n";
}
#ifndef MINIMAX_H3_SPARSE_REFERENCE_ONLY
struct CpuGraph {
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(h3_test_cpu_reg(),0),nullptr);
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    GGMLRunnerContext runner;
    std::vector<std::pair<ggml_tensor*,const void*>> bindings;
    CpuGraph() {
        check(backend != nullptr,"CPU backend init");
        ggml_backend_cpu_set_n_threads(backend,2);
        ggml_init_params init{64u*1024u*1024u,nullptr,true};
        context=ggml_init(init); check(context != nullptr,"graph context");
        runner.ggml_ctx=context; runner.backend=backend;
        runner.bind_backend_tensor_data=[this](ggml_tensor* t,const void* p) { bindings.emplace_back(t,p); };
    }
    ~CpuGraph() { if (buffer) ggml_backend_buffer_free(buffer); if(context) ggml_free(context); if(backend) ggml_backend_free(backend); }
    std::vector<float> run(const Input& x,const SparseAttentionConfig& config,
                           const SparseAttentionLayout& layout,bool flash,bool baseline=false,bool strided=false) {
        runner.flash_attn_enabled=flash;
        ggml_tensor* packed=nullptr;
        ggml_tensor *q=nullptr,*k=nullptr,*v=nullptr;
        std::vector<float> packed_data;
        if (strided) {
            // The real H3 projected-QKV views have contiguous d, interleaved
            // heads, and a 3*inner token stride; q/k are [d,L,H], v [d,H,L].
            const int64_t inner=D*x.heads;
            packed=ggml_new_tensor_2d(context,GGML_TYPE_F32,3*inner,x.length);
            q=ggml_view_3d(context,packed,D,x.length,x.heads,size_t(3*inner)*sizeof(float),D*sizeof(float),0);
            k=ggml_view_3d(context,packed,D,x.length,x.heads,size_t(3*inner)*sizeof(float),D*sizeof(float),size_t(inner)*sizeof(float));
            v=ggml_view_3d(context,packed,D,x.heads,x.length,D*sizeof(float),size_t(3*inner)*sizeof(float),size_t(2*inner)*sizeof(float));
            packed_data.resize(size_t(3*inner*x.length));
            for(int64_t t=0;t<x.length;++t) for(int64_t h=0;h<x.heads;++h) for(int d=0;d<D;++d) {
                const size_t base=size_t(d+D*h+3*inner*t);
                packed_data[base]=x.q[x.qi(d,t,h)];
                packed_data[base+size_t(inner)]=x.k[x.qi(d,t,h)];
                packed_data[base+size_t(2*inner)]=x.v[x.vi(d,t,h)];
            }
        } else {
            q=ggml_new_tensor_3d(context,GGML_TYPE_F32,D,x.length,x.heads);
            k=ggml_new_tensor_3d(context,GGML_TYPE_F32,D,x.length,x.heads);
            v=ggml_new_tensor_3d(context,GGML_TYPE_F32,D,x.heads,x.length);
        }
        ggml_tensor* output = baseline
            ? ggml_ext_attention_ext(context,backend,q,k,v,x.heads,nullptr,true,flash,KV,false)
            : h3_subblock_attention(&runner,q,k,v,x.heads,KV,config,layout);
        check(output != nullptr,"CPU sparse graph unsupported");
        check(output->ne[0]==D*x.heads && output->ne[1]==x.length && output->ne[2]==1,"graph output layout");
        auto* graph=ggml_new_graph_custom(context,32768,false);
        ggml_build_forward_expand(graph,output);
        int topks=0,flash_ops=0;
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* node=ggml_graph_node(graph,i);
            check(ggml_backend_supports_op(backend,node),"graph op unsupported on CPU");
            topks += node->op==GGML_OP_TOP_K;
            flash_ops += node->op==GGML_OP_FLASH_ATTN_EXT;
        }
        if (!baseline) {
            check(topks>0,"graph silently substituted dense instead of dynamic topk");
            if (flash) check(flash_ops>0,"CPU flash unexpectedly fell back");
        }
        if (flash) {
            for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
                auto* node=ggml_graph_node(graph,i);
                if (node->op!=GGML_OP_FLASH_ATTN_EXT) continue;
                float attention_scale=0;
                std::memcpy(&attention_scale,node->op_params,sizeof(float));
                check(std::fabs(attention_scale-(1.f/std::sqrt(128.f))/KV)<1e-6f,
                      "flash must use softmax_scale / kv_scale");
                check(node->src[1]->type==GGML_TYPE_F16 && node->src[2]->type==GGML_TYPE_F16,
                      "baseline flash K/V rounding must be F16");
                for (int source : {1,2}) {
                    auto* scaled=node->src[source]->src[0];
                    check(scaled && scaled->op==GGML_OP_SCALE,"baseline KV prep missing scale before cast");
                    float factor=0; std::memcpy(&factor,scaled->op_params,sizeof(float));
                    check(factor==KV,"baseline KV prep scale changed");
                }
            }
        }
        buffer=ggml_backend_alloc_ctx_tensors(context,backend); check(buffer != nullptr,"CPU allocation");
        if (strided) ggml_backend_tensor_set(packed,packed_data.data(),0,packed_data.size()*sizeof(float));
        else {
            ggml_backend_tensor_set(q,x.q.data(),0,x.q.size()*sizeof(float));
            ggml_backend_tensor_set(k,x.k.data(),0,x.k.size()*sizeof(float));
            ggml_backend_tensor_set(v,x.v.data(),0,x.v.size()*sizeof(float));
        }
        for (const auto& b : bindings) ggml_backend_tensor_set(b.first,b.second,0,ggml_nbytes(b.first));
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"CPU compute");
        std::vector<float> result(size_t(D*x.length*x.heads));
        ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
        if (!baseline) {
            Input changed=x;
            for (float& f : changed.k) f=-f;
            if (strided) {
                const int64_t inner=D*x.heads;
                for(int64_t t=0;t<x.length;++t) for(int64_t h=0;h<x.heads;++h) for(int d=0;d<D;++d)
                    packed_data[size_t(d+D*h+3*inner*t+inner)]=changed.k[changed.qi(d,t,h)];
                ggml_backend_tensor_set(packed,packed_data.data(),0,packed_data.size()*sizeof(float));
            } else ggml_backend_tensor_set(k,changed.k.data(),0,changed.k.size()*sizeof(float));
            check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"CPU replay compute");
            std::vector<float> replay(result.size());
            ggml_backend_tensor_get(output,replay.data(),0,replay.size()*sizeof(float));
            check(max_error(replay,reference(changed,config,layout).output)<(flash?2e-3f:5e-4f),
                  "dynamic router failed activation-changing graph replay");
        }
        return result;
    }
};
std::vector<float> graph_run(const Input& x,const SparseAttentionConfig& config,
                              const SparseAttentionLayout& layout,bool flash,bool baseline=false,bool strided=false) {
    CpuGraph g; return g.run(x,config,layout,flash,baseline,strided);
}
void test_graph() {
    Input x(146,2); SparseAttentionConfig config; auto layout=mixed_layout();
    config.sparse_keep_ratio=1;
    auto mathematical=reference(x,config,layout);
    auto actual=graph_run(x,config,layout,false);
    const float graph_error=max_error(actual,mathematical.output);
    // CPU ggml mul_mat may quantize its RHS to F16 (build dependent).
    check(graph_error<5e-4f,"ggml graph vs F32 oracle");
    const float dense_error=max_error(actual,graph_run(x,config,layout,false,true));
    check(dense_error<5e-4f,"all-selected graph vs baseline dense");
    config.sparse_keep_ratio=.5f;
    const float sparse_error=max_error(graph_run(x,config,layout,false),reference(x,config,layout).output);
    check(sparse_error<5e-4f,"selected sparse graph vs independent F32 reference");
    check(max_error(graph_run(x,config,layout,false,false,true),reference(x,config,layout).output)<5e-4f,
          "H3 strided projected QKV views changed sparse output");
    // Exercise an entirely video sequence, K-tail padding without protected KV.
    Input tail(65,2); SparseAttentionLayout tail_layout{{{0,65,SparseTokenKind::TargetVideo}}};
    config.sparse_topk=1;
    check(max_error(graph_run(tail,config,tail_layout,false),reference(tail,config,tail_layout).output)<5e-4f,
          "graph padded tail diluted softmax");
    // Opposing routes make a shared-head or frozen router numerically obvious.
    Input heads(128,2); std::fill(heads.q.begin(),heads.q.end(),1.f);
    for (int64_t h=0;h<2;++h) for(int64_t t=0;t<128;++t) for(int d=0;d<D;++d) {
        heads.k[heads.qi(d,t,h)]=((t<64)==(h==0))?1.f:-1.f;
        heads.v[heads.vi(d,t,h)]=t<64?2.f:7.f;
    }
    SparseAttentionLayout video{{{0,128,SparseTokenKind::TargetVideo}}};
    check(max_error(graph_run(heads,config,video,false),reference(heads,config,video).output)<5e-4f,
          "graph route must be dynamic and perhead");
    Input lse(128,1); std::fill(lse.q.begin(),lse.q.end(),1.f);
    for(int64_t t=0;t<128;++t) for(int d=0;d<D;++d)
        lse.k[lse.qi(d,t,0)]=t<64?((t/16)%2==0?4.f:-4.f):.25f;
    check(max_error(graph_run(lse,config,video,false),reference(lse,config,video).output)<5e-4f,
          "ggml router must use SubBlock LSE not whole-block mean");
    Input negative_tail(65,1); std::fill(negative_tail.q.begin(),negative_tail.q.end(),1.f);
    for(int64_t t=0;t<65;++t) for(int d=0;d<D;++d)
        negative_tail.k[negative_tail.qi(d,t,0)]=t<64?-2.f:-3.f;
    check(max_error(graph_run(negative_tail,config,tail_layout,false),reference(negative_tail,config,tail_layout).output)<5e-4f,
          "ggml padded route cells must not contribute exp(0)");
    Input forced(135,1);
    std::fill(forced.q.begin(),forced.q.end(),0);
    std::fill(forced.k.begin(),forced.k.end(),0);
    std::fill(forced.v.begin(),forced.v.end(),0);
    for(int64_t t=128;t<135;++t) for(int d=0;d<D;++d) forced.v[forced.vi(d,t,0)]=float(t-127);
    SparseAttentionLayout forced_layout{{{0,128,SparseTokenKind::TargetVideo},
        {128,131,SparseTokenKind::Audio},{131,133,SparseTokenKind::Reference},{133,135,SparseTokenKind::Query}}};
    check(max_error(graph_run(forced,config,forced_layout,false),reference(forced,config,forced_layout).output)<5e-4f,
          "ggml must force late audio/reference/query KV and update all Q");
    Input large=heads;
    for(float& f : large.q) f*=100.f;
    check(max_error(graph_run(large,config,video,false),reference(large,config,video).output)<5e-4f,
          "ggml stable LSE/softmax failed large finite logits");
    // Exact baseline SEMANTICS: compare to the same helper with flash enabled,
    // rather than pretend a mathematical F32 oracle reproduces F16 KV rounding.
    config.sparse_topk=-1; config.sparse_keep_ratio=1;
    const float flash_error=max_error(graph_run(x,config,layout,true),graph_run(x,config,layout,true,true));
    // Small masked chunks use the CPU vec kernel (F16 PV accumulator); the
    // full baseline uses its tiled kernel. KV reordering also changes rounding.
    check(flash_error<1e-3f,"all-selected flash with baseline kv_scale semantics");
    CpuGraph invalid;
    auto* q=ggml_new_tensor_3d(invalid.context,GGML_TYPE_F32,D,128,2);
    auto* k=ggml_new_tensor_3d(invalid.context,GGML_TYPE_F32,D,128,2);
    auto* v=ggml_new_tensor_3d(invalid.context,GGML_TYPE_F32,D,2,128);
    config.sparse_topk=0;
    check(!h3_subblock_attention(&invalid.runner,q,k,v,2,KV,config,video),"graph topk0 not failclosed");
    config.sparse_topk=-1;
    check(!h3_subblock_attention(&invalid.runner,q,k,v,2,.125f,config,video),"graph kv_scale not failclosed");
    auto* bf16=ggml_new_tensor_3d(invalid.context,GGML_TYPE_BF16,D,128,2);
    check(!h3_subblock_attention(&invalid.runner,q,bf16,v,2,KV,config,video),
          "unsupported BF16 prep rounding not failclosed");
    // Fake only the capability query, NEVER execute a fake/GPU backend. The
    // helper must reject unavailable topk and unavailable flash, not return dense.
    auto device=*ggml_backend_get_device(invalid.backend);
    auto stream=*invalid.backend;
    stream.device=&device;
    auto* original=invalid.runner.backend;
    invalid.runner.backend=&stream;
    device.iface.supports_op=[](ggml_backend_dev_t,const ggml_tensor* op) {
        return op->op!=GGML_OP_TOP_K;
    };
    check(!h3_subblock_attention(&invalid.runner,q,k,v,2,KV,config,video),"unsupported topk silently dense");
    device.iface.supports_op=[](ggml_backend_dev_t,const ggml_tensor* op) {
        return op->op!=GGML_OP_FLASH_ATTN_EXT;
    };
    invalid.runner.flash_attn_enabled=true;
    check(!h3_subblock_attention(&invalid.runner,q,k,v,2,KV,config,video),"unsupported flash silently dense");
    invalid.runner.backend=original;
    invalid.runner.flash_attn_enabled=false;
    invalid.runner.bind_backend_tensor_data={};
    check(!h3_subblock_attention(&invalid.runner,q,k,v,2,KV,config,video),"missing graph input lifetime/binding not failclosed");
    check(!h3_subblock_attention(nullptr,q,k,v,2,KV,config,video),"null runner accepted");
    std::cout << "CPU ggml: reference max_abs=" << graph_error << ", dense=" << dense_error
              << ", sparse=" << sparse_error << ", flash-baseline=" << flash_error << "\n";
}
#endif
} // namespace
int main(int argc, char** argv) {
    try {
#if !defined(MINIMAX_H3_SPARSE_REFERENCE_ONLY) && defined(GGML_BACKEND_DL)
        check(argc==2,"backend-DL test requires explicit CPU-only module path as its sole argument");
        h3_test_cpu_module=ggml_backend_load(argv[1]);
        check(h3_test_cpu_module && std::strcmp(ggml_backend_reg_name(h3_test_cpu_module),"CPU")==0,
              "could not load CPU-only backend module");
#else
        (void)argc; (void)argv;
#endif
        test_reference();
#ifndef MINIMAX_H3_SPARSE_REFERENCE_ONLY
        test_graph();
#endif
        std::cout << "H3 SubBlock sparse tests passed (CPU only)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
