// Default --cpu never initializes CUDA. --run requires the existing guarded
// single-GPU wrapper. This file covers route IDs, all output rows, graph size,
// tails, protected spans, view strides, changing inputs and allocation bounds.
#define MINIMAX_H3_SPARSE_REFERENCE_ONLY
#ifndef ED_ENABLE_CUDA_H3_SUBBLOCK
#define ED_ENABLE_CUDA_H3_SUBBLOCK
#endif
#include "dit_models/models/minimax_h3_cuda_sparse.hpp"
#include "core/backend/ggml/cuda/h3_subblock.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <cstdlib>
#include <array>
#include <random>

using namespace MiniMaxH3;
namespace native = edgedit::h3_subblock;
static int checks=0, comparisons=0;
static void check(bool ok,const char* what) { ++checks;if(!ok)throw std::runtime_error(what); }
static void near(const std::vector<float>& a,const std::vector<float>& b,float atol,float rtol) {
    check(a.size()==b.size(),"output size mismatch");
    for(size_t i=0;i<a.size();++i) if(!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
        std::abs(a[i]-b[i])>atol+rtol*std::abs(b[i])) {
        std::cerr<<"index="<<i<<" actual="<<a[i]<<" reference="<<b[i]<<'\n';
        throw std::runtime_error("numeric mismatch / nonfinite / unwritten output");
    }
    ++comparisons;
}
struct Fixture {
    int length,heads; SparseAttentionLayout layout; SparseAttentionConfig options;
    h3_sparse_detail::Plan plan; std::vector<int32_t> blocks,chunks,protected_ids;
    std::vector<float> q,k,v;
    Fixture(int n,int h,bool mixed,int topk=-1,bool allocate=true):length(n),heads(h),q(allocate?size_t(n)*h*128:0),k(q.size()),v(q.size()) {
        if(!mixed)layout.spans={{0,n,SparseTokenKind::TargetVideo}};
        else {
            check(n>=149,"mixed fixture too short");
            layout.spans={{0,3,SparseTokenKind::Text},{3,68,SparseTokenKind::TargetVideo},
                         {68,73,SparseTokenKind::Audio},{73,90,SparseTokenKind::Reference},
                         {90,n-2,SparseTokenKind::TargetVideo},{n-2,n,SparseTokenKind::Query}};
        }
        options.sparse_topk=topk;
        std::string reason;
        check(h3_sparse_detail::make_plan(n,h,128,1.f/128,options,layout,&plan,&reason),"fixture plan");
        for(const auto& b:plan.video) {blocks.push_back(int(b.begin));blocks.push_back(int(b.end));}
        for(const auto& span:layout.spans) if(span.kind!=SparseTokenKind::TargetVideo) {
            for(int64_t t=span.begin;t<span.end;++t)protected_ids.push_back(int(t));
            for(int64_t t=span.begin;t<span.end;t+=64) {
                chunks.push_back(int(t));chunks.push_back(int(std::min(t+64,span.end)));
            }
        }
        chunks.push_back(0);chunks.push_back(0);protected_ids.push_back(-1);
    }
    void fill(int seed) {
        std::mt19937 rng(seed);std::uniform_real_distribution<float> random(-.75f,.75f);
        for(auto* x:{&q,&k,&v})for(float& a:*x)a=random(rng);
    }
    SparseAttentionReference reference(bool dense=false)const {
        SparseAttentionReference ref;std::string reason;auto config=options;
        if(dense)config.sparse_topk=int(plan.video.size());
        check(h3_subblock_attention_reference(q.data(),k.data(),v.data(),length,heads,128,1.f/128,
                                              config,layout,&ref,&reason),"CPU original reference failed");
        return ref;
    }
};
// Independent host emulation of the CUDA lane reduction and bounded gather.
// It deliberately doesn't call the original reference's block_score/attention.
static SparseAttentionReference emulate(const Fixture& f) {
    const int B=int(f.plan.video.size()),K=f.plan.topk,N=f.length,H=f.heads;
    SparseAttentionReference out;out.video_blocks=B;out.topk=K;
    out.selected_blocks.resize(size_t(H)*B*K);out.output.resize(size_t(H)*N*128);
    std::vector<float> qm(size_t(H)*B*4*128),km(qm.size());
    for(int h=0;h<H;++h)for(int b=0;b<B;++b)for(int cell=0;cell<4;++cell)for(int d=0;d<128;++d) {
        const auto span=f.plan.video[b];int first=int(span.begin)+16*cell,last=std::min(int(span.end),first+16);
        float sq=0,sk=0;for(int t=first;t<last;++t) {
            sq+=f.q[size_t(d+128*(t+N*h))];sk+=f.k[size_t(d+128*(t+N*h))];
        }
        const size_t i=((size_t(h)*B+b)*4+cell)*128+d;
        qm[i]=last>first?sq/(last-first):0;km[i]=last>first?sk/(last-first):0;
    }
    for(int h=0;h<H;++h)for(int b=0;b<B;++b) {
        const int nq=int(f.plan.video[b].end-f.plan.video[b].begin+15)/16;
        std::vector<float> scores(B);std::vector<int> order(B);std::iota(order.begin(),order.end(),0);
        for(int kb=0;kb<B;++kb) {
            int nk=int(f.plan.video[kb].end-f.plan.video[kb].begin+15)/16;
            float maximum=-std::numeric_limits<float>::infinity(),mass=0;
            for(int a=0;a<nq;++a)for(int z=0;z<nk;++z) {
                std::array<float,32> lane{};
                for(int l=0;l<32;++l)for(int d=l;d<128;d+=32)
                    lane[l]+=qm[((size_t(h)*B+b)*4+a)*128+d]*km[((size_t(h)*B+kb)*4+z)*128+d];
                for(int shift=16;shift;shift/=2) {auto old=lane;for(int l=0;l+shift<32;++l)lane[l]+=old[l+shift];}
                float dot=lane[0]/std::sqrt(128.f),m=std::max(maximum,dot);
                mass=mass*std::exp(maximum-m)+std::exp(dot-m);maximum=m;
            }
            scores[kb]=maximum+std::log(mass);
        }
        std::sort(order.begin(),order.end(),[&](int a,int b){return scores[a]==scores[b]?a<b:scores[a]>scores[b];});
        for(int rank=0;rank<K;++rank)out.selected_blocks[(size_t(h)*B+b)*K+rank]=order[rank];
    }
    native::Capacity cap;check(native::capacity(N,H,B,K,f.plan.protected_tokens,&cap),"emulator capacity");
    for(int phase=0;phase<2;++phase) {
        const bool sparse=phase==0;int count=sparse?B:int(f.chunks.size()/2)-1;
        if(!count)continue;
        for(int start=0;start<H*count;start+=cap.batch)for(int job=start;job<std::min(start+cap.batch,H*count);++job) {
            const int h=job/count,b=job%count;
            const int first=sparse?f.blocks[2*b]:f.chunks[2*b];
            const int last=sparse?f.blocks[2*b+1]:f.chunks[2*b+1];
            std::vector<int> keys;
            if(sparse) {
                // Build the same padded gather, then remove -1 masked slots.
                for(int rank=0;rank<K;++rank) {
                    int id=out.selected_blocks[(size_t(h)*B+b)*K+rank];
                    for(int t=f.blocks[2*id];t<f.blocks[2*id+1];++t)keys.push_back(t);
                }
                keys.insert(keys.end(),f.protected_ids.begin(),f.protected_ids.end()-1);
            } else {keys.resize(N);std::iota(keys.begin(),keys.end(),0);}
            for(int t=first;t<last;++t) {
                std::vector<double> weights;double maxlog=-std::numeric_limits<double>::infinity();
                for(int s:keys) {double dot=0;for(int d=0;d<128;++d)
                    dot+=double(f.q[size_t(d+128*(t+N*h))])*f.k[size_t(d+128*(s+N*h))];
                    weights.push_back(dot/std::sqrt(128.));maxlog=std::max(maxlog,weights.back());}
                double total=0;for(double& w:weights){w=std::exp(w-maxlog);total+=w;}
                for(int d=0;d<128;++d){double sum=0;for(size_t i=0;i<keys.size();++i)
                    sum+=weights[i]*f.v[size_t(d+128*(h+H*keys[i]))];
                    out.output[size_t(d+128*(h+H*t))]=float(sum/total);}
            }
        }
    }
    return out;
}
// CPU verification of the exact ldmatrix address functions shared with the
// indexed CUDA kernel. This proves the intended fragment layout, NOT execution
// of the hardware instructions (the guarded CUDA matrix below is still needed).
static void indexed_layout_tests() {
    using Fragments=std::array<std::array<float,8>,32>;
    auto load_matrix=[](auto address,bool transpose) {
        Fragments f{};
        for(int lane=0;lane<32;++lane)for(int reg=0;reg<4;++reg)for(int half=0;half<2;++half) {
            int row=lane/4,col=2*(lane%4)+half;
            auto a=address(reg*8+(transpose?col:row));
            f[lane][reg*2+half]=float(a.first*1000+a.second+(transpose?row:col));
        }
        return f;
    };
    auto decode=[](const Fragments& f,bool a) {
        std::array<float,256> matrix{};std::array<int,256> written{};
        for(int lane=0;lane<32;++lane)for(int e=0;e<8;++e) {
            const int row=a?lane/4+((e/2)%2)*8:2*(lane%4)+e%2+((e/2)%2)*8;
            const int col=a?2*(lane%4)+e%2+(e/4)*8:lane/4+(e/4)*8;
            check(++written[row*16+col]==1,"MMA fragment aliases an element");
            matrix[row*16+col]=f[lane][e];
        }
        return matrix;
    };
    for(int warp=0;warp<4;++warp)for(int d=0;d<8;++d) {
        auto q=decode(load_matrix([&](int l){return std::make_pair(native::indexed_q_row(warp,l),native::indexed_q_col(d,l));},false),true);
        for(int r=0;r<16;++r)for(int c=0;c<16;++c)
            check(q[r*16+c]==float((warp*16+r)*1000+d*16+c),"Q ldmatrix layout");
    }
    for(int kt=0;kt<4;++kt)for(int d=0;d<8;++d) {
        auto k=decode(load_matrix([&](int l){return std::make_pair(native::indexed_k_row(kt,l),native::indexed_k_col(d,l));},false),false);
        auto v=decode(load_matrix([&](int l){return std::make_pair(native::indexed_v_row(kt,l),native::indexed_v_col(d,l));},true),false);
        for(int r=0;r<16;++r)for(int c=0;c<16;++c) {
            check(k[r*16+c]==float((kt*16+c)*1000+d*16+r),"K-transpose ldmatrix layout");
            check(v[r*16+c]==float((kt*16+r)*1000+d*16+c),"V transposed-ldmatrix layout");
        }
    }
    Fragments probability{};
    for(int l=0;l<32;++l)for(int e=0;e<8;++e)
        probability[l][e]=float(native::indexed_acc_row(l,e)*1000+native::indexed_acc_col(l,e));
    auto p=decode(probability,true);
    for(int r=0;r<16;++r)for(int c=0;c<16;++c)
        check(p[r*16+c]==float(r*1000+c),"P accumulator-to-A packing");
    std::array<int,64*128> written{};
    for(int w=0;w<4;++w)for(int d=0;d<8;++d)for(int l=0;l<32;++l)for(int e=0;e<8;++e) {
        const int row=w*16+native::indexed_acc_row(l,e),col=d*16+native::indexed_acc_col(l,e);
        check(++written[row*128+col]==1,"indexed output alias / duplicate scatter");
    }
    for(int n:written)check(n==1,"indexed output hole");
    check(native::indexed_shared_bytes==34816,"indexed scratch size changed");
    check(native::indexed_attention_tag!=native::attention_tag && native::indexed_attention_tag!=native::router_tag,"indexed tag collision");
}

static float as_half(float x) {return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));}
// Per-K64 online max/sum/PV oracle, with the indexed kernel's baseline half
// prep and half probabilities. No call to its original dense reference math.
static std::vector<float> indexed_online_oracle(const Fixture& f,const SparseAttentionReference& routes) {
    const int N=f.length,H=f.heads,B=int(f.plan.video.size()),K=f.plan.topk;
    std::vector<float> q(f.q.size()),k(q.size()),v(q.size()),output(q.size());
    const float scale=as_half(128.f/std::sqrt(128.f));
    for(size_t i=0;i<q.size();++i) {
        q[i]=as_half(as_half(f.q[i])*scale);k[i]=as_half(f.k[i]/128.f);v[i]=as_half(f.v[i]/128.f);
    }
    for(int h=0;h<H;++h) {
        int qb=0;
        for(const auto& chunk:f.plan.chunks) {
            std::vector<std::vector<int>> tiles;
            if(chunk.kind==SparseTokenKind::TargetVideo) {
                for(int rank=0;rank<K;++rank) {
                    const int id=routes.selected_blocks[(size_t(h)*B+qb)*K+rank];
                    std::vector<int> tokens;for(int t=int(f.plan.video[id].begin);t<f.plan.video[id].end;++t)tokens.push_back(t);
                    tiles.push_back(tokens);
                }
                for(size_t p=0;p+1<f.protected_ids.size();p+=64) {
                    std::vector<int> tokens;for(size_t i=p;i<std::min(p+64,f.protected_ids.size()-1);++i)tokens.push_back(f.protected_ids[i]);
                    tiles.push_back(tokens);
                }
                ++qb;
            } else for(int start=0;start<N;start+=64) {
                std::vector<int> tokens;for(int t=start;t<std::min(start+64,N);++t)tokens.push_back(t);tiles.push_back(tokens);
            }
            for(int t=int(chunk.begin);t<chunk.end;++t) {
                float maximum=-std::numeric_limits<float>::infinity(),denominator=0;
                std::array<float,128> numerator{};
                for(const auto& keys:tiles) {
                    std::vector<float> scores;float next=maximum;
                    for(int key:keys) {
                        float dot=0;for(int d=0;d<128;++d)dot+=q[size_t(d+128*(t+N*h))]*k[size_t(d+128*(key+N*h))];
                        scores.push_back(dot);next=std::max(next,dot);
                    }
                    const float alpha=std::exp(maximum-next);maximum=next;denominator*=alpha;
                    for(float& x:numerator)x*=alpha;
                    for(size_t key=0;key<keys.size();++key) {
                        const float weight=std::exp(scores[key]-maximum);denominator+=weight;
                        const float rounded=as_half(weight);
                        for(int d=0;d<128;++d)numerator[d]+=rounded*v[size_t(d+128*(h+H*keys[key]))];
                    }
                }
                for(int d=0;d<128;++d)output[size_t(d+128*(h+H*t))]=numerator[d]*(128.f/denominator);
            }
        }
    }
    return output;
}
static void indexed_online_tests() {
    for(bool mixed:{false,true})for(int replay=0;replay<2;++replay) {
        Fixture f(mixed?193:65,2,mixed);f.fill(299+replay);
        const auto ref=f.reference();near(indexed_online_oracle(f,ref),ref.output,4e-4f,4e-3f);
    }
    // Full keep, deliberately reversed route order and large moving row maxima.
    // A dense global softmax reference is order independent; a broken online
    // rescale (including resetting between video/protected tiles) is not.
    Fixture f(193,2,true,4);f.fill(125);
    for(int h=0;h<2;++h)for(int t=0;t<f.length;++t) {
        f.q[size_t(128*(t+f.length*h))]=1;
        f.k[size_t(128*(t+f.length*h))]=float((t%97)-48)*2;
    }
    auto ref=f.reference();auto reordered=ref;
    for(size_t row=0;row<reordered.selected_blocks.size();row+=4)
        std::reverse(reordered.selected_blocks.begin()+row,reordered.selected_blocks.begin()+row+4);
    near(indexed_online_oracle(f,reordered),ref.output,4e-4f,4e-3f);
}

struct Graph {
    ggml_context* ctx=nullptr;ggml_cgraph* graph=nullptr;
    ggml_tensor *q=nullptr,*k=nullptr,*v=nullptr,*blocks=nullptr,*chunks=nullptr,*ids=nullptr,*route=nullptr,*out=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    const Fixture& fixture;bool strides;
    Graph(const Fixture& f,bool view,bool indexed=false):fixture(f),strides(view) {
        ctx=ggml_init({4u<<20,nullptr,true});check(ctx,"ggml metadata context");
        auto input=[&](bool value) {
            const int64_t a=value?f.heads:f.length,b=value?f.length:f.heads;
            if(!view)return ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,a,b);
            auto* raw=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,132,a+1,b);
            return ggml_view_3d(ctx,raw,128,a,b,raw->nb[1],raw->nb[2],4);
        };
        q=input(false);k=input(false);v=input(true);
        blocks=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,f.blocks.size()/2);
        chunks=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,2,f.chunks.size()/2);
        ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,f.protected_ids.size());
        route=native::router(ctx,q,k,blocks,f.plan.topk);
        out=native::attention(ctx,q,k,v,route,blocks,chunks,ids,
                              indexed?native::Implementation::Indexed:native::Implementation::Gather);
        check(native::tag(out)==(indexed?native::indexed_attention_tag:native::attention_tag),"wrong implementation tag / silent fallback");
        graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
        check(ggml_graph_n_nodes(graph)<=5,"native graph node count grew with blocks/heads");
        check(ed_cuda_h3_subblock_shape_supported(route),"native router shape rejected");
        check(ed_cuda_h3_subblock_shape_supported(out),"native attention shape rejected");
    }
    ~Graph(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
    void alloc(ggml_backend_t backend) {
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            check(ggml_backend_supports_op(backend,ggml_graph_node(graph,i)),"CUDA dispatch missing / op unsupported");
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"CUDA allocation failed");
        auto device=ggml_backend_get_device(backend);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* t=ggml_graph_node(graph,i);auto* source=t->view_src?t->view_src:t;
            check(source->buffer && !ggml_backend_buffer_is_host(source->buffer),"host fallback allocation");
            check(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(source->buffer))==device,"wrong GPU allocation");
        }
        ggml_backend_tensor_set(blocks,fixture.blocks.data(),0,fixture.blocks.size()*4);
        ggml_backend_tensor_set(chunks,fixture.chunks.data(),0,fixture.chunks.size()*4);
        ggml_backend_tensor_set(ids,fixture.protected_ids.data(),0,fixture.protected_ids.size()*4);
    }
    void upload(ggml_tensor* t,const std::vector<float>& x) {
        if(!strides){ggml_backend_tensor_set(t,x.data(),0,x.size()*4);return;}
        for(int64_t b=0;b<t->ne[2];++b)for(int64_t a=0;a<t->ne[1];++a)
            ggml_backend_tensor_set(t,x.data()+size_t((b*t->ne[1]+a)*128),size_t(a)*t->nb[1]+size_t(b)*t->nb[2],128*4);
    }
    SparseAttentionReference run(ggml_backend_t backend) {
        upload(q,fixture.q);upload(k,fixture.k);upload(v,fixture.v);
        std::vector<float> nan(size_t(fixture.length)*fixture.heads*128,std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(out,nan.data(),0,nan.size()*4);
        std::vector<int32_t> sentinel(size_t(route->ne[0]*route->ne[1]*route->ne[2]),-1);
        ggml_backend_tensor_set(route,sentinel.data(),0,sentinel.size()*4);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"CUDA graph execution failed");
        ggml_backend_synchronize(backend);
        SparseAttentionReference result;result.output=nan;result.selected_blocks=sentinel;
        ggml_backend_tensor_get(out,result.output.data(),0,result.output.size()*4);
        ggml_backend_tensor_get(route,result.selected_blocks.data(),0,result.selected_blocks.size()*4);
        return result;
    }
};
static void capacity_tests() {
    native::Capacity cap;std::string why;
    check(h3_subblock_graph_budget({{{0,9472,SparseTokenKind::TargetVideo}}},56,50,131072,&why),"9472x56 production bounded graph rejected");
    check(h3_subblock_graph_budget({{{0,512,SparseTokenKind::Text},{512,50732,SparseTokenKind::TargetVideo}}},56,50,131072,&why)==false,"large scratch bound not enforced");
    check(!h3_subblock_graph_budget({{{0,INT64_MAX,SparseTokenKind::TargetVideo}}},INT64_MAX,50,131072),"overflow graph accepted");
    check(!h3_subblock_graph_budget({{{1,65,SparseTokenKind::TargetVideo}}},1,1,131072),"gapped native layout accepted");
    check(!h3_subblock_graph_budget({{{0,64,SparseTokenKind::TargetVideo}}},1,1000,8192),"node capacity gate lost");
    check(native::capacity(9472,56,148,74,0,&cap),"production scratch capacity");
    check(cap.batch>=1 && cap.batch_bytes<=native::gather_limit,"gather scratch bound");
    check(cap.means_bytes+cap.scores_bytes<=native::router_limit,"router scratch bound");
    std::cout<<"9472/H56: router="<<cap.means_bytes+cap.scores_bytes<<" gather="<<cap.batch_bytes<<" batch="<<cap.batch<<'\n';
    check(native::capacity(13000,56,197,99,445,&cap),"864x480/124f-size geometry capacity");
    check(cap.batch_bytes<=native::gather_limit,"larger production gather bound");
    native::Traffic traffic;
    check(native::traffic(15000,56,235,118,0,0,&traffic),"production traffic estimator");
    check(traffic.kv_stage_bytes==uint64_t(56)*235*7552*128*4,"KV staging accounting");
    check(traffic.dense_jobs==0 && traffic.sparse_jobs==13160,"production job accounting");
    check(!native::traffic(15000,56,235,118,0,1,&traffic),"invalid protected chunk accounting");
    for(int protected_tokens:{0,500}) {
        const int blocks=(15000-protected_tokens+63)/64,topk=(blocks+1)/2;
        check(native::capacity(15000,56,blocks,topk,protected_tokens,&cap),"15k capacity");
        check(native::traffic(15000,56,blocks,topk,protected_tokens,(protected_tokens+63)/64,&traffic),"15k traffic");
        std::cout<<"15000/H56 P="<<protected_tokens<<" B="<<blocks<<" topk="<<topk
                 <<" batch="<<cap.batch<<" batches="<<traffic.sparse_batches<<"+"<<traffic.dense_batches
                 <<" kv_stage="<<traffic.kv_stage_bytes<<" mask_stage="<<traffic.mask_stage_bytes
                 <<" q_stage="<<traffic.q_stage_bytes<<'\n';
    }
    check(!native::capacity(65537,1,1024,1,0,&cap),"sequence limit");
    check(!native::capacity(65536,64,1024,512,0,&cap),"router scratch limit");
    check(!native::capacity(64,65,1,1,0,&cap),"head limit");
    check(!native::capacity(64,1,1,0,0,&cap),"topk zero");
    check(!native::capacity(64,1,1,2,0,&cap),"topk out of range");
    check(!native::capacity(64,1,2,1,64,&cap),"no target video");
    Fixture f(9472,56,false,-1,false);Graph g(f,false);
    const auto save=g.route->type;g.route->type=GGML_TYPE_F32;
    check(!ed_cuda_h3_subblock_shape_supported(g.out),"malformed route accepted");g.route->type=save;
    auto* previous=g.out->src[2];g.out->src[2]=g.q;
    check(!ed_cuda_h3_subblock_shape_supported(g.out),"wrong V layout accepted");g.out->src[2]=previous;
    // Actual graph cuts may turn a previously computed route into an op=NONE
    // leaf. Capability checks must use its shape, not depend on producer identity.
    const auto op=g.route->op;g.route->op=GGML_OP_NONE;
    check(ed_cuda_h3_subblock_shape_supported(g.out),"external graph-cut route leaf rejected");g.route->op=op;
}
static std::string env(const char* key){const char* p=std::getenv(key);return p?p:"";}
static bool permit(int argc,char** argv) {
    if(argc!=2 || std::string(argv[1])!="--run" || env("H3_GPU_PARITY_ALLOW")!="1")return false;
    auto uuid=env("CUDA_VISIBLE_DEVICES");
    if(uuid.size()!=40 || uuid.substr(0,4)!="GPU-" || uuid!=env("H3_GPU_PARITY_PREFLIGHT_DEVICE"))return false;
    for(size_t i=4;i<uuid.size();++i) {
        if(i==12||i==17||i==22||i==27){if(uuid[i]!='-')return false;}
        else if(std::string("0123456789abcdefABCDEF").find(uuid[i])==std::string::npos)return false;
    }
    auto stamp=env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    if(stamp.empty() || stamp.find_first_not_of("0123456789")!=std::string::npos)return false;
    char* end=nullptr;double at=std::strtod(stamp.c_str(),&end),age=double(std::time(nullptr))-at;
    return end && !*end && std::isfinite(age) && age>=0 && age<=30;
}
int main(int argc,char** argv) {
    try {
        const bool gpu=argc==2 && std::string(argv[1])=="--run";
        if(!gpu && !(argc==2 && std::string(argv[1])=="--cpu")) {
            std::cerr<<"NOT RUN: use --cpu or guarded --run\n";return 2;
        }
        if(gpu && !permit(argc,argv)){std::cerr<<"NOT RUN: missing/stale single-GPU guard\n";return 2;}
        capacity_tests();
        indexed_layout_tests();
        indexed_online_tests();
        ggml_backend_t backend=nullptr;
        if(gpu) {
            check(ggml_backend_cuda_get_device_count()==1,"exactly one visible CUDA device required");
            backend=ggml_backend_cuda_init(0);check(backend&&ggml_backend_is_cuda(backend),"CUDA backend unavailable");
        }
        for(bool use_indexed:{false,true}) {
        for(bool mixed:{false,true})for(int n: mixed?std::vector<int>{149,193,257}:std::vector<int>{1,15,16,17,63,64,65,127,128,129,257})
            for(int topk:{-1,1})for(bool views:{false,true}) {
                Fixture f(n,3,mixed,topk);Graph graph(f,views,use_indexed);if(gpu)graph.alloc(backend);
                for(int replay=0;replay<2;++replay) {
                    f.fill(977+n+replay*1000);auto ref=f.reference();
                    auto actual=gpu?graph.run(backend):emulate(f);
                    check(actual.selected_blocks==ref.selected_blocks,"per-head/per-Q route IDs differ from original CPU estimator");
                    near(actual.output,ref.output,gpu?4e-4f:2e-6f,gpu?4e-3f:2e-6f);
                    if(mixed){auto dense=f.reference(true);for(const auto& span:f.layout.spans)
                        if(span.kind!=SparseTokenKind::TargetVideo) {
                            size_t first=size_t(span.begin)*f.heads*128,last=size_t(span.end)*f.heads*128;
                            near({actual.output.begin()+first,actual.output.begin()+last},
                                 {dense.output.begin()+first,dense.output.begin()+last},gpu?4e-4f:2e-6f,gpu?4e-3f:2e-6f);
                        }}
                }
            }
        // More than one native batch in BOTH video and protected phases;
        // heads/Q chunks cross a batch boundary and reuse the same scratch.
        {
            Fixture f(193,17,true);Graph graph(f,true,use_indexed);if(gpu)graph.alloc(backend);
            for(int replay=0;replay<2;++replay) {
                f.fill(779+replay);auto ref=f.reference();
                auto actual=gpu?graph.run(backend):emulate(f);
                check(actual.selected_blocks==ref.selected_blocks,"batch-boundary route mismatch");
                near(actual.output,ref.output,gpu?4e-4f:2e-6f,gpu?4e-3f:2e-6f);
            }
        }
        // Explicit adversarial routing: both head identity and Q-block identity
        // reverse the winning K block, so global/head-shared TopK cannot pass.
        {
            Fixture f(128,2,false,1);f.fill(193);
            std::fill(f.q.begin(),f.q.end(),0);std::fill(f.k.begin(),f.k.end(),0);
            for(int h=0;h<2;++h)for(int t=0;t<128;++t) {
                f.q[size_t(128*(t+128*h))]=(t<64?1.f:-1.f);
                f.k[size_t(128*(t+128*h))]=(t<64?2.f:-2.f)*(h==0?1.f:-1.f);
            }
            auto ref=f.reference();check(ref.selected_blocks==std::vector<int32_t>({0,1,1,0}),"adversarial reference route");
            Graph graph(f,true,use_indexed);if(gpu)graph.alloc(backend);
            auto actual=gpu?graph.run(backend):emulate(f);
            check(actual.selected_blocks==ref.selected_blocks,"per-query or per-head routing collapsed");
            near(actual.output,ref.output,gpu?4e-4f:2e-6f,gpu?4e-3f:2e-6f);
        }
        // Sub-cell variation matters even when whole-block means are zero.
        {
            Fixture f(128,1,false,1);f.fill(888);
            std::fill(f.q.begin(),f.q.end(),0);std::fill(f.k.begin(),f.k.end(),0);
            for(int t=0;t<128;++t) {
                f.q[size_t(128*t)]=(t/16)%2?8.f:-8.f;
                f.k[size_t(128*t)]=t<64?0.f:((t/16)%2?8.f:-8.f);
            }
            auto ref=f.reference();check(ref.selected_blocks==std::vector<int32_t>({1,1}),"sub-cell reference route");
            Graph graph(f,false,use_indexed);if(gpu)graph.alloc(backend);
            auto actual=gpu?graph.run(backend):emulate(f);
            check(actual.selected_blocks==ref.selected_blocks,"whole-block / mean-query-only proxy substituted");
            near(actual.output,ref.output,gpu?4e-4f:2e-6f,gpu?4e-3f:2e-6f);
        }
        // All-zero routing ties must pick low IDs; all K blocks must equal dense.
        for(bool full:{false,true}) {
            Fixture f(193,2,true,full?4:1);f.fill(422);std::fill(f.q.begin(),f.q.end(),0);
            auto ref=f.reference();Graph graph(f,true,use_indexed);if(gpu)graph.alloc(backend);
            auto actual=gpu?graph.run(backend):emulate(f);
            check(actual.selected_blocks==ref.selected_blocks,"equal-score deterministic selection");
            near(actual.output,ref.output,gpu?4e-4f:2e-6f,gpu?4e-3f:2e-6f);
        }
        } // gather and explicitly tagged indexed implementation
        if(backend)ggml_backend_free(backend);
        std::cout<<(gpu?"CUDA":"CPU-only")<<" SubBlock: checks="<<checks<<" output comparisons="<<comparisons<<" PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
