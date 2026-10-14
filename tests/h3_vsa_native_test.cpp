// TEST ONLY: default/--cpu is device-free. --run requires the existing fresh
// single-GPU guard AND a build with ED_ENABLE_CUDA_H3_VSA + its CUDA DSO.
// This tests native fine VSA, not compression/gates or production enablement.
#include "core/backend/ggml/cuda/h3_vsa.h"
#include "core/backend/ggml/cuda/h3_subblock.h"
#include "core/backend/ggml/cuda/ed_cuda_sage_attn.h"
#include "dit_models/models/minimax_h3_vsa_reference.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef ED_ENABLE_CUDA_H3_VSA
#include "ggml-cuda.h"
#endif
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
namespace native = edgedit::h3_vsa;
namespace ref = MiniMaxH3::VSAReference;
static int checks=0, comparisons=0;
static void check(bool ok,const char* why) { ++checks;if(!ok)throw std::runtime_error(why); }
static void near(const std::vector<float>& a,const std::vector<float>& b,float atol=6e-3f,float rtol=2e-3f) {
    check(a.size()==b.size(),"output size");
    for(size_t i=0;i<a.size();++i)if(!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i]-b[i])>atol+rtol*std::abs(b[i])) {
        std::cerr<<"index="<<i<<" actual="<<a[i]<<" reference="<<b[i]<<'\n';throw std::runtime_error("nonfinite/unwritten/numeric mismatch");
    }
    ++comparisons;
}
static float bf16(float x) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)); }
struct Fixture {
    ref::Geometry geo;int n,h,topk;ref::Tensor q,k,v;std::vector<int32_t> slots,counts;int32_t prefix;
    Fixture(ref::Grid grid,int dense,int heads,double keep):
        geo(ref::make_geometry(dense?std::vector<ref::Segment>{ref::Segment::text(dense),ref::Segment::video(grid)}:
                                    std::vector<ref::Segment>{ref::Segment::video(grid)},64)),
        n(int(geo.packed_tokens)),h(heads),topk(int(ref::compute_topk(1-keep,geo.tiles()-geo.num_prefix_tiles))),
        q(1,n,h,128),k(q),v(q),prefix(int32_t(geo.num_prefix_tiles)) {
        check(geo.video_tile_spans.size()==1,"native supports one video region only");
        std::vector<int> seen(n);
        for(size_t t=0;t<geo.tiles();++t) {
            int count=int(geo.variable_block_sizes[t]);check(count>0 && count<=64,"tile valid count");counts.push_back(count);
            for(int p=0;p<64;++p) {
                size_t row=geo.source_index[t*64+p];int32_t id=row==ref::invalid_token?-1:int32_t(row);slots.push_back(id);
                if(p<count) { check(id>=0 && id<n && ++seen[id]==1,"slot scatter overlap/range");
                    check(geo.untile_combined_index[id]==t*64+p,"slot inverse map"); }
                else check(id==-1,"padding must be -1, NOT the graph pool's dummy token S");
            }
        }
        for(int count:seen)check(count==1,"slot map omitted an original token");
        if(grid==ref::Grid{5,5,5})check(slots[size_t(prefix)*64+4]==dense+5,"3D geometry became linear K64 chunks");
    }
    void fill(int seed,int mode,int replay) { // 1: zero-Q ties; 2: head/query adversary; 3: F32-only score gap
        std::mt19937 rng(seed);std::uniform_real_distribution<float> random(-.75f,.75f);
        for(auto* x:{&q,&k,&v})for(float& a:x->data)a=random(rng);
        for(int t=0;t<n;++t)for(int head=0;head<h;++head)for(int d=0;d<128;++d)v.at(0,t,head,d)+=.05f*head;
        if(mode)std::fill(q.data.begin(),q.data.end(),0);
        if(mode>=2) {
            std::fill(k.data.begin(),k.data.end(),0);
            for(int t=0;t<n;++t)for(int head=0;head<h;++head) {
                int tile=int(geo.untile_combined_index[t]/64)-prefix;
                q.at(0,t,head,0)=mode==3?1.f:(tile%2==0?4.f:-4.f);
                k.at(0,t,head,0)=tile<0?32.f:((tile+head+replay)%2==0?4.f:-4.f);
                if(mode==3 && tile>=0)k.at(0,t,head,0)=1.f+((tile+head+replay)%2)*1e-4f;
            }
        }
    }
    ref::Scores scores()const { return ref::pooled_scores(ref::tile(q,geo),ref::tile(k,geo),geo); }
    std::vector<int32_t> routes(const ref::Scores& s)const {
        int tiles=int(geo.tiles());std::vector<int32_t> result;
        for(int head=0;head<h;++head)for(int qt=0;qt<tiles;++qt) {
            std::vector<int32_t> ids(tiles-prefix);std::iota(ids.begin(),ids.end(),prefix);
            std::sort(ids.begin(),ids.end(),[&](int a,int b) { float x=s.at(0,head,qt,a),y=s.at(0,head,qt,b);return x==y?a<b:x>y; });
            result.insert(result.end(),ids.begin(),ids.begin()+topk);
        }
        return result;
    }
    std::vector<float> oracle(const ref::Scores& s,const std::vector<int32_t>& ids,bool dense=false)const {
        ref::RoutingOptions opts;opts.region_budgets={ref::RegionBudget::topk(topk)};opts.force_dense=dense;
        auto mask=ref::route(s,geo,opts);int tiles=int(geo.tiles());
        for(int head=0;head<h;++head)for(int qt=0;qt<tiles;++qt)for(int kt=0;kt<tiles;++kt) {
            auto first=ids.begin()+(head*tiles+qt)*topk;
            bool wanted=dense || qt<prefix || kt<prefix || std::find(first,first+topk,kt)!=first+topk;
            check(mask.keeps(0,head,qt,kt)==wanted,"CPU mask vs independent sorted route IDs");
        }
        auto rq=q,rk=k,rv=v;
        for(auto* x:{&rq,&rk,&rv})for(float& a:x->data)a=bf16(a);
        // Independent VSAReference softmax/PV uses F32 probabilities, native uses
        // BF16 unnormalized probabilities + F32 accumulation. |V|<=1.15 here:
        // BF16 relative rounding <=1/256 gives <=0.0045 absolute PV error.
        // 0.006 + 0.002*|reference| allows that error and online/MMA order, NOT
        // SubBlock's 1/128 scaling. Q/K/V round to BF16; logits scale 1/sqrt(128).
        return ref::untile(ref::fine_attention(ref::tile(rq,geo),ref::tile(rk,geo),ref::tile(rv,geo),geo,mask),geo).data;
    }
};
struct Graph {
    ggml_context* ctx=nullptr;ggml_cgraph* graph=nullptr;ggml_backend_buffer_t buffer=nullptr;
    ggml_tensor *q,*k,*v,*scores,*slots,*counts,*prefix,*route,*out;
    Graph(int n,int h,int tiles,int topk) {
        ctx=ggml_init({1u<<20,nullptr,true});check(ctx,"metadata context");
        q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,n,h);k=ggml_dup_tensor(ctx,q);
        v=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,h,n);scores=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,tiles,tiles,h);
        slots=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,64,tiles);counts=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tiles);
        prefix=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);route=native::router(ctx,scores,prefix,topk);
        out=native::fine(ctx,q,k,v,route,slots,counts,prefix);
        check(native::tag(route)==native::router_tag && native::tag(out)==native::fine_tag,"native op tags");
        check(ed_cuda_h3_vsa_shape_supported(route) && ed_cuda_h3_vsa_shape_supported(out),"valid native shapes rejected");
        graph=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(graph,out);
        check(ggml_graph_n_nodes(graph)==2,"expanded-Q graph / non-native fallback");
        check(ggml_graph_node(graph,0)==route && ggml_graph_node(graph,1)==out,"node map/order");
        int tensors=0;for(auto* t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            ++tensors;check(!t->data && !t->buffer && !t->view_src,"CPU metadata allocated activation storage");
        }
        check(tensors==9,"hidden expanded-Q/scratch graph tensors");
    }
    ~Graph() { if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx); }
    void alloc(ggml_backend_t backend) {
        check(ggml_backend_supports_op(backend,route) && ggml_backend_supports_op(backend,out),"native CUDA dispatcher/DSO missing");
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"GPU allocation");
        std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
        for(auto* t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            check(t->buffer==buffer && !ggml_backend_buffer_is_host(t->buffer),"all inputs/outputs must share GPU buffer");
            check(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer))==ggml_backend_get_device(backend),"wrong GPU");
            uintptr_t begin=reinterpret_cast<uintptr_t>(t->data);ranges.emplace_back(begin,begin+ggml_nbytes(t));
        }
        std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)check(ranges[i-1].second<=ranges[i].first,"tensor allocation overlap");
    }
    void run(ggml_backend_t backend,const Fixture& f,const ref::Scores& s,const std::vector<int32_t>& ids,const std::vector<float>& expected) {
        auto upload=[](ggml_tensor* t,const auto& data) { ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(data[0])); };
        auto transpose=[&](const ref::Tensor& x) {
            std::vector<float> data(x.data.size());
            for(int head=0;head<f.h;++head)for(int t=0;t<f.n;++t)for(int d=0;d<128;++d)data[d+128*(t+f.n*head)]=x.at(0,t,head,d);
            return data;
        };
        upload(q,transpose(f.q));upload(k,transpose(f.k));upload(v,f.v.data);upload(scores,s.data);
        upload(slots,f.slots);upload(counts,f.counts);ggml_backend_tensor_set(prefix,&f.prefix,0,sizeof(f.prefix));
        std::vector<int32_t> got_ids(ids.size(),-1);std::vector<float> actual(expected.size(),std::numeric_limits<float>::quiet_NaN());
        upload(route,got_ids);upload(out,actual);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"actual native graph execution failed");ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(route,got_ids.data(),0,got_ids.size()*4);ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
        check(got_ids==ids,"per-head/per-Q top-k mismatch (including tie-ID ordering)");near(actual,expected);
    }
};
static void cpu_contracts() {
    check(!ref::production_switch_available,"test must not enable production VSA");
    check(!ed_cuda_h3_vsa_shape_supported(nullptr) && native::tag(nullptr)==0,"null op");
    for(int tiles=1;tiles<=native::max_tiles;++tiles)for(int heads:{1,3,9,native::max_heads}) {
        size_t per_head=size_t(tiles)*64*128*2*sizeof(uint16_t);int batch=native::stage_heads(tiles,heads);
        check(batch==std::min({heads,8,int(native::staging_limit/per_head)}),"stage_heads planner");
        check(batch>0 && size_t(batch)*per_head<=native::staging_limit,"KV staging exceeds 128MiB");
    }
    for(int tiles:{-1,0,native::max_tiles+1})check(native::stage_heads(tiles,3)==0,"invalid tile budget");
    for(int heads:{-1,0,native::max_heads+1})check(native::stage_heads(2,heads)==0,"invalid head budget");
    check(native::stage_heads(4,9)==8 && native::stage_heads(1024,64)==4,"head batching boundaries");
    Graph large(native::max_tokens,native::max_heads,native::max_tiles,1); // metadata ONLY, no huge activations
    Graph g(67,3,4,1);
    auto reject=[&](ggml_tensor* root,ggml_tensor* t,auto mutate) { auto save=*t;mutate(*t);bool rejected=!ed_cuda_h3_vsa_shape_supported(root);*t=save;check(rejected,"malformed shape accepted"); };
    for(auto* root:{g.route,g.out}) {
        reject(root,root,[](auto& t){t.op=GGML_OP_NONE;});reject(root,root,[](auto& t){t.ne[3]=2;});
        reject(root,root,[](auto& t){t.type=GGML_TYPE_F16;});
        reject(root,root,[](auto& t){t.nb[1]+=4;});reject(root,root,[](auto& t){t.src[0]=nullptr;});
        for(uint32_t tag:{0u,0x48335341u /* Sage H3SA */,edgedit::h3_subblock::router_tag,edgedit::h3_subblock::attention_tag,
                          edgedit::h3_subblock::indexed_attention_tag,ED_CUDA_H3_SAGE_FP8_MAGIC}) {
            check(tag!=native::router_tag && tag!=native::fine_tag,"VSA tag collision");
            reject(root,root,[&](auto& t){native::CustomParams p{};std::memcpy(&p,t.op_params,sizeof(p));p.userdata=reinterpret_cast<void*>(uintptr_t(tag));std::memcpy(t.op_params,&p,sizeof(p));});
        }
        reject(root,g.prefix,[](auto& t){t.type=GGML_TYPE_F32;});reject(root,g.prefix,[](auto& t){t.ne[0]=2;});
    }
    for(auto* t:{g.q,g.k,g.v,g.scores})reject(t==g.scores?g.route:g.out,t,[](auto& x){x.type=GGML_TYPE_F16;});
    for(auto* t:{g.q,g.k,g.v}) {
        reject(g.out,t,[](auto& x){x.ne[0]=64;});reject(g.out,t,[](auto& x){x.nb[0]=8;});
        reject(g.out,t,[](auto& x){x.nb[2]=0;});reject(g.out,t,[](auto& x){x.nb[2]=size_t(1)<<41;});
    }
    reject(g.out,g.out,[](auto& x){x.src[2]=x.src[0];});
    for(int64_t n:{int64_t(0),int64_t(native::max_tokens+1)})reject(g.out,g.q,[&](auto& x){x.ne[1]=n;});
    for(int64_t h:{int64_t(0),int64_t(native::max_heads+1)})reject(g.route,g.scores,[&](auto& x){x.ne[2]=h;});
    reject(g.route,g.scores,[](auto& x){x.ne[0]=native::max_tiles+1;});reject(g.route,g.scores,[](auto& x){x.ne[1]+=1;});
    for(auto* root:{g.route,g.out})for(int64_t topk:{int64_t(0),int64_t(5)})reject(root,g.route,[&](auto& x){x.ne[0]=topk;});
    for(auto* t:{g.route,g.slots,g.counts}) {
        reject(g.out,t,[](auto& x){x.type=GGML_TYPE_F32;});reject(g.out,t,[&](auto& x){x.nb[t==g.counts?0:1]+=4;});
    }
    for(int64_t h:{int64_t(0),int64_t(native::max_heads+1)})reject(g.out,g.q,[&](auto& x){x.ne[2]=h;});
    reject(g.out,g.route,[](auto& x){x.ne[2]+=1;});
    for(int64_t t:{int64_t(0),int64_t(native::max_tiles+1)})reject(g.out,g.slots,[&](auto& x){x.ne[1]=t;});
    reject(g.out,g.slots,[](auto& x){x.ne[0]=63;});reject(g.out,g.counts,[](auto& x){x.ne[0]+=1;});
    auto op=g.route->op;g.route->op=GGML_OP_NONE;check(ed_cuda_h3_vsa_shape_supported(g.out),"external route leaf rejected");g.route->op=op;
    // Valid channel-contiguous strided views; shape validation does not inspect device metadata values.
    for(auto* t:{g.q,g.k,g.v}) {
        auto save=*t;t->nb[1]+=16;t->nb[2]=t->nb[1]*t->ne[1];t->nb[3]=t->nb[2]*t->ne[2];
        check(ed_cuda_h3_vsa_shape_supported(g.out),"channel-contiguous strided input rejected");*t=save;
    }
}
static std::string env(const char* name) { const char* p=std::getenv(name);return p?p:""; }
static bool permit(int argc,char** argv) { // Kept identical to h3_subblock_test's guard.
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
        bool gpu=argc==2 && std::string(argv[1])=="--run";
        if((argc!=1 && !(argc==2 && std::string(argv[1])=="--cpu") && !gpu) || (gpu && !permit(argc,argv))) {
            std::cerr<<"NOT RUN: use --cpu or fresh guarded single-GPU --run\n";return 2;
        }
        ggml_backend_t backend=nullptr;
#ifndef ED_ENABLE_CUDA_H3_VSA
        if(gpu){std::cerr<<"NOT RUN: compile with ED_ENABLE_CUDA_H3_VSA and native CUDA DSO\n";return 2;}
#else
        if(gpu) {
            check(ggml_backend_cuda_get_device_count()==1,"exactly one visible CUDA device required");
            backend=ggml_backend_cuda_init(0);check(backend && ggml_backend_is_cuda(backend),"CUDA backend unavailable");
        }
#endif
        cpu_contracts();
        struct Case { ref::Grid grid;int prefix,heads,mode; };
        for(const auto& c:std::vector<Case>{{{1,1,1},0,1,0},{{4,4,4},0,2,0},{{5,5,5},67,3,0},
                                          {{5,5,5},3,2,1},{{1,1,8},3,2,2},{{1,1,8},3,2,3},{{1,1,9},3,9,0}})for(double keep:{0.,1.}) {
            Fixture f(c.grid,c.prefix,c.heads,keep);Graph graph(f.n,f.h,int(f.geo.tiles()),f.topk);if(gpu)graph.alloc(backend);
            check(f.topk==(keep==0?1:int(f.geo.tiles())-f.prefix),"zero/full keep policy");std::vector<float> previous;
            for(int replay=0;replay<2;++replay) {
                f.fill(971+replay,c.mode,replay);auto scores=f.scores();auto ids=f.routes(scores);auto expected=f.oracle(scores,ids);
                if(c.mode==1)for(size_t i=0;i<ids.size();++i)check(ids[i]==f.prefix+int(i%f.topk),"lower-ID tie policy");
                if(c.mode==2 && f.topk==1)for(int head=0;head<f.h;++head)for(int qt=f.prefix;qt<int(f.geo.tiles());++qt)
                    check(ids[head*f.geo.tiles()+qt]==f.prefix+((qt-f.prefix+head+replay)%2),"adversarial head/query winner");
                if(c.mode==3 && f.topk==1)for(int head=0;head<f.h;++head)for(size_t qt=0;qt<f.geo.tiles();++qt)
                    check(ids[head*f.geo.tiles()+qt]==f.prefix+((1+head+replay)%2),"F32 pooled score gap lost to BF16 rounding");
                auto dense=f.oracle(scores,ids,true);size_t prefix_values=size_t(c.prefix)*f.h*128;
                near({expected.begin(),expected.begin()+prefix_values},{dense.begin(),dense.begin()+prefix_values},0,0);
                if(keep==1)near(expected,dense,0,0);
                if(!previous.empty())check(previous!=expected,"replay did not change inputs/reference");
                previous=expected;
                if(gpu)graph.run(backend,f,scores,ids,expected);
            }
        }
        if(backend)ggml_backend_free(backend);
        std::cout<<(gpu?"CUDA actual":"CPU-only shape/planning/oracle")<<" VSA: checks="<<checks<<" comparisons="<<comparisons<<" PASS\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
