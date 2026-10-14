// Device-free full native graph assembly: no kernels execute or devices load.
#include "dit_models/models/minimax_h3_vsa_native.hpp"
#include <iostream>
#include <set>
#include <stdexcept>
namespace V=MiniMaxH3::VSAReference;
namespace G=MiniMaxH3::VSAGraph;
namespace N=edgedit::h3_vsa;
static void check(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main()try {
    namespace Contract = MiniMaxH3::VSAContract;
    check(Contract::video_topk(10)==1 && Contract::video_topk(100)==10 && Contract::video_topk(11)==2,
          "exact V1 quota must not round .9f into an extra block");
    auto keys=Contract::ordinary_keys();check(keys.size()==598,"258pairs+82deltas required");
    Contract::validate_ordinary(keys);
    keys.erase("model.diffusion_model.blocks.3.norm1.weight.delta");bool missing=false;
    try{Contract::validate_ordinary(keys);}catch(const std::invalid_argument&){missing=true;}
    check(missing,"runtime must reject missing trained norm even if caller rehashes file");
    keys.insert("model.diffusion_model.blocks.49.norm1.weight.delta");missing=false;
    try{Contract::validate_ordinary(keys);}catch(const std::invalid_argument&){missing=true;}
    check(missing,"equal tensor count cannot hide a replaced target");
    ggml_context* ctx=ggml_init({16u<<20,nullptr,true});check(ctx,"context");
    auto geometry=V::make_geometry({V::Segment::text(67),V::Segment::audio(5),V::Segment::video({5,5,5})},64);
    const int64_t s=geometry.packed_tokens,h=3,d=128;
    auto* q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,s,h);
    auto* k=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,s,h);
    auto* v=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,h,s);
    auto* gate=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,d*h,s);
    G::GraphInputs inputs;std::vector<ggml_tensor*> constants;
    inputs.bind_backend_tensor_data=[&](ggml_tensor* t,const void*){constants.push_back(t);};
    auto* out=MiniMaxH3::VSANative::build(ctx,q,k,v,gate,geometry,inputs,.9,1u<<28);
    auto* graph=ggml_new_graph_custom(ctx,512,false);ggml_build_forward_expand(graph,out);
    int routes=0,fine=0;ggml_tensor* routing=nullptr;ggml_tensor* attention=nullptr;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto* t=ggml_graph_node(graph,i);
        if(N::tag(t)==N::router_tag){routes++;routing=t;check(ed_cuda_h3_vsa_shape_supported(t),"router shape");}
        if(N::tag(t)==N::fine_tag){fine++;attention=t;check(ed_cuda_h3_vsa_shape_supported(t),"fine shape");}
    }
    check(routes==1&&fine==1&&routing&&attention,"one true router and fine node");
    check(attention->src[3]==routing,"fine consumes actual route");
    check(routing->src[0]->type==GGML_TYPE_F32,"full pooled scores F32");
    check(ggml_graph_n_nodes(graph)<80,"bounded node count");
    check(out->ne[0]==d*h&&out->ne[1]==s,"packed output shape");
    check(constants.size()==6 && inputs.buffers.size()==6,"all metadata owned through execution");
    auto bad=geometry;bad.video_tile_spans.push_back({0,1});bool rejected=false;
    try{MiniMaxH3::VSANative::build(ctx,q,k,v,gate,bad,inputs,.9,1u<<28);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"multiple regions cannot silently use single-region kernel");
    std::cout<<"VSA assembly PASS nodes="<<ggml_graph_n_nodes(graph)<<"; no CUDA run\n";
    ggml_free(ctx);return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
