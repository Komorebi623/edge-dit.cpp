#pragma once
// SPDX-License-Identifier: Apache-2.0
// Explicit native VSA graph assembly; model runtime admission is separate.
// One generated-video region with text/audio/keyframes exempt from sparsity.
#include "minimax_h3_vsa_graph.hpp"
#include "minimax_h3_vsa_contract.hpp"
#include <map>
#include <string>
#include "core/backend/ggml/cuda/h3_vsa.h"
namespace MiniMaxH3 { namespace VSANative {
// Model-owned parameters, request-graph-owned constants; never shared globally.
struct State {
    std::vector<ggml_tensor*> gates;
    std::map<std::string,ggml_tensor*> gate_tensors;
    VSAReference::Geometry geometry;
    VSAGraph::GraphInputs inputs;
    double sparsity = .9;
    size_t graph_budget = size_t(12) << 30;
    void reset_graph() { inputs.buffers.clear(); geometry = {}; }
};
inline ggml_tensor* build(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,
                          ggml_tensor* raw_gate,const VSAReference::Geometry& geometry,
                          VSAGraph::GraphInputs& inputs,double sparsity,size_t budget_bytes) {
    namespace Native = edgedit::h3_vsa;
    VSAGraph::require(q && q->ne[0]==128 && q->ne[2]<=Native::max_heads &&
        geometry.tiles()<=Native::max_tiles && geometry.packed_tokens<=Native::max_tokens,
        "VSA native supports D128 and bounded tile64 geometry");
    VSAReference::validate_geometry(geometry);
    VSAGraph::require(geometry.video_tile_spans.size()==1 && geometry.tile_elems==64 &&
        geometry.video_tile_spans[0].first==geometry.num_prefix_tiles && geometry.video_tile_spans[0].second==geometry.tiles(),
        "VSA native currently requires one target video region and an exempt prefix");
    VSAGraph::require(std::isfinite(sparsity) && sparsity>=0.f && sparsity<1.f,"invalid native VSA sparsity");
    std::vector<int32_t> slots,counts;
    slots.reserve(geometry.source_index.size());counts.reserve(geometry.tiles());
    for(size_t index:geometry.source_index)slots.push_back(index==VSAReference::invalid_token?-1:int32_t(index));
    for(size_t n:geometry.variable_block_sizes)counts.push_back(int32_t(n));
    auto* map=VSAGraph::detail::constant(ctx,inputs,GGML_TYPE_I32,64,int64_t(geometry.tiles()),slots);
    auto* valid=VSAGraph::detail::constant(ctx,inputs,GGML_TYPE_I32,int64_t(counts.size()),1,counts);
    auto* prefix=VSAGraph::detail::constant(ctx,inputs,GGML_TYPE_I32,1,1,std::vector<int32_t>{int32_t(geometry.num_prefix_tiles)});
    const size_t video_tiles=geometry.tiles()-geometry.num_prefix_tiles;
    const int topk=int(sparsity==.9?VSAContract::video_topk(video_tiles):VSAReference::compute_topk(sparsity,video_tiles));
    // Apply the same explicit BF16 operand rounding to routing/compression and
    // fine Q/K/V, rather than rounding only inside the fine kernel.
    auto round=[&](ggml_tensor* t){return ggml_cast(ctx,ggml_cast(ctx,t,GGML_TYPE_BF16),GGML_TYPE_F32);};
    q=round(q);k=round(k);v=round(v);
    auto result=VSAGraph::build_with_fine(ctx,q,k,v,raw_gate,[&](ggml_tensor* scores){
        auto* route=Native::router(ctx,scores,prefix,topk);
        auto* fine=Native::fine(ctx,q,k,v,route,map,valid,prefix);
        VSAGraph::require(ed_cuda_h3_vsa_shape_supported(route) && ed_cuda_h3_vsa_shape_supported(fine),
                          "VSA native op shape not supported");
        return fine;
    },geometry,inputs,budget_bytes);
    return result.output;
}
}} // namespace MiniMaxH3::VSANative
