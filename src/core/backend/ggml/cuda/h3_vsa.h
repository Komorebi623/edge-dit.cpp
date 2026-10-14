#pragma once
// SPDX-License-Identifier: Apache-2.0
// Independent FastH3 tile-64 VSA fine attention. This is NOT SubBlock routing.
// The graph supplies F32 pooled-Q/K scores and true 3D tile gather metadata.
// Runtime VSA remains gated separately until complete model validation.
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

struct ggml_backend_cuda_context;
namespace edgedit::h3_vsa {
constexpr uint32_t router_tag=0x48335652u, fine_tag=0x48335646u; // H3VR/H3VF
constexpr int dim=128, tile=64, max_tiles=1024, max_heads=64, max_tokens=65536;
constexpr size_t staging_limit=size_t(128)<<20;
struct CustomParams { ggml_custom_op_t fun; int n_tasks; void* userdata; };
inline uint32_t tag(const ggml_tensor* t) {
    if(!t || t->op!=GGML_OP_CUSTOM)return 0;
    CustomParams p{};static_assert(sizeof(p)<=GGML_MAX_OP_PARAMS,"CUSTOM ABI");
    std::memcpy(&p,t->op_params,sizeof(p));return uint32_t(reinterpret_cast<uintptr_t>(p.userdata));
}
inline int stage_heads(int tiles,int heads) {
    if(tiles<1 || tiles>max_tiles || heads<1 || heads>max_heads)return 0;
    return std::min(heads,std::min(8,int(staging_limit/(size_t(tiles)*tile*dim*2*sizeof(uint16_t)))));
}
inline void no_cpu(ggml_tensor*,int,int,void*) { GGML_ABORT("H3 VSA CUDA op cannot execute on CPU"); }
// scores F32[Ktiles,Qtiles,H], prefix I32[1]. One target-video region after
// dense prefix (T2VA/FL2VA). Result contains absolute video-K tile IDs per Q/H.
inline ggml_tensor* router(ggml_context* c,ggml_tensor* scores,ggml_tensor* prefix,int topk) {
    ggml_tensor* args[]={scores,prefix};
    auto* t=ggml_custom_4d(c,GGML_TYPE_I32,topk,scores->ne[1],scores->ne[2],1,args,2,no_cpu,1,
                         reinterpret_cast<void*>(uintptr_t(router_tag)));
    ggml_set_name(t,"h3.vsa.perhead.pertile.topk");return t;
}
// Q/K F32[D,S,H], V F32[D,H,S]; slots I32[64,T] uses -1 for padding;
// counts I32[T]. All Q rows are updated and scattered back to original order.
inline ggml_tensor* fine(ggml_context* c,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,
                         ggml_tensor* routes,ggml_tensor* slots,ggml_tensor* counts,ggml_tensor* prefix) {
    ggml_tensor* args[]={q,k,v,routes,slots,counts,prefix};
    auto* t=ggml_custom_4d(c,GGML_TYPE_F32,dim*q->ne[2],q->ne[1],1,1,args,7,no_cpu,1,
                         reinterpret_cast<void*>(uintptr_t(fine_tag)));
    ggml_set_name(t,"h3.vsa.tile64.bf16.fine");return t;
}
}
bool ed_cuda_h3_vsa_shape_supported(const ggml_tensor* dst);
bool ed_cuda_h3_vsa_supported(int device,const ggml_tensor* dst);
bool ed_cuda_h3_vsa_compute(ggml_backend_cuda_context& ctx,ggml_tensor* dst);
