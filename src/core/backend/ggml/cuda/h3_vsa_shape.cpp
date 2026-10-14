#include "h3_vsa.h"
namespace {
using namespace edgedit::h3_vsa;
bool matrix(const ggml_tensor* t,ggml_type type,int64_t a,int64_t b=1) {
    return t && t->type==type && t->ne[0]==a && t->ne[1]==b && t->ne[2]==1 && t->ne[3]==1 && ggml_is_contiguous(t);
}
bool qshape(const ggml_tensor* t) {
    if(!t || t->type!=GGML_TYPE_F32 || t->ne[0]!=dim || t->ne[1]<1 || t->ne[1]>max_tokens ||
       t->ne[2]<1 || t->ne[2]>max_heads || t->ne[3]!=1 || t->nb[0]!=4)return false;
    for(auto n:t->nb)if(!n || n%4 || n>(size_t(1)<<40))return false;
    return true;
}
}
bool ed_cuda_h3_vsa_shape_supported(const ggml_tensor* dst) {
    using namespace edgedit::h3_vsa;
    if(!dst || !ggml_is_contiguous(dst))return false;
    if(tag(dst)==router_tag) {
        const auto* s=dst->src[0];
        return s && s->type==GGML_TYPE_F32 && s->ne[0]>=1 && s->ne[0]<=max_tiles && s->ne[1]==s->ne[0] &&
               s->ne[2]>=1 && s->ne[2]<=max_heads && s->ne[3]==1 && ggml_is_contiguous(s) &&
               matrix(dst->src[1],GGML_TYPE_I32,1) && dst->type==GGML_TYPE_I32 &&
               dst->ne[0]>=1 && dst->ne[0]<=s->ne[0] && dst->ne[1]==s->ne[1] && dst->ne[2]==s->ne[2] && dst->ne[3]==1;
    }
    if(tag(dst)!=fine_tag)return false;
    const auto *q=dst->src[0],*k=dst->src[1],*v=dst->src[2],*route=dst->src[3],*slots=dst->src[4];
    if(!qshape(q)||!qshape(k)||!ggml_are_same_shape(q,k)||!v||v->type!=GGML_TYPE_F32||v->ne[0]!=dim||
       v->ne[1]!=q->ne[2]||v->ne[2]!=q->ne[1]||v->ne[3]!=1||v->nb[0]!=4)return false;
    for(auto n:v->nb)if(!n || n%4 || n>(size_t(1)<<40))return false;
    if(!slots || slots->ne[1]<1 || slots->ne[1]>max_tiles || !matrix(slots,GGML_TYPE_I32,64,slots->ne[1]))return false;
    const auto n=slots->ne[1];
    return route && route->type==GGML_TYPE_I32 && ggml_is_contiguous(route) && route->ne[0]>=1 && route->ne[0]<=n &&
        route->ne[1]==n && route->ne[2]==q->ne[2] && route->ne[3]==1 &&
        matrix(dst->src[5],GGML_TYPE_I32,n) && matrix(dst->src[6],GGML_TYPE_I32,1) &&
        dst->type==GGML_TYPE_F32 && dst->ne[0]==dim*q->ne[2] && dst->ne[1]==q->ne[1] && dst->ne[2]==1 && dst->ne[3]==1;
}
