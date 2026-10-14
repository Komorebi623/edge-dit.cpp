// SPDX-License-Identifier: Apache-2.0
// FastH3 VSA: pooled-score routing and actual indexed BF16 QK/BF16 PV.
// No SubBlock subcell estimator, no whole-model quantization switch.
#include "h3_vsa.h"
#include "common.cuh"
#include "sage/mma.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>
namespace {
using namespace edgedit::h3_vsa;
constexpr int stride=136;
struct Input { const float* data; int64_t token_stride,head_stride; };
Input input(const ggml_tensor* t,bool v=false) {
    return {static_cast<const float*>(t->data),int64_t(t->nb[v?2:1]/4),int64_t(t->nb[v?1:2]/4)};
}
__device__ __forceinline__ float load(Input x,int d,int token,int head) {
    return x.data[int64_t(head)*x.head_stride+int64_t(token)*x.token_stride+d];
}
__device__ __forceinline__ void demand(bool value) { if(!value)asm volatile("trap;"); }
__device__ __forceinline__ bool better(float a,int ai,float b,int bi) { return a>b || (a==b && ai<bi); }
// Exact per-head/per-Q tile top-k over the video region only. Prefix keys are
// always retained, prefix queries bypass routes and visit every tile.
__global__ void route_rows(const float* scores,const int32_t* prefix,int32_t* out,int tiles,int topk,int width) {
    __shared__ float values[max_tiles];__shared__ int ids[max_tiles];
    const int row=blockIdx.x,p=*prefix,n=tiles-p;
    demand(p>=0 && p<tiles && topk<=n);
    for(int i=threadIdx.x;i<width;i+=blockDim.x) {
        const float value=i<n?scores[int64_t(row)*tiles+p+i]:-CUDART_INF_F;
        demand(i>=n || isfinite(value));values[i]=value;ids[i]=p+i;
    }
    __syncthreads();
    for(int size=2;size<=width;size*=2)for(int dist=size/2;dist;dist/=2) {
        for(int i=threadIdx.x;i<width;i+=blockDim.x) {
            int j=i^dist;if(j>i) {
                bool swap=(i&size)==0?better(values[j],ids[j],values[i],ids[i]):better(values[i],ids[i],values[j],ids[j]);
                if(swap) { float v=values[i];values[i]=values[j];values[j]=v;int id=ids[i];ids[i]=ids[j];ids[j]=id; }
            }
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<topk;i+=blockDim.x)out[int64_t(row)*topk+i]=ids[i];
}
__global__ void pack(Input k,Input v,const int32_t* slots,const int32_t* counts,nv_bfloat16* pk,nv_bfloat16* pv,
                     int tokens,int tiles,int first_head,int heads) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(heads)*tiles*64*dim;
    if(i>=n)return;
    const int d=i%dim,row=(i/dim)%(tiles*64),h=first_head+int(i/(int64_t(tiles)*64*dim));
    const int count=counts[row/64],token=slots[row];demand(count>0&&count<=64);
    demand(row%64<count ? token>=0&&token<tokens : token==-1);
    pk[i]=__float2bfloat16_rn(token>=0?load(k,d,token,h):0.f);
    pv[i]=__float2bfloat16_rn(token>=0?load(v,d,token,h):0.f);
}
struct Params {
    Input q; const nv_bfloat16 *k,*v;const int32_t *routes,*slots,*counts,*prefix;
    float* out;int tokens,tiles,heads,topk,first_head;
};
__device__ __forceinline__ uint32_t pair(float a,float b) {
    return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
           (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b)))<<16);
}
// PTX fragment layout agrees with the existing Apache-2.0 Sage/FlashInfer
// ldmatrix wrappers. Only the BF16 operand type differs; accumulators are F32.
__device__ __forceinline__ void mm(float* c,const uint32_t* a,const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(c[4]),"=f"(c[5]),"=f"(c[6]),"=f"(c[7])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[2]),"r"(b[3]),"f"(c[4]),"f"(c[5]),"f"(c[6]),"f"(c[7]));
#endif
}
__device__ __forceinline__ float row_max(float x) { x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1));return fmaxf(x,__shfl_xor_sync(0xffffffff,x,2)); }
__device__ __forceinline__ float row_sum(float x) { x+=__shfl_xor_sync(0xffffffff,x,1);return x+__shfl_xor_sync(0xffffffff,x,2); }
__device__ __forceinline__ int key_tile(const Params& p,int query,int iter) {
    int pre=*p.prefix;demand(pre>=0&&pre<p.tiles&&p.topk<=p.tiles-pre);
    int id=query<pre || iter<pre ? iter : p.routes[(int64_t(blockIdx.y+p.first_head)*p.tiles+query)*p.topk+iter-pre];
    demand(id>=0&&id<p.tiles);return id;
}
template<bool Value>
__device__ __forceinline__ void prefetch(const Params& p,int key,nv_bfloat16* shared) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const nv_bfloat16* source=(Value?p.v:p.k)+(int64_t(blockIdx.y)*p.tiles+key)*64*dim;
    for(int i=threadIdx.x;i<64*dim/8;i+=128) {
        const int row=i/16,d=(i%16)*8;
        unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(shared+row*stride+d));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(address),"l"(source+row*dim+d));
    }
    asm volatile("cp.async.commit_group;");
#endif
}
__global__ __launch_bounds__(128,2) void fine_attention(Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) nv_bfloat16 storage[2*64*stride];
    auto* sk=storage;auto* sv=storage+64*stride;
    const int qtile=blockIdx.x,h=blockIdx.y+p.first_head,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int qcount=p.counts[qtile],pre=*p.prefix;demand(qcount>0&&qcount<=64);
    for(int i=threadIdx.x;i<64*dim;i+=128) {
        int row=i/dim,d=i%dim,token=p.slots[qtile*64+row];
        demand(row<qcount ? token>=0&&token<p.tokens : token==-1);
        sk[row*stride+d]=__float2bfloat16_rn(token>=0?load(p.q,d,token,h):0.f);
    }
    __syncthreads();uint32_t qr[8][4];
#pragma unroll
    for(int d=0;d<8;++d)mma::ldmatrix_m8n8x4(qr[d],sk+(warp*16+lane%16)*stride+d*16+(lane/16)*8);
    __syncthreads();
    float out[8][8]={},maximum[2]={-CUDART_INF_F,-CUDART_INF_F},denom[2]={};
    const int iterations=qtile<pre?p.tiles:pre+p.topk;
    int first=key_tile(p,qtile,0);prefetch<false>(p,first,sk);prefetch<true>(p,first,sv);
    for(int iter=0;iter<iterations;++iter) {
        asm volatile("cp.async.wait_group 1;");__syncthreads();
        int key=key_tile(p,qtile,iter),valid=p.counts[key];demand(valid>0&&valid<=64);
        float scores[4][8]={};
#pragma unroll
        for(int d=0;d<8;++d) {
#pragma unroll
            for(int kt=0;kt<4;++kt) {
                uint32_t kr[4];
                mma::ldmatrix_m8n8x4(kr,sk+(kt*16+lane%8+(lane/16)*8)*stride+d*16+((lane/8)%2)*8);
                mm(scores[kt],qr[d],kr);
            }
        }
        __syncthreads();if(iter+1<iterations)prefetch<false>(p,key_tile(p,qtile,iter+1),sk);
#pragma unroll
        for(int kt=0;kt<4;++kt) {
#pragma unroll
            for(int e=0;e<8;++e) {
                int col=kt*16+(lane%4)*2+e%2+(e/4)*8;
                scores[kt][e]=col<valid?scores[kt][e]*0.08838834764831844f:-CUDART_INF_F;
            }
        }
#pragma unroll
        for(int row=0;row<2;++row) {
            float next=maximum[row];
#pragma unroll
            for(int kt=0;kt<4;++kt)next=fmaxf(next,fmaxf(fmaxf(scores[kt][2*row],scores[kt][2*row+1]),fmaxf(scores[kt][2*row+4],scores[kt][2*row+5])));
            next=row_max(next);const float alpha=expf(maximum[row]-next);float sum=0;
#pragma unroll
            for(int kt=0;kt<4;++kt) {
#pragma unroll
                for(int part=0;part<2;++part) {
                    int e=2*row+part*4;scores[kt][e]=expf(scores[kt][e]-next);scores[kt][e+1]=expf(scores[kt][e+1]-next);
                    sum+=scores[kt][e]+scores[kt][e+1];
                }
            }
            denom[row]=denom[row]*alpha+row_sum(sum);maximum[row]=next;
#pragma unroll
            for(int d=0;d<8;++d){out[d][2*row]*=alpha;out[d][2*row+1]*=alpha;out[d][2*row+4]*=alpha;out[d][2*row+5]*=alpha;}
        }
        if(iter+1<iterations)asm volatile("cp.async.wait_group 1;");else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
#pragma unroll
        for(int kt=0;kt<4;++kt) {
            uint32_t prob[4]={pair(scores[kt][0],scores[kt][1]),pair(scores[kt][2],scores[kt][3]),pair(scores[kt][4],scores[kt][5]),pair(scores[kt][6],scores[kt][7])};
#pragma unroll
            for(int d=0;d<8;++d) {
                uint32_t vr[4];
                mma::ldmatrix_m8n8x4_trans(vr,sv+(kt*16+lane%16)*stride+d*16+(lane/16)*8);
                mm(out[d],prob,vr);
            }
        }
        __syncthreads();if(iter+1<iterations)prefetch<true>(p,key_tile(p,qtile,iter+1),sv);
    }
#pragma unroll
    for(int d=0;d<8;++d) {
#pragma unroll
        for(int e=0;e<8;++e) {
            const int row=warp*16+lane/4+((e%4)/2)*8;
            if(row<qcount) { int token=p.slots[qtile*64+row],channel=d*16+(lane%4)*2+e%2+(e/4)*8;
                p.out[(int64_t(token)*p.heads+h)*dim+channel]=out[d][e]/denom[(e%4)/2]; }
        }
    }
#else
    asm volatile("trap;");
#endif
}
}
bool ed_cuda_h3_vsa_supported(int device,const ggml_tensor* dst) {
    return ed_cuda_h3_vsa_shape_supported(dst) && device>=0 && device<ggml_cuda_info().device_count &&
        GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[device].cc) && ggml_cuda_info().devices[device].cc>=GGML_CUDA_CC_AMPERE &&
        ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[device].cc)>=GGML_CUDA_CC_AMPERE;
}
bool ed_cuda_h3_vsa_compute(ggml_backend_cuda_context& ctx,ggml_tensor* dst) {
    if(!ed_cuda_h3_vsa_supported(ctx.device,dst))return false;
    ggml_cuda_set_device(ctx.device);auto stream=ctx.stream();
    if(tag(dst)==router_tag) {
        int tiles=dst->ne[1],heads=dst->ne[2],width=1;while(width<tiles)width*=2;
        route_rows<<<tiles*heads,256,0,stream>>>(static_cast<const float*>(dst->src[0]->data),static_cast<const int32_t*>(dst->src[1]->data),static_cast<int32_t*>(dst->data),tiles,dst->ne[0],width);
    } else {
        auto* q=dst->src[0];int tiles=dst->src[4]->ne[1],heads=q->ne[2],batch=stage_heads(tiles,heads);
        size_t n=size_t(batch)*tiles*64*dim;ggml_cuda_pool_alloc<nv_bfloat16> pk(ctx.pool(),n),pv(ctx.pool(),n);
        Params p{input(q),pk.get(),pv.get(),static_cast<const int32_t*>(dst->src[3]->data),static_cast<const int32_t*>(dst->src[4]->data),
            static_cast<const int32_t*>(dst->src[5]->data),static_cast<const int32_t*>(dst->src[6]->data),static_cast<float*>(dst->data),
            int(q->ne[1]),tiles,heads,int(dst->src[3]->ne[0]),0};
        for(int begin=0;begin<heads;begin+=batch) {
            int count=std::min(batch,heads-begin);p.first_head=begin;int64_t elements=int64_t(count)*tiles*64*dim;
            pack<<<(elements+255)/256,256,0,stream>>>(input(dst->src[1]),input(dst->src[2],true),p.slots,p.counts,pk.get(),pv.get(),p.tokens,tiles,begin,count);
            fine_attention<<<dim3(tiles,count),128,0,stream>>>(p);
        }
    }
    CUDA_CHECK(cudaGetLastError());return true;
}
