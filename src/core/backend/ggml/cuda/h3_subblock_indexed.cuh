#pragma once
// SPDX-License-Identifier: Apache-2.0
// H3-only Q64/K64 indexed attention, independently implemented around the
// existing Apache-2.0 SageAttention/FlashInfer MMA wrappers in sage/mma.cuh.
// The wrappers retain their original copyright/license notices unchanged.
// Algorithm/layout references: sage/attn_utils.cuh (Apache-2.0) and fixed ggml
// fattn-mma-f16.cuh (MIT). No Sage INT8/FP8 quantization is used here.
#include "h3_subblock.h"
#include "sage/mma.cuh"
#include <math_constants.h>

namespace edgedit::h3_subblock::indexed {
constexpr int stride = 136; // +8 half padding: ldmatrix row addresses avoid bank aliasing
constexpr size_t shared_bytes = 2*64*stride*sizeof(half); // 34 KiB, no global staging
static_assert(shared_bytes==indexed_shared_bytes,"shared allocation contract");
struct Input { const float* data; int64_t token_stride, head_stride; };
struct Params {
    Input q, k, v;
    const int32_t *blocks, *route, *protected_chunks, *protected_ids;
    float* output;
    int sequence, heads, video_blocks, topk, protected_count, protected_chunk_count;
    const half *packed_k = nullptr, *packed_v = nullptr;
    int head_begin = 0;
};
static __device__ __forceinline__ float load(Input x,int d,int token,int head) {
    return x.data[int64_t(head)*x.head_stride+int64_t(token)*x.token_stride+d];
}
static __device__ __forceinline__ uint32_t pair(float a,float b) {
    return uint32_t(__half_as_ushort(__float2half_rn(a))) |
           (uint32_t(__half_as_ushort(__float2half_rn(b))) << 16);
}
// m16n16 accumulator = two m16n8 fragments. Each four-thread group owns two
// query rows (group,group+8); each lane owns two columns in each n8 half.
// This is the PTX m16n8 layout used by the pinned Sage wrappers, NOT an opaque
// WMMA fragment layout assumption.
static __device__ __forceinline__ float row_max(float x) {
    x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,1));
    return fmaxf(x,__shfl_xor_sync(0xffffffffu,x,2));
}
static __device__ __forceinline__ float row_sum(float x) {
    x+=__shfl_xor_sync(0xffffffffu,x,1);
    return x+__shfl_xor_sync(0xffffffffu,x,2);
}

// One async group per K or V tile. K(next) overlaps softmax/PV(current),
// V(next) overlaps QK(next), with only the original two shared-memory tiles.
// wait_group<1> waits the older group, never a tile still consumed by a warp.
template<bool Sparse, bool Value>
static __device__ __forceinline__ void prefetch_tile(const Params& p, int block, int iter, half* shared) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int begin=0,end=0;
    if constexpr(Sparse) if(iter<p.topk) {
        const int id=p.route[(int64_t(blockIdx.y+p.head_begin)*p.video_blocks+block)*p.topk+iter];
        if(id<0 || id>=p.video_blocks) asm volatile("trap;");
        begin=p.blocks[2*id];end=p.blocks[2*id+1];
    }
    for(int i=threadIdx.x;i<64*128/8;i+=128) {
        const int row=i/16,d=(i%16)*8;
        int token;
        if constexpr(Sparse) {
            if(iter<p.topk) token=begin+row<end?begin+row:-1;
            else {
                const int index=(iter-p.topk)*64+row;
                token=index<p.protected_count?p.protected_ids[index]:-1;
            }
        } else token=iter*64+row;
        const bool valid=token>=0 && token<p.sequence;
        const half* source=(Value?p.packed_v:p.packed_k)+
            (int64_t(blockIdx.y)*p.sequence+(valid?token:0))*128+d;
        const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(shared+row*stride+d));
        const int bytes=valid?16:0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(address),"l"(source),"r"(bytes));
    }
    asm volatile("cp.async.commit_group;");
#endif
}
static __device__ __forceinline__ void wait_older_tile() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;");
#endif
}
static __device__ __forceinline__ void wait_all_tiles() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}

// Sparse CTA: (one actual video Q64, one head), reading ONLY its topk K64
// blocks, then all protected keys. Dense CTA is ONLY for protected Q chunks.
// One online-softmax state spans every selected/protected tile. No full-N mask,
// no per-Q global K/V copy, no route download or output missing query rows.
template<bool Sparse>
__global__ __launch_bounds__(128,2) void attention(Params p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) half storage[2*64*stride];
    half* sk=storage;
    half* sv=storage+64*stride;
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const int block=blockIdx.x, head=blockIdx.y+p.head_begin;
    const int32_t* spans=Sparse?p.blocks:p.protected_chunks;
    const int first=spans[2*block],last=spans[2*block+1];
    // Same prep order as pinned stock ggml MMA: half(Q) * half(scale),
    // rounded to half. K/V are half(input/128); output is compensated by 128.
    const half qscale=__float2half_rn(128.0f/sqrtf(128.0f));
    for(int i=threadIdx.x;i<64*128;i+=128) {
        const int row=i/128,d=i%128,t=first+row;
        const float value=t>=0 && t<last && t<p.sequence ? load(p.q,d,t,head):0;
        sk[row*stride+d]=__hmul(__float2half_rn(value),qscale);
    }
    __syncthreads();
    uint32_t qreg[8][4];
#pragma unroll
    for(int d=0;d<8;++d) {
        // x4 order: Q top-left,bottom-left,top-right,bottom-right.
        mma::ldmatrix_m8n8x4(qreg[d],sk+indexed_q_row(warp,lane)*stride+indexed_q_col(d,lane));
    }
    // Q remains in registers. Reuse its shared tile as K for every key block.
    __syncthreads();
    float out[8][8]={};
    float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},denominator[2]={0,0};
    const int iterations=Sparse ? p.topk+(p.protected_count+63)/64 : (p.sequence+63)/64;
    prefetch_tile<Sparse,false>(p,block,0,sk);
    prefetch_tile<Sparse,true>(p,block,0,sv);
    for(int iter=0;iter<iterations;++iter) {
        wait_older_tile(); // K(current) is older than V(current).
        __syncthreads();
        int key_begin=0,key_end=0;
        if constexpr(Sparse) {
            if(iter<p.topk) {
                const int id=p.route[(int64_t(head)*p.video_blocks+block)*p.topk+iter];
                // A corrupt route is a hard device error, never a gather/dense
                // fallback or silent omission of a selected block.
                if(id<0 || id>=p.video_blocks) asm volatile("trap;");
                key_begin=p.blocks[2*id];key_end=p.blocks[2*id+1];
            }
        }
        float scores[4][8]={}; // four K16 blocks, all 64 keys for 16 query rows
#pragma unroll
        for(int d=0;d<8;++d) {
#pragma unroll
            for(int kt=0;kt<4;++kt) {
                uint32_t kr[4];
                // K is [token,d]; B is K^T. x4 must group both d8 pieces
                // of tokens0..7 before both d8 pieces of tokens8..15.
                mma::ldmatrix_m8n8x4(kr,sk+indexed_k_row(kt,lane)*stride+indexed_k_col(d,lane));
                mma::mma_sync_m16n16k16_row_col_f16f16f32(scores[kt],qreg[d],kr);
            }
        }
        __syncthreads(); // K(current) must be dead in every warp before reuse.
        if(iter+1<iterations) prefetch_tile<Sparse,false>(p,block,iter+1,sk);
        // Mask real K64 tails (including a partial packed protected tile).
#pragma unroll
        for(int kt=0;kt<4;++kt) {
#pragma unroll
            for(int e=0;e<8;++e) {
                const int key=kt*16+indexed_acc_col(lane,e);
                bool valid;
                if constexpr(Sparse) valid=iter<p.topk ? key_begin+key<key_end : (iter-p.topk)*64+key<p.protected_count;
                else valid=iter*64+key<p.sequence;
                if(!valid)scores[kt][e]=-CUDART_INF_F;
            }
        }
#pragma unroll
        for(int r=0;r<2;++r) {
            float next=maximum[r];
#pragma unroll
            for(int kt=0;kt<4;++kt) {
                next=fmaxf(next,fmaxf(fmaxf(scores[kt][2*r],scores[kt][2*r+1]),
                                    fmaxf(scores[kt][2*r+4],scores[kt][2*r+5])));
            }
            next=row_max(next);
            const float alpha=expf(maximum[r]-next);
            float total=0;
#pragma unroll
            for(int kt=0;kt<4;++kt) {
#pragma unroll
                for(int part=0;part<2;++part) {
                    const int e=2*r+part*4;
                    scores[kt][e]=expf(scores[kt][e]-next);
                    scores[kt][e+1]=expf(scores[kt][e+1]-next);
                    total+=scores[kt][e]+scores[kt][e+1];
                }
            }
            denominator[r]=denominator[r]*alpha+row_sum(total);
            maximum[r]=next;
#pragma unroll
            for(int d=0;d<8;++d) {
                out[d][2*r]*=alpha;out[d][2*r+1]*=alpha;
                out[d][2*r+4]*=alpha;out[d][2*r+5]*=alpha;
            }
        }
        if(iter+1<iterations) wait_older_tile(); // V(current), not K(next).
        else wait_all_tiles();
        __syncthreads();
        // Convert normalized numerator weights to F16 as an A fragment.
        // Denominator and running output remain FP32 (no F16 output accumulator).
#pragma unroll
        for(int kt=0;kt<4;++kt) {
            uint32_t probability[4]={pair(scores[kt][0],scores[kt][1]),
                                     pair(scores[kt][2],scores[kt][3]),
                                     pair(scores[kt][4],scores[kt][5]),
                                     pair(scores[kt][6],scores[kt][7])};
#pragma unroll
            for(int d=0;d<8;++d) {
                uint32_t vr[4];
                // V [key,d] is B [K,N], requiring transposed ldmatrix.
                mma::ldmatrix_m8n8x4_trans(vr,sv+indexed_v_row(kt,lane)*stride+indexed_v_col(d,lane));
                mma::mma_sync_m16n16k16_row_col_f16f16f32(out[d],probability,vr);
            }
        }
        __syncthreads(); // V(current) must be dead in every warp before reuse.
        if(iter+1<iterations) prefetch_tile<Sparse,true>(p,block,iter+1,sv);
    }
#pragma unroll
    for(int d=0;d<8;++d) {
#pragma unroll
        for(int e=0;e<8;++e) {
            const int row=warp*16+indexed_acc_row(lane,e),token=first+row;
            if(token>=0 && token<last && token<p.sequence) {
                const int channel=d*16+indexed_acc_col(lane,e),r=(e%4)/2;
                p.output[(int64_t(token)*p.heads+head)*128+channel]=out[d][e]*(128.f/denominator[r]);
            }
        }
    }
#else
    // This specialization requires native F16 MMA m16n8k16 (SM80+). Host
    // capability gating rejects older GPUs; never substitute another method.
    asm volatile("trap;");
#endif
}
} // namespace edgedit::h3_subblock::indexed
