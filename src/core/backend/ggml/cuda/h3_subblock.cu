#include "h3_subblock.h"
#include "common.cuh"
#include "fattn.cuh"
#include <math_constants.h>
#include "h3_subblock_indexed.cuh"

// Bounded batching, not a dense LxL mask or per-head/per-Q expanded ggml graph.
// Uses stock ggml FlashAttention with the baseline's K/V F16 * (1/128) prep.
// Only the router and gather/scatter kernels below are new. Pooling stays F32
// to match this project's original CPU SubBlock estimator, not SGLang's BF16
// pool rounding. No model weights, global flags, cache or default paths change.
namespace {
using namespace edgedit::h3_subblock;
struct Input { const float* p; int64_t token_stride, head_stride; };
Input input(const ggml_tensor* t, bool v = false) {
    return {static_cast<const float*>(t->data), int64_t(t->nb[v ? 2 : 1]/4), int64_t(t->nb[v ? 1 : 2]/4)};
}
__device__ float read(Input x, int d, int t, int h) {
    return x.p[int64_t(h)*x.head_stride + int64_t(t)*x.token_stride + d];
}
__global__ void pool_cells(Input q, Input k, const int32_t* blocks, float* pq, float* pk, int b, int sequence) {
    const int cell = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int begin = blocks[2*(cell/4)], end = blocks[2*(cell/4)+1];
    float sq = 0, sk = 0;
    const int first = begin + (cell%4)*16;
    const int last = min(end, first+16);
    const bool valid = begin >= 0 && end > begin && end <= sequence && end-begin <= 64;
    if (valid) for (int t = first; t < last; ++t) { sq += read(q,d,t,h); sk += read(k,d,t,h); }
    const int n = max(0,last-first);
    const int64_t i = (int64_t(h)*b*4+cell)*dimension+d;
    pq[i] = valid ? (n ? sq/n : 0) : CUDART_NAN_F;
    pk[i] = valid ? (n ? sk/n : 0) : CUDART_NAN_F;
}
// Bounded head-major K/V packing. Values follow the old indexed kernel's
// exact F32*(1/128) -> F16 conversion, but only once per actual token/head.
__global__ void pack_indexed_kv(Input k, Input v, half* pk, half* pv,
                               int sequence, int head_begin, int head_count) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(sequence)*head_count*dimension) return;
    const int d=i%dimension, token=(i/dimension)%sequence;
    const int head=head_begin+int(i/(int64_t(sequence)*dimension));
    pk[i]=__float2half_rn(read(k,d,token,head)*(1.0f/128.0f));
    pv[i]=__float2half_rn(read(v,d,token,head)*(1.0f/128.0f));
}
__device__ float warp_sum(float x) {
    for (int offset=16; offset; offset/=2) x += __shfl_down_sync(0xffffffffu,x,offset);
    return __shfl_sync(0xffffffffu,x,0);
}
// One warp per (Q64,K64) pair; all 4x4 cell dot products and the full LSE.
// No query-only pooling, no normalization over key blocks, no global TopK.
__global__ void score_cells(const float* q, const float* k, const int32_t* blocks,
                           float* scores, int b) {
    const int kb = blockIdx.x*4 + threadIdx.x/32, qb = blockIdx.y, h = blockIdx.z;
    if (kb >= b) return;
    const int lane = threadIdx.x%32;
    const int nq = max(0,min(4,(blocks[2*qb+1]-blocks[2*qb]+15)/16));
    const int nk = max(0,min(4,(blocks[2*kb+1]-blocks[2*kb]+15)/16));
    float maximum = -CUDART_INF_F, mass = 0;
    for (int a=0;a<nq;++a) for (int z=0;z<nk;++z) {
        float dot = 0;
        for (int d=lane;d<dimension;d+=32)
            dot += q[((int64_t(h)*b+qb)*4+a)*dimension+d] *
                   k[((int64_t(h)*b+kb)*4+z)*dimension+d];
        dot = warp_sum(dot) * 0.08838834764831844f;
        const float next = fmaxf(maximum,dot);
        mass = mass*expf(maximum-next)+expf(dot-next);
        maximum = next;
    }
    if (lane==0) scores[(int64_t(h)*b+qb)*b+kb] = maximum+logf(mass);
}
__device__ bool better(float a,int ai,float b,int bi) {
    // Nonfinite input deliberately survives as NaN output in the attention path;
    // finite tie breaking exactly matches the local CPU reference (lower ID).
    return a>b || (a==b && ai<bi);
}
__global__ void sort_rows(const float* scores, int32_t* ids, int b, int topk, int width) {
    __shared__ float values[max_blocks];
    __shared__ int index[max_blocks];
    const int row=blockIdx.x;
    for (int i=threadIdx.x;i<width;i+=blockDim.x) {
        values[i]=i<b ? scores[int64_t(row)*b+i] : -CUDART_INF_F;
        index[i]=i;
    }
    __syncthreads();
    for(int size=2;size<=width;size*=2) for(int distance=size/2;distance;distance/=2) {
        for(int i=threadIdx.x;i<width;i+=blockDim.x) {
            const int j=i^distance;
            if(j>i) {
                const bool descending=(i&size)==0;
                const bool swap=descending ? better(values[j],index[j],values[i],index[i])
                                           : better(values[i],index[i],values[j],index[j]);
                if(swap) { float v=values[i];values[i]=values[j];values[j]=v;
                           int id=index[i];index[i]=index[j];index[j]=id; }
            }
        }
        __syncthreads();
    }
    for(int i=threadIdx.x;i<topk;i+=blockDim.x) ids[int64_t(row)*topk+i]=index[i];
}
struct Jobs {
    const int32_t* spans;
    const int32_t* blocks;
    const int32_t* route;
    const int32_t* protected_ids;
    int count, video_blocks, topk, protected_count, sequence, heads, start;
    bool sparse;
};
__device__ int query_start(Jobs j,int batch) { return j.spans[2*((j.start+batch)%j.count)]; }
__device__ int query_end(Jobs j,int batch) { return j.spans[2*((j.start+batch)%j.count)+1]; }
__device__ int key_token(Jobs j,int batch,int slot) {
    if(!j.sparse) return slot<j.sequence ? slot : -1;
    if(slot<j.topk*64) {
        const int row=j.start+batch;
        const int id=j.route[int64_t(row)*j.topk+slot/64];
        if(id<0 || id>=j.video_blocks) return -1;
        const int begin=j.blocks[2*id], end=j.blocks[2*id+1];
        const int t=begin+slot%64;
        return begin>=0 && end<=j.sequence && end>begin && t<end ? t : -1;
    }
    const int p=slot-j.topk*64;
    if(p>=j.protected_count) return -1;
    const int t=j.protected_ids[p];
    return t>=0 && t<j.sequence ? t : -1;
}
__global__ void gather_q(Input q, Jobs jobs, float* out, int batch) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*64*dimension) return;
    const int d=i%dimension, row=(i/dimension)%64, job=i/(dimension*64);
    const int h=(jobs.start+job)/jobs.count;
    const int t=query_start(jobs,job)+row;
    out[i]=t>=0 && t<query_end(jobs,job) && t<jobs.sequence ? read(q,d,t,h) : 0;
}
__global__ void gather_kv(Input k, Input v, Jobs jobs, half* ko, half* vo, int slots, int batch) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(batch)*slots*dimension) return;
    const int d=i%dimension, slot=(i/dimension)%slots, job=i/(int64_t(dimension)*slots);
    const int h=(jobs.start+job)/jobs.count, t=key_token(jobs,job,slot);
    ko[i]=__float2half_rn(t>=0 ? read(k,d,t,h)*(1.0f/128.0f) : 0);
    vo[i]=__float2half_rn(t>=0 ? read(v,d,t,h)*(1.0f/128.0f) : 0);
}
__global__ void gather_mask(Jobs jobs, half* mask, int slots, int batch) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(batch)*slots*64) return;
    const int slot=i%slots, job=i/(int64_t(slots)*64);
    mask[i]=__float2half_rn(key_token(jobs,job,slot)>=0 ? 0 : -CUDART_INF_F);
}
__global__ void scatter(const float* in, float* out, Jobs jobs, int batch) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*64*dimension) return;
    const int d=i%dimension, row=(i/dimension)%64, job=i/(dimension*64);
    const int h=(jobs.start+job)/jobs.count, t=query_start(jobs,job)+row;
    if(t>=0 && t<query_end(jobs,job) && t<jobs.sequence)
        out[(int64_t(t)*jobs.heads+h)*dimension+d]=in[i]*128.0f;
}
void tensor(ggml_tensor& t, ggml_type type,int64_t n0,int64_t n1,int64_t n2,int64_t n3,void* data) {
    t={};t.type=type;t.ne[0]=n0;t.ne[1]=n1;t.ne[2]=n2;t.ne[3]=n3;
    t.nb[0]=ggml_type_size(type);for(int i=1;i<4;++i)t.nb[i]=t.nb[i-1]*t.ne[i-1];
    t.data=data;
}
struct Flash {
    ggml_tensor q{}, k{}, v{}, mask{}, out{};
    void init(int slots,int batch,void* qp,void* kp,void* vp,void* mp,void* op) {
        tensor(q,GGML_TYPE_F32,128,64,1,batch,qp);
        tensor(k,GGML_TYPE_F16,128,slots,1,batch,kp);
        tensor(v,GGML_TYPE_F16,128,slots,1,batch,vp);
        tensor(mask,GGML_TYPE_F16,slots,64,1,batch,mp);
        tensor(out,GGML_TYPE_F32,128,1,64,batch,op);
        out.op=GGML_OP_FLASH_ATTN_EXT;
        out.src[0]=&q;out.src[1]=&k;out.src[2]=&v;out.src[3]=&mask;
        const float scale=128.0f/std::sqrt(128.0f);
        std::memcpy(out.op_params,&scale,sizeof(scale));
    }
};
}

bool ed_cuda_h3_subblock_supported(int device,const ggml_tensor* dst) {
    if(!ed_cuda_h3_subblock_shape_supported(dst))return false;
    if(device<0 || device>=ggml_cuda_info().device_count ||
       !GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[device].cc) ||
       ggml_cuda_info().devices[device].cc<GGML_CUDA_CC_TURING)return false;
    if(tag(dst)==router_tag)return true;
    if(tag(dst)==indexed_attention_tag)
        return ggml_cuda_info().devices[device].cc >= GGML_CUDA_CC_AMPERE &&
               ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[device].cc) >= GGML_CUDA_CC_AMPERE;
    // Probe the actual batch/mask/precision layout before allocation. No fallback.
    Capacity c;
    if(!capacity(dst->src[0]->ne[1],dst->src[0]->ne[2],dst->src[4]->ne[1],
                 dst->src[3]->ne[0],dst->src[6]->ne[0]-1,&c))return false;
    Flash flash;flash.init(c.key_slots,c.batch,nullptr,nullptr,nullptr,nullptr,nullptr);
    return ggml_cuda_flash_attn_ext_supported(device,&flash.out);
}

bool ed_cuda_h3_subblock_compute(ggml_backend_cuda_context& ctx,ggml_tensor* dst) {
    if(!ed_cuda_h3_subblock_supported(ctx.device,dst))return false;
    ggml_cuda_set_device(ctx.device);
    const auto stream=ctx.stream();
    const auto* q=dst->src[0]; const auto* k=dst->src[1];
    const int heads=int(q->ne[2]), sequence=int(q->ne[1]);
    if(tag(dst)==router_tag) {
        const int blocks=int(dst->ne[1]), topk=int(dst->ne[0]);
        const size_t cells=size_t(heads)*blocks*4*dimension;
        ggml_cuda_pool_alloc<float> pq(ctx.pool(),cells),pk(ctx.pool(),cells);
        ggml_cuda_pool_alloc<float> scores(ctx.pool(),size_t(heads)*blocks*blocks);
        const auto* spans=static_cast<const int32_t*>(dst->src[2]->data);
        pool_cells<<<dim3(blocks*4,heads),128,0,stream>>>(input(q),input(k),spans,pq.get(),pk.get(),blocks,sequence);
        score_cells<<<dim3((blocks+3)/4,blocks,heads),128,0,stream>>>(pq.get(),pk.get(),spans,scores.get(),blocks);
        int width=1;while(width<blocks)width*=2;
        sort_rows<<<heads*blocks,256,0,stream>>>(scores.get(),static_cast<int32_t*>(dst->data),blocks,topk,width);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if(tag(dst)==indexed_attention_tag) {
        const auto qi=input(q),ki=input(k),vi=input(dst->src[2],true);
        indexed::Params p{{qi.p,qi.token_stride,qi.head_stride},{ki.p,ki.token_stride,ki.head_stride},
                          {vi.p,vi.token_stride,vi.head_stride},
                          static_cast<const int32_t*>(dst->src[4]->data),
                          static_cast<const int32_t*>(dst->src[3]->data),
                          static_cast<const int32_t*>(dst->src[5]->data),
                          static_cast<const int32_t*>(dst->src[6]->data),
                          static_cast<float*>(dst->data),sequence,heads,int(dst->src[4]->ne[1]),
                          int(dst->src[3]->ne[0]),int(dst->src[6]->ne[0]-1),int(dst->src[5]->ne[1]-1)};
        const int batch=indexed_stage_heads(sequence,heads);
        GGML_ASSERT(batch>0);
        const size_t elements=size_t(batch)*sequence*dimension;
        ggml_cuda_pool_alloc<half> packed_k(ctx.pool(),elements),packed_v(ctx.pool(),elements);
        p.packed_k=packed_k.get();p.packed_v=packed_v.get();
        for(int begin=0;begin<heads;begin+=batch) {
            const int count=std::min(batch,heads-begin);
            p.head_begin=begin;
            const int64_t n=int64_t(count)*sequence*dimension;
            pack_indexed_kv<<<(n+255)/256,256,0,stream>>>(ki,vi,packed_k.get(),packed_v.get(),sequence,begin,count);
            indexed::attention<true><<<dim3(p.video_blocks,count),128,0,stream>>>(p);
            if(p.protected_chunk_count)
                indexed::attention<false><<<dim3(p.protected_chunk_count,count),128,0,stream>>>(p);
        }
        CUDA_CHECK(cudaGetLastError());
        return true; // explicitly indexed; no fallback to gather on failure
    }
    Capacity c;
    const int blocks=int(dst->src[4]->ne[1]), topk=int(dst->src[3]->ne[0]);
    const int protected_count=int(dst->src[6]->ne[0]-1);
    GGML_ASSERT(capacity(sequence,heads,blocks,topk,protected_count,&c));
    // Caller-owned ggml CUDA pool, same stream. Bounded scratch is reused for
    // every batch; no global state, per-layer leaks, D2H route reads or cudaMalloc.
    ggml_cuda_pool_alloc<float> qbuf(ctx.pool(),size_t(c.batch)*64*dimension);
    ggml_cuda_pool_alloc<half> kbuf(ctx.pool(),size_t(c.batch)*c.key_slots*dimension);
    ggml_cuda_pool_alloc<half> vbuf(ctx.pool(),size_t(c.batch)*c.key_slots*dimension);
    ggml_cuda_pool_alloc<half> mask(ctx.pool(),size_t(c.batch)*c.key_slots*64);
    ggml_cuda_pool_alloc<float> out(ctx.pool(),size_t(c.batch)*64*dimension);
    for(int phase=0;phase<2;++phase) {
        const bool sparse=phase==0;
        const int count=sparse ? blocks : int(dst->src[5]->ne[1]-1);
        if(!count)continue;
        const int slots=((sparse ? topk*64+protected_count : sequence)+63)/64*64;
        Jobs jobs{static_cast<const int32_t*>(dst->src[sparse?4:5]->data),
                  static_cast<const int32_t*>(dst->src[4]->data),
                  static_cast<const int32_t*>(dst->src[3]->data),
                  static_cast<const int32_t*>(dst->src[6]->data),
                  count,blocks,topk,protected_count,sequence,heads,0,sparse};
        for(int start=0;start<heads*count;start+=c.batch) {
            const int batch=std::min(c.batch,heads*count-start);jobs.start=start;
            gather_q<<<(batch*64*dimension+255)/256,256,0,stream>>>(input(q),jobs,qbuf.get(),batch);
            gather_kv<<<(int64_t(batch)*slots*dimension+255)/256,256,0,stream>>>(input(k),input(dst->src[2],true),jobs,kbuf.get(),vbuf.get(),slots,batch);
            gather_mask<<<(int64_t(batch)*slots*64+255)/256,256,0,stream>>>(jobs,mask.get(),slots,batch);
            CUDA_CHECK(cudaGetLastError());
            Flash flash;flash.init(slots,batch,qbuf.get(),kbuf.get(),vbuf.get(),mask.get(),out.get());
            GGML_ASSERT(ggml_cuda_flash_attn_ext_supported(ctx.device,&flash.out));
            ggml_cuda_flash_attn_ext(ctx,&flash.out);
            scatter<<<(batch*64*dimension+255)/256,256,0,stream>>>(out.get(),static_cast<float*>(dst->data),jobs,batch);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return true;
}
