// edge-dit: standalone A/B correctness harness for the vectorized conv3d im2col.
// Builds several conv3d graphs (f32/f16/bf16 kernels, with and without padding,
// dilation, and a large Wan-VAE-like case), runs them on the CPU backend, and
// dumps a stable digest of the output. Run once with the default (vectorized) path
// and once with ED_CONV3D_VEC=0; the digests must match bit-for-bit (bf16/f16
// paths) or within rounding (they are computed the same way, so also exact).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

static int calc_out(int ins, int ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

struct Case {
    const char * name;
    int N, IC, ID, IH, IW;
    int OC, KD, KH, KW;
    int s0, s1, s2, p0, p1, p2, d0, d1, d2;
    ggml_type ktype;
};

static void run_case(const Case & c, bool quiet = false) {
    const size_t mem = (size_t)1 << 30; // 1 GiB scratch for tensors
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const int64_t ne_in[]  = { c.IW, c.IH, c.ID, (int64_t)c.IC * c.N };
    const int64_t ne_knl[] = { c.KW, c.KH, c.KD, (int64_t)c.IC * c.OC };
    ggml_tensor * input  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_in);
    ggml_tensor * kernel = ggml_new_tensor(ctx, c.ktype,       4, ne_knl);

    ggml_tensor * out = ggml_conv_3d_direct(ctx, kernel, input,
        c.s0, c.s1, c.s2, c.p0, c.p1, c.p2, c.d0, c.d1, c.d2, c.IC, c.N, c.OC);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // deterministic inputs
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    {
        std::vector<float> tmp(ggml_nelements(input));
        for (auto & v : tmp) v = dist(rng);
        ggml_backend_tensor_set(input, tmp.data(), 0, tmp.size() * sizeof(float));
    }
    {
        const int64_t n = ggml_nelements(kernel);
        std::vector<float> f(n);
        for (auto & v : f) v = dist(rng);
        if (c.ktype == GGML_TYPE_F32) {
            ggml_backend_tensor_set(kernel, f.data(), 0, n * sizeof(float));
        } else if (c.ktype == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(n);
            ggml_fp32_to_fp16_row(f.data(), h.data(), n);
            ggml_backend_tensor_set(kernel, h.data(), 0, n * sizeof(ggml_fp16_t));
        } else { // bf16
            std::vector<ggml_bf16_t> h(n);
            ggml_fp32_to_bf16_row(f.data(), h.data(), n);
            ggml_backend_tensor_set(kernel, h.data(), 0, n * sizeof(ggml_bf16_t));
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(backend, gf);

    const int64_t n = ggml_nelements(out);
    std::vector<float> o(n);
    ggml_backend_tensor_get(out, o.data(), 0, n * sizeof(float));

    // stable digest: FNV-1a over raw bits, plus sum and max-abs
    uint64_t h = 1469598103934665603ULL;
    double sum = 0.0, maxa = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits; std::memcpy(&bits, &o[i], 4);
        h ^= bits; h *= 1099511628211ULL;
        sum += o[i];
        double a = std::fabs((double)o[i]);
        if (a > maxa) maxa = a;
    }
    if (!quiet) {
        printf("%-14s type=%d nout=%lld  sum=%.6f  maxabs=%.6f  hash=%016llx  first=[%.6f %.6f %.6f]\n",
               c.name, (int)c.ktype, (long long)n, sum, maxa, (unsigned long long)h,
               n>0?o[0]:0, n>1?o[1]:0, n>2?o[2]:0);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    std::vector<Case> cases = {
        // small, no pad, f32/f16/bf16
        { "small_f32",  1, 4, 8, 8, 8,  8, 3,3,3, 1,1,1, 0,0,0, 1,1,1, GGML_TYPE_F32 },
        { "small_f16",  1, 4, 8, 8, 8,  8, 3,3,3, 1,1,1, 0,0,0, 1,1,1, GGML_TYPE_F16 },
        { "small_bf16", 1, 4, 8, 8, 8,  8, 3,3,3, 1,1,1, 0,0,0, 1,1,1, GGML_TYPE_BF16 },
        // padding exercises the boundary/zero-fill path
        { "pad_bf16",   1, 6, 5, 7, 9,  8, 3,3,3, 1,1,1, 1,1,1, 1,1,1, GGML_TYPE_BF16 },
        { "pad_f16",    1, 6, 5, 7, 9,  8, 3,3,3, 1,1,1, 1,1,1, 1,1,1, GGML_TYPE_F16 },
        { "pad_f32",    1, 6, 5, 7, 9,  8, 3,3,3, 1,1,1, 1,1,1, 1,1,1, GGML_TYPE_F32 },
        // dilation>1 (disables the memcpy fast path -> per-element bound check)
        { "dil_bf16",   1, 4, 8, 8, 8,  8, 3,3,3, 1,1,1, 2,2,2, 2,2,2, GGML_TYPE_BF16 },
        // asymmetric kernel / stride
        { "asym_bf16",  1, 5, 6, 8,10,  7, 3,3,3, 2,1,1, 1,1,1, 1,1,1, GGML_TYPE_BF16 },
        // larger Wan-VAE-like: many channels, kernel 3^3 (heavy im2col)
        { "wan_bf16",   1, 96,4,32,32, 96, 3,3,3, 1,1,1, 1,1,1, 1,1,1, GGML_TYPE_BF16 },
        // Wan causal decode shape: temporal left padding is materialized before
        // conv, so the conv itself has p2=0 and ID=OD+KD-1.
        { "wan_causal_bf16", 1, 96,6,32,32, 96, 3,3,3, 1,1,1, 1,1,0, 1,1,1, GGML_TYPE_BF16 },
    };
    const char * env = getenv("ED_CONV3D_VEC");
    printf("=== ED_CONV3D_VEC=%s ===\n", env ? env : "(unset->vec)");
    for (auto & c : cases) run_case(c);

    // Optional timing mode: build the heavy Wan-like conv3d once, then loop only
    // graph_compute so setup/alloc overhead is excluded. im2col is the only path
    // that differs between vec and scalar (GEMM + permute identical), so the delta
    // is the im2col speedup.
    if (getenv("ED_CONV3D_TIME")) {
        // large: patch_total = 16*64*64 = 65536, IC=96, 3^3 kernel (heavy im2col).
        // OC is set from env: OC=96 (full conv) or OC=1 to isolate im2col, since the
        // im2col fill cost is independent of OC while GEMM/permute scale with OC.
        const int OCv = getenv("ED_CONV3D_OC") ? atoi(getenv("ED_CONV3D_OC")) : 96;
        Case big = { "wan_big", 1, 96,16,64,64, OCv, 3,3,3, 1,1,1, 1,1,1, 1,1,1, GGML_TYPE_BF16 };

        const size_t mem = (size_t)2 << 30;
        ggml_init_params ip = { mem, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        const int64_t ne_in[]  = { big.IW, big.IH, big.ID, (int64_t)big.IC * big.N };
        const int64_t ne_knl[] = { big.KW, big.KH, big.KD, (int64_t)big.IC * big.OC };
        ggml_tensor * input  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_in);
        ggml_tensor * kernel = ggml_new_tensor(ctx, big.ktype, 4, ne_knl);
        ggml_tensor * out = ggml_conv_3d_direct(ctx, kernel, input,
            big.s0,big.s1,big.s2, big.p0,big.p1,big.p2, big.d0,big.d1,big.d2, big.IC, big.N, big.OC);
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 32);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::mt19937 rng(7); std::uniform_real_distribution<float> dist(-1.f, 1.f);
        { std::vector<float> t(ggml_nelements(input)); for (auto&v:t) v=dist(rng);
          ggml_backend_tensor_set(input, t.data(), 0, t.size()*sizeof(float)); }
        { int64_t nk=ggml_nelements(kernel); std::vector<float> f(nk); for(auto&v:f)v=dist(rng);
          std::vector<ggml_bf16_t> h(nk); ggml_fp32_to_bf16_row(f.data(), h.data(), nk);
          ggml_backend_tensor_set(kernel, h.data(), 0, nk*sizeof(ggml_bf16_t)); }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        const int warm = 3, reps = 20;
        for (int r = 0; r < warm; ++r) ggml_backend_graph_compute(backend, gf);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(backend, gf);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        printf("[timing] wan_big conv3d (compute only, 32thr) avg = %.3f ms/iter over %d iters\n", ms, reps);
        ggml_backend_buffer_free(buf); ggml_backend_free(backend); ggml_free(ctx);
    }
    return 0;
}
