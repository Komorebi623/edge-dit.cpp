// CPU ONLY. Links only ggml's CPU build. Never executes a Sage custom node,
// loads CUDA, discovers GPUs, or initializes a CUDA backend.
#include "dit_models/models/minimax_h3_cuda_sage.hpp"
#include "backend/ggml/ggml_extend.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

// Optional exact-source host predicate harness (generated under /tmp from the
// authorized CUDA TU; never linked to CUDA). See h3-sage-progress.md.
#ifdef ED_H3_SAGE_HOST_PREDICATE_HEADER
#include ED_H3_SAGE_HOST_PREDICATE_HEADER
#endif

using namespace MiniMaxH3;

static void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

struct Context {
    ggml_context* ctx;
    explicit Context(bool no_alloc = false) {
        ctx = ggml_init({32 * 1024 * 1024, nullptr, no_alloc});
        check(ctx != nullptr, "CPU context creation failed");
    }
    ~Context() { ggml_free(ctx); }
};

// Environment changes are confined to THIS test process and restored. The
// production helper never calls setenv/putenv or changes global Sage options.
struct TestEnv {
    const char* key;
    std::string old;
    bool existed;
    TestEnv(const char* key_, const char* value) : key(key_) {
        const char* p = std::getenv(key);
        existed = p != nullptr;
        if (p != nullptr) old = p;
        check(setenv(key, value, 1) == 0, "test environment setup failed");
    }
    ~TestEnv() {
        if (existed) setenv(key, old.c_str(), 1);
        else unsetenv(key);
    }
};

static void compute_cpu(ggml_context* ctx, ggml_tensor* out) {
    auto graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, out);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        check(ggml_graph_node(graph, i)->op != GGML_OP_CUSTOM,
              "CPU test attempted to execute a custom Sage node");
    check(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS,
          "CPU graph computation failed");
}

static float get(const ggml_tensor* t, int64_t a, int64_t b = 0,
                 int64_t c = 0, int64_t d = 0) {
    const char* p = static_cast<const char*>(t->data) +
                    a*t->nb[0] + b*t->nb[1] + c*t->nb[2] + d*t->nb[3];
    if (t->type == GGML_TYPE_F32) {
        float f;
        std::memcpy(&f, p, sizeof(f));
        return f;
    }
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t h;
        std::memcpy(&h, p, sizeof(h));
        return ggml_fp16_to_fp32(h);
    }
    throw std::runtime_error("unsupported test read type");
}

static void put(ggml_tensor* t, int64_t a, int64_t b, int64_t c, int64_t d, float f) {
    char* p = static_cast<char*>(t->data) +
              a*t->nb[0] + b*t->nb[1] + c*t->nb[2] + d*t->nb[3];
    if (t->type == GGML_TYPE_F32) std::memcpy(p, &f, sizeof(f));
    else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(f);
        std::memcpy(p, &h, sizeof(h));
    } else throw std::runtime_error("unsupported test write type");
}

struct Qkv { ggml_tensor* q; ggml_tensor* k; ggml_tensor* v; };

static Qkv qkv(ggml_context* ctx, int64_t length = 5, int64_t heads = 2,
               int64_t batch = 1, ggml_type qt = GGML_TYPE_F32,
               ggml_type kt = GGML_TYPE_F32, ggml_type vt = GGML_TYPE_F32,
               bool packed = false, int64_t dim = 128) {
    return {ggml_new_tensor_3d(ctx, qt, dim, length, heads*batch),
            ggml_new_tensor_3d(ctx, kt, dim, length, heads*batch),
            packed ? ggml_new_tensor_3d(ctx, vt, dim, length, heads) :
                     ggml_new_tensor_4d(ctx, vt, dim, heads, length, batch)};
}

static H3SageRequest candidate() {
    H3SageRequest r;
    r.enabled = r.is_cuda = r.flash_attn = true;
    r.scope = H3SageScope::DiT;
    r.layer_idx = 1;
    r.total_layers = 50;
    return r;
}

static void test_policy(ggml_backend_t cpu) {
    check(h3_sage_decide({}).route == H3SageRoute::Dense, "default must stay dense");
    auto r = candidate();
    check(h3_sage_decide(r).route == H3SageRoute::Sage, "interior DiT layer rejected");
    r.layer_idx = 0;
    check(h3_sage_decide(r).route == H3SageRoute::Dense, "first layer used Sage");
    r.layer_idx = 49;
    check(h3_sage_decide(r).route == H3SageRoute::Dense, "last layer used Sage");
    r.layer_idx = -1;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "missing layer accepted");
    r.layer_idx = 50;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "out-of-range layer accepted");
    r.total_layers = 0;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "empty DiT stack accepted");
    r = candidate();
    r.scope = H3SageScope::TokenRefiner;
    check(h3_sage_decide(r).route == H3SageRoute::Dense, "token refiner used Sage");
    r = candidate(); r.subblock_enabled = true;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "Sage+subblock accepted");
    r = candidate(); r.pv_fp8 = true;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "unvalidated FP8 PV accepted");
    r = candidate(); r.is_cuda = false;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "CPU Sage accepted");
    r = candidate(); r.flash_attn = false;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "flash-off silently fell back to dense");
    r = candidate(); r.dense_fallback = true;
    check(h3_sage_decide(r).route == H3SageRoute::Reject, "hidden dense fallback accepted");
    for (int count : {1, 2}) for (int i = 0; i < count; ++i) {
        r = candidate(); r.total_layers = count; r.layer_idx = i;
        check(h3_sage_decide(r).route == H3SageRoute::Dense, "tiny stack used Sage");
    }
    check(!h3_sage_cuda_backend(cpu) && !h3_sage_cuda_backend(nullptr), "CPU/backend-null CUDA gate");
    Context c(true); auto t = qkv(c.ctx);
    auto result = h3_sage_try_build(c.ctx, cpu, t.q, t.k, t.v, 2, candidate());
    check(result.route == H3SageRoute::Reject && result.output == nullptr,
          "caller spoofed CUDA/backend support");
    r = candidate(); r.enabled = false;
    check(h3_sage_try_build(c.ctx, cpu, t.q, t.k, t.v, 2, r).route == H3SageRoute::Dense,
          "flag-off changed baseline");

    for (const char* global_value : {"0", "1"}) {
        TestEnv global("ED_SAGE_ATTN", global_value);
        TestEnv skip_first("ED_SAGE_SKIP_FIRST", "50");
        TestEnv skip_last("ED_SAGE_SKIP_LAST", "50");
        std::map<std::string, std::string> env = {{"ED_SAGE_ATTN", global_value}};
        auto lookup = [&](const char* key) { return env[ key ]; };
        CudaOptimizationOptions opts; std::string error;
        check(parse_cuda_optimization_options(lookup, &opts, &error) && !opts.sage_attn,
              "global Sage enabled H3 policy");
        env["ED_H3_SAGE_ATTN"] = "1";
        check(parse_cuda_optimization_options(lookup, &opts, &error) && opts.sage_attn,
              "H3 independent flag parse failed");
        check(h3_sage_decide(candidate()).route == H3SageRoute::Sage,
              "global skip options changed H3 layer policy");
        auto request = h3_sage_request(&opts, cpu, true, H3SageScope::DiT, 1, 50, false);
        check(request.enabled && !request.is_cuda, "request construction lost backend gate");
        opts.sparse_attn = true;
        check(h3_sage_request(&opts, cpu, true, H3SageScope::DiT, 1, 50, false).subblock_enabled,
              "parsed Sage/SubBlock conflict lost in request");
        auto raw = h3_sage_custom_graph(c.ctx, t.q, t.k, t.v, 2);
        check(raw != nullptr, "global gate changed H3 graph construction");
        check(edgedit::ggml_ext::sage_attn_enabled() == (std::string(global_value) == "1"),
              "H3 builder mutated ED_SAGE_ATTN");
    }
    std::cout << "PASS policy/global isolation/CPU fail-closed\n";
}

static void test_tags() {
    using namespace edgedit::ggml_ext;
    for (int64_t heads : {int64_t(1), int64_t(56), int64_t(65535), int64_t(INT32_MAX)}) {
        auto h3 = sage_attn_params_from_userdata(h3_sage_params_to_userdata(heads));
        check(h3_sage_params_valid(h3) && h3.n_head == heads, "H3 tag round-trip");
        auto old = sage_attn_params_from_userdata(sage_attn_params_to_userdata(heads));
        check(sage_attn_params_valid(old) && !h3_sage_params_valid(old), "legacy tag semantics");
    }
    check(h3_sage_params_to_userdata(0) == nullptr &&
          h3_sage_params_to_userdata(-1) == nullptr &&
          h3_sage_params_to_userdata(int64_t(INT32_MAX) + 1) == nullptr, "invalid packed head count");
    check(!h3_sage_params_valid(sage_attn_params_from_userdata(nullptr)), "null tag accepted");
    std::cout << "PASS shared H3SA/legacy SAGQ tagged userdata ABI\n";
}

struct CustomParamsLayout {
    ggml_custom_op_t fun;
    int n_tasks;
    void* userdata;
};


static void test_dispatch_predicate() {
#ifdef ED_H3_SAGE_HOST_PREDICATE_HEADER
    using h3_sage_host_test::ed_cuda_sage_attn_custom_supported;
    Context c(true);
    for (const char* value : {"", "0", "false", "FALSE", "off", "OFF", "1", "yes"}) {
        TestEnv gate("ED_SAGE_ATTN", value);
        for (int64_t dim : {64, 128}) for (auto qt : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16})
            for (auto kt : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16})
                for (auto vt : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16}) {
                    auto t = qkv(c.ctx, 5, 2, 1, qt, kt, vt, false, dim);
                    ggml_tensor* args[] = {t.q, t.k, t.v};
                    for (uint32_t magic : {edgedit::ggml_ext::kSageAttnCustomMagic,
                                           kH3SageCustomMagic, uint32_t(0xDEADBEEF)}) {
                        const auto userdata = reinterpret_cast<void*>(uintptr_t(magic) | (uintptr_t(2) << 32));
                        auto raw = ggml_custom_4d(c.ctx, GGML_TYPE_F32, dim, 2, 5, 1, args, 3,
                                                   h3_sage_cpu_custom_op_stub, 1, userdata);
                        const bool actual = ed_cuda_sage_attn_custom_supported(raw);
                        if (magic == kH3SageCustomMagic) {
                            check(actual == (dim == 128 && qt == GGML_TYPE_F32 &&
                                              kt == GGML_TYPE_F32 && vt == GGML_TYPE_F32),
                                  "H3SA gate/head/dtype acceptance matrix");
                        } else {
                            check(actual == h3_sage_host_test::legacy_sage_supported(raw),
                                  "old/unknown tag differs from original CUDA predicate");
                        }
                    }
                }
    }
    auto t = qkv(c.ctx);
    auto raw = h3_sage_custom_graph(c.ctx, t.q, t.k, t.v, 2);
    TestEnv gate("ED_SAGE_ATTN", "0");
    check(ed_cuda_sage_attn_custom_supported(raw), "H3SA requires global ED_SAGE_ATTN");
    for (const char* key : {"ED_SAGE_FALLBACK", "ED_SAGE_PV_FP8"}) {
        TestEnv override(key, "1");
        check(!ed_cuda_sage_attn_custom_supported(raw), "H3SA accepted unvalidated FP8/hidden dense override");
    }
    check(ed_cuda_sage_attn_custom_supported(raw), "H3SA did not recover after override rejection");
    check(!ed_cuda_sage_attn_custom_supported(nullptr), "null dispatch accepted");
    auto op = raw->op; raw->op = GGML_OP_NONE;
    check(!ed_cuda_sage_attn_custom_supported(raw), "non-custom dispatch accepted"); raw->op = op;
    auto src = raw->src[2]; raw->src[2] = nullptr;
    check(!ed_cuda_sage_attn_custom_supported(raw), "missing V dispatch accepted"); raw->src[2] = src;
    auto nb = raw->src[0]->nb[0]; raw->src[0]->nb[0] *= 2;
    check(!ed_cuda_sage_attn_custom_supported(raw), "bad channel stride dispatch accepted");
    raw->src[0]->nb[0] = nb+1;
    check(!ed_cuda_sage_attn_custom_supported(raw), "fractional channel stride dispatch accepted");
    raw->src[0]->nb[0] = nb;
    auto dtype = raw->type; raw->type = GGML_TYPE_F16;
    check(!ed_cuda_sage_attn_custom_supported(raw), "bad output dtype accepted"); raw->type = dtype;
    auto axis = raw->ne[1]; raw->ne[1]++;
    check(!ed_cuda_sage_attn_custom_supported(raw), "bad output layout accepted"); raw->ne[1] = axis;
    auto sequence = raw->src[1]->ne[1]; raw->src[1]->ne[1]--;
    check(!ed_cuda_sage_attn_custom_supported(raw), "H3 cross-attn dispatch accepted"); raw->src[1]->ne[1] = sequence;
    CustomParamsLayout params{};
    std::memcpy(&params, raw->op_params, sizeof(params));
    const auto original_params = params;
    params.userdata = reinterpret_cast<void*>(uintptr_t(kH3SageCustomMagic));
    std::memcpy(raw->op_params, &params, sizeof(params));
    check(!ed_cuda_sage_attn_custom_supported(raw), "zero-head H3SA dispatch accepted");
    std::memcpy(raw->op_params, &original_params, sizeof(params));
    std::cout << "PASS EXACT CUDA host predicate: H3SA gate/types and legacy differential matrix (1296 cases)\n";
#else
    std::cout << "SKIP exact CUDA host predicate harness (optional build macro not supplied)\n";
#endif
}

static void test_shapes() {
    Context c(true);
    auto t = qkv(c.ctx, 5, 2, 2);
    check(h3_sage_shape_supported(t.q, t.k, t.v, 2), "batched 4D V rejected");
    auto packed = qkv(c.ctx, 7, 2, 1, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, true);
    check(h3_sage_shape_supported(packed.q, packed.k, packed.v, 2), "packed/mixed inputs rejected before normalization");
    auto d64 = qkv(c.ctx, 5, 2, 1, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, false, 64);
    check(edgedit::ggml_ext::sage_attn_shape_supported(d64.q, d64.k, d64.v, 2) &&
          !h3_sage_shape_supported(d64.q, d64.k, d64.v, 2), "H3 dimension gate");
    auto mask = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 5, 5);
    check(!h3_sage_shape_supported(t.q, t.k, t.v, 2, mask), "masked attention accepted");
    check(!h3_sage_shape_supported(nullptr, t.k, t.v, 2) &&
          !h3_sage_shape_supported(t.q, t.k, t.v, 0), "null/zero-head guard");
    for (auto tensor : {t.q, t.k, t.v}) {
        auto type = tensor->type; tensor->type = GGML_TYPE_BF16;
        check(!h3_sage_shape_supported(t.q, t.k, t.v, 2), "BF16 low-memory path escaped");
        tensor->type = type;
        auto nb0 = tensor->nb[0]; tensor->nb[0] *= 2;
        check(!h3_sage_shape_supported(t.q, t.k, t.v, 2), "non-unit channel stride accepted");
        tensor->nb[0] = nb0;
    }
    auto ne = t.k->ne[1]; t.k->ne[1]--;
    check(!h3_sage_shape_supported(t.q, t.k, t.v, 2), "cross-attention accepted"); t.k->ne[1] = ne;
    t.q->ne[3] = 2;
    check(!h3_sage_shape_supported(t.q, t.k, t.v, 2), "unflattened Q accepted"); t.q->ne[3] = 1;
    ne = t.k->ne[2]; t.k->ne[2] = 2;
    check(!h3_sage_shape_supported(t.q, t.k, t.v, 2), "GQA accepted"); t.k->ne[2] = ne;
    auto huge = qkv(c.ctx, int64_t(UINT32_MAX) / 128 + 1, 1, 1);
    check(!h3_sage_shape_supported(huge.q, huge.k, huge.v, 1), "u32 stride overflow accepted");
    auto grid = qkv(c.ctx, 1, 1, 65536);
    check(!h3_sage_shape_supported(grid.q, grid.k, grid.v, 1), "CUDA grid.z overflow accepted");
    auto below = qkv(c.ctx, 1, 1, 65535);
    check(h3_sage_shape_supported(below.q, below.k, below.v, 1), "valid grid.z boundary rejected");
    auto ambiguous = qkv(c.ctx, 2, 2, 1, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16, true);
    check(h3_sage_shape_supported(ambiguous.q, ambiguous.k, ambiguous.v, 2), "L==heads packed shape rejected");
    check(h3_sage_custom_graph(nullptr, t.q, t.k, t.v, 2) == nullptr, "null graph accepted");
    std::cout << "PASS H3 shape/type/layout/overflow guards\n";
}


static void test_custom_graph() {
    Context c;
    auto t = qkv(c.ctx);
    auto raw = h3_sage_custom_graph(c.ctx, t.q, t.k, t.v, 2);
    check(raw != nullptr && raw->op == GGML_OP_CUSTOM && raw->type == GGML_TYPE_F32,
          "H3 did not emit a real custom dispatch node");
    check(raw->ne[0] == 128 && raw->ne[1] == 2 && raw->ne[2] == 5 && raw->ne[3] == 1,
          "custom output layout");
    check(raw->src[0] == t.q && raw->src[1] == t.k, "Q/K incorrectly scaled/copied");
    check(raw->src[2]->op == GGML_OP_SCALE, "V conditioning missing");
    float factor; std::memcpy(&factor, raw->src[2]->op_params, sizeof(factor));
    check(factor == 1.f/128.f, "V conditioning scale");
    CustomParamsLayout params{};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "custom params test ABI");
    std::memcpy(&params, raw->op_params, sizeof(params));
    check(params.fun == h3_sage_cpu_custom_op_stub && params.n_tasks == 1 &&
          h3_sage_params_valid(edgedit::ggml_ext::sage_attn_params_from_userdata(params.userdata)),
          "H3 custom tag/stub not encoded");
    for (int64_t d = 0; d < 128; ++d) for (int64_t h = 0; h < 2; ++h)
        for (int64_t j = 0; j < 5; ++j) put(t.v, d, h, j, 0, float(1+d+128*h+256*j));
    compute_cpu(c.ctx, raw->src[2]); // ONLY V preparation, never raw
    check(get(raw->src[2], 7, 1, 3) == get(t.v, 7, 1, 3)/128.f,
          "V preparation math changed magnitude/layout");
    for (auto qt : {GGML_TYPE_F16, GGML_TYPE_F32})
        for (auto kt : {GGML_TYPE_F16, GGML_TYPE_F32})
            for (auto vt : {GGML_TYPE_F16, GGML_TYPE_F32}) {
                auto mixed = qkv(c.ctx, 3, 2, 1, qt, kt, vt, true);
                auto op = h3_sage_custom_graph(c.ctx, mixed.q, mixed.k, mixed.v, 2);
                check(op != nullptr, "mixed F16/F32 custom build");
                for (int i = 0; i < 3; ++i)
                    check(op->src[i]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[i]),
                          "run_sage<T> mixed-type reinterpretation risk");
            }
    // A truly strided token-major H3 V (as produced by QKV projection views).
    auto backing = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 3*128*2, 5, 1);
    std::memset(backing->data, 0, ggml_nbytes(backing));
    auto strided = ggml_view_4d(c.ctx, backing, 128, 2, 5, 1,
                                128*sizeof(ggml_fp16_t), backing->nb[1], backing->nb[2],
                                2*128*2*sizeof(ggml_fp16_t));
    auto strided_op = h3_sage_custom_graph(c.ctx, t.q, t.k, strided, 2);
    check(strided_op != nullptr && ggml_is_contiguous(strided_op->src[2]), "strided H3 V normalization");
    put(strided, 7, 1, 3, 0, 256.f);
    compute_cpu(c.ctx, strided_op->src[2]);
    check(get(strided_op->src[2], 7, 1, 3) == 2.f, "strided V preparation math");
    std::cout << "PASS custom-node graph/type normalization/V conditioning (no custom execution)\n";
}

static void test_output_layout() {
    for (int64_t batch : {1, 2, 3}) {
        Context c;
        constexpr int64_t heads = 2, length = 5;
        // Synthetic kernel output: NONE tensor, not the CUDA-only custom node.
        auto raw = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F32, 128, heads*batch, length);
        for (int64_t b = 0; b < batch; ++b) for (int64_t h = 0; h < heads; ++h)
            for (int64_t j = 0; j < length; ++j) for (int64_t d = 0; d < 128; ++d)
                put(raw, d, h+heads*b, j, 0, float(d + 128*h + 1000*j + 10000*b)/128.f);
        auto out = h3_sage_output_graph(c.ctx, raw, heads);
        check(out != nullptr && out->ne[0] == 128*heads && out->ne[1] == length &&
              out->ne[2] == batch && ggml_is_contiguous(out), "final [C,L,batch] layout");
        compute_cpu(c.ctx, out);
        for (int64_t b = 0; b < batch; ++b) for (int64_t j = 0; j < length; ++j)
            for (int64_t h = 0; h < heads; ++h) for (int64_t d = 0; d < 128; ++d)
                check(get(out, d+128*h, j, b) == float(d + 128*h + 1000*j + 10000*b),
                      "head/token/batch output permutation or inverse scale");
        check(h3_sage_output_graph(c.ctx, raw, 3) == nullptr || batch == 3,
              "non-divisible head count accepted");
    }
    std::cout << "PASS output inverse-scale and batch 1/2/3 CPU graphs\n";
}

// Mathematical reference is TEST-ONLY, not an alleged Sage implementation.
static std::vector<float> reference(Qkv t, int64_t heads, float sm_scale) {
    const int64_t length = t.q->ne[1], batch = t.q->ne[2]/heads;
    std::vector<float> out(size_t(128*heads*length*batch));
    for (int64_t b = 0; b < batch; ++b) for (int64_t h = 0; h < heads; ++h)
        for (int64_t i = 0; i < length; ++i) {
            std::vector<double> scores(size_t(length), 0.0);
            for (int64_t j = 0; j < length; ++j) {
                for (int64_t d = 0; d < 128; ++d)
                    scores[size_t(j)] += double(get(t.q, d, i, h+heads*b))*get(t.k, d, j, h+heads*b);
                scores[size_t(j)] *= sm_scale;
            }
            const double max_score = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (auto& s : scores) { s = std::exp(s-max_score); sum += s; }
            for (int64_t d = 0; d < 128; ++d) {
                double value = 0;
                for (int64_t j = 0; j < length; ++j)
                    value += scores[size_t(j)]/sum * get(t.v, d, h, j, b);
                out[size_t(d+128*h+128*heads*(i+length*b))] = float(value);
            }
        }
    return out;
}

static float max_error(ggml_tensor* out, const std::vector<float>& ref) {
    float result = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        result = std::max(result, std::fabs(static_cast<float*>(out->data)[i]-ref[i]));
    return result;
}

static void test_shared_scaling(ggml_backend_t cpu) {
    TestEnv global("ED_SAGE_ATTN", "0");
    Context c; constexpr int64_t heads = 2, length = 5;
    auto t = qkv(c.ctx, length, heads);
    for (int64_t h = 0; h < heads; ++h) for (int64_t j = 0; j < length; ++j)
        for (int64_t d = 0; d < 128; ++d) {
            put(t.q, d, j, h, 0, float(((d+3*j+5*h)%19)-9)/8.f);
            put(t.k, d, j, h, 0, float(((3*d+7*j+h)%23)-11)/8.f);
            put(t.v, d, h, j, 0, float(((d+7*j+11*h)%31)-15)/4.f);
        }
    const auto correct = reference(t, heads, h3_sage_sm_scale());
    const auto wrong = reference(t, heads, 1.f/128.f);
    auto dense = ggml_ext_attention_ext(c.ctx, cpu, t.q, t.k, t.v, heads, nullptr,
                                         true, false, kH3SageKvScale, false);
    compute_cpu(c.ctx, dense);
    check(max_error(dense, correct) < 2e-5f, "shared dense effective scale is not 1/sqrt128");
    check(max_error(dense, wrong) > 0.05f, "fixture cannot detect mistaken 1/128 logits");
    auto dense_one = ggml_ext_attention_ext(c.ctx, cpu, t.q, t.k, t.v, heads, nullptr,
                                             true, false, 1.f, false);
    compute_cpu(c.ctx, dense_one);
    check(max_error(dense_one, correct) < 2e-5f, "dense kv_scale unexpectedly changes result");

    auto flash = ggml_ext_attention_ext(c.ctx, cpu, t.q, t.k, t.v, heads, nullptr,
                                         true, true, kH3SageKvScale, false);
    auto graph = ggml_new_graph_custom(c.ctx, 256, false);
    ggml_build_forward_expand(graph, flash);
    ggml_tensor* flash_node = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        if (ggml_graph_node(graph, i)->op == GGML_OP_FLASH_ATTN_EXT)
            flash_node = ggml_graph_node(graph, i);
    check(flash_node != nullptr, "shared flash graph was silently dense");
    float scale; std::memcpy(&scale, flash_node->op_params, sizeof(scale));
    check(std::fabs(scale-h3_sage_sm_scale()/kH3SageKvScale) < 1e-6f,
          "shared flash sm_scale compensation changed");
    check(flash_node->src[0] == t.q && flash_node->src[1]->type == GGML_TYPE_F16 &&
          flash_node->src[2]->type == GGML_TYPE_F16, "shared flash preparation graph changed");
    compute_cpu(c.ctx, flash);
    check(max_error(flash, correct) < 0.01f, "flash K/V scale + output inverse-scale math");
    // Values chosen on binary fractions survive the 1/128 F16 preparation.
    check(get(flash_node->src[1], 5, 3, 1) == get(t.k, 5, 3, 1)/128.f,
          "shared flash K does not shrink by kv_scale");
    check(get(flash_node->src[2], 5, 3, 1) == get(t.v, 5, 1, 3)/128.f,
          "shared flash V does not shrink by kv_scale");
    // Precise legacy caveat: V conditioning is inside the shared helper's
    // type/contiguity branch, NOT unconditional. A prepacked contiguous F16 V
    // bypasses its scale but the final inverse scale still runs. Record this
    // actual behavior; do not claim algebraic equivalence for every dtype.
    auto packed_f16 = ggml_new_tensor_3d(c.ctx, GGML_TYPE_F16, 128, length, heads);
    for (int64_t h = 0; h < heads; ++h) for (int64_t j = 0; j < length; ++j)
        for (int64_t d = 0; d < 128; ++d) put(packed_f16, d, j, h, 0, get(t.v, d, h, j));
    auto legacy_v_no_scale = ggml_ext_attention_ext(c.ctx, cpu, t.q, t.k, packed_f16,
                                                    heads, nullptr, true, true,
                                                    kH3SageKvScale, false);
    compute_cpu(c.ctx, legacy_v_no_scale);
    std::vector<float> amplified = correct;
    for (auto& value : amplified) value *= 128.f;
    check(max_error(legacy_v_no_scale, amplified) < 0.3f,
          "legacy contiguous F16 V bypass caveat changed");
    check(max_error(legacy_v_no_scale, correct) > 10.f, "legacy V bypass fixture is weak");
    std::cout << "PASS legacy contiguous-F16 V conditional-scale caveat recorded (NOT transplanted)\n";
    std::cout << "PASS ACTUAL ggml_ext_attention_ext CPU dense/flash scaling; max abs error dense="
              << max_error(dense, correct) << ", flash=" << max_error(flash, correct)
              << ", wrong 1/128=" << max_error(dense, wrong) << '\n';
}

// Build the ACTUAL cut plan on CPU without executing its CUSTOM node. This
// proves the planner keeps H3SA atomic and preserves packed userdata; execution,
// QKV rebinding and reuse are separately covered by the opt-in CUDA sidecar.
static void test_real_cut_plan(ggml_backend_t cpu) {
    namespace Cut = sd::ggml_graph_cut;
    Context params(true), graph_context(true);
    auto t = qkv(params.ctx, 65, 2, 2);
    auto* buffer = ggml_backend_alloc_ctx_tensors(params.ctx, cpu);
    check(buffer != nullptr, "CPU cut-plan parameter buffer");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> own(buffer, ggml_backend_buffer_free);
    auto* ctx = graph_context.ctx;
    auto* q = ggml_scale(ctx, t.q, 1.f);
    auto* k = ggml_scale(ctx, t.k, 1.f);
    auto* v = ggml_scale(ctx, t.v, 1.f);
    Cut::mark_graph_cut(q, "qkv-producer", "q");
    Cut::mark_graph_cut(k, "qkv-producer", "k");
    Cut::mark_graph_cut(v, "qkv-producer", "v");
    auto* raw = h3_sage_custom_graph(ctx, q, k, v, 2);
    check(raw != nullptr, "cut H3SA graph");
    auto* out = h3_sage_output_graph(ctx, raw, 2);
    Cut::mark_graph_cut(out, "attention-consumer", "out");
    auto* final = ggml_scale(ctx, out, 1.f);
    auto* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, final);
    const auto plan = Cut::build_plan(cpu, graph, {t.q, t.k, t.v}, "H3SA-CPU-plan-only");
    check(plan.available && plan.valid && plan.has_cuts && plan.segments.size() == 3,
          "producer/attention/output must form three actual segments");
    size_t owners = 0;
    CustomParamsLayout before{}; std::memcpy(&before, raw->op_params, sizeof(before));
    for (const auto& segment : plan.segments) {
        check(segment.comm_ops.empty(), "unexpected communications");
        ggml_context* part_ctx = nullptr;
        auto* part = Cut::build_segment_graph(graph, segment, &part_ctx);
        check(part != nullptr, "real segment graph");
        for (int i = 0; i < ggml_graph_n_nodes(part); ++i) {
            if (ggml_graph_node(part, i) != raw) continue;
            ++owners;
            CustomParamsLayout after{}; std::memcpy(&after, raw->op_params, sizeof(after));
            check(after.fun == before.fun && after.userdata == before.userdata && after.n_tasks == before.n_tasks,
                  "cut builder corrupted H3SA tag");
        }
        ggml_free(part_ctx);
    }
    check(owners == 1, "H3SA must execute in exactly one segment");
    Cut::clear_graph_cut_marks();
    std::cout << "PASS actual CPU cut planner: three segments, H3SA atomic/tag preserved (NO CUDA execution)\n";
}

int main() {
    try {
        // Explicit CPU init only. No load_all(), no CUDA dependencies or calls.
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_init_by_name("CPU", nullptr), ggml_backend_free);
        check(cpu != nullptr, "explicit CPU backend unavailable");
        TestEnv pv("ED_SAGE_PV_FP8", "0");
        test_policy(cpu.get());
        test_tags();
        test_dispatch_predicate();
        test_shapes();
        test_real_cut_plan(cpu.get());
        test_custom_graph();
        test_output_layout();
        test_shared_scaling(cpu.get());
        std::cout << "H3 Sage CPU tests passed. CUDA dispatch/accuracy/performance NOT tested.\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
