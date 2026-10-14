// Diagnostic only: no model, production edits, CPU attention oracle or sweeps.
// Reuse the tiny test's layouts, shape checks and EXACT single-GPU guard. Its
// renamed main is never called (in particular, none of its CPU or GPU tests).
#define main h3_subblock_tiny_test_main_unused
#include "h3_subblock_test.cpp"
#undef main
// The helper's private logger collides with ggml-impl.h's C declaration.
// Rename only that header-local utility, not any attention implementation.
#define ggml_log_callback_default h3_perf_extend_log_callback_default
#include "core/backend/ggml/ggml_extend.hpp" // the actual H3 production helper
#undef ggml_log_callback_default
#include "core/backend/cuDNN/ed_cudnn_sdpa.h"
#include <dlfcn.h>
#include "ggml-impl.h" // graph_view: immutable execution slices, not rebuilt graphs
#include <chrono>
#include <iomanip>
#include <sstream>


// Linux diagnostic-only interposition. cuDNN enablement is cached inside the
// DSO, so toggling ED_DISABLE_CUDNN_SDPA between graphs would silently misroute.
// Bypass ONLY this probe's stock tensor; production calls the real DSO function
// unchanged. Observe both actual dispatches without cuDNN profiling/events.
namespace dispatch {
using Cudnn = ed_cudnn_sdpa_result_t (*)(ggml_tensor*, ed_cudnn_sdpa_stream_t);
using Stock = void (*)(ggml_backend_cuda_context&, ggml_tensor*);
using Supported = bool (*)(int, const ggml_tensor*);
static Cudnn cudnn = nullptr;
static Stock stock = nullptr;
static Supported stock_supported = nullptr;
static ggml_tensor *stock_tensor = nullptr, *production_tensor = nullptr;
static std::string library;
enum class Expected { Unchecked, Stock, Cudnn };
struct Counts { int cudnn_calls = 0, cudnn_success = 0, stock_calls = 0, bypasses = 0; };
static thread_local Counts counts, warmup_counts, measured_counts;
static void resolve() {
    cudnn = reinterpret_cast<Cudnn>(dlsym(RTLD_NEXT, "_Z21ed_cudnn_sdpa_computeP11ggml_tensorPv"));
    stock = reinterpret_cast<Stock>(dlsym(RTLD_NEXT, "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor"));
    stock_supported = reinterpret_cast<Supported>(dlsym(RTLD_NEXT, "_Z34ggml_cuda_flash_attn_ext_supportediPK11ggml_tensor"));
    check(cudnn && stock && stock_supported, "missing real CUDA/cuDNN dispatch symbols");
    Dl_info info{};
    check(dladdr(reinterpret_cast<void*>(cudnn), &info) && info.dli_fname, "cannot attest CUDA library path");
    library = info.dli_fname;
}
static void verify(Expected expected) {
    if (expected == Expected::Cudnn)
        check(counts.cudnn_calls == 1 && counts.cudnn_success == 1 && counts.stock_calls == 0 && counts.bypasses == 0,
              "production cuDNN route NOT attested (fallback or dispatch interposition unavailable)");
    if (expected == Expected::Stock)
        check(counts.stock_calls == 1 && counts.cudnn_calls == 0 && counts.bypasses == 1,
              "stock ggml route NOT attested");
}
} // namespace dispatch

ed_cudnn_sdpa_result_t ed_cudnn_sdpa_compute(ggml_tensor* dst, ed_cudnn_sdpa_stream_t stream) {
    if (dst == dispatch::stock_tensor) {
        ++dispatch::counts.bypasses;
        return ED_CUDNN_SDPA_DISABLED;
    }
    // resolve() runs outside graph timing, before any CUDA backend calls.
    const auto result = dispatch::cudnn(dst, stream);
    if (dst == dispatch::production_tensor) {
        ++dispatch::counts.cudnn_calls;
        if (result == ED_CUDNN_SDPA_SUCCESS) ++dispatch::counts.cudnn_success;
    }
    return result;
}
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    if (dst == dispatch::stock_tensor || dst == dispatch::production_tensor) ++dispatch::counts.stock_calls;
    dispatch::stock(ctx, dst);
}

namespace perf {
constexpr size_t MiB = size_t(1) << 20;
constexpr size_t tensor_limit = size_t(12) << 30;
constexpr size_t reserve_bytes = 256 * MiB;
constexpr size_t host_chunk_elements = MiB; // 4 MiB, independent of sequence

struct Options {
    int sequence = 15000, heads = 56, dimension = 128;
    int video = -1, protected_tokens = 0, topk = -1, seed = 977;
    bool run = false, describe = false, skip_dense = false, skip_stock = false, skip_production = false;
};

static std::string quote(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

static int integer(const char* value) {
    const std::string text(value);
    size_t used = 0;
    const long long n = std::stoll(text, &used);
    if (used != text.size() || n < -1 || n > INT32_MAX)
        throw std::invalid_argument("invalid bounded integer: " + text);
    return int(n);
}

static Options parse(int argc, char** argv) {
    Options o;
    bool sequence_set = false, protected_set = false;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        if (key == "--seq") key = "--sequence";
        if (std::find(seen.begin(), seen.end(), key) != seen.end())
            throw std::invalid_argument("duplicate option: " + key);
        seen.push_back(key);
        if (key == "--run") o.run = true;
        else if (key == "--describe") o.describe = true;
        else if (key == "--skip-dense") o.skip_dense = true;
        else if (key == "--skip-stock") o.skip_stock = true;
        else if (key == "--skip-production") o.skip_production = true;
        else {
            if (i + 1 == argc) throw std::invalid_argument("missing value: " + key);
            const int n = integer(argv[++i]);
            if (key == "--sequence") { o.sequence = n; sequence_set = true; }
            else if (key == "--heads") o.heads = n;
            else if (key == "--head-dim") o.dimension = n;
            else if (key == "--video-tokens") o.video = n;
            else if (key == "--protected-tokens") { o.protected_tokens = n; protected_set = true; }
            else if (key == "--topk") o.topk = n;
            else if (key == "--seed") o.seed = n;
            else throw std::invalid_argument("unknown option: " + key);
        }
    }
    if (o.run == o.describe) throw std::invalid_argument("choose guarded --run or device-free --describe");
    if (o.sequence < 1 || o.sequence > native::max_sequence || o.protected_tokens < 0 ||
        o.protected_tokens >= native::max_sequence)
        throw std::invalid_argument("sequence/protected count outside native bounds");
    if (o.video != -1) {
        if (o.video <= 0) throw std::invalid_argument("video token count must be positive");
        if (!sequence_set) {
            const int64_t total = int64_t(o.video) + o.protected_tokens;
            if (total > native::max_sequence) throw std::invalid_argument("sequence exceeds native bound");
            o.sequence = int(total);
        } else if (!protected_set) o.protected_tokens = o.sequence - o.video;
    } else o.video = o.sequence - o.protected_tokens;
    if (o.sequence < 1 || o.sequence > native::max_sequence || o.heads < 1 || o.heads > native::max_heads ||
        o.dimension != native::dimension || o.video < 1 || o.protected_tokens < 0 ||
        int64_t(o.video) + o.protected_tokens != o.sequence || o.seed < 0 || o.topk < -1 || o.topk == 0)
        throw std::invalid_argument("require N=1..65536, H=1..64, head-dim=128, video>0, protected>=0, "
                                    "N=video+protected, topk=-1 or positive, seed>=0");
    return o;
}

static void configure(Fixture& f, const Options& o) {
    // A protected prefix followed by video, including non-64-aligned boundaries.
    // Metadata encoding and the trailing dummy entries exactly match Fixture.
    f.layout.spans.clear();
    if (o.protected_tokens) f.layout.spans.push_back({0, o.protected_tokens, SparseTokenKind::Text});
    f.layout.spans.push_back({o.protected_tokens, o.sequence, SparseTokenKind::TargetVideo});
    std::string why;
    check(h3_sparse_detail::make_plan(o.sequence, o.heads, o.dimension, 1.f/128,
                                     f.options, f.layout, &f.plan, &why), why.c_str());
    f.blocks.clear(); f.chunks.clear(); f.protected_ids.clear();
    for (const auto& b : f.plan.video) {
        f.blocks.push_back(int32_t(b.begin)); f.blocks.push_back(int32_t(b.end));
    }
    for (const auto& span : f.layout.spans) if (span.kind != SparseTokenKind::TargetVideo) {
        for (int64_t t = span.begin; t < span.end; ++t) f.protected_ids.push_back(int32_t(t));
        for (int64_t t = span.begin; t < span.end; t += 64) {
            f.chunks.push_back(int32_t(t)); f.chunks.push_back(int32_t(std::min(t + 64, span.end)));
        }
    }
    f.chunks.insert(f.chunks.end(), {0, 0}); f.protected_ids.push_back(-1);
}

static size_t context_bytes(ggml_context* ctx) {
    size_t bytes = 0;
    for (auto* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
        if (!t->view_src) bytes += ggml_nbytes(t) + 256; // conservative per-tensor alignment
    return bytes;
}

static void device_buffer(ggml_backend_buffer_t buffer, ggml_backend_t backend) {
    check(buffer && !ggml_backend_buffer_is_host(buffer), "missing CUDA allocation / host fallback");
    check(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) == ggml_backend_get_device(backend),
          "wrong GPU allocation");
}

struct Dense {
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *mask = nullptr, *flash = nullptr, *out = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    int kv_length, mask_queries;
    explicit Dense(const Graph& sparse) {
        const int n = sparse.fixture.length;
        kv_length = (n + 255) / 256 * 256;
#ifdef GGML_KQ_MASK_PAD
        mask_queries = (n + GGML_KQ_MASK_PAD - 1) / GGML_KQ_MASK_PAD * GGML_KQ_MASK_PAD;
#else
        mask_queries = n;
#endif
        ctx = ggml_init({4u << 20, nullptr, true}); check(ctx, "dense metadata context");
        auto* k = sparse.k;
        // Match ggml_ext_attention_ext's baseline F32 scale -> F16 cast, V
        // layout conversion, KV-256 padding/mask, F32 precision and output *128.
        // Do NOT call that helper: it can choose cuDNN or a quadratic fallback.
        auto* v = ggml_cont(ctx, ggml_permute(ctx, sparse.v, 0, 2, 1, 3));
        if (kv_length != n) {
            k = ggml_pad(ctx, k, 0, kv_length - n, 0, 0);
            v = ggml_pad(ctx, v, 0, kv_length - n, 0, 0);
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_length, mask_queries);
            ggml_set_name(mask, "perf.dense.padding_mask");
        }
        k = ggml_cast(ctx, ggml_scale(ctx, k, 1.f/128), GGML_TYPE_F16);
        v = ggml_cast(ctx, ggml_scale(ctx, v, 1.f/128), GGML_TYPE_F16);
        flash = ggml_flash_attn_ext(ctx, sparse.q, k, v, mask, 128.f/std::sqrt(128.f), 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        ggml_set_name(flash, "perf.dense.stock_flash_f32");
        out = ggml_scale(ctx, flash, 128.f);
        graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, out);
        check(ggml_graph_n_nodes(graph) <= 16, "dense graph must remain constant-size");
        check(ggml_nelements(out) == ggml_nelements(sparse.out), "dense output layout mismatch");
    }
    ~Dense() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
    std::string unsupported(ggml_backend_t backend, size_t sparse_bytes) const {
        if (!dispatch::stock_supported(0, flash)) return "stock ggml FLASH_ATTN_EXT unsupported";
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto* t = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, t)) return std::string("CUDA does not support ") + ggml_op_name(t->op);
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total);
        const size_t bytes = context_bytes(ctx);
        if (bytes + sparse_bytes > tensor_limit) return "dense + sparse tensors exceed diagnostic 12 GiB bound";
        if (bytes + reserve_bytes > free) return "insufficient free VRAM for dense graph plus reserve";
        return {};
    }
    bool alloc(ggml_backend_t backend, int real_keys) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) return false;
        device_buffer(buffer, backend);
        // Mask is fixed input metadata. Its allocation/upload is NOT timed.
        if (mask) {
            const size_t rows_per_chunk = std::max(size_t(1), host_chunk_elements / size_t(kv_length));
            std::vector<ggml_fp16_t> chunk(rows_per_chunk * kv_length, ggml_fp32_to_fp16(0.f));
            for (size_t row = 0; row < rows_per_chunk; ++row)
                std::fill(chunk.begin() + row * kv_length + real_keys, chunk.begin() + (row + 1) * kv_length,
                          ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int row = 0; row < mask_queries; row += int(rows_per_chunk)) {
                const size_t rows = std::min(rows_per_chunk, size_t(mask_queries - row));
                ggml_backend_tensor_set(mask, chunk.data(), size_t(row) * kv_length * sizeof(ggml_fp16_t),
                                        rows * kv_length * sizeof(ggml_fp16_t));
            }
        }
        return true;
    }
};

// Same call as MiniMax-H3 Attention::forward, NOT a hand-built approximation.
// Its backend-dependent branch requires a real guarded CUDA backend; --describe
// therefore never invokes this helper. Unsupported tiny geometries are reported
// rather than silently benchmarking ggml as the production cuDNN baseline.
struct ProductionDense {
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *flash = nullptr, *out = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ProductionDense(const Graph& sparse, ggml_backend_t backend) {
        ctx = ggml_init({4u << 20, nullptr, true}); check(ctx, "production metadata context");
        out = ggml_ext_attention_ext(ctx, backend, sparse.q, sparse.k, sparse.v,
                                    sparse.fixture.heads, nullptr, true, true, 1.f/128.f,
                                    sparse.fixture.length < 65536);
        graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, out);
        check(ggml_graph_n_nodes(graph) <= 20, "production graph must remain constant-size");
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            check(node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_SOFT_MAX,
                  "production helper took quadratic dense fallback; refusing execution");
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                check(!flash, "multiple production attention ops"); flash = node;
            }
        }
        check(flash && ggml_nelements(out) == ggml_nelements(sparse.out), "production FLASH_ATTN_EXT/layout missing");
        check(flash->type == GGML_TYPE_F32 && flash->src[0] == sparse.q &&
              flash->src[1]->type == GGML_TYPE_F16 && flash->src[2]->type == GGML_TYPE_F16 &&
              !flash->src[3] && flash->src[1]->ne[1] == sparse.fixture.length &&
              ggml_flash_attn_ext_get_prec(flash) == GGML_PREC_F32,
              "production helper no longer matches unpadded F32 self-attention with F16 KV prep");
    }
    ~ProductionDense() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
    std::string prepare(ggml_backend_t backend, size_t sparse_bytes) {
        if (!ed_cudnn_sdpa_supported(flash, 0)) return "production cuDNN shape/device unsupported";
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto* t = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, t)) return std::string("production CUDA op unsupported: ") + ggml_op_name(t->op);
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &free, &total);
        // cuDNN has cached Q/O conversion buffers in addition to ggml tensors.
        const size_t conversion_bytes = size_t(ggml_nelements(flash)) * 2 * sizeof(ggml_fp16_t);
        const size_t bytes = context_bytes(ctx);
        if (sparse_bytes + bytes + conversion_bytes > tensor_limit) return "production + sparse tensors exceed diagnostic bound";
        if (bytes + conversion_bytes + reserve_bytes > free) return "insufficient VRAM for production tensors/conversion/reserve";
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) return "production CUDA allocation failed";
        device_buffer(buffer, backend);
        float scale = 0; std::memcpy(&scale, flash->op_params, sizeof(scale));
        // Builds/allocates the SAME production plan but does not execute it.
        // Plan construction is outside both the one warmup and timed pass.
        const auto ready = ed_cudnn_sdpa_prewarm_self_attn(0, flash->type, flash->src[0]->ne[0],
                            flash->src[0]->ne[2], flash->src[0]->ne[1], scale, true);
        if (ready != ED_CUDNN_SDPA_SUCCESS) return std::string("production sync plan: ") + ed_cudnn_sdpa_result_name(ready);
        ggml_backend_synchronize(backend);
        return {};
    }
};

// All synthetic data and validation transfers are streamed through a bounded
// host buffer. Dense and sparse read the SAME device Q/K/V, never new samples.
static void upload(Graph& g, int seed, std::vector<float>& chunk) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> random(-.75f, .75f);
    for (auto* t : {g.q, g.k, g.v}) {
        const size_t count = size_t(ggml_nelements(t));
        for (size_t offset = 0; offset < count; offset += chunk.size()) {
            const size_t n = std::min(chunk.size(), count - offset);
            for (size_t i = 0; i < n; ++i) chunk[i] = random(rng);
            ggml_backend_tensor_set(t, chunk.data(), offset * sizeof(float), n * sizeof(float));
        }
    }
}

static void poison(ggml_tensor* out, std::vector<float>& chunk) {
    // IEEE quiet NaN for F32, -1 for I32. Use bytes, not an aliasing type pun.
    if (out->type == GGML_TYPE_I32) std::memset(chunk.data(), 0xff, chunk.size() * sizeof(float));
    else std::fill(chunk.begin(), chunk.end(), std::numeric_limits<float>::quiet_NaN());
    const size_t bytes = ggml_nbytes(out), step = chunk.size() * sizeof(float);
    for (size_t offset = 0; offset < bytes; offset += step)
        ggml_backend_tensor_set(out, chunk.data(), offset, std::min(step, bytes - offset));
}

static double measure(ggml_backend_t backend, ggml_cgraph* graph, ggml_tensor* out, std::vector<float>& chunk,
                      dispatch::Expected expected = dispatch::Expected::Unchecked) {
    poison(out, chunk);
    ggml_backend_synchronize(backend);
    dispatch::counts = {};
    const auto warmup = ggml_backend_graph_compute(backend, graph); // exactly ONE warmup
    ggml_backend_synchronize(backend);
    check(warmup == GGML_STATUS_SUCCESS, "warmup failed");
    dispatch::verify(expected); dispatch::warmup_counts = dispatch::counts;
    poison(out, chunk); // a warmup's valid output must not hide an unwritten measured row
    ggml_backend_synchronize(backend);
    dispatch::counts = {};
    const auto begin = std::chrono::steady_clock::now();
    const auto status = ggml_backend_graph_compute(backend, graph); // exactly ONE measurement
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    check(status == GGML_STATUS_SUCCESS, "measured CUDA graph failed");
    dispatch::verify(expected); dispatch::measured_counts = dispatch::counts;
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static std::string finite_output(ggml_tensor* out, std::vector<float>& chunk) {
    float max_abs = 0; double sum = 0;
    const size_t count = size_t(ggml_nelements(out));
    for (size_t offset = 0; offset < count; offset += chunk.size()) {
        const size_t n = std::min(chunk.size(), count - offset);
        ggml_backend_tensor_get(out, chunk.data(), offset * sizeof(float), n * sizeof(float));
        for (size_t i = 0; i < n; ++i) {
            check(std::isfinite(chunk[i]), "nonfinite / unwritten output");
            max_abs = std::max(max_abs, std::abs(chunk[i])); sum += chunk[i];
        }
    }
    std::ostringstream s;
    s << std::setprecision(12) << "\"finite\":true,\"elements_checked\":" << count
      << ",\"max_abs\":" << max_abs << ",\"sum\":" << sum;
    return s.str();
}

static std::vector<int32_t> capture_routes(const Graph& g) {
    std::vector<int32_t> routes(size_t(ggml_nelements(g.route)));
    ggml_backend_tensor_get(g.route, routes.data(), 0, routes.size() * sizeof(int32_t));
    const int blocks = int(g.fixture.plan.video.size()), topk = g.fixture.plan.topk;
    std::vector<int> seen(blocks, -1);
    for (size_t i = 0; i < routes.size(); ++i) {
        check(routes[i] >= 0 && routes[i] < blocks, "invalid / unwritten route ID");
        const int row = int(i / topk);
        check(seen[routes[i]] != row, "duplicate block within one route"); seen[routes[i]] = row;
    }
    return routes;
}

static void result(const char* name, double ms, const std::string& validation) {
    std::cout << std::setprecision(12) << "{\"event\":\"timing\",\"kernel\":" << quote(name)
              << ",\"status\":\"ok\",\"warmups\":1,\"measurements\":1,\"ms\":" << ms
              << ',' << validation << "}" << std::endl;
}

static std::string attestation(dispatch::Expected expected) {
    const auto& w = dispatch::warmup_counts; const auto& m = dispatch::measured_counts;
    std::ostringstream out;
    out << "\"route_attested\":true,\"actual_attention_backend\":" << quote(expected == dispatch::Expected::Cudnn ? "cudnn_sdpa" : "ggml_cuda_flash")
        << ",\"warmup_cudnn_successes\":" << w.cudnn_success << ",\"measured_cudnn_successes\":" << m.cudnn_success
        << ",\"warmup_stock_flash_calls\":" << w.stock_calls << ",\"measured_stock_flash_calls\":" << m.stock_calls
        << ",\"attestation\":\"real_DSO_dispatch_return_counters\",\"cuda_library\":" << quote(dispatch::library);
    return out.str();
}
static void unavailable(const char* name, bool skip, const std::string& reason) {
    std::cout << "{\"event\":\"timing\",\"kernel\":" << quote(name) << ",\"status\":"
              << quote(skip ? "skipped" : "unsupported") << ",\"ms\":null,\"reason\":" << quote(reason) << "}" << std::endl;
}
static void production_measure(const Options& o, Graph& g, ggml_backend_t backend, size_t sparse_bytes,
                               std::vector<float>& chunk) {
    const char* name = "dense_production_cudnn";
    if (o.skip_dense || o.skip_production) { unavailable(name, true, "production dense skipped"); return; }
    if (o.sequence < 4096) { unavailable(name, false, "production long F32 cuDNN self-attention requires N>=4096"); return; }
    ProductionDense production(g, backend);
    const std::string why = production.prepare(backend, sparse_bytes);
    if (!why.empty()) { unavailable(name, false, why); return; }
    std::cout << "{\"event\":\"production_graph\",\"helper\":\"ggml_ext_attention_ext\",\"ggml_nodes\":"
              << ggml_graph_n_nodes(production.graph) << ",\"tensor_bytes_estimate\":" << context_bytes(production.ctx)
              << ",\"flash_attn\":true,\"skip_reshape\":true,\"kv_scale\":0.0078125,\"pad_kv_arg\":"
              << (o.sequence < 65536 ? "true" : "false") << ",\"actual_kv_padding\":0,\"mask\":false"
              << ",\"flash_q_type\":\"F32\",\"flash_kv_type\":\"F16\",\"flash_output_type\":\"F32\""
              << ",\"precision\":\"GGML_PREC_F32\",\"sync_plan_ready\":true,\"plan_build_timed\":false"
              << ",\"shared_device_qkv\":true,\"includes_production_kv_prep_and_output_scaling\":true}" << std::endl;
    check(setenv("ED_CUDNN_SDPA_NO_FALLBACK", "1", 1) == 0, "require cuDNN route failed");
    dispatch::production_tensor = production.flash;
    const double ms = measure(backend, production.graph, production.out, chunk, dispatch::Expected::Cudnn);
    result(name, ms, "\"role\":\"production_baseline\"," + attestation(dispatch::Expected::Cudnn) + ',' + finite_output(production.out, chunk));
    dispatch::production_tensor = nullptr;
    check(setenv("ED_CUDNN_SDPA_NO_FALLBACK", "0", 1) == 0, "restore stock dispatch policy failed");
} // production tensors released before the stock diagnostic allocation

struct Backend {
    ggml_backend_t value = nullptr;
    ~Backend() { if (value) ggml_backend_free(value); }
};

static int probe(const Options& o) {
    // Match production's synchronous F32 self-attention route. These are set
    // BEFORE any backend/support/plan calls because the DSO caches some flags.
    // Keep graph capture and intrusive cuDNN profiling outside the diagnostic.
    if (o.run) {
        check(setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1) == 0, "disable capture failed");
        check(setenv("ED_DISABLE_CUDNN_SDPA", "0", 1) == 0, "enable production cuDNN failed");
        check(setenv("ED_DISABLE_CUDNN_SDPA_UNPAD", "0", 1) == 0, "enable production unpadded route failed");
        check(setenv("ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN", "1", 1) == 0, "enable synchronous production plan failed");
        check(setenv("ED_PROFILE_CUDNN_SDPA", "0", 1) == 0, "disable intrusive cuDNN profiler failed");
        check(setenv("ED_CUDNN_SDPA_NO_FALLBACK", "0", 1) == 0, "allow explicit stock dispatch failed");
        dispatch::resolve();
    }
    Backend backend; // outlives graph buffers
    Fixture f(o.sequence, o.heads, false, o.topk, false); configure(f, o);
    native::Capacity cap;
    check(native::capacity(o.sequence, o.heads, f.plan.video.size(), f.plan.topk, o.protected_tokens, &cap),
          "native bounded capacity rejected");
    Graph g(f, false, true);
    check(ggml_graph_n_nodes(g.graph) == 2 && ggml_graph_node(g.graph, 0) == g.route &&
          ggml_graph_node(g.graph, 1) == g.out, "expected exactly router then indexed attention");
    // Neither slice mutates op/src/route metadata. The attention slice contains
    // ONLY the indexed op; it consumes the router slice's device output directly.
    auto router = ggml_graph_view(g.graph, 0, 1);
    auto attention = ggml_graph_view(g.graph, 1, 2);
    Dense dense(g); // metadata only; no dense allocation until sparse timings finish
    const size_t sparse_bytes = context_bytes(g.ctx);
    const size_t router_scratch = cap.means_bytes + cap.scores_bytes;
    std::cout << "{\"event\":\"metadata\",\"mode\":" << quote(o.run ? "run" : "describe")
              << ",\"sequence\":" << o.sequence << ",\"heads\":" << o.heads << ",\"head_dim\":128"
              << ",\"video_tokens\":" << o.video << ",\"protected_tokens\":" << o.protected_tokens
              << ",\"layout\":\"protected_prefix_then_video\",\"video_blocks\":" << f.plan.video.size()
              << ",\"protected_chunks\":" << f.chunks.size()/2-1 << ",\"topk\":" << f.plan.topk
              << ",\"seed\":" << o.seed << ",\"input\":\"mt19937_uniform_f32_minus0.75_plus0.75\""
              << ",\"qk_layout\":\"D,N,H\",\"v_and_output_layout\":\"D,H,N\",\"shared_device_qkv\":true"
              << ",\"router_nodes\":1,\"attention_nodes\":1,\"dense_nodes\":" << ggml_graph_n_nodes(dense.graph)
              << ",\"sparse_tensor_bytes_estimate\":" << sparse_bytes << ",\"router_scratch_bytes\":" << router_scratch
              << ",\"indexed_shared_bytes_per_cta\":" << native::indexed_shared_bytes
              << ",\"dense_tensor_bytes_estimate\":" << context_bytes(dense.ctx)
              << ",\"dense_kv_tokens\":" << dense.kv_length << ",\"dense_mask_queries\":" << (dense.mask ? dense.mask_queries : 0)
              << ",\"dense_precision\":\"GGML_PREC_F32\",\"dense_includes\":\"V_contiguous,KV_pad,KV_f32_div128_to_f16,FLASH_ATTN_EXT,output_mul128\""
              << ",\"mask_setup_timed\":false,\"cuda_graph_capture\":false,\"production_cudnn_sdpa\":true"
              << ",\"production_sync_f32_self_attn\":true,\"production_helper\":\"core/backend/ggml/ggml_extend.hpp:ggml_ext_attention_ext\""
              << ",\"stock_dense_role\":\"diagnostic_not_production_baseline\",\"production_graph_requires_guarded_backend\":true"
              << ",\"clock\":\"steady_clock_backend_synchronized\",\"timed_scope\":\"graph_compute_plus_completion_sync\""
              << ",\"allocation_upload_download_validation_timed\":false,\"large_cpu_reference\":false}" << std::endl;
    if (o.describe) return 0; // no CUDA discovery, allocations, upload or execution
    check(ggml_backend_cuda_get_device_count() == 1, "exactly one visible CUDA device required");
    backend.value = ggml_backend_cuda_init(0);
    check(backend.value && ggml_backend_is_cuda(backend.value), "CUDA backend unavailable");
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend.value), &free, &total);
    std::cout << "{\"event\":\"device\",\"uuid\":" << quote(env("CUDA_VISIBLE_DEVICES"))
              << ",\"description\":" << quote(ggml_backend_dev_description(ggml_backend_get_device(backend.value)))
              << ",\"free_bytes\":" << free << ",\"total_bytes\":" << total << "}" << std::endl;
    check(sparse_bytes + router_scratch <= tensor_limit && sparse_bytes + router_scratch + reserve_bytes <= free,
          "insufficient VRAM / diagnostic sparse bound exceeded");
    g.alloc(backend.value); device_buffer(g.buffer, backend.value);
    std::vector<float> chunk(host_chunk_elements);
    upload(g, o.seed, chunk);
    const double router_ms = measure(backend.value, &router, g.route, chunk);
    const auto routes = capture_routes(g);
    result("router", router_ms, "\"valid_route_ids\":true,\"unique_per_row\":true,\"route_ids_checked\":" + std::to_string(routes.size()));
    const double attention_ms = measure(backend.value, &attention, g.out, chunk);
    check(capture_routes(g) == routes, "attention altered the already-produced route");
    result("indexed_attention", attention_ms, "\"router_in_timed_graph\":false,\"route_unchanged\":true," + finite_output(g.out, chunk));
    production_measure(o, g, backend.value, sparse_bytes, chunk);
    const bool skip_stock = o.skip_dense || o.skip_stock;
    std::string why = skip_stock ? "stock dense skipped" : dense.unsupported(backend.value, sparse_bytes);
    if (why.empty() && !dense.alloc(backend.value, o.sequence)) why = "dense CUDA allocation failed";
    if (!why.empty()) {
        unavailable("dense_stock_ggml_diagnostic", skip_stock, why);
        return 0;
    }
    dispatch::stock_tensor = dense.flash;
    const double dense_ms = measure(backend.value, dense.graph, dense.out, chunk, dispatch::Expected::Stock);
    result("dense_stock_ggml_diagnostic", dense_ms, "\"role\":\"diagnostic_not_production_baseline\"," +
           attestation(dispatch::Expected::Stock) + ',' + finite_output(dense.out, chunk));
    dispatch::stock_tensor = nullptr;
    return 0;
}
} // namespace perf

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Bounded H3 CUDA graph probe (default OFF; no model / CPU attention reference).\n"
                     "  --run | --describe (metadata only, never initializes CUDA)\n"
                     "  --sequence/--seq N (15000) --heads H (56) --head-dim D (128 ONLY)\n"
                     "  --video-tokens V --protected-tokens P (0) --topk K (-1 = native half keep)\n"
                     "  --seed S (977) --skip-dense (both) --skip-stock --skip-production\n"
                     "With video+protected and no sequence, N=V+P; with sequence+video, P=N-V.\n"
                     "Run requires the exact fresh UUID/env guard from test_h3_gpu_parity.sh.\n"
                     "Production cuDNN baseline uses the real H3 helper; stock ggml is a separate diagnostic.\n"
                     "ONE warmup + ONE measured pass per graph. Real route counters; no intrusive profiler.\n"
                     "JSONL stdout, diagnostics stderr. Production shape/plan unavailable is explicit.\n"
                     "Exit 0: results/description (dense may be unsupported); 1: failure; 2: not run.\n";
        return 0;
    }
    perf::Options options;
    try { options = perf::parse(argc, argv); }
    catch (const std::exception& e) {
        std::cout << "{\"status\":\"not_run\",\"reason\":" << perf::quote(e.what()) << "}" << std::endl;
        return 2;
    }
    if (options.run) {
        // permit is reused verbatim. Only normalize argc/argv after strict CLI
        // parsing, so shape options do not weaken the original --run-only guard.
        char flag[] = "--run"; char* guarded[] = {argv[0], flag, nullptr};
        if (!permit(2, guarded)) {
            std::cout << "{\"status\":\"not_run\",\"reason\":\"missing/stale single-GPU guard\"}" << std::endl;
            return 2;
        }
    }
    try { return perf::probe(options); }
    catch (const std::exception& e) {
        std::cout << "{\"status\":\"failed\",\"reason\":" << perf::quote(e.what()) << "}" << std::endl;
        return 1;
    }
}
