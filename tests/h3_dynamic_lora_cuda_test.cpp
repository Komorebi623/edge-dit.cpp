// Optional sidecar; the original CPU test and frozen adapter header are untouched.
// No arguments: skip without backend initialization. --run requires the same
// fresh, single-UUID approval contract as h3_subblock_test.cpp. Never auto-fallback
// to CPU. Synthetic Q4_0/Q4_K weights are quantized, not relabelled float buffers.
#define main h3_dynamic_lora_cpu_test_main
#include "h3_dynamic_lora_test.cpp"
#undef main
#include "ggml-cuda.h"
#include <ctime>

namespace {
std::string env(const char* key) { const char* p = std::getenv(key); return p ? p : ""; }
bool permit(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "--run" || env("H3_GPU_PARITY_ALLOW") != "1") return false;
    const auto uuid = env("CUDA_VISIBLE_DEVICES");
    if (uuid.size() != 40 || uuid.substr(0, 4) != "GPU-" || uuid != env("H3_GPU_PARITY_PREFLIGHT_DEVICE")) return false;
    for (size_t i = 4; i < uuid.size(); ++i) {
        if (i == 12 || i == 17 || i == 22 || i == 27) { if (uuid[i] != '-') return false; }
        else if (std::string("0123456789abcdefABCDEF").find(uuid[i]) == std::string::npos) return false;
    }
    const auto stamp = env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    if (stamp.empty() || stamp.find_first_not_of("0123456789") != std::string::npos) return false;
    char* end = nullptr;
    const double at = std::strtod(stamp.c_str(), &end), age = double(std::time(nullptr)) - at;
    return end && !*end && std::isfinite(age) && age >= 0 && age <= 30;
}
struct Buffer {
    ggml_backend_buffer_t value = nullptr;
    ~Buffer() { ggml_backend_buffer_free(value); }
};
struct Case {
    ggml_type base, a, b;
    int64_t in, rank, out, tokens, d2 = 1, d3 = 1;
    float coefficient = 1.f, conditioning = 1.f;
    bool strided = false;
};
int cases = 0, comparisons = 0;
void near(float got, float expected, float atol, float rtol, const char* label, size_t index) {
    ++comparisons;
    check(std::isfinite(got) && std::isfinite(expected) && std::abs(got - expected) <= atol + rtol * std::abs(expected),
          std::string(label) + " index=" + std::to_string(index) + " got=" + std::to_string(got) +
          " expected=" + std::to_string(expected));
}
std::vector<float> read_f32(ggml_tensor* t) {
    const auto raw = bytes(t);
    std::vector<float> result(ggml_nelements(t));
    check(t->type == GGML_TYPE_F32 && raw.size() == result.size() * sizeof(float), "F32 output expected");
    std::memcpy(result.data(), raw.data(), raw.size());
    return result;
}
void run_case(ggml_backend_t backend, const Case& s) {
    std::cout << "CUDA case=" << cases << " base=" << ggml_type_name(s.base)
              << " A=" << ggml_type_name(s.a) << " B=" << ggml_type_name(s.b)
              << " in/rank/out=" << s.in << '/' << s.rank << '/' << s.out
              << " tokens/batch=" << s.tokens << '/' << s.d2 << '/' << s.d3
              << " coefficient=" << s.coefficient << " strided=" << s.strided << std::endl;
    Context pc(true), gc(true);
    auto* w = ggml_new_tensor_2d(pc.ctx, s.base, s.in, s.out);
    auto* bias = ggml_new_tensor_1d(pc.ctx, GGML_TYPE_F32, s.out);
    ggml_set_name(w, (module + ".weight").c_str());
    ggml_set_name(bias, (module + ".bias").c_str());
    Adapter::TensorMap base{{module + ".weight", w}};
    String2TensorStorage storage;
    add_pair(storage, module, s.a, s.b, s.in, s.rank, s.out);
    Adapter adapter(pc.ctx, base, storage, s.coefficient);
    auto* a = adapter.tensors().at(module + ".lora_A.weight");
    auto* b = adapter.tensors().at(module + ".lora_B.weight");
    Buffer params;
    params.value = ggml_backend_alloc_ctx_tensors(pc.ctx, backend);
    check(params.value != nullptr, "CUDA runner parameter allocation");
    check(w->buffer == a->buffer && w->buffer == b->buffer, "factors belong to runner buffer");
    check(a->type == s.a && b->type == s.b, "factor dtypes unchanged");
    // Reuse the CPU test's encoding/decoding helpers: the scalar oracle consumes
    // the actual quantized base and actual rounded BF16/F16 factor values.
    const auto wr = fill(w, .018f), ar = fill(a, .008f), br = fill(b, .02f), biasr = fill(bias, .007f);
    const auto wbytes = bytes(w), abytes = bytes(a), bbytes = bytes(b);
    const int64_t raw_tokens = s.tokens + (s.strided ? 2 : 0);
    auto* raw_x = ggml_new_tensor_4d(gc.ctx, GGML_TYPE_F32, s.in, raw_tokens, s.d2, s.d3);
    auto* x = s.strided ? ggml_view_4d(gc.ctx, raw_x, s.in, s.tokens, s.d2, s.d3,
                                      raw_x->nb[1], raw_x->nb[2], raw_x->nb[3], raw_x->nb[1]) : raw_x;
    auto* baseline = ggml_ext_linear(gc.ctx, x, w, bias, true, s.conditioning, (module + ".weight").c_str());
    adapter.begin_graph();
    auto* out = adapter.forward_with_lora(gc.ctx, backend, x, w, bias, module + ".", linear(s.conditioning));
    // Independent CUDA oracle: native matmul/reshape/scale/add only, no call to
    // the adapter or ggml_ext_linear for either factor. Match backend operand
    // conversion/dispatch rather than assuming F32 storage means F32 products.
    auto* oracle = baseline;
    ggml_tensor* oracle_delta = nullptr;
    if (s.coefficient != 0.f) {
        auto* ox = x;
        if (ox->ne[2] * ox->ne[3] > 1024 && !ggml_is_contiguous(ox)) ox = ggml_cont(gc.ctx, ox);
        const auto project = [&](ggml_tensor* input, ggml_tensor* weight) {
            const int64_t d2 = input->ne[2], d3 = input->ne[3];
            const bool flatten = d2 * d3 > 1024;
            if (flatten) input = ggml_reshape_2d(gc.ctx, input, input->ne[0], input->ne[1] * d2 * d3);
            auto* y = ggml_mul_mat(gc.ctx, weight, input);
            ggml_mul_mat_set_prec(y, GGML_PREC_F32);
            return flatten ? ggml_reshape_4d(gc.ctx, y, y->ne[0], y->ne[1] / d2 / d3, d2, d3) : y;
        };
        oracle_delta = project(project(ox, a), b);
        if (s.coefficient != 1.f) oracle_delta = ggml_scale(gc.ctx, oracle_delta, s.coefficient);
        oracle = ggml_add(gc.ctx, baseline, oracle_delta);
    }
    auto* g = graph(gc, {baseline, out, oracle});
    adapter.validate_coverage(g);
    int matmuls = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto* node = ggml_graph_node(g, i);
        check(ggml_backend_supports_op(backend, node), std::string("CUDA unsupported op: ") + ggml_op_name(node->op));
        if (node->op == GGML_OP_MUL_MAT) {
            ++matmuls;
            check(node->src[0] == w || node->src[0] == a || node->src[0] == b, "unexpected weight materialization");
            if (node->src[0] == w) check(std::string(node->name) == module + ".weight", "original linear label lost");
        }
        check(!(node->op == GGML_OP_CPY && node->src[0] == w), "full base dequantization in graph");
    }
    check(matmuls == (s.coefficient == 0.f ? 2 : 6), "baseline plus adapter plus independent CUDA oracle matmul count");
    check(out->ne[0] == s.out && out->ne[1] == s.tokens && out->ne[2] == s.d2 && out->ne[3] == s.d3,
          "CUDA batch/output shape");
    Buffer scratch;
    scratch.value = ggml_backend_alloc_ctx_tensors(gc.ctx, backend);
    check(scratch.value != nullptr, "CUDA graph allocation");
    for (int replay = 0; replay < 2; ++replay) {
        const auto xr = fill(raw_x, replay ? -.009f : .013f);
        std::vector<float> poison(ggml_nelements(out), std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(out, poison.data(), 0, poison.size() * sizeof(float));
        ggml_backend_tensor_set(baseline, poison.data(), 0, poison.size() * sizeof(float));
        if (oracle != baseline) ggml_backend_tensor_set(oracle, poison.data(), 0, poison.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "CUDA execution failed (no fallback)");
        ggml_backend_synchronize(backend);
        ++graphs;
        const auto got = read_f32(out), original = read_f32(baseline);
        const auto oracle_values = read_f32(oracle);
        const auto cuda_delta = oracle_delta ? read_f32(oracle_delta) : std::vector<float>(got.size(), 0.f);
        // Only native-CUDA oracle equality gates adapter arithmetic. Scalar
        // comparisons remain explicit diagnostics for fixed-library rounding:
        // ggml-cuda enables CUBLAS_TF32_TENSOR_OP_MATH; F16/BF16/quantized paths
        // also convert operands. Do not relax a scalar tolerance to hide this.
        double max_base_error = 0., max_total_scalar_error = 0., max_residual_error = 0.;
        double max_cuda_delta_scalar_error = 0.;
        size_t max_base_index = 0;
        for (int64_t h = 0; h < s.d3; ++h) for (int64_t z = 0; z < s.d2; ++z) for (int64_t t = 0; t < s.tokens; ++t) {
            const size_t xi = size_t(s.in * ((s.strided ? 1 : 0) + t + raw_tokens * (z + s.d2 * h)));
            std::vector<double> ax(s.rank, 0.);
            for (int64_t r = 0; r < s.rank; ++r) for (int64_t k = 0; k < s.in; ++k) ax[r] += double(ar[r*s.in+k]) * xr[xi+k];
            for (int64_t j = 0; j < s.out; ++j) {
                double delta = 0., ref = biasr[j];
                for (int64_t r = 0; r < s.rank; ++r) delta += double(br[j*s.rank+r]) * ax[r];
                for (int64_t k = 0; k < s.in; ++k) ref += double(wr[j*s.in+k]) * xr[xi+k];
                const size_t i = size_t(j + s.out * (t + s.tokens * (z + s.d2 * h)));
                const float residual = float(s.coefficient * delta);
                near(got[i], oracle_values[i], 0.f, 0.f, "independent native CUDA oracle", i);
                check(std::isfinite(cuda_delta[i]), "nonfinite independent CUDA residual");
                max_cuda_delta_scalar_error = std::max(max_cuda_delta_scalar_error,
                                                       std::abs(double(cuda_delta[i]) - residual));
                check(std::isfinite(ref) && std::isfinite(original[i]), "nonfinite baseline diagnostic");
                const double base_error = std::abs(double(original[i]) - ref);
                if (base_error > max_base_error) { max_base_error = base_error; max_base_index = i; }
                max_total_scalar_error = std::max(max_total_scalar_error, std::abs(double(got[i]) - (ref + residual)));
                max_residual_error = std::max(max_residual_error, std::abs(double(got[i]) - original[i] - residual));
            }
        }
        std::cout << "APPROXIMATE_BASELINE_DIAGNOSTIC case=" << cases << " replay=" << replay
                  << " cuda_base_vs_scalar_max_abs=" << max_base_error << " index=" << max_base_index
                  << " cuda_total_vs_scalar_max_abs=" << max_total_scalar_error
                  << " residual_vs_independent_BAx_max_abs=" << max_residual_error
                  << " independent_cuda_BAx_vs_scalar_max_abs=" << max_cuda_delta_scalar_error
                  << " (scalar comparisons diagnostic; independent CUDA oracle is byte-exact gated)" << std::endl;
        check(bytes(out) == bytes(oracle), "adapter must byte-match independently built native CUDA oracle");
        if (s.coefficient == 0.f) check(bytes(out) == bytes(baseline), "CUDA coefficient zero must be bit-exact");
        check(bytes(w) == wbytes && bytes(a) == abytes && bytes(b) == bbytes, "CUDA modified immutable base/factors");
    }
    ++cases;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "NOT RUN: optional CUDA sidecar requires guarded --run; use the original binary for CPU tests\n";
        return 0;
    }
    if (!permit(argc, argv)) {
        std::cerr << "NOT RUN: missing/stale single-GPU guard (same contract as h3_subblock_test.cpp)\n";
        return 2;
    }
    try {
        static_assert(GGML_MAX_NAME == 128, "candidate requires maxname128");
        check(ggml_backend_cuda_get_device_count() == 1, "exactly one visible CUDA device required");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cuda_init(0), ggml_backend_free);
        check(backend && ggml_backend_is_cuda(backend.get()), "CUDA backend unavailable");
        // All input widths respect the quant block size; rank/output/token tails
        // intentionally do not respect CUDA tile widths. Include Light's r128.
        const std::vector<Case> matrix = {
            {GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  37, 5, 19, 1},
            {GGML_TYPE_F32,  GGML_TYPE_F32,  GGML_TYPE_F32,  37, 5, 19, 17, 3, 2, -1.25f, 1.f/128.f, true},
            {GGML_TYPE_Q4_0, GGML_TYPE_F32,  GGML_TYPE_F32,  96, 7, 33, 65, 1, 1, .5f, 1.f/128.f},
            {GGML_TYPE_Q4_0, GGML_TYPE_F16,  GGML_TYPE_F16,  96, 7, 33, 17, 2, 1, .7f, 1.f/128.f, true},
            {GGML_TYPE_Q4_0, GGML_TYPE_BF16, GGML_TYPE_BF16, 96, 7, 33, 1, 1, 1, 1.f, 1.f/128.f},
            {GGML_TYPE_Q4_K, GGML_TYPE_BF16, GGML_TYPE_BF16, 256, 13, 37, 17, 2, 3, 1.f, 1.f/128.f, true},
            {GGML_TYPE_Q4_K, GGML_TYPE_BF16, GGML_TYPE_BF16, 256, 128, 37, 33, 1, 1, 8.f/128.f, 1.f/128.f},
            {GGML_TYPE_Q4_K, GGML_TYPE_F32,  GGML_TYPE_BF16, 256, 13, 37, 65, 1, 1, -.5f, 1.f/128.f},
            {GGML_TYPE_Q4_K, GGML_TYPE_BF16, GGML_TYPE_BF16, 256, 13, 37, 17, 3, 2, 0.f, 1.f/128.f},
            {GGML_TYPE_Q4_0, GGML_TYPE_BF16, GGML_TYPE_BF16, 32, 3, 7, 1, 1025, 1, .5f, 1.f/128.f, true},
        };
        for (const auto& s : matrix) run_case(backend.get(), s);
        std::cout << "h3_dynamic_lora_cuda_test: PASS (" << cases << " cases, " << graphs << " CUDA graphs, "
                  << comparisons << " scalar comparisons, " << checks << " checks)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "h3_dynamic_lora_cuda_test: FAIL: " << e.what() << '\n'; return 1;
    }
}
