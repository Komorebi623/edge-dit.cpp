// Opt-in, tiny real-CUDA parity for the primitives used by ED_H3_FUSED_OPS.
// This does NOT test request dispatch, model quality, offload, or performance.
// Run only through scripts/test_h3_gpu_parity.sh after the main agent releases
// its GPU baseline. No scheduler, backend discovery, CPU fallback or load_all.
// Link the matching checkout's ggml-cuda (ED_ENABLE_CUDA_MODULATION=ON),
// ggml-cpu and ggml-base; see the script's --help for a standalone command.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "backend/ggml/ed_ggml_modulation_ext.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(GGML_BACKEND_DL) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#error "Use directly linked CPU/NVIDIA CUDA backends (GGML_BACKEND_DL=OFF)"
#endif
static_assert(GGML_MAX_NAME == 128, "Match this project's GGML_MAX_NAME=128 ABI");

namespace {
using Shape = std::array<int64_t, 4>;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using namespace edgedit::ggml_ext;

struct Failure : std::runtime_error {
    int code;
    Failure(const std::string& message, int status) : std::runtime_error(message), code(status) {}
};
void require(bool ok, const std::string& message, int code = 1) {
    if (!ok) throw Failure(message, code);
}
std::string env(const char* key) {
    const char* value = std::getenv(key);
    return value ? value : "";
}
bool gpu_uuid(const std::string& value) {
    if (value.size() != 40 || value.compare(0, 4, "GPU-") != 0) return false;
    for (size_t i = 4; i < value.size(); ++i) {
        if (i == 12 || i == 17 || i == 22 || i == 27) {
            if (value[i] != '-') return false;
        } else if (std::string("0123456789abcdefABCDEF").find(value[i]) == std::string::npos) {
            return false;
        }
    }
    return true;
}
void require_opt_in(int argc, char** argv) {
    require(argc == 2 && std::string(argv[1]) == "--run" && env("H3_GPU_PARITY_ALLOW") == "1",
            "NOT RUN: explicit opt-in required; use scripts/test_h3_gpu_parity.sh --help", 2);
    const auto visible = env("CUDA_VISIBLE_DEVICES");
    require(gpu_uuid(visible) && visible == env("H3_GPU_PARITY_PREFLIGHT_DEVICE"),
            "NOT RUN: exactly one full GPU UUID and matching read-only preflight required", 2);
    const auto stamp = env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    require(!stamp.empty() && stamp.find_first_not_of("0123456789") == std::string::npos,
            "NOT RUN: missing preflight timestamp", 2);
    char* end = nullptr;
    const double checked = std::strtod(stamp.c_str(), &end);
    const double age = std::difftime(std::time(nullptr), 0) - checked;
    require(end && *end == '\0' && std::isfinite(age) && age >= 0 && age <= 30,
            "NOT RUN: stale preflight (>30 seconds); rerun the guard", 2);
}

struct Devices {
    Backend cpu{nullptr, ggml_backend_free};
    Backend cuda{nullptr, ggml_backend_free};
    Devices() {
        // Direct registration only; never enumerate/load other backend plugins.
        auto reg = ggml_backend_cuda_reg();
        require(reg && std::string(ggml_backend_reg_name(reg)) == "CUDA",
                "CUDA backend absent (not a successful skip)", 3);
        require(ggml_backend_reg_dev_count(reg) == 1 && ggml_backend_cuda_get_device_count() == 1,
                "CUDA device absent or visibility is not exactly one device", 3);
        auto dev = ggml_backend_reg_dev_get(reg, 0);
        require(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU,
                "selected CUDA device is not a GPU", 3);
        cuda.reset(ggml_backend_cuda_init(0)); // logical 0 == the single visible UUID
        require(cuda && ggml_backend_is_cuda(cuda.get()) && ggml_backend_get_device(cuda.get()) == dev,
                "CUDA initialization/device identity failed", 3);
        auto cpu_reg = ggml_backend_cpu_reg();
        require(cpu_reg && ggml_backend_reg_dev_count(cpu_reg) == 1, "CPU reference backend absent", 3);
        auto cpu_dev = ggml_backend_reg_dev_get(cpu_reg, 0);
        require(cpu_dev && ggml_backend_dev_type(cpu_dev) == GGML_BACKEND_DEVICE_TYPE_CPU,
                "reference device is not CPU", 3);
        cpu.reset(ggml_backend_dev_init(cpu_dev, nullptr));
        require(bool(cpu), "CPU reference initialization failed", 3);
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        std::cout << "DEVICE CUDA logical=0 visible=" << env("CUDA_VISIBLE_DEVICES")
                  << " description=" << ggml_backend_dev_description(dev)
                  << "; reference=CPU/1-thread\n";
    }
};

void upload(ggml_tensor* tensor, const std::vector<float>& values) {
    require(ggml_is_contiguous(tensor) && size_t(ggml_nelements(tensor)) == values.size(),
            "upload must address a complete contiguous root/output");
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    } else {
        require(tensor->type == GGML_TYPE_F16, "unsupported upload type");
        std::vector<ggml_fp16_t> half(values.size());
        for (size_t i = 0; i < values.size(); ++i) half[i] = ggml_fp32_to_fp16(values[i]);
        ggml_backend_tensor_set(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    }
}
std::vector<float> download(const ggml_tensor* tensor) {
    require(ggml_is_contiguous(tensor), "comparison output must be contiguous");
    std::vector<float> values(size_t(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    } else {
        require(tensor->type == GGML_TYPE_F16, "unsupported comparison type");
        std::vector<ggml_fp16_t> half(values.size());
        ggml_backend_tensor_get(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < values.size(); ++i) values[i] = ggml_fp16_to_fp32(half[i]);
    }
    return values;
}

struct Graph {
    ggml_backend_t backend;
    bool on_cuda;
    bool round_inputs_to_half;
    Context context{ggml_init({2u * 1024u * 1024u, nullptr, true}), ggml_free};
    Buffer buffer{nullptr, ggml_backend_buffer_free};
    ggml_cgraph* graph = nullptr;
    std::vector<ggml_tensor*> roots, outputs;
    Graph(ggml_backend_t b, bool gpu, bool half = false)
        : backend(b), on_cuda(gpu), round_inputs_to_half(half) {
        require(bool(context), "context allocation failed");
        require(ggml_backend_is_cuda(b) == gpu, "graph backend identity mismatch");
    }
    ggml_context* ctx() const { return context.get(); }
    ggml_tensor* input(Shape shape, ggml_type type = GGML_TYPE_F32, bool padded = false) {
        Shape parent = shape;
        if (padded) { parent[0] += 3; parent[1] += 2; }
        auto root = ggml_new_tensor_4d(ctx(), type, parent[0], parent[1], parent[2], parent[3]);
        roots.push_back(root);
        if (!padded) return root;
        // Nonzero offset, row gaps and (for modulation) batch gaps; unaligned vec4.
        return ggml_view_4d(ctx(), root, shape[0], shape[1], shape[2], shape[3],
                            root->nb[1], root->nb[2], root->nb[3], root->nb[1] + root->nb[0]);
    }
    void check_storage(ggml_tensor* tensor) const {
        while (tensor->view_src) tensor = tensor->view_src;
        require(tensor->buffer != nullptr, "tensor has no backend allocation");
        const auto type = ggml_backend_buffer_get_type(tensor->buffer);
        if (on_cuda) {
            require(!ggml_backend_buffer_is_host(tensor->buffer) &&
                    type == ggml_backend_cuda_buffer_type(0) &&
                    ggml_backend_buft_get_device(type) == ggml_backend_get_device(backend),
                    "CUDA graph tensor is on host, split, or another device; no fallback permitted");
        } else {
            require(ggml_backend_buffer_is_host(tensor->buffer), "CPU reference not in host memory");
        }
    }
    void prepare(std::initializer_list<ggml_tensor*> result, int expected_custom, int expected_glu) {
        outputs.assign(result.begin(), result.end());
        graph = ggml_new_graph_custom(ctx(), 256, false);
        for (auto output : outputs) {
            require(output != nullptr, "fusion construction rejected (not a successful skip)");
            ggml_build_forward_expand(graph, output);
        }
        int custom = 0, glu = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            require(ggml_backend_supports_op(backend, node),
                    std::string(on_cuda ? "CUDA" : "CPU") + " unsupported op " + ggml_op_name(node->op) +
                    " / " + ggml_get_name(node) + " (no fallback)");
            if (node->op == GGML_OP_CUSTOM) ++custom;
            if (node->op == GGML_OP_GLU) {
                require(ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU, "not actual SwiGLU");
                ++glu;
            }
        }
        require(custom == expected_custom && glu == expected_glu, "expected fused nodes absent");
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx(), backend));
        require(bool(buffer), "backend tensor allocation failed");
        for (auto tensor = ggml_get_first_tensor(ctx()); tensor; tensor = ggml_get_next_tensor(ctx(), tensor))
            check_storage(tensor);
        // Recheck with allocated buffers, as supports_op can depend on placement.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)),
                    "allocated graph contains unsupported op");
    }
    void fill(int replay) {
        for (size_t r = 0; r < roots.size(); ++r) {
            std::vector<float> values(size_t(ggml_nelements(roots[r])));
            for (size_t i = 0; i < values.size(); ++i) {
                float value = 1.5f * std::sin(float(i % 113) * .17f + float(r) * .43f + replay * .71f);
                // Include signed zero, small values, positive/negative saturated SiLU inputs.
                switch ((i + 13 * r + 7 * size_t(replay)) % 41) {
                    case 0: value = 0.f; break;
                    case 1: value = -0.f; break;
                    case 2: value = 8.f; break;
                    case 3: value = -8.f; break;
                    case 4: value = 0.000244140625f; break;
                    case 5: value = -0.000244140625f; break;
                }
                // F16 GPU and F32 CPU reference see exactly the same rounded inputs.
                if (round_inputs_to_half) value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
                values[i] = value;
            }
            upload(roots[r], values);
        }
        // A missing launch or unwritten tail must not inherit a previous passing output.
        for (auto output : outputs)
            upload(output, std::vector<float>(size_t(ggml_nelements(output)),
                                             std::numeric_limits<float>::quiet_NaN()));
    }
    void compute() {
        // Direct execution on this backend only. A scheduler could hide CPU fallback.
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                std::string(on_cuda ? "CUDA" : "CPU") + " graph_compute failed");
        ggml_backend_synchronize(backend);
    }
};

int comparisons = 0;
size_t elements = 0;
double worst_scaled_error = 0;
void compare(const ggml_tensor* actual, const ggml_tensor* reference,
             ggml_type expected_type, const std::string& label) {
    require(actual->type == expected_type && reference->type == GGML_TYPE_F32, label + ": type mismatch");
    require(ggml_are_same_shape(actual, reference), label + ": shape mismatch");
    const auto got = download(actual), want = download(reference);
    // Fixed tolerances: FP32 arithmetic/exp differences; FP16 input/output rounding.
    const double atol = expected_type == GGML_TYPE_F16 ? 2e-3 : 2e-5;
    const double rtol = expected_type == GGML_TYPE_F16 ? 2e-3 : 2e-5;
    double max_abs = 0, max_scaled = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(want[i]),
                label + ": non-finite at element " + std::to_string(i));
        const double error = std::abs(double(got[i]) - double(want[i]));
        const double allowed = atol + rtol * std::abs(double(want[i]));
        max_abs = std::max(max_abs, error);
        max_scaled = std::max(max_scaled, error / allowed);
        require(error <= allowed, label + ": element=" + std::to_string(i) +
                " CUDA=" + std::to_string(got[i]) + " CPU=" + std::to_string(want[i]) +
                " abs_error=" + std::to_string(error) + " allowed=" + std::to_string(allowed));
    }
    ++comparisons;
    elements += got.size();
    worst_scaled_error = std::max(worst_scaled_error, max_scaled);
    std::cout << "PASS " << label << " n=" << got.size() << " atol=" << atol << " rtol=" << rtol
              << " max_abs=" << max_abs << " max_scaled=" << max_scaled << '\n';
}

void modulation_case(Devices& devices, Shape shape, Shape shift_shape, Shape scale_shape,
                     bool padded, const std::string& label) {
    Graph gpu(devices.cuda.get(), true), cpu(devices.cpu.get(), false);
    auto build = [&](Graph& g, bool fused) {
        auto x = g.input(shape, GGML_TYPE_F32, padded);
        auto shift = g.input(shift_shape, GGML_TYPE_F32, padded);
        auto scale = g.input(scale_shape, GGML_TYPE_F32, padded);
        auto residual = g.input(shape, GGML_TYPE_F32, padded);
        auto gate = g.input(shift_shape, GGML_TYPE_F32, padded);
        ggml_tensor *mod, *add;
        if (fused) {
            mod = fused_modulate_custom(g.ctx(), x, shift, scale);
            add = fused_residual_gate_custom(g.ctx(), residual, x, gate);
            require(mod && add && mod->op == GGML_OP_CUSTOM && add->op == GGML_OP_CUSTOM &&
                    std::string(ggml_get_name(mod)) == "ed_fused_modulate" &&
                    std::string(ggml_get_name(add)) == "ed_fused_residual_gate", "custom fusion missing");
        } else {
            // Independent, unfused CPU graph, not the custom op's CPU callback.
            mod = ggml_add(g.ctx(), ggml_add(g.ctx(), x, ggml_mul(g.ctx(), x, scale)), shift);
            add = ggml_add(g.ctx(), residual, ggml_mul(g.ctx(), x, gate));
        }
        g.prepare({mod, add}, fused ? 2 : 0, 0);
    };
    build(gpu, true);
    build(cpu, false);
    for (int replay = 0; replay < 2; ++replay) {
        gpu.fill(replay); cpu.fill(replay);
        cpu.compute(); gpu.compute();
        const auto suffix = label + "/replay=" + std::to_string(replay);
        compare(gpu.outputs[0], cpu.outputs[0], GGML_TYPE_F32, "modulate/F32/" + suffix);
        compare(gpu.outputs[1], cpu.outputs[1], GGML_TYPE_F32, "residual-gate/F32/" + suffix);
    }
}

void swiglu_case(Devices& devices, int64_t hidden, int64_t tokens, int64_t batch,
                 bool padded, bool split, ggml_type type) {
    Graph gpu(devices.cuda.get(), true, type == GGML_TYPE_F16);
    Graph cpu(devices.cpu.get(), false, type == GGML_TYPE_F16);
    // Row-padded GLU is supported; gaps between batches are NOT contiguous_1.
    require(!padded || batch == 1, "invalid GLU test layout");
    auto build = [&](Graph& g, bool fused) {
        const auto input_type = fused ? type : GGML_TYPE_F32;
        auto uv = g.input({(split ? hidden : 2 * hidden), tokens, batch, 1}, input_type, padded);
        auto v = split ? g.input({hidden, tokens, batch, 1}, input_type, padded) : nullptr;
        ggml_tensor* output;
        require(ggml_is_contiguous_1(uv), "GLU input layout unsupported");
        if (fused) {
            output = split ? ggml_swiglu_split(g.ctx(), uv, v) : ggml_swiglu(g.ctx(), uv);
        } else {
            auto u = uv;
            if (!split) {
                u = ggml_view_4d(g.ctx(), uv, hidden, tokens, batch, 1, uv->nb[1], uv->nb[2], uv->nb[3], 0);
                v = ggml_view_4d(g.ctx(), uv, hidden, tokens, batch, 1, uv->nb[1], uv->nb[2], uv->nb[3],
                                 size_t(hidden) * uv->nb[0]);
            }
            output = ggml_mul(g.ctx(), ggml_silu(g.ctx(), ggml_cont(g.ctx(), u)), ggml_cont(g.ctx(), v));
        }
        g.prepare({output}, 0, fused ? 1 : 0);
    };
    build(gpu, true);
    build(cpu, false);
    const std::string label = std::string("swiglu/") + ggml_type_name(type) + (split ? "/split" : "/packed") +
        (padded ? "/row-view" : "/dense") + "/h=" + std::to_string(hidden) +
        "/tokens=" + std::to_string(tokens) + "/batch=" + std::to_string(batch);
    for (int replay = 0; replay < 2; ++replay) {
        gpu.fill(replay); cpu.fill(replay);
        cpu.compute(); gpu.compute();
        compare(gpu.outputs[0], cpu.outputs[0], type, label + "/replay=" + std::to_string(replay));
    }
}

void reject_unsupported() {
    Context ctx{ggml_init({1024u * 1024u, nullptr, true}), ggml_free};
    require(bool(ctx), "rejection context allocation failed");
    auto x = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 8, 6, 2, 1);
    auto good = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 8, 1, 1, 1);
    auto bad = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 3, 1, 1, 1);
    auto half = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 8, 1, 1, 1);
    auto half_x = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 8, 6, 2, 1);
    require(!fused_modulate_custom(ctx.get(), x, bad, good) &&
            !fused_residual_gate_custom(ctx.get(), x, x, bad), "invalid broadcast accepted");
    require(!fused_modulate_custom(ctx.get(), half_x, good, good) &&
            !fused_modulate_custom(ctx.get(), x, half, good) &&
            !fused_modulate_custom(ctx.get(), x, good, half) &&
            !fused_residual_gate_custom(ctx.get(), half_x, x, good) &&
            !fused_residual_gate_custom(ctx.get(), x, half_x, good) &&
            !fused_residual_gate_custom(ctx.get(), x, x, half), "unsupported F16 modulation/gate accepted");
    require(!fused_modulate_custom(ctx.get(), nullptr, good, good) &&
            !fused_residual_gate_custom(ctx.get(), x, nullptr, good), "null input accepted");
    std::cout << "PASS rejection-only: F16 modulation/residual, invalid broadcast and null; "
                 "NOT F16 CUDA modulation coverage\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        require_opt_in(argc, argv); // before any backend/device access
        // Do not silently turn a caller's scalar-only override into vec4 coverage.
        require(env("ED_CUDA_FUSED_MODULATION_DISABLE_VEC4").empty() ||
                env("ED_CUDA_FUSED_MODULATION_DISABLE_VEC4") == "0",
                "NOT RUN: unset ED_CUDA_FUSED_MODULATION_DISABLE_VEC4 for the default-path matrix", 2);
        Devices devices;
        reject_unsupported();
        modulation_case(devices, {64,16,2,1}, {64,1,2,1}, {64,1,1,1}, false, "vec4-full-blocks");
        modulation_case(devices, {12,7,1,1}, {12,1,1,1}, {12,1,1,1}, false, "vec4-block-tail");
        modulation_case(devices, {7,5,2,1}, {7,1,2,1}, {1,5,1,1}, false, "scalar-tail");
        modulation_case(devices, {12,6,4,2}, {3,2,2,1}, {4,3,1,2}, false, "4d-divisor-broadcast");
        modulation_case(devices, {12,6,4,2}, {3,2,2,1}, {4,3,1,2}, true, "4d-strided-offset");
        modulation_case(devices, {5,3,2,1}, {5,3,2,1}, {5,3,2,1}, false, "equal-shape");
        modulation_case(devices, {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, false, "single-element");
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            swiglu_case(devices, 32, 8, 2, false, false, type);
            swiglu_case(devices, 7, 5, 2, false, false, type);
            swiglu_case(devices, 16, 7, 1, true, false, type);
            swiglu_case(devices, 33, 3, 2, false, true, type);
            swiglu_case(devices, 7, 5, 1, true, true, type);
            swiglu_case(devices, 1, 1, 1, false, false, type);
        }
        require(comparisons == 52, "incomplete matrix (no successful skips permitted)");
        std::cout << "PASS REAL CUDA tiny parity: comparisons=" << comparisons << " elements=" << elements
                  << " worst_scaled_error=" << worst_scaled_error
                  << "; no CPU fallback; NOT a full-model/baseline/performance result\n";
        return 0;
    } catch (const Failure& error) {
        std::cerr << "FAIL H3 CUDA parity: " << error.what() << '\n';
        return error.code;
    } catch (const std::exception& error) {
        std::cerr << "FAIL H3 CUDA parity: " << error.what() << '\n';
        return 1;
    }
}
