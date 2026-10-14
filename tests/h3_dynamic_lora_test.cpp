// Standalone CPU test: compile with the existing build-h3-cpu flags/libraries,
// including GGML_MAX_NAME=128. No CMake changes, GPU, or trained weights needed.
#include "dit_models/models/minimax_h3_dynamic_lora.hpp"

#include <cstdio>
#include <unistd.h>

using Adapter = MiniMaxH3::DynamicLoraAdapter;
static const std::string module = "model.diffusion_model.blocks.0.attn.qkv_proj";
static int checks = 0, graphs = 0;
static void check(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> static void rejects(F f, const std::string& message) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}
static void close(float got, float expected, float tol, const std::string& message) {
    check(std::isfinite(got) && std::abs(got - expected) <= tol * (1.f + std::abs(expected)),
          message + ": got=" + std::to_string(got) + " expected=" + std::to_string(expected));
}
struct Context {
    ggml_context* ctx;
    explicit Context(bool params = false) : ctx(ggml_init({16 * 1024 * 1024, nullptr, params})) {
        check(ctx != nullptr, "context allocation");
    }
    ~Context() { ggml_free(ctx); }
    Context(const Context&) = delete;
};
struct Params : Context {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = nullptr;
    Adapter::TensorMap base;
    Params() : Context(true) { check(backend != nullptr, "CPU backend"); }
    ~Params() { ggml_backend_buffer_free(buffer); ggml_backend_free(backend); }
    ggml_tensor* add(const std::string& name, ggml_type type = GGML_TYPE_F32, int64_t in = 32, int64_t out = 4) {
        auto* w = ggml_new_tensor_2d(ctx, type, in, out);
        ggml_set_name(w, (name + ".weight").c_str());
        base.emplace(name + ".weight", w);
        return w;
    }
    void alloc() { buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); check(buffer != nullptr, "runner allocation"); }
};
static void add_pair(String2TensorStorage& storage, const std::string& name = module,
                     ggml_type a = GGML_TYPE_F32, ggml_type b = GGML_TYPE_F32,
                     int64_t in = 32, int64_t rank = 3, int64_t out = 4) {
    const int64_t an[] = {in, rank}, bn[] = {rank, out};
    const std::string ak = name + ".lora_A.weight", bk = name + ".lora_B.weight";
    storage.emplace(ak, TensorStorage(ak, a, an, 2, 0));
    storage.emplace(bk, TensorStorage(bk, b, bn, 2, 0));
}
static std::vector<uint8_t> bytes(const ggml_tensor* t) {
    std::vector<uint8_t> result(ggml_nbytes(t));
    if (t->buffer) ggml_backend_tensor_get(t, result.data(), 0, result.size());
    else std::memcpy(result.data(), t->data, result.size());
    return result;
}
static std::vector<uint8_t> encode(ggml_type type, const std::vector<float>& values, int64_t width) {
    std::vector<uint8_t> result(ggml_row_size(type, width) * (values.size() / width));
    if (type == GGML_TYPE_F32) std::memcpy(result.data(), values.data(), result.size());
    else if (ggml_is_quantized(type)) {
        check(ggml_quantize_chunk(type, values.data(), result.data(), 0, values.size() / width, width, nullptr) == result.size(),
              "quantized fixture size");
    } else ggml_get_type_traits(type)->from_float_ref(values.data(), result.data(), values.size());
    return result;
}
static std::vector<float> fill(ggml_tensor* t, float factor) {
    std::vector<float> values(ggml_nelements(t));
    for (size_t i = 0; i < values.size(); ++i) values[i] = factor * float(int((i * 13 + 7) % 29) - 14);
    const auto raw = encode(t->type, values, t->ne[0]);
    if (t->buffer) ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
    else std::memcpy(t->data, raw.data(), raw.size());
    if (t->type != GGML_TYPE_F32) ggml_get_type_traits(t->type)->to_float(raw.data(), values.data(), values.size());
    return values;
}
static ggml_cgraph* graph(Context& c, std::initializer_list<ggml_tensor*> outputs) {
    auto* g = ggml_new_graph_custom(c.ctx, 2048, false);
    for (auto* out : outputs) ggml_build_forward_expand(g, out);
    return g;
}
static void compute(Context& c, ggml_cgraph* g) {
    check(ggml_graph_compute_with_ctx(c.ctx, g, 1) == GGML_STATUS_SUCCESS, "CPU graph compute");
    ++graphs;
}
static WeightAdapter::ForwardParams linear(float scale = 1.f) {
    WeightAdapter::ForwardParams p{};
    p.op_type = WeightAdapter::ForwardParams::op_type_t::OP_LINEAR;
    p.linear.force_prec_f32 = true;
    p.linear.scale = scale;
    return p;
}
static float at(const ggml_tensor* t, int64_t i, int64_t token, int64_t d2, int64_t d3) {
    const char* p = static_cast<const char*>(t->data) + i*t->nb[0] + token*t->nb[1] + d2*t->nb[2] + d3*t->nb[3];
    if (t->type == GGML_TYPE_F16) { ggml_fp16_t v; std::memcpy(&v, p, sizeof(v)); return ggml_fp16_to_fp32(v); }
    float v; std::memcpy(&v, p, sizeof(v)); return v;
}

// F32 is exercised first. Compare against both a scalar reference and the
// untouched original base graph + scalar BAx (isolates q4 activation rounding).
static void numerical(ggml_type base_type, ggml_type a_type, ggml_type b_type,
                      float coefficient, float conditioning, int64_t d2 = 1, int64_t d3 = 1,
                      bool strided = false, ggml_type input_type = GGML_TYPE_F32) {
    Params params;
    auto* w = params.add(module, base_type);
    auto* bias = ggml_new_tensor_1d(params.ctx, GGML_TYPE_F32, 4);
    String2TensorStorage storage;
    add_pair(storage, module, a_type, b_type);
    Adapter adapter(params.ctx, params.base, storage, coefficient);
    for (const auto& entry : adapter.tensors()) {
        check(entry.first == ggml_get_name(entry.second) && entry.first.size() < 128, "full factor name");
        check((entry.second->flags & GGML_TENSOR_FLAG_PARAM) == 0, "inference factor must not be a trainable graph node");
        check(entry.second->data == nullptr && entry.second->buffer == nullptr, "factor allocated too early");
    }
    params.alloc();
    auto* a = adapter.tensors().at(module + ".lora_A.weight");
    auto* b = adapter.tensors().at(module + ".lora_B.weight");
    check(a->type == a_type && b->type == b_type, "factor dtypes preserved");
    check(a->buffer == w->buffer && b->buffer == w->buffer, "runner owns factors");
    auto wr = fill(w, .018f), ar = fill(a, .008f), br = fill(b, .02f), biasr = fill(bias, .007f);
    const auto wbytes = bytes(w), abytes = bytes(a), bbytes = bytes(b);
    Context c;
    auto* input = ggml_new_tensor_4d(c.ctx, input_type, 32, strided ? d2 : 2, strided ? 2 : d2, d3);
    fill(input, .013f);
    auto* x = strided ? ggml_permute(c.ctx, input, 0, 2, 1, 3) : input;
    auto* baseline = ggml_ext_linear(c.ctx, x, w, bias, true, conditioning, (module + ".weight").c_str());
    adapter.begin_graph();
    auto* out = adapter.forward_with_lora(c.ctx, params.backend, x, w, bias, module + ".", linear(conditioning));
    auto* g = graph(c, {out});
    const auto coverage = adapter.report(g);
    check(coverage.complete() && coverage.target_count == 1 && coverage.used.size() == 1, "reachable target coverage");
    adapter.validate_coverage(g);
    for (int d = 0; d < 4; ++d) check(out->ne[d] == baseline->ne[d], "output shape including batch dimensions");
    int matmuls = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto* node = ggml_graph_node(g, i);
        if (node->op == GGML_OP_MUL_MAT) {
            ++matmuls;
            if (node->src[0] == w) check(std::string(ggml_get_name(node)) == module + ".weight", "base matmul label preserved");
            check(node->src[0] == w || node->src[0] == a || node->src[0] == b, "unexpected weight materialization");
        }
        check(!(node->op == GGML_OP_CPY && node->src[0] == w), "full base dequantization forbidden");
    }
    check(matmuls == (coefficient == 0.f ? 1 : 3), "only W(x), A(x), B(A(x))");
    compute(c, graph(c, {baseline}));
    compute(c, g);
    const float factor_tol = (a_type == GGML_TYPE_BF16 || b_type == GGML_TYPE_BF16) ? .002f :
                             (a_type == GGML_TYPE_F16 || b_type == GGML_TYPE_F16) ? .0002f : .00003f;
    for (int64_t h = 0; h < d3; ++h) for (int64_t z = 0; z < d2; ++z) for (int64_t t = 0; t < 2; ++t) {
        float ax[3] = {};
        for (int r = 0; r < 3; ++r) for (int k = 0; k < 32; ++k) ax[r] += ar[r*32+k] * at(x, k, t, z, h);
        for (int j = 0; j < 4; ++j) {
            float delta = 0, ref = biasr[j];
            for (int r = 0; r < 3; ++r) delta += br[j*3+r] * ax[r];
            for (int k = 0; k < 32; ++k) ref += wr[j*32+k] * at(x, k, t, z, h);
            const float got = at(out, j, t, z, h);
            close(got, at(baseline, j, t, z, h) + coefficient * delta, factor_tol, "dynamic residual reference");
            close(got, ref + coefficient * delta, ggml_is_quantized(base_type) ? .008f : factor_tol, "full scalar reference");
        }
    }
    check(bytes(w) == wbytes && bytes(a) == abytes && bytes(b) == bbytes, "base/factors changed during forward");
}

static int node_index(ggml_cgraph* g, ggml_tensor* t) {
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) if (ggml_graph_node(g, i) == t) return i;
    return -1;
}
static void zero_exact() {
    Params p; auto* w = p.add(module, GGML_TYPE_Q4_0);
    String2TensorStorage storage; add_pair(storage);
    Adapter adapter(p.ctx, p.base, storage, 0.f);
    p.alloc(); fill(w, .019f);
    const auto original = bytes(w);
    Context c;
    auto* x = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 32, 2, 3, 2); fill(x, .015f);
    auto* bias = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 4); fill(bias, .004f);
    auto* base = ggml_ext_linear(c.ctx, x, w, bias, true, 1.f/128.f, (module + ".weight").c_str());
    adapter.begin_graph();
    auto* out = adapter.forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", linear(1.f/128.f));
    auto* expected = graph(c, {base}); auto* actual = graph(c, {out});
    adapter.validate_coverage(actual);
    check(adapter.get_extra_graph_size() == 0, "zero coefficient graph overhead");
    check(ggml_graph_n_nodes(expected) == ggml_graph_n_nodes(actual), "zero coefficient node count");
    for (int i = 0; i < ggml_graph_n_nodes(expected); ++i) {
        auto* a = ggml_graph_node(expected, i); auto* b = ggml_graph_node(actual, i);
        check(a->op == b->op && a->type == b->type && std::string(a->name) == b->name &&
              !std::memcmp(a->ne, b->ne, sizeof(a->ne)) && !std::memcmp(a->nb, b->nb, sizeof(a->nb)) &&
              !std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)), "zero coefficient original graph structure/name");
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const int ai = node_index(expected, a->src[s]), bi = node_index(actual, b->src[s]);
            check(ai == bi && (ai >= 0 || a->src[s] == b->src[s]), "zero coefficient original graph edges");
        }
    }
    compute(c, expected); compute(c, actual);
    check(bytes(base) == bytes(out), "zero coefficient bit-exact original result");
    check(bytes(w) == original, "zero coefficient q4 original bytes");
}

static void rejections() {
    const std::string ak = module + ".lora_A.weight", bk = module + ".lora_B.weight";
    auto bad = [&](const std::string& label, auto mutate) {
        Params p; p.add(module); String2TensorStorage s; add_pair(s);
        mutate(p, s);
        const auto before = ggml_used_mem(p.ctx);
        rejects([&] { Adapter a(p.ctx, p.base, s); }, label);
        check(ggml_used_mem(p.ctx) == before, "validation partially modified runner: " + label);
    };
    bad("incomplete A", [&](auto&, auto& s) { s.erase(ak); });
    bad("incomplete B", [&](auto&, auto& s) { s.erase(bk); });
    bad("empty adapter", [&](auto&, auto& s) { s.clear(); });
    bad("unsupported alpha", [&](auto&, auto& s) { auto t = s.at(ak); t.name = module + ".alpha"; s.emplace(t.name, t); });
    bad("unsupported DoRA", [&](auto&, auto& s) { auto t = s.at(ak); t.name = module + ".lora_magnitude_vector"; s.emplace(t.name, t); });
    bad("unnormalized prefix", [&](auto&, auto& s) { add_pair(s, "diffusion_model.blocks.0.attn.qkv_proj"); });
    bad("unknown target", [&](auto&, auto& s) { add_pair(s, "model.diffusion_model.not_a_module"); });
    bad("duplicate storage alias", [&](auto&, auto& s) { s.emplace("duplicate", s.at(ak)); });
    bad("key/name mismatch", [&](auto&, auto& s) { s.at(ak).name = bk; });
    bad("zero dimension", [&](auto&, auto& s) { s.at(ak).ne[0] = 0; });
    bad("negative dimension", [&](auto&, auto& s) { s.at(bk).ne[1] = -1; });
    bad("rank over 2048", [&](auto&, auto& s) { s.at(ak).ne[1] = s.at(bk).ne[0] = 2049; });
    bad("rank zero", [&](auto&, auto& s) { s.at(ak).ne[1] = s.at(bk).ne[0] = 0; });
    bad("unequal rank", [&](auto&, auto& s) { s.at(bk).ne[0] = 2; });
    bad("input mismatch", [&](auto&, auto& s) { s.at(ak).ne[0] = 31; });
    bad("output mismatch", [&](auto&, auto& s) { s.at(bk).ne[1] = 5; });
    bad("factor not matrix", [&](auto&, auto& s) { s.at(ak).n_dims = 3; });
    bad("hidden dimension", [&](auto&, auto& s) { s.at(ak).ne[4] = 2; });
    bad("size overflow", [&](auto&, auto& s) { s.at(ak).ne[0] = std::numeric_limits<int64_t>::max(); });
    bad("quantized factor", [&](auto&, auto& s) { s.at(ak).type = GGML_TYPE_Q4_0; });
    bad("factor conversion", [&](auto&, auto& s) { s.at(ak).expected_type = GGML_TYPE_F16; });
    bad("FP8 disguised as F16", [&](auto&, auto& s) { s.at(ak).is_f8_e4m3 = true; });
    bad("swapped halves", [&](auto&, auto& s) { s.at(ak).swap_swiglu_halves = true; });
    bad("missing base", [&](auto& p, auto&) { p.base.clear(); });
    bad("null base", [&](auto& p, auto&) { p.base[module + ".weight"] = nullptr; });
    bad("base not matrix", [&](auto& p, auto&) { p.base.at(module + ".weight")->ne[2] = 2; });
    bad("base alias", [&](auto& p, auto& s) {
        auto alias = module + "_alias"; p.base[alias + ".weight"] = p.base.at(module + ".weight"); add_pair(s, alias);
    });
    bad("existing factor name", [&](auto& p, auto&) {
        auto* t = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, 1); ggml_set_name(t, ak.c_str());
    });
    bad("overlong factor name", [&](auto& p, auto& s) {
        auto name = "model.diffusion_model." + std::string(110, 'a'); p.add(name); add_pair(s, name);
    });
    bad("allocated params", [&](auto& p, auto&) { p.alloc(); });
    {
        Params p; p.add(module); String2TensorStorage s; add_pair(s);
        for (float coeff : {NAN, INFINITY, -INFINITY}) rejects([&] { Adapter a(p.ctx, p.base, s, coeff); }, "nonfinite coefficient");
        rejects([&] { Adapter a(nullptr, p.base, s); }, "null params context");
        Context allocated;
        rejects([&] { Adapter a(allocated.ctx, p.base, s); }, "context must be no_alloc");
        Adapter a(p.ctx, p.base, s);
        rejects([&] { Adapter duplicate(p.ctx, p.base, s); }, "duplicate adapter tensor names");
    }
    for (size_t length : {size_t(127), size_t(128)}) {
        Params p; String2TensorStorage s;
        const std::string root = "model.diffusion_model.";
        const std::string name = root + std::string(length - root.size() - std::string(".lora_A.weight").size(), 'a');
        p.add(name); add_pair(s, name);
        if (length == 128) rejects([&] { Adapter a(p.ctx, p.base, s); }, "128-byte name rejected, not truncated");
        else {
            Adapter a(p.ctx, p.base, s);
            check(std::strlen(a.tensors().begin()->second->name) == 127, "127-byte name retained exactly");
        }
    }
    {
        auto* ctx = ggml_init({2 * ggml_tensor_overhead(), nullptr, true});
        check(ctx != nullptr, "small metadata context");
        auto* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 4);
        Adapter::TensorMap base{{module + ".weight", w}};
        String2TensorStorage s; add_pair(s);
        rejects([&] { Adapter a(ctx, base, s); }, "metadata capacity must fail before ggml abort");
        ggml_free(ctx);
    }
    for (int rank : {1, 2048}) {
        Params p; p.add(module); String2TensorStorage s; add_pair(s, module, GGML_TYPE_F32, GGML_TYPE_F32, 32, rank);
        Adapter a(p.ctx, p.base, s);
        check(a.tensors().at(ak)->ne[1] == rank, "valid rank boundary");
    }
}

static void coverage_and_paths() {
    Params p; auto* w0 = p.add(module);
    const std::string other = "model.diffusion_model.blocks.1.attn.out_proj";
    auto* w1 = p.add(other);
    String2TensorStorage storage; add_pair(storage); add_pair(storage, other);
    Adapter a(p.ctx, p.base, storage);
    Context c; auto* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 2);
    auto forward = [&](ggml_tensor* w, const std::string& name) {
        return a.forward_with_lora(c.ctx, p.backend, x, w, nullptr, name + ".", linear());
    };
    rejects([&] { forward(w0, module); }, "begin_graph required");
    a.begin_graph(); auto* y0 = forward(w0, module);
    rejects([&] { a.validate_coverage(); }, "partial coverage must fail");
    auto* y1 = forward(w1, other);
    check(a.report().complete(), "per-graph call coverage");
    rejects([&] { a.validate_coverage(graph(c, {y0})); }, "discarded target branch must fail reachability");
    auto* full = graph(c, {y0, y1});
    a.validate_coverage(full);
    rejects([&] { forward(w0, module); }, "validated graph scope is closed");
    a.begin_graph();
    check(a.report().used.empty(), "begin_graph resets all usage");
    rejects([&] { a.validate_coverage(full); }, "previous graph cannot validate an empty new scope");
    forward(w0, module);
    rejects([&] { a.validate_coverage(); }, "previous graph coverage leaked");
    forward(w1, other); a.validate_coverage();
    check(!a.report(graph(c, {x})).complete(), "graph-independent coverage leaked");
    a.begin_graph();
    rejects([&] { a.patch_weight(c.ctx, p.backend, w0, module + ".weight"); }, "nonlinear patch must fail");
    rejects([&] { a.patch_weight(c.ctx, p.backend, w0, "alias.weight"); }, "nonlinear pointer alias must fail");
    auto conv = linear(); conv.op_type = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
    rejects([&] { a.forward_with_lora(c.ctx, p.backend, x, w0, nullptr, module + ".", conv); }, "target convolution must fail");
    auto* other_w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 4);
    rejects([&] { forward(other_w, module); }, "changed base identity must fail");
    rejects([&] { forward(w0, "alias"); }, "wrong linear prefix must fail");
    auto* wrong_x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 31, 2);
    rejects([&] { a.forward_with_lora(c.ctx, p.backend, wrong_x, w0, nullptr, module + ".", linear()); }, "activation mismatch");
    for (const auto& name : {module + ".bias", module + ".norm.weight", module + ".unrelated.weight"}) {
        check(a.patch_weight(c.ctx, p.backend, other_w, name) == other_w, "unrelated patch changes weight");
    }
    auto* plain = a.forward_with_lora(c.ctx, p.backend, x, other_w, nullptr, "unrelated.", linear());
    check(plain->op == GGML_OP_MUL_MAT && std::string(plain->name) == "unrelated.weight", "unrelated linear baseline");
    check(a.report().used.empty(), "unrelated or failed paths count toward coverage");
    Params p2; p2.add(module); String2TensorStorage s2; add_pair(s2);
    Adapter independent(p2.ctx, p2.base, s2); independent.begin_graph();
    rejects([&] { independent.validate_coverage(); }, "cross-instance usage leaked");
}

// Real inference graph-cut regression: resident numerical parity cannot detect
// factors incorrectly marked trainable and therefore omitted from offload.
static void graph_cut_param_staging() {
    namespace Cut = sd::ggml_graph_cut;
    struct Marks {
        Marks() { Cut::clear_graph_cut_marks(); }
        ~Marks() { Cut::clear_graph_cut_marks(); }
    };
    size_t previous_staged_bytes = 0;
    for (int rank : {16, 128}) {
        Marks marks;
        Params p;
        const std::string second = "model.diffusion_model.blocks.1.attn.out_proj";
        auto* w0 = p.add(module, GGML_TYPE_Q4_0, 32, 32);
        auto* w1 = p.add(second, GGML_TYPE_Q4_0, 32, 19);
        auto* unused = p.add("model.diffusion_model.unused", GGML_TYPE_F32, 32, 7);
        String2TensorStorage storage;
        add_pair(storage, module, GGML_TYPE_BF16, GGML_TYPE_BF16, 32, rank, 32);
        add_pair(storage, second, GGML_TYPE_BF16, GGML_TYPE_BF16, 32, rank, 19);
        Adapter adapter(p.ctx, p.base, storage);
        p.alloc();
        std::unordered_set<const ggml_tensor*> parameter_set;
        for (auto* t = ggml_get_first_tensor(p.ctx); t; t = ggml_get_next_tensor(p.ctx, t)) {
            parameter_set.insert(t);
            check(t->buffer && t->data, "runner inference parameter must own backing storage");
            check(!(t->flags & GGML_TENSOR_FLAG_PARAM), "runner inference parameter must remain a graph leaf");
            fill(t, .001f);
        }
        Context c(true); // Same metadata-only graph construction as the runner.
        auto* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 3);
        ggml_set_input(x);
        const auto build = [&] {
            Cut::clear_graph_cut_marks();
            adapter.begin_graph();
            auto* y0 = adapter.forward_with_lora(c.ctx, p.backend, x, w0, nullptr, module + ".", linear(1.f/128.f));
            auto* y1 = adapter.forward_with_lora(c.ctx, p.backend, y0, w1, nullptr, second + ".", linear(1.f/128.f));
            Cut::mark_graph_cut(y0, "block0", "out");
            Cut::mark_graph_cut(y1, "block1", "out");
            auto* g = graph(c, {y1});
            adapter.validate_coverage(g);
            return g;
        };
        auto* g = build();
        std::set<ggml_tensor*> leaves;
        for (int i = 0; i < Cut::leaf_count(g); ++i) leaves.insert(Cut::leaf_tensor(g, i));
        for (const auto& entry : adapter.tensors()) {
            check(leaves.count(entry.second) == 1, "reachable A/B must be graph leaf: " + entry.first);
            check(node_index(g, entry.second) == -1, "A/B must not be an internal producer: " + entry.first);
        }
        const auto plan = Cut::build_plan(p.backend, g, parameter_set, "h3-lora-staging");
        check(plan.available && plan.valid && plan.has_cuts && plan.segments.size() == 2, "real two-cut LoRA plan");
        std::set<ggml_tensor*> all_staged;
        size_t staged_bytes = 0;
        for (const auto& segment : plan.segments) {
            const bool first = segment.group_name == "block0";
            const auto& name = first ? module : second;
            const std::set<ggml_tensor*> expected{
                first ? w0 : w1, adapter.tensors().at(name + ".lora_A.weight"),
                adapter.tensors().at(name + ".lora_B.weight")};
            std::set<ggml_tensor*> classified;
            bool previous_cut = false;
            for (const auto& input : segment.input_refs) {
                auto* t = Cut::input_tensor(g, input);
                if (parameter_set.count(t)) {
                    check(input.type == Cut::Segment::INPUT_PARAM && input.leaf_index >= 0 && input.node_index == -1,
                          "runner weight/factor must classify as leaf INPUT_PARAM");
                    classified.insert(t);
                }
                previous_cut |= input.type == Cut::Segment::INPUT_PREVIOUS_CUT;
            }
            check(classified == expected, "segment must classify its base AND both LoRA factors");
            check(previous_cut == !first, "second segment consumes previous-cut activation");
            const auto selected = Cut::runtime_param_tensors(g, segment, "h3-lora-staging");
            const std::set<ggml_tensor*> staged(selected.begin(), selected.end());
            check(staged == expected && selected.size() == expected.size(), "runtime staging omitted or duplicated a factor");
            size_t segment_bytes = 0;
            for (auto* t : selected) {
                check(Cut::tensor_buffer(t) == p.buffer && t->data, "staged parameter lost runner buffer");
                segment_bytes += ggml_nbytes(t);
                all_staged.insert(t);
            }
            check(segment.input_param_bytes == segment_bytes, "graph-cut byte accounting omits factors");
            staged_bytes += segment_bytes;
        }
        check(all_staged.size() == 6 && !all_staged.count(unused), "exact reachable params, not whole-model fallback");
        for (const auto& entry : adapter.tensors()) check(all_staged.count(entry.second) == 1, "A/B absent from runtime staging");
        check(staged_bytes > previous_staged_bytes, "larger-rank adapter must increase staged bytes");
        previous_staged_bytes = staged_bytes;

        // Negative control reproduces the actual bug: allocated trainable A/B
        // still pass coverage/resident binding, but disappear from runtime params.
        for (const auto& entry : adapter.tensors()) ggml_set_param(entry.second);
        auto* broken = build();
        const auto bad_plan = Cut::build_plan(p.backend, broken, parameter_set, "h3-lora-staging-negative");
        check(bad_plan.valid && bad_plan.segments.size() == 2, "negative-control graph-cut plan");
        std::set<ggml_tensor*> bad_staged;
        for (const auto& segment : bad_plan.segments) {
            const auto selected = Cut::runtime_param_tensors(broken, segment, "h3-lora-staging-negative");
            bad_staged.insert(selected.begin(), selected.end());
        }
        check(bad_staged == std::set<ggml_tensor*>({w0, w1}), "negative control must reproduce missing adapter staging");
        for (const auto& entry : adapter.tensors()) {
            check(node_index(broken, entry.second) >= 0 && !bad_staged.count(entry.second), "trainable factor bug not reproduced");
            entry.second->flags &= ~GGML_TENSOR_FLAG_PARAM;
        }
    }
}

// Real init_from_file -> constructor -> runner allocation -> load_tensors,
// proving all three storage dtypes survive the normalized loader path bytewise.
static void loader_roundtrip() {
    struct TemporaryFile {
        char path[64] = "/tmp/h3_dynamic_lora_XXXXXX.safetensors";
        TemporaryFile() { int fd = mkstemps(path, 12); check(fd >= 0, "temporary safetensors"); ::close(fd); }
        ~TemporaryFile() { std::remove(path); }
    } file;
    Params p;
    String2TensorStorage storage;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        const auto name = module + "_" + ggml_type_name(type);
        p.add(name); add_pair(storage, name, type, type);
    }
    std::ostringstream header; header << '{';
    std::map<std::string, std::vector<uint8_t>> payloads;
    size_t offset = 0;
    for (const auto& entry : storage) {
        const auto& s = entry.second;
        std::vector<float> values(s.nelements());
        for (size_t i = 0; i < values.size(); ++i) values[i] = .0037f * float(int(i % 17) - 8);
        auto raw = encode(s.type, values, s.ne[0]);
        if (offset) header << ',';
        header << '"' << s.name << "\":{\"dtype\":\"" << (s.type == GGML_TYPE_F32 ? "F32" : s.type == GGML_TYPE_F16 ? "F16" : "BF16")
               << "\",\"shape\":[" << s.ne[1] << ',' << s.ne[0] << "],\"data_offsets\":[" << offset << ',' << offset + raw.size() << "]}";
        offset += raw.size(); payloads.emplace(s.name, std::move(raw));
    }
    header << '}'; std::string json = header.str(); while (json.size() % 8) json += ' ';
    {
        std::ofstream out(file.path, std::ios::binary);
        uint64_t length = json.size();
        for (int i = 0; i < 8; ++i) out.put(char((length >> (i*8)) & 255));
        out.write(json.data(), json.size());
        for (const auto& entry : storage) {
            const auto& raw = payloads.at(entry.first); out.write(reinterpret_cast<const char*>(raw.data()), raw.size());
        }
        check(out.good(), "write normalized safetensors");
    }
    ModelLoader loader;
    check(loader.init_from_file(file.path), "normalized init_from_file without converter");
    Adapter a(p.ctx, p.base, loader.get_tensor_storage_map(), 8.f/128.f);
    p.alloc();
    check(loader.load_tensors(a.tensors(), {}, 1, false), "separate adapter loader after runner allocation");
    for (const auto& entry : a.tensors()) {
        check(bytes(entry.second) == payloads.at(entry.first), "loader rescaled or converted factors");
        check(entry.second->type == storage.at(entry.first).type, "loader factor dtype changed");
        check(std::string(entry.second->name) == entry.first, "loader normalized name changed");
    }
}

// Dense opt-in regressions stay deliberately small and require no factor file.
static const std::string norm_key = "model.diffusion_model.blocks.0.norm.weight";
static ggml_tensor* add_vector(Params& p, const std::string& key, ggml_type type = GGML_TYPE_BF16,
                               int64_t size = 4) {
    auto* t = ggml_new_tensor_1d(p.ctx, type, size);
    ggml_set_name(t, key.c_str());
    check(p.base.emplace(key, t).second, "duplicate vector fixture");
    return t;
}
static void add_delta(String2TensorStorage& storage, const std::string& key,
                      ggml_type type = GGML_TYPE_F32, int64_t in = 32, int64_t out = 4,
                      int dims = 2) {
    const int64_t ne[] = {in, out};
    const auto name = key + ".delta";
    check(storage.emplace(name, TensorStorage(name, type, ne, dims, 0)).second, "duplicate delta fixture");
}
using Snapshot = std::map<ggml_tensor*, std::vector<uint8_t>>;
static Snapshot snapshot(Params& p) {
    Snapshot result;
    for (auto* t = ggml_get_first_tensor(p.ctx); t; t = ggml_get_next_tensor(p.ctx, t)) result.emplace(t, bytes(t));
    return result;
}
static void unchanged(const Snapshot& before, const std::string& label) {
    for (const auto& entry : before) check(bytes(entry.first) == entry.second, label + ": " + entry.first->name);
}
static void store(ggml_tensor* t, const std::vector<float>& values) {
    check(int64_t(values.size()) == ggml_nelements(t), "fixture value count");
    const auto raw = encode(t->type, values, t->ne[0]);
    if (t->buffer) ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
    else std::memcpy(t->data, raw.data(), raw.size());
}
// Same parameter/input pointers: compare the complete graph, not just its size.
static void identical_graph(ggml_cgraph* expected, ggml_cgraph* actual, const std::string& label) {
    check(ggml_graph_n_nodes(expected) == ggml_graph_n_nodes(actual), label + " node count");
    namespace Cut = sd::ggml_graph_cut;
    check(Cut::leaf_count(expected) == Cut::leaf_count(actual), label + " leaf count");
    for (int i = 0; i < Cut::leaf_count(expected); ++i)
        check(Cut::leaf_tensor(expected, i) == Cut::leaf_tensor(actual, i), label + " leaf identity/order");
    for (int i = 0; i < ggml_graph_n_nodes(expected); ++i) {
        auto* a = ggml_graph_node(expected, i); auto* b = ggml_graph_node(actual, i);
        check(a->op == b->op && a->type == b->type && a->flags == b->flags &&
              std::string(a->name) == b->name && !std::memcmp(a->ne, b->ne, sizeof(a->ne)) &&
              !std::memcmp(a->nb, b->nb, sizeof(a->nb)) &&
              !std::memcmp(a->op_params, b->op_params, sizeof(a->op_params)) && a->view_offs == b->view_offs,
              label + " node metadata at " + std::to_string(i));
        auto same_edge = [&](ggml_tensor* x, ggml_tensor* y) {
            const int xi = node_index(expected, x), yi = node_index(actual, y);
            check(xi == yi && (xi >= 0 || x == y), label + " graph edge");
        };
        for (int s = 0; s < GGML_MAX_SRC; ++s) same_edge(a->src[s], b->src[s]);
        same_edge(a->view_src, b->view_src);
    }
}

static void dense_numerical(ggml_type base_type, float coefficient, float conditioning, bool with_ab,
                            bool strided = false) {
    Params p; auto* w = p.add(module, base_type);
    auto* bias = add_vector(p, module + ".bias", GGML_TYPE_F32);
    String2TensorStorage storage;
    add_delta(storage, module + ".weight");
    if (with_ab) add_pair(storage);
    Adapter adapter(p.ctx, p.base, storage, coefficient, true);
    check(adapter.tensors().size() == (with_ab ? 3u : 1u), "dense payload allocation count");
    for (const auto& entry : adapter.tensors()) {
        check(!entry.second->data && !entry.second->buffer, "dense allocated before runner");
        check(entry.first == entry.second->name && entry.second->type == storage.at(entry.first).type,
              "dense payload name/type preserved");
        check(!(entry.second->flags & GGML_TENSOR_FLAG_PARAM), "dense payload is an inference leaf");
    }
    p.alloc();
    auto* d = adapter.tensors().at(module + ".weight.delta");
    const auto wr = fill(w, .018f), dr = fill(d, .007f), biasr = fill(bias, .003f);
    std::vector<float> ar, br;
    if (with_ab) {
        ar = fill(adapter.tensors().at(module + ".lora_A.weight"), .008f);
        br = fill(adapter.tensors().at(module + ".lora_B.weight"), .02f);
    }
    for (const auto& entry : adapter.tensors()) check(entry.second->buffer == p.buffer, "runner owns dense payload");
    const auto original = snapshot(p);
    Context c;
    auto* input = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 32, strided ? 3 : 2, strided ? 2 : 3, 2);
    fill(input, .013f);
    auto* x = strided ? ggml_permute(c.ctx, input, 0, 2, 1, 3) : input;
    const auto input_bytes = bytes(input);
    auto* baseline = ggml_ext_linear(c.ctx, x, w, bias, true, conditioning, (module + ".weight").c_str());
    adapter.begin_graph();
    auto* out = adapter.forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", linear(conditioning));
    auto* g = graph(c, {out});
    const auto coverage = adapter.report(g);
    check(coverage.complete() && coverage.target_count == (with_ab ? 2u : 1u) &&
          coverage.used.size() == coverage.target_count, "dense and A/B are separate covered targets");
    adapter.validate_coverage(g);
    std::set<ggml_tensor*> expected_weights{w, d};
    if (with_ab) {
        expected_weights.insert(adapter.tensors().at(module + ".lora_A.weight"));
        expected_weights.insert(adapter.tensors().at(module + ".lora_B.weight"));
    }
    std::multiset<ggml_tensor*> matmul_weights;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto* node = ggml_graph_node(g, i);
        if (node->op == GGML_OP_MUL_MAT) {
            matmul_weights.insert(node->src[0]);
            check(expected_weights.count(node->src[0]), "dense residual must not materialize merged weights");
            if (node->src[0] == w) check(std::string(node->name) == module + ".weight", "dense preserves base matmul name");
        }
        // Matrix weights may only enter matmuls, never a cast/add/copy/requantization.
        for (auto* src : node->src) if (src == w || src == d)
            check(node->op == GGML_OP_MUL_MAT && node->src[0] == src, "no dense merge or full base dequantization");
    }
    check(matmul_weights == std::multiset<ggml_tensor*>(expected_weights.begin(), expected_weights.end()),
          "exactly W*x, delta*x, and optional A*x/B*A*x");
    for (int dim = 0; dim < 4; ++dim) check(out->ne[dim] == baseline->ne[dim], "dense 4D output shape");
    unchanged(original, "dense graph construction changed parameter");
    compute(c, graph(c, {baseline}));
    compute(c, g);
    for (int h = 0; h < 2; ++h) for (int z = 0; z < 3; ++z) for (int t = 0; t < 2; ++t) {
        float ax[3] = {};
        if (with_ab) for (int r = 0; r < 3; ++r) for (int k = 0; k < 32; ++k)
            ax[r] += ar[r*32+k] * at(x, k, t, z, h);
        for (int j = 0; j < 4; ++j) {
            float dx = 0.f, bax = 0.f, wx = biasr[j];
            for (int k = 0; k < 32; ++k) {
                dx += dr[j*32+k] * at(x, k, t, z, h);
                wx += wr[j*32+k] * at(x, k, t, z, h);
            }
            if (with_ab) for (int r = 0; r < 3; ++r) bax += br[j*3+r] * ax[r];
            // For Q4, use the SAME untouched CPU base graph, not a dequantized
            // scalar approximation or a newly quantized W+delta oracle.
            close(at(out, j, t, z, h), at(baseline, j, t, z, h) + coefficient * (dx + bax),
                  .00003f, "dense residual vs same CPU baseline");
            if (base_type == GGML_TYPE_F32)
                close(at(out, j, t, z, h), wx + coefficient * (dx + bax), .00003f, "dense+AB scalar oracle");
        }
    }
    const auto first_result = bytes(out);
    compute(c, g);
    check(bytes(out) == first_result, "dense repeated compute drift");
    unchanged(original, "dense execution changed base/payload");
    check(bytes(input) == input_bytes, "dense execution changed input");
}

// Independent finite FP32 -> BF16 round-to-nearest-even oracle. In particular,
// do not derive expected bytes from the adapter's cast or graph output.
static uint16_t bf16_bits(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
static float bf16_value(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
static void dense_vectors(float coefficient, ggml_type payload_type) {
    Params p; auto* w = p.add(module, GGML_TYPE_F32, 32, 8);
    auto* bias = add_vector(p, module + ".bias", GGML_TYPE_BF16, 8);
    auto* norm = add_vector(p, norm_key, GGML_TYPE_BF16, 8);
    String2TensorStorage storage;
    add_delta(storage, module + ".bias", payload_type, 8, 1, 1);
    add_delta(storage, norm_key, payload_type, 8, 1, 1);
    Adapter adapter(p.ctx, p.base, storage, coefficient, true);
    p.alloc(); fill(w, .01f);
    const std::vector<float> base_values{1.f, 1.0078125f, -1.f, -1.0078125f, .5f, .50390625f, -.5f, -.50390625f};
    // Just above/below a BF16 halfway point, exact ties with even/odd LSBs,
    // both signs. F32 cases also distinguish an erroneous early BF16 cast.
    std::vector<float> delta_values{.0039215087890625f, .0038909912109375f, -.0039215087890625f,
                                  -.0038909912109375f, .001953125f, .001953125f, -.001953125f, -.001953125f};
    for (auto& v : delta_values) v /= coefficient;
    const auto encoded_delta = encode(payload_type, delta_values, 8);
    if (payload_type != GGML_TYPE_F32)
        ggml_get_type_traits(payload_type)->to_float(encoded_delta.data(), delta_values.data(), delta_values.size());
    store(bias, base_values); store(norm, base_values);
    for (const auto& entry : adapter.tensors()) store(entry.second, delta_values);
    const auto original = snapshot(p);
    std::vector<uint16_t> expected_bits;
    std::vector<float> expected_values;
    bool premature_cast_differs = false, missing_cast_differs = false;
    for (size_t i = 0; i < base_values.size(); ++i) {
        const float update = coefficient * delta_values[i];
        const float sum = base_values[i] + update;
        const auto bits = bf16_bits(sum);
        expected_bits.push_back(bits); expected_values.push_back(bf16_value(bits));
        premature_cast_differs |= bits != bf16_bits(base_values[i] + bf16_value(bf16_bits(update)));
        missing_cast_differs |= sum != expected_values.back();
    }
    check(missing_cast_differs, "BF16 fixture must expose omitted cast-back");
    if (payload_type == GGML_TYPE_F32) check(premature_cast_differs, "BF16 fixture must expose premature rounding");
    std::vector<uint8_t> expected_bytes(expected_bits.size() * sizeof(uint16_t));
    std::memcpy(expected_bytes.data(), expected_bits.data(), expected_bytes.size());
    Context c; auto* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 2); fill(x, .03f);
    auto* expected_bias = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 8); store(expected_bias, expected_values);
    auto* baseline = ggml_ext_linear(c.ctx, x, w, expected_bias, true, 1.f, (module + ".weight").c_str());
    adapter.begin_graph();
    auto* patched_norm = adapter.patch_weight(c.ctx, p.backend, norm, norm_key);
    auto* patched_bias = adapter.patch_weight(c.ctx, p.backend, bias, module + ".bias");
    auto* out = adapter.forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", linear());
    check(patched_norm != norm && patched_bias != bias, "vector update must be out-of-place");
    check(patched_norm->type == GGML_TYPE_BF16 && patched_bias->type == GGML_TYPE_BF16, "vector retains base dtype");
    auto* g = graph(c, {out, patched_norm, patched_bias});
    adapter.validate_coverage(g);
    for (auto* patched : {patched_norm, patched_bias}) {
        check(patched->op == GGML_OP_CPY && patched->src[0]->op == GGML_OP_ADD &&
              patched->src[0]->type == GGML_TYPE_F32 && patched->src[0]->src[0]->type == GGML_TYPE_F32 &&
              patched->src[0]->src[1]->type == GGML_TYPE_F32, "vector FP32 add before BF16 cast-back");
    }
    unchanged(original, "vector graph construction changed parameter");
    compute(c, graph(c, {patched_norm, patched_bias}));
    check(bytes(patched_norm) == expected_bytes && bytes(patched_bias) == expected_bytes,
          "standalone BF16 vector rounding oracle");
    unchanged(original, "standalone vector execution changed base/payload");
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto* node = ggml_graph_node(g, i);
        // supports_op() accepts ADD too broadly in this CPU build; reject its
        // known unsupported F32 + BF16 combination before ggml can abort.
        const bool unsupported_add = node->op == GGML_OP_ADD && node->src[0]->type == GGML_TYPE_F32 &&
                                     node->src[1]->type == GGML_TYPE_BF16;
        check(!unsupported_add && ggml_backend_supports_op(p.backend, node),
              std::string("dense BF16 linear CPU-unsupported op: ") + ggml_op_name(node->op) +
              " dst=" + ggml_type_name(node->type) +
              " src0=" + (node->src[0] ? ggml_type_name(node->src[0]->type) : "none") +
              " src1=" + (node->src[1] ? ggml_type_name(node->src[1]->type) : "none"));
    }
    compute(c, graph(c, {baseline})); compute(c, g);
    check(bytes(patched_norm) == expected_bytes, "BF16 norm FP32-add/RNE-cast oracle");
    check(bytes(patched_bias) == expected_bytes, "BF16 bias FP32-add/RNE-cast oracle");
    check(bytes(out) == bytes(baseline), "linear bias must use rounded BF16 update, not FP32 bias");
    compute(c, g);
    check(bytes(patched_norm) == expected_bytes && bytes(patched_bias) == expected_bytes && bytes(out) == bytes(baseline),
          "vector repeated compute drift");
    unchanged(original, "vector execution changed base/payload");
}


static void dense_zero_exact(bool with_ab) {
    Params p; auto* w = p.add(module, GGML_TYPE_Q4_0);
    auto* bias = add_vector(p, module + ".bias", GGML_TYPE_F32);
    auto* norm = add_vector(p, norm_key);
    String2TensorStorage storage;
    add_delta(storage, module + ".weight");
    add_delta(storage, module + ".bias", GGML_TYPE_BF16, 4, 1, 1);
    add_delta(storage, norm_key, GGML_TYPE_F32, 4, 1, 1);
    if (with_ab) add_pair(storage);
    Adapter adapter(p.ctx, p.base, storage, 0.f, true);
    p.alloc();
    for (auto* t = ggml_get_first_tensor(p.ctx); t; t = ggml_get_next_tensor(p.ctx, t)) fill(t, .019f);
    // Unused payloads contain NaNs: a zero-gain matmul/add is NOT an exact no-op.
    for (const auto& entry : adapter.tensors()) store(entry.second, std::vector<float>(ggml_nelements(entry.second), NAN));
    const auto original = snapshot(p);
    Context c; auto* x = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 32, 2, 3, 2); fill(x, .015f);
    auto* baseline = ggml_ext_linear(c.ctx, x, w, bias, true, 1.f/128.f, (module + ".weight").c_str());
    auto* expected_norm = ggml_cast(c.ctx, norm, GGML_TYPE_F32);
    adapter.begin_graph();
    check(adapter.patch_weight(c.ctx, p.backend, norm, norm_key) == norm, "zero dense norm returns identical pointer");
    check(adapter.patch_weight(c.ctx, p.backend, bias, module + ".bias") == bias, "zero dense bias returns identical pointer");
    auto* out = adapter.forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", linear(1.f/128.f));
    auto* actual_norm = ggml_cast(c.ctx, norm, GGML_TYPE_F32);
    auto* expected = graph(c, {baseline, expected_norm}); auto* actual = graph(c, {out, actual_norm});
    const auto coverage = adapter.report(actual);
    check(coverage.complete() && coverage.target_count == (with_ab ? 4u : 3u), "zero dense-only coverage");
    adapter.validate_coverage(actual);
    check(adapter.get_extra_graph_size() == 0, "zero dense extra graph size");
    identical_graph(expected, actual, "zero dense exact baseline");
    namespace Cut = sd::ggml_graph_cut;
    for (const auto& entry : adapter.tensors()) {
        check(node_index(actual, entry.second) < 0, "zero dense payload cannot be a node");
        for (int i = 0; i < Cut::leaf_count(actual); ++i)
            check(Cut::leaf_tensor(actual, i) != entry.second, "zero dense payload cannot be reachable");
    }
    compute(c, expected); compute(c, actual);
    check(bytes(out) == bytes(baseline) && bytes(actual_norm) == bytes(expected_norm), "zero dense output bytes");
    unchanged(original, "zero dense changed base or NaN payload");
}

// Hand-built pre-dense A/B graph is the oracle for both the omitted flag and
// explicit false; enabling the new feature without deltas must not change it.
static void ab_graph_unchanged(int mode) {
    Params p; auto* w = p.add(module, GGML_TYPE_Q4_0);
    auto* bias = add_vector(p, module + ".bias", GGML_TYPE_F32);
    String2TensorStorage storage; add_pair(storage);
    const float coefficient = -.75f;
    std::unique_ptr<Adapter> adapter;
    if (mode == 0) adapter.reset(new Adapter(p.ctx, p.base, storage, coefficient));
    else adapter.reset(new Adapter(p.ctx, p.base, storage, coefficient, mode == 2));
    p.alloc();
    for (auto* t = ggml_get_first_tensor(p.ctx); t; t = ggml_get_next_tensor(p.ctx, t)) fill(t, .007f);
    const auto original = snapshot(p);
    Context c; auto* input = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 32, 3, 2, 2); fill(input, .013f);
    auto* x = ggml_permute(c.ctx, input, 0, 2, 1, 3);
    auto* base = ggml_ext_linear(c.ctx, x, w, bias, true, 1.f/128.f, (module + ".weight").c_str());
    auto* ax = ggml_ext_linear(c.ctx, x, adapter->tensors().at(module + ".lora_A.weight"), nullptr, true, 1.f);
    auto* bax = ggml_ext_linear(c.ctx, ax, adapter->tensors().at(module + ".lora_B.weight"), nullptr, true, 1.f);
    auto* reference = ggml_add(c.ctx, base, ggml_scale(c.ctx, bax, coefficient));
    adapter->begin_graph();
    auto* out = adapter->forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", linear(1.f/128.f));
    auto* expected = graph(c, {reference}); auto* actual = graph(c, {out});
    adapter->validate_coverage(actual);
    identical_graph(expected, actual, "legacy A/B mode " + std::to_string(mode));
    compute(c, expected); compute(c, actual);
    check(bytes(reference) == bytes(out), "A/B exact result unchanged by opt-in API");
    unchanged(original, "A/B opt-in API changed parameter");
}

static void dense_rejections() {
    const auto wk = module + ".weight", dk = wk + ".delta";
    auto bad = [&](const std::string& label, auto mutate) {
        Params p; p.add(module); String2TensorStorage storage; add_delta(storage, wk);
        mutate(p, storage);
        const auto before = ggml_used_mem(p.ctx);
        rejects([&] { Adapter a(p.ctx, p.base, storage, 1.f, true); }, "dense " + label);
        check(ggml_used_mem(p.ctx) == before, "dense rejection partially allocated runner: " + label);
    };
    for (bool with_ab : {false, true}) for (float coefficient : {0.f, 1.f}) {
        Params p; p.add(module); String2TensorStorage s; add_delta(s, wk);
        if (with_ab) add_pair(s);
        const auto before = ggml_used_mem(p.ctx);
        rejects([&] { Adapter a(p.ctx, p.base, s, coefficient); }, "default false rejects .delta");
        rejects([&] { Adapter a(p.ctx, p.base, s, coefficient, false); }, "explicit false rejects .delta");
        check(ggml_used_mem(p.ctx) == before, "guarded opt-in must validate before allocation");
    }
    bad("missing base", [&](auto& p, auto&) { p.base.clear(); });
    bad("null base", [&](auto& p, auto&) { p.base[wk] = nullptr; });
    bad("key/name mismatch", [&](auto&, auto& s) { s.at(dk).name = module + ".bias.delta"; });
    bad("duplicate storage alias", [&](auto&, auto& s) { s.emplace("alias", s.at(dk)); });
    bad("base identity alias", [&](auto& p, auto& s) { p.base[module + "_alias.weight"] = p.base.at(wk); add_delta(s, module + "_alias.weight"); });
    bad("A/B and dense base identity alias", [&](auto& p, auto& s) { p.base[module + "_alias.weight"] = p.base.at(wk); add_pair(s, module + "_alias"); });
    bad("base type", [&](auto& p, auto&) { p.base.at(wk)->type = GGML_TYPE_I32; });
    bad("nonleaf base", [&](auto& p, auto&) { p.base.at(wk)->op = GGML_OP_ADD; });
    bad("view base", [&](auto& p, auto&) { p.base[wk] = ggml_view_tensor(p.ctx, p.base.at(wk)); });
    bad("noncontiguous base", [&](auto& p, auto&) { p.base.at(wk)->nb[1] *= 2; });
    bad("input shape", [&](auto&, auto& s) { s.at(dk).ne[0] = 31; });
    bad("output shape", [&](auto&, auto& s) { s.at(dk).ne[1] = 5; });
    bad("zero dimension", [&](auto&, auto& s) { s.at(dk).ne[0] = 0; });
    bad("negative dimension", [&](auto&, auto& s) { s.at(dk).ne[1] = -1; });
    bad("rank-3 payload", [&](auto&, auto& s) { s.at(dk).n_dims = 3; });
    bad("rank-0 payload", [&](auto&, auto& s) { s.at(dk).n_dims = 0; });
    bad("hidden dimension", [&](auto&, auto& s) { s.at(dk).ne[4] = 2; });
    bad("overflow", [&](auto&, auto& s) { s.at(dk).ne[0] = std::numeric_limits<int64_t>::max(); });
    bad("quantized payload", [&](auto&, auto& s) { s.at(dk).type = GGML_TYPE_Q4_0; });
    bad("integer payload", [&](auto&, auto& s) { s.at(dk).type = GGML_TYPE_I32; });
    bad("dtype conversion", [&](auto&, auto& s) { s.at(dk).expected_type = GGML_TYPE_F16; });
    bad("FP8 payload", [&](auto&, auto& s) { s.at(dk).is_f8_e5m2 = true; });
    bad("F64 payload", [&](auto&, auto& s) { s.at(dk).is_f64 = true; });
    bad("reordered payload", [&](auto&, auto& s) { s.at(dk).swap_scale_shift = true; });
    bad("existing delta name", [&](auto& p, auto&) { ggml_set_name(ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, 1), dk.c_str()); });
    bad("unnormalized target", [&](auto&, auto& s) { s.clear(); add_delta(s, "diffusion_model.blocks.0.attn.qkv_proj.weight"); });
    bad("unsupported dense suffix", [&](auto&, auto& s) { s.clear(); add_delta(s, module + ".alpha"); });
    bad("incomplete A/B beside delta", [&](auto&, auto& s) { add_pair(s); s.erase(module + ".lora_B.weight"); });
    for (int rank : {0, 2049}) bad("invalid A/B rank beside delta", [&](auto&, auto& s) { add_pair(s, module, GGML_TYPE_F32, GGML_TYPE_F32, 32, rank); });
    bad("mismatched A/B rank beside delta", [&](auto&, auto& s) { add_pair(s); s.at(module + ".lora_B.weight").ne[0] = 2; });
    {
        Params p, foreign; p.base[wk] = foreign.add(module);
        String2TensorStorage s; add_delta(s, wk);
        const auto before = ggml_used_mem(p.ctx);
        rejects([&] { Adapter a(p.ctx, p.base, s, 1.f, true); }, "dense foreign runner identity");
        check(ggml_used_mem(p.ctx) == before, "foreign runner rejection allocated payload");
    }
    for (const auto& key : {module + ".bias", norm_key}) {
        Params p; add_vector(p, key); String2TensorStorage s; add_delta(s, key, GGML_TYPE_F32, 5, 1, 1);
        const auto before = ggml_used_mem(p.ctx);
        rejects([&] { Adapter a(p.ctx, p.base, s, 1.f, true); }, "dense vector shape mismatch");
        check(ggml_used_mem(p.ctx) == before, "vector shape rejection allocated payload");
    }
}

static void dense_coverage_and_paths() {
    // Reject a discarded delta even when BOTH its base and payload are still
    // graph-reachable. Mere leaf presence or an earlier forward call is not use.
    for (int kind = 0; kind < 3; ++kind) {
        Params p; auto* w = p.add(module);
        auto* bias = add_vector(p, module + ".bias", GGML_TYPE_F32);
        auto* norm = add_vector(p, norm_key, GGML_TYPE_F32);
        const std::string key = kind == 0 ? module + ".weight" : kind == 1 ? module + ".bias" : norm_key;
        auto* base = p.base.at(key);
        String2TensorStorage s; add_delta(s, key, GGML_TYPE_F32, base->ne[0], base->ne[1], kind ? 1 : 2);
        Adapter a(p.ctx, p.base, s, 1.f, true);
        Context c; auto* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 2);
        auto forward = [&] { return a.forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", linear()); };
        auto use = [&] { return kind == 2 ? a.patch_weight(c.ctx, p.backend, norm, norm_key) : forward(); };
        rejects([&] { use(); }, "dense begin_graph required");
        a.begin_graph();
        rejects([&] { a.validate_coverage(); }, "dense unused coverage");
        auto* out = use();
        check(a.report().complete() && a.report().target_count == 1, "dense call coverage");
        auto* plain = ggml_ext_linear(c.ctx, x, w, bias, true, 1.f);
        auto* payload_decoy = ggml_sum(c.ctx, a.tensors().at(key + ".delta"));
        auto* base_decoy = ggml_sum(c.ctx, norm);
        auto* discarded = graph(c, {plain, payload_decoy, base_decoy});
        const auto missing = a.report(discarded);
        check(missing.unused == std::vector<std::string>{key + ".delta"}, "discarded dense output must report exact target");
        rejects([&] { a.validate_coverage(discarded); }, "discarded dense output with reachable base/payload");
        auto* full = graph(c, {out});
        a.validate_coverage(full);
        rejects([&] { use(); }, "dense validated graph scope closes");
        a.begin_graph();
        check(a.report().used.empty(), "dense begin_graph resets all calls");
        rejects([&] { a.validate_coverage(full); }, "dense previous output must not leak across graph scopes");
        auto* replacement = ggml_new_tensor_2d(c.ctx, base->type, base->ne[0], base->ne[1]);
        if (kind == 0) {
            rejects([&] { a.forward_with_lora(c.ctx, p.backend, x, replacement, bias, module + ".", linear()); }, "dense changed matrix identity");
            rejects([&] { a.forward_with_lora(c.ctx, p.backend, x, w, bias, "alias.", linear()); }, "dense matrix key alias");
            rejects([&] { a.patch_weight(c.ctx, p.backend, w, key); }, "dense matrix cannot merge through patch_weight");
        } else {
            rejects([&] { a.patch_weight(c.ctx, p.backend, replacement, key); }, "dense changed vector identity");
            rejects([&] { a.patch_weight(c.ctx, p.backend, base, "alias.weight"); }, "dense vector key alias");
            if (kind == 1) {
                rejects([&] { a.forward_with_lora(c.ctx, p.backend, x, w, replacement, module + ".", linear()); }, "dense changed forward bias identity");
                auto* other_w = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 4);
                rejects([&] { a.forward_with_lora(c.ctx, p.backend, x, other_w, bias, "alias.", linear()); }, "dense forward bias key alias");
            }
        }
        if (kind < 2) {
            auto conv = linear(); conv.op_type = WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;
            rejects([&] { a.forward_with_lora(c.ctx, p.backend, x, w, bias, module + ".", conv); }, "dense convolution target rejected");
            auto* bad_x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 31, 2);
            rejects([&] { a.forward_with_lora(c.ctx, p.backend, bad_x, w, bias, module + ".", linear()); }, "dense activation shape mismatch");
        }
        check(a.report().used.empty(), "failed dense calls must not count as use");
    }
    Params p; auto* w = p.add(module); auto* norm = add_vector(p, norm_key, GGML_TYPE_F32);
    String2TensorStorage s; add_pair(s); add_delta(s, module + ".weight"); add_delta(s, norm_key, GGML_TYPE_F32, 4, 1, 1);
    Adapter a(p.ctx, p.base, s, 1.f, true);
    Context c; auto* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 2);
    a.begin_graph();
    auto* y = a.forward_with_lora(c.ctx, p.backend, x, w, nullptr, module + ".", linear());
    check(a.report().target_count == 3 && a.report().unused == std::vector<std::string>{norm_key + ".delta"}, "partial mixed A/B+dense coverage");
    rejects([&] { a.validate_coverage(graph(c, {y})); }, "incomplete mixed dense coverage");
    auto* n = a.patch_weight(c.ctx, p.backend, norm, norm_key);
    rejects([&] { a.validate_coverage(graph(c, {y})); }, "called but discarded norm delta in mixed graph");
    rejects([&] { a.validate_coverage(graph(c, {n})); }, "called but discarded matrix delta in mixed graph");
    a.validate_coverage(graph(c, {y, n}));
}

static void dense_graph_cut_staging() {
    namespace Cut = sd::ggml_graph_cut;
    struct Marks { Marks() { Cut::clear_graph_cut_marks(); } ~Marks() { Cut::clear_graph_cut_marks(); } } marks;
    Params p;
    const std::string second = "model.diffusion_model.blocks.1.attn.out_proj";
    auto* w0 = p.add(module, GGML_TYPE_Q4_0, 32, 32);
    auto* w1 = p.add(second, GGML_TYPE_Q4_0, 32, 4);
    auto* b0 = add_vector(p, module + ".bias", GGML_TYPE_F32, 32);
    auto* norm = add_vector(p, norm_key, GGML_TYPE_BF16, 32);
    auto* unused = p.add("model.diffusion_model.unused");
    String2TensorStorage s;
    add_pair(s, module, GGML_TYPE_F32, GGML_TYPE_F32, 32, 3, 32);
    add_pair(s, second);
    add_delta(s, module + ".weight", GGML_TYPE_F32, 32, 32);
    add_delta(s, second + ".weight");
    add_delta(s, module + ".bias", GGML_TYPE_BF16, 32, 1, 1);
    add_delta(s, norm_key, GGML_TYPE_F32, 32, 1, 1);
    Adapter adapter(p.ctx, p.base, s, .5f, true);
    p.alloc();
    std::unordered_set<const ggml_tensor*> parameter_set;
    for (auto* t = ggml_get_first_tensor(p.ctx); t; t = ggml_get_next_tensor(p.ctx, t)) {
        parameter_set.insert(t); fill(t, .001f);
        check(t->buffer == p.buffer && t->data && !(t->flags & GGML_TENSOR_FLAG_PARAM), "dense inference leaf ownership");
    }
    const auto original = snapshot(p);
    Context c(true); auto* x = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, 32, 2); ggml_set_input(x);
    adapter.begin_graph();
    auto* patched = adapter.patch_weight(c.ctx, p.backend, norm, norm_key);
    auto* normed = ggml_mul(c.ctx, x, ggml_cast(c.ctx, patched, GGML_TYPE_F32));
    auto* y0 = adapter.forward_with_lora(c.ctx, p.backend, normed, w0, b0, module + ".", linear(1.f/128.f));
    auto* y1 = adapter.forward_with_lora(c.ctx, p.backend, y0, w1, nullptr, second + ".", linear(1.f/128.f));
    Cut::mark_graph_cut(y0, "block0", "out"); Cut::mark_graph_cut(y1, "block1", "out");
    auto* g = graph(c, {y1}); adapter.validate_coverage(g);
    std::set<ggml_tensor*> expected_all{w0, w1, b0, norm}, leaves;
    for (const auto& entry : adapter.tensors()) expected_all.insert(entry.second);
    for (int i = 0; i < Cut::leaf_count(g); ++i) leaves.insert(Cut::leaf_tensor(g, i));
    for (auto* t : expected_all) check(leaves.count(t) && node_index(g, t) == -1, "base/A/B/delta must all be graph leaves");
    const auto plan = Cut::build_plan(p.backend, g, parameter_set, "h3-dense-staging");
    check(plan.available && plan.valid && plan.has_cuts && plan.segments.size() == 2, "dense two-cut plan");
    std::set<ggml_tensor*> all_staged;
    size_t total_bytes = 0;
    for (const auto& segment : plan.segments) {
        const bool first = segment.group_name == "block0";
        check(first || segment.group_name == "block1", "unexpected dense cut group");
        const auto& name = first ? module : second;
        std::set<ggml_tensor*> expected{first ? w0 : w1, adapter.tensors().at(name + ".lora_A.weight"),
            adapter.tensors().at(name + ".lora_B.weight"), adapter.tensors().at(name + ".weight.delta")};
        if (first) { expected.insert(b0); expected.insert(norm); expected.insert(adapter.tensors().at(module + ".bias.delta")); expected.insert(adapter.tensors().at(norm_key + ".delta")); }
        std::set<ggml_tensor*> classified;
        bool previous_cut = false;
        for (const auto& input : segment.input_refs) {
            auto* t = Cut::input_tensor(g, input);
            if (parameter_set.count(t)) {
                check(input.type == Cut::Segment::INPUT_PARAM && input.leaf_index >= 0 && input.node_index == -1,
                      "dense base/A/B/delta classification must be leaf INPUT_PARAM");
                classified.insert(t);
            }
            previous_cut |= input.type == Cut::Segment::INPUT_PREVIOUS_CUT;
        }
        check(previous_cut == !first, "dense cut activation boundary");
        check(classified == expected, "dense cut parameter classification");
        const auto selected = Cut::runtime_param_tensors(g, segment, "h3-dense-staging");
        check(std::set<ggml_tensor*>(selected.begin(), selected.end()) == expected && selected.size() == expected.size(),
              "dense runtime staging must include exact base+A+B+deltas");
        size_t expected_bytes = 0;
        for (auto* t : expected) expected_bytes += ggml_nbytes(t);
        check(segment.input_param_bytes == expected_bytes, "dense INPUT_PARAM byte accounting");
        total_bytes += segment.input_param_bytes;
        all_staged.insert(selected.begin(), selected.end());
    }
    size_t expected_bytes = 0;
    for (auto* t : expected_all) expected_bytes += ggml_nbytes(t);
    check(all_staged == expected_all && !all_staged.count(unused), "dense staging excludes unreachable base");
    check(total_bytes == expected_bytes, "dense total staged bytes");
    unchanged(original, "dense graph-cut planning changed parameter");
}

static void dense_zero_leaf_only() {
    Params p;
    auto* w = add_vector(p, norm_key, GGML_TYPE_BF16, 4);
    String2TensorStorage storage;
    add_delta(storage, norm_key, GGML_TYPE_BF16, 4, 1, 1);
    Adapter adapter(p.ctx, p.base, storage, 0.f, true);
    p.alloc();
    Context c;
    adapter.begin_graph();
    auto* result = adapter.patch_weight(c.ctx, p.backend, w, norm_key);
    check(result == w, "zero vector patch should return original leaf");
    auto* g = graph(c, {result});
    check(ggml_graph_n_nodes(g) == 0, "leaf-only regression must have no compute nodes");
    adapter.validate_coverage(g);
}

int main() {
    dense_zero_leaf_only();
    static_assert(GGML_MAX_NAME == 128, "compile against the existing maxname128 build");
    int failures = 0;
    const auto run = [&](const std::string& name, auto test) {
        try { test(); }
        catch (const std::exception& e) {
            ++failures;
            std::cerr << "h3_dynamic_lora_test: FAIL [" << name << "]: " << e.what() << '\n';
        }
    };
    run("dense F32+AB", [] { dense_numerical(GGML_TYPE_F32, 1.f, 1.f, true); });
    run("dense F32+AB strided negative", [] { dense_numerical(GGML_TYPE_F32, -1.25f, 1.f/128.f, true, true); });
    run("dense Q4", [] { dense_numerical(GGML_TYPE_Q4_0, .7f, 1.f/128.f, false); });
    run("dense Q4+AB strided negative", [] { dense_numerical(GGML_TYPE_Q4_0, -.5f, 1.f/128.f, true, true); });
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) for (float coefficient : {1.f, -.75f})
        run(std::string("BF16 norm/bias ") + ggml_type_name(type) + " coefficient=" + std::to_string(coefficient),
            [&] { dense_vectors(coefficient, type); });
    run("zero dense-only", [] { dense_zero_exact(false); });
    run("zero dense+AB", [] { dense_zero_exact(true); });
    for (int mode = 0; mode < 3; ++mode) run("A/B exact graph mode=" + std::to_string(mode), [&] { ab_graph_unchanged(mode); });
    run("dense invalid metadata/opt-in", dense_rejections);
    run("dense coverage/identity", dense_coverage_and_paths);
    run("dense graph-cut staging", dense_graph_cut_staging);
    run("legacy A/B regression suite", [] {
        graph_cut_param_staging();
        numerical(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, 1.f, 1.f);
        numerical(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, -1.25f, 1.f/128.f, 3, 2, true);
        numerical(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, 8.f/128.f, 1.f/128.f, 3, 2);
        numerical(GGML_TYPE_Q4_0, GGML_TYPE_F16, GGML_TYPE_F16, .7f, 1.f/128.f);
        numerical(GGML_TYPE_Q4_0, GGML_TYPE_BF16, GGML_TYPE_BF16, .7f, 1.f/128.f);
        numerical(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, -.5f, 1.f, 2, 2);
        numerical(GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F32, .5f, 1.f, 1, 1, false, GGML_TYPE_F16);
        numerical(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, .5f, 1.f/128.f, 1025);
        numerical(GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_F32, .5f, 1.f/128.f, 1025, 1, true);
        zero_exact(); rejections(); coverage_and_paths(); loader_roundtrip();
    });
    std::cout << "h3_dynamic_lora_test: " << (failures ? "FAIL" : "PASS") << " (" << checks << " checks, "
              << graphs << " CPU graphs, " << failures << " failed groups)\n";
    return failures ? 1 : 0;
}
