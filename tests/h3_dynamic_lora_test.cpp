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

int main() {
    try {
        static_assert(GGML_MAX_NAME == 128, "compile against the existing maxname128 build");
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
        std::cout << "h3_dynamic_lora_test: PASS (" << checks << " checks, " << graphs << " CPU graphs)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "h3_dynamic_lora_test: FAIL: " << e.what() << '\n'; return 1;
    }
}
