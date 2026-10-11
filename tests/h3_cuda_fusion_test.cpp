// CPU-only numerical graph regression, despite the historical CUDA filename.
// Standalone: see tests/h3-fusion-progress.md. No GPU discovery/load-all calls.
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Avoid the shared backend's convenience wrapper (which can load GPU plugins).
#ifdef GGML_BACKEND_DL
#error "This standalone test requires the directly linked CPU backend (GGML_BACKEND_DL=OFF)"
#endif
extern "C" ggml_backend_reg_t h3_fusion_cpu_reg() asm("ggml_backend_cpu_reg");

namespace {
using Shape = std::array<int64_t, 4>;
using namespace edgedit::ggml_ext;
constexpr float tolerance = 1e-6f;
int comparisons = 0;
float worst_error = 0;
void check(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
struct Env {
    std::string name, old;
    bool existed;
    Env(const char* key, const char* value) : name(key), existed(std::getenv(key) != nullptr) {
        if (existed) old = std::getenv(key);
        check(setenv(key, value, 1) == 0, "setenv failed");
    }
    ~Env() { if (existed) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};
struct CpuGraph {
    ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    std::vector<ggml_tensor*> roots;
    explicit CpuGraph(int threads) {
        auto reg = h3_fusion_cpu_reg(); // ONLY the directly linked CPU registration.
        check(reg != nullptr && ggml_backend_reg_dev_count(reg) == 1, "CPU registration unavailable");
        auto dev = ggml_backend_reg_dev_get(reg, 0);
        check(ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU, "not a CPU device");
        backend = ggml_backend_dev_init(dev, nullptr);
        check(backend != nullptr, "CPU initialization failed");
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        check(set_threads != nullptr, "CPU thread setter unavailable");
        set_threads(backend, threads);
        ctx = ggml_init({8u * 1024u * 1024u, nullptr, true});
        check(ctx != nullptr, "ggml context allocation failed");
    }
    ~CpuGraph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    ggml_tensor* input(Shape s, bool padded = false, ggml_type type = GGML_TYPE_F32) {
        Shape parent = s;
        if (padded) { parent[0] += 3; parent[1] += 2; }
        auto base = ggml_new_tensor_4d(ctx, type, parent[0], parent[1], parent[2], parent[3]);
        roots.push_back(base);
        if (!padded) return base;
        // Nonzero offset plus gaps between rows and between batches.
        auto view = ggml_view_4d(ctx, base, s[0], s[1], s[2], s[3],
                                 base->nb[1], base->nb[2], base->nb[3],
                                 base->nb[1] + base->nb[0]);
        check(view->view_offs > 0, "missing view offset");
        return view;
    }
    void prepare(const std::vector<ggml_tensor*>& outputs) {
        graph = ggml_new_graph_custom(ctx, 4096, false);
        for (auto out : outputs) { check(out != nullptr, "unsupported output: no silent fallback"); ggml_build_forward_expand(graph, out); }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            check(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)), "unsupported CPU graph op");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr, "CPU tensor allocation failed");
        fill(0);
    }
    void fill(int replay) {
        for (size_t r = 0; r < roots.size(); ++r) {
            auto t = roots[r];
            check(t->type == GGML_TYPE_F32, "non-F32 numerical input");
            std::vector<float> values(size_t(ggml_nelements(t)));
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = .5f * std::sin(float(i % 113) * .17f + float(r) * .43f + replay * .71f);
            if (!values.empty()) values[0] = 0; // Include zero, positive and negative values.
            ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
        }
    }
    void compute() { check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "CPU graph execution failed"); }
    int custom_count(const char* name) const {
        int n = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto t = ggml_graph_node(graph, i);
            if (t->op == GGML_OP_CUSTOM && std::strcmp(ggml_get_name(t), name) == 0) ++n;
        }
        return n;
    }
};
float at(const ggml_tensor* t, int64_t a, int64_t b, int64_t c, int64_t d) {
    float value;
    const size_t offset = (a % t->ne[0]) * t->nb[0] + (b % t->ne[1]) * t->nb[1] +
                          (c % t->ne[2]) * t->nb[2] + (d % t->ne[3]) * t->nb[3];
    ggml_backend_tensor_get(t, &value, offset, sizeof(value));
    return value;
}
void compare(ggml_tensor* actual, ggml_tensor* expected, const std::string& name) {
    check(actual && expected && actual->type == GGML_TYPE_F32 && expected->type == GGML_TYPE_F32, name + ": wrong type");
    check(ggml_are_same_shape(actual, expected), name + ": wrong shape");
    float error = 0;
    for (int64_t d = 0; d < actual->ne[3]; ++d) for (int64_t c = 0; c < actual->ne[2]; ++c)
    for (int64_t b = 0; b < actual->ne[1]; ++b) for (int64_t a = 0; a < actual->ne[0]; ++a) {
        const float x = at(actual,a,b,c,d), y = at(expected,a,b,c,d);
        check(std::isfinite(x) && std::isfinite(y), name + ": nonfinite result");
        error = std::max(error, std::fabs(x-y));
    }
    check(error <= tolerance, name + ": max_abs=" + std::to_string(error));
    worst_error = std::max(worst_error, error);
    ++comparisons;
    std::cout << "PASS " << name << " max_abs=" << error << '\n';
}
void direct_case(Shape shape, Shape shift_shape, Shape scale_shape, bool padded, int threads) {
    CpuGraph g(threads);
    auto x = g.input(shape,padded), residual = g.input(shape,padded);
    auto shift = g.input(shift_shape,padded), scale = g.input(scale_shape,padded);
    auto gate = g.input(scale_shape,padded);
    auto mod = fused_modulate_custom(g.ctx,x,shift,scale);
    auto res = fused_residual_gate_custom(g.ctx,residual,x,gate);
    check(mod && res && mod->op == GGML_OP_CUSTOM && res->op == GGML_OP_CUSTOM,
          "supported fusion must build actual custom ops");
    auto baseline_mod = ggml_add(g.ctx,ggml_add(g.ctx,x,ggml_mul(g.ctx,x,scale)),shift);
    auto baseline_res = ggml_add(g.ctx,residual,ggml_mul(g.ctx,x,gate));
    g.prepare({mod,baseline_mod,res,baseline_res});
    check(g.custom_count("ed_fused_modulate") == 1 && g.custom_count("ed_fused_residual_gate") == 1,
          "fusion absent from executable graph");
    for (int replay = 0; replay < 2; ++replay) {
        g.fill(replay); g.compute();
        auto label = std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + "x" +
                     std::to_string(shape[2]) + "x" + std::to_string(shape[3]) +
                     (padded ? "/view" : "/dense") + "/threads=" + std::to_string(threads) + "/replay=" + std::to_string(replay);
        compare(mod,baseline_mod,"modulate/"+label);
        compare(res,baseline_res,"residual/"+label);
        // Independent scalar check supplements (does not replace) both CPU graphs.
        for (int64_t d=0; d<shape[3]; ++d) for (int64_t c=0; c<shape[2]; ++c)
        for (int64_t b=0; b<shape[1]; ++b) for (int64_t a=0; a<shape[0]; ++a) {
            const float xv=at(x,a,b,c,d), sv=at(scale,a,b,c,d), sh=at(shift,a,b,c,d);
            const float expected_mod=(xv+xv*sv)+sh;
            const float expected_res=at(residual,a,b,c,d)+xv*at(gate,a,b,c,d);
            check(std::fabs(at(mod,a,b,c,d)-expected_mod)<=tolerance,"modulation scalar oracle");
            check(std::fabs(at(res,a,b,c,d)-expected_res)<=tolerance,"residual scalar oracle");
        }
    }
}
void segmented_case(int64_t hidden, int64_t batch, bool padded, int threads) {
    CpuGraph g(threads);
    auto x=g.input({hidden,11,batch,1},padded), update=g.input({hidden,11,batch,1},padded);
    // 6 AdaLN fields, 3 modalities, 2 timestep rows: all row/chunk offsets nontrivial.
    auto projection=g.input({hidden*6*3,2,1,1});
    const std::vector<MiniMaxH3::TokenModulationSpan> spans={{0,1,4},{1,4,0},{4,6,5},{6,7,2},{7,11,1}};
    ggml_tensor *bm, *br, *fm, *fr;
    {
        Env all("ED_MINIMAX_H3_FUSED_SEGMENT_MODULATION","0");
        Env mod("ED_MINIMAX_H3_FUSED_SEGMENT_MODULATE","0");
        Env res("ED_MINIMAX_H3_FUSED_SEGMENT_RESIDUAL_GATE","0");
        Env views("ED_MINIMAX_H3_DISABLE_SEGMENT_VIEW","1");
        bm=MiniMaxH3::modulate_segments(g.ctx,x,projection,spans,hidden,6,3,0,1,nullptr,g.backend);
        br=MiniMaxH3::gated_residual_segments(g.ctx,x,update,projection,spans,hidden,2,nullptr,g.backend);
    }
    {
        Env all("ED_MINIMAX_H3_FUSED_SEGMENT_MODULATION","1");
        Env views("ED_MINIMAX_H3_DISABLE_SEGMENT_VIEW","0");
        fm=MiniMaxH3::modulate_segments(g.ctx,x,projection,spans,hidden,6,3,0,1,nullptr,g.backend);
        fr=MiniMaxH3::gated_residual_segments(g.ctx,x,update,projection,spans,hidden,2,nullptr,g.backend);
    }
    g.prepare({bm,br,fm,fr});
    check(g.custom_count("ed_fused_modulate")==int(spans.size()),"segmented modulation silently fell back");
    check(g.custom_count("ed_fused_residual_gate")==int(spans.size()),"segmented residual silently fell back");
    for (int replay=0; replay<2; ++replay) {
        g.fill(replay); g.compute();
        compare(fm,bm,"segmented/modulate/hidden="+std::to_string(hidden)+"/batch="+std::to_string(batch));
        compare(fr,br,"segmented/residual/hidden="+std::to_string(hidden)+"/batch="+std::to_string(batch));
    }
}
void swiglu_case(int64_t hidden, int64_t tokens, int64_t batch, bool padded, int threads) {
    CpuGraph g(threads);
    auto uv=g.input({2*hidden,tokens,batch,1},padded);
    auto fused=ggml_swiglu(g.ctx,uv);
    auto chunks=ggml_ext_chunk(g.ctx,uv,2,0,true);
    auto baseline=ggml_mul(g.ctx,ggml_silu(g.ctx,chunks[0]),chunks[1]);
    check(fused->op==GGML_OP_GLU && ggml_get_glu_op(fused)==GGML_GLU_OP_SWIGLU,"not actual SwiGLU");
    g.prepare({fused,baseline});
    for(int replay=0; replay<2; ++replay) {
        g.fill(replay); g.compute();
        compare(fused,baseline,"swiglu/hidden="+std::to_string(hidden)+(padded?"/view":"/dense"));
    }
}
void reject_unsupported() {
    CpuGraph g(1);
    auto x=g.input({8,6,2,1}), good=g.input({8,1,1,1});
    auto bad0=g.input({3,1,1,1}), bad1=g.input({8,4,1,1}), bad2=g.input({8,1,3,1});
    auto bad3=g.input({8,1,1,2}), wrong_res=g.input({8,5,2,1});
    auto half=g.input({8,1,1,1},false,GGML_TYPE_F16);
    auto integer=g.input({8,1,1,1},false,GGML_TYPE_I32);
    for (auto bad : {bad0,bad1,bad2,bad3,half,integer}) {
        check(!fused_modulate_shape_supported(x,bad,good),"invalid shift accepted");
        check(!fused_modulate_custom(g.ctx,x,bad,good),"invalid shift constructed fusion");
        check(!fused_modulate_custom(g.ctx,x,good,bad),"invalid scale constructed fusion");
        check(!fused_residual_gate_shape_supported(x,x,bad),"invalid gate accepted");
        check(!fused_residual_gate_custom(g.ctx,x,x,bad),"invalid gate constructed fusion");
    }
    check(!fused_residual_gate_custom(g.ctx,wrong_res,x,good),"mismatched residual accepted");
    check(!fused_modulate_custom(g.ctx,nullptr,good,good),"null x accepted");
    check(!fused_residual_gate_custom(g.ctx,x,nullptr,good),"null update accepted");
    check(!fused_modulate_custom(g.ctx,half,good,good),"F16 x accepted");
    check(!fused_residual_gate_custom(g.ctx,half,x,good),"F16 residual accepted");
    std::cout << "PASS unsupported shapes/types/null rejected (no fallback)\n";
}
} // namespace
int main() {
    try {
        // Request-scoped CUDA fusion intentionally remains untested on CPU;
        // segmented CPU custom ops are selected via scoped legacy flags above.
        for (int threads : {1,2}) {
            direct_case({7,5,1,1},{7,1,1,1},{1,5,1,1},false,threads);
            direct_case({8,6,3,1},{8,1,3,1},{8,1,1,1},true,threads);
            direct_case({12,6,4,2},{3,2,2,1},{4,3,1,2},true,threads);
            direct_case({5,3,2,1},{5,3,2,1},{5,3,2,1},false,threads);
            segmented_case(7,1,false,threads);
            segmented_case(16,2,true,threads);
            swiglu_case(7,5,2,false,threads);
            swiglu_case(16,7,1,true,threads);
        }
        reject_unsupported();
        std::cout << "PASS CPU-only actual ggml graphs: " << comparisons
                  << " comparisons; tolerance=" << tolerance << "; worst_max_abs=" << worst_error
                  << ". CUDA NOT validated.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL H3 CPU fusion regression: " << e.what() << '\n';
        return 1;
    }
}
