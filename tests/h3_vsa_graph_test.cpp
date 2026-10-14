// Standalone CPU-only ggml regression; link ggml-cpu + ggml-base, no model/CUDA.
#include "dit_models/models/minimax_h3_vsa_graph.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <iostream>
#include <set>
#include <sstream>

namespace G = MiniMaxH3::VSAGraph;
namespace R = MiniMaxH3::VSAReference;
static int assertions = 0, graphs = 0, fixed_nodes = -1;
static float max_error = 0;
static void check(bool ok, const char* message) {
    ++assertions;
    if (!ok) throw std::runtime_error(message);
}
static void close(float got, float expected, const char* message) {
    float error = std::abs(got - expected);
    max_error = std::max(max_error, error);
    if (!std::isfinite(got) || error > 3e-5f * (1.f + std::abs(expected))) {
        std::ostringstream ss; ss << message << ": got " << got << ", expected " << expected;
        throw std::runtime_error(ss.str());
    }
    ++assertions;
}
struct Context {
    ggml_context* ctx;
    explicit Context(bool deferred = false) : ctx(ggml_init({32*1024*1024,nullptr,deferred})) {
        check(ctx != nullptr, "context allocation");
    }
    ~Context() { ggml_free(ctx); }
};
static ggml_tensor* tensor(ggml_context* c, int64_t a, int64_t b, int64_t h = 1) {
    return ggml_new_tensor_3d(c, GGML_TYPE_F32, a,b,h);
}
static void put(ggml_tensor* t, int64_t a, int64_t b, int64_t h, float v) {
    std::memcpy(static_cast<char*>(t->data)+a*t->nb[0]+b*t->nb[1]+h*t->nb[2], &v, sizeof(v));
}
static float get(ggml_tensor* t, int64_t a, int64_t b, int64_t h = 0) {
    float v;
    std::memcpy(&v, static_cast<char*>(t->data)+a*t->nb[0]+b*t->nb[1]+h*t->nb[2], sizeof(v));
    return v;
}
static ggml_cgraph* graph(ggml_context* c, const G::Result& out, bool strided = false) {
    auto* gf = ggml_new_graph_custom(c, 256, false);
    ggml_build_forward_expand(gf, out.output);
    std::set<ggml_op> ops;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto* node = ggml_graph_node(gf, i);
        check(node->op != GGML_OP_CUSTOM && node->op != GGML_OP_UNARY, "no host math/sigmoid");
        ops.insert(node->op);
    }
    for (auto op : {GGML_OP_GET_ROWS, GGML_OP_PAD, GGML_OP_MEAN, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX})
        check(ops.count(op) != 0, "missing native primitive");
    const int n = ggml_graph_n_nodes(gf);
    check(n <= G::max_graph_nodes, "bounded node count");
    if (!strided) {
        if (fixed_nodes < 0) fixed_nodes = n;
        check(n == fixed_nodes, "node count must not grow with heads/tiles/dimension");
    }
    ++graphs;
    return gf;
}
// Modes: 0 varied, 1 zero QKV, 2 constant QKV, 3 zero scores/nonzero V.
static void parity(const R::Geometry& geometry, int d, int h, int mode = 0,
                   bool zero_gate = false, bool strided = false, bool deferred = false) {
    Context c(deferred);
    const int64_t s = int64_t(geometry.packed_tokens);
    auto activation = [&](int64_t a, int64_t b, int64_t e) {
        auto* x = tensor(c.ctx,a,strided ? e : b,strided ? b : e);
        return strided ? ggml_permute(c.ctx,x,0,2,1,3) : x;
    };
    auto *q = activation(d,s,h), *k = activation(d,s,h), *v = activation(d,h,s);
    auto *gate = tensor(c.ctx,d*h,s), *fine = tensor(c.ctx,d*h,s);
    std::vector<std::pair<ggml_tensor*, const void*>> bindings;
    G::GraphInputs inputs;
    inputs.bind_backend_tensor_data = [&](ggml_tensor* t, const void* data) {
        check(t->data == nullptr && (t->flags & GGML_TENSOR_FLAG_INPUT), "deferred input constant");
        bindings.emplace_back(t,data);
    };
    auto out = G::build(c.ctx,q,k,v,gate,fine,geometry,inputs);
    auto* gf = graph(c.ctx,out,strided);
    // Allocate/upload only AFTER build has destroyed its local metadata vectors.
    ggml_backend_t backend = deferred ? ggml_backend_cpu_init() : nullptr;
    ggml_backend_buffer_t buffer = deferred ? ggml_backend_alloc_ctx_tensors(c.ctx,backend) : nullptr;
    if (deferred) {
        check(backend && buffer && bindings.size() == 3 && inputs.buffers.size() == 3, "owned constants");
        // Reallocating the owner's pointer vector must not invalidate registered data.
        inputs.buffers.reserve(64);
        for (auto binding : bindings)
            ggml_backend_tensor_set(binding.first,binding.second,0,ggml_nbytes(binding.first));
    } else check(bindings.empty() && inputs.buffers.empty(), "CPU storage needs no binding");
    R::Tensor rq(1,s,h,d), rk(1,s,h,d), rv(1,s,h,d), rg(1,s,h,d), rf(1,s,h,d);
    for (int64_t i = 0; i < s; ++i) for (int j = 0; j < h; ++j) for (int a = 0; a < d; ++a) {
        float seed = float((i*17+j*11+a*7)%41)-20.f;
        float x = mode == 1 || mode == 3 ? 0.f : mode == 2 ? .75f : seed*.07f+.3f;
        float y = mode == 1 ? 0.f : mode == 2 ? -1.25f : float((i*13+j*3+a*11)%29)*.09f-.8f;
        float z = mode == 1 ? 0.f : mode == 2 ? 2.5f : seed*.13f+float(j)*.17f+.9f;
        float g = zero_gate ? 0.f : (i+j+a)%3 == 0 ? -2.f : (i+j+a)%3 == 1 ? 0.f : 1.75f;
        float f = seed*.017f-.2f;
        rq.at(0,i,j,a)=x; rk.at(0,i,j,a)=y; rv.at(0,i,j,a)=z;
        rg.at(0,i,j,a)=g; rf.at(0,i,j,a)=f;
        put(q,a,i,j,x); put(k,a,i,j,y); put(v,a,j,i,z);
        put(gate,j*d+a,i,0,g); put(fine,j*d+a,i,0,f);
    }
    auto tq=R::tile(rq,geometry), tk=R::tile(rk,geometry), tv=R::tile(rv,geometry);
    // Independent oracle explicitly ignores padding; poison it to detect an
    // accidental reference dependency on the graph's zero-padding assumption.
    for (size_t i=0;i<geometry.padded_tokens();++i) if (geometry.source_index[i]==R::invalid_token)
        for (int j=0;j<h;++j) for (int a=0;a<d;++a)
            tq.at(0,i,j,a)=tk.at(0,i,j,a)=tv.at(0,i,j,a)=10000.f;
    auto pq=R::pool_tiles(tq,geometry), pk=R::pool_tiles(tk,geometry), pv=R::pool_tiles(tv,geometry);
    auto scores=R::pooled_scores(tq,tk,geometry);
    auto compressed=R::compressed_attention(tv,geometry,scores);
    auto expected=R::untile(R::fuse_compression(R::tile(rf,geometry),compressed,R::tile(rg,geometry),geometry),geometry);
    for (int replay=0;replay<(deferred ? 2 : 1);++replay) {
        auto status = deferred ? ggml_backend_graph_compute(backend,gf) : ggml_graph_compute_with_ctx(c.ctx,gf,1);
        check(status==GGML_STATUS_SUCCESS,"CPU graph compute");
        for (size_t i=0;i<geometry.tiles();++i) for (int j=0;j<h;++j) {
            std::vector<float> logits(geometry.tiles());
            for (size_t b=0;b<geometry.tiles();++b) logits[b]=scores.at(0,j,i,b);
            auto weights=R::softmax(logits);
            float sum=0;
            for (size_t b=0;b<geometry.tiles();++b) {
                close(get(out.scores,b,i,j),logits[b],"scaled score");
                close(get(out.probabilities,b,i,j),weights[b],"all-tile softmax");
                sum+=get(out.probabilities,b,i,j);
            }
            close(sum,1.f,"normalized over logical tiles only");
            for (int a=0;a<d;++a) {
                close(get(out.pooled_q,a,i,j),pq.at(0,i,j,a),"pooled Q");
                close(get(out.pooled_k,a,i,j),pk.at(0,i,j,a),"pooled K");
                close(get(out.pooled_v,a,i,j),pv.at(0,i,j,a),"pooled V/valid divisor");
                close(get(out.compressed_tiles,a,i,j),compressed.at(0,i,j,a),"compressed tile");
            }
        }
        check(out.output->ne[0]==d*h && out.output->ne[1]==s,"preserve all packed rows");
        for (int64_t i=0;i<s;++i) for (int j=0;j<h;++j) for (int a=0;a<d;++a) {
            close(get(out.compressed_rows,j*d+a,i),compressed.at(0,geometry.untile_combined_index[i]/64,j,a),"untile broadcast");
            close(get(out.output,j*d+a,i),expected.at(0,i,j,a),"raw-gate fusion");
            if (zero_gate || mode==1) check(get(out.output,j*d+a,i)==rf.at(0,i,j,a),"zero branch preserves fine exactly");
        }
    }
    if (buffer) ggml_backend_buffer_free(buffer);
    if (backend) ggml_backend_free(backend);
}
// Metadata only: no backend allocation or compute, even for real-size tensors.
static void planning(const R::Geometry& geometry, int d, int h) {
    Context c(true);
    const int64_t s=int64_t(geometry.packed_tokens);
    auto *q=tensor(c.ctx,d,s,h), *k=tensor(c.ctx,d,s,h), *v=tensor(c.ctx,d,h,s);
    auto *gate=tensor(c.ctx,d*h,s), *fine=tensor(c.ctx,d*h,s);
    G::GraphInputs inputs;
    size_t bindings=0;
    inputs.bind_backend_tensor_data=[&](ggml_tensor* t,const void* ptr) {
        check(!t->data && ptr,"metadata-only constant binding"); ++bindings;
    };
    const size_t bytes=G::estimate_bytes(geometry,d,h), before=ggml_used_mem(c.ctx);
    for (size_t budget : {size_t(0),bytes-1,std::min(bytes-1,G::default_budget_bytes)}) {
        bool failed=false;
        try { G::build(c.ctx,q,k,v,gate,fine,geometry,inputs,budget); }
        catch (const std::invalid_argument&) { failed=true; }
        check(failed && before==ggml_used_mem(c.ctx) && bindings==0,"byte budget rejected before construction");
    }
    auto out=G::build(c.ctx,q,k,v,gate,fine,geometry,inputs,bytes);
    size_t actual=ggml_used_mem(c.ctx)-before;
    for (auto* t=ggml_get_first_tensor(c.ctx);t;t=ggml_get_next_tensor(c.ctx,t)) {
        check(!t->data,"large test must stay metadata-only");
        if (!t->view_src && t!=q && t!=k && t!=v && t!=gate && t!=fine) actual+=ggml_nbytes_pad(t);
    }
    check(actual<=bytes && bindings==3,"estimate covers all owned tensors/metadata");
    graph(c.ctx,out);
    check(out.scores->ne[0]==int64_t(geometry.tiles()) && out.scores->ne[2]==h,"large routing layout");
    std::cout << "PLAN S=" << s << " T=" << geometry.tiles() << " H=" << h << " D=" << d
              << " conservative_bytes=" << bytes << " (not peak VRAM)\n";
}
static void bounds() {
    Context c(true);
    auto geometry=R::make_geometry({R::Segment::video({1,1,1})},64);
    auto *q=tensor(c.ctx,4,1,2), *k=tensor(c.ctx,4,1,2), *v=tensor(c.ctx,4,2,1);
    auto *gate=tensor(c.ctx,8,1), *fine=tensor(c.ctx,8,1);
    G::GraphInputs inputs;
    auto reject = [&](auto fn) {
        size_t before=ggml_used_mem(c.ctx);
        bool failed=false;
        try { fn(); } catch (const std::invalid_argument&) { failed=true; }
        check(failed && before==ggml_used_mem(c.ctx),"reject before graph allocation");
    };
    auto build = [&] { return G::build(c.ctx,q,k,v,gate,fine,geometry,inputs); };
    reject(build); // no binder
    inputs.bind_backend_tensor_data=[](ggml_tensor*,const void*) { throw std::runtime_error("unexpected binding"); };
    reject([&] { G::build(nullptr,q,k,v,gate,fine,geometry,inputs); });
    reject([&] { G::build(c.ctx,nullptr,k,v,gate,fine,geometry,inputs); });
    for (int tile : {32,128,256}) { geometry.tile_elems=tile; reject(build); }
    geometry.tile_elems=64;
    geometry.source_index[1]=0; reject(build); geometry.source_index[1]=R::invalid_token;
    geometry.variable_block_sizes[0]=0; reject(build); geometry.variable_block_sizes[0]=1;
    geometry.untile_combined_index[0]=1; reject(build); geometry.untile_combined_index[0]=0;
    auto original=*q;
    for (int64_t d : {int64_t(0), G::max_head_dim+1, INT64_MAX}) { q->ne[0]=d; reject(build); }
    *q=original; q->ne[2]=G::max_heads+1; reject(build); *q=original;
    q->type=GGML_TYPE_F16; reject(build); *q=original;
    q->nb[0]=8; reject(build); *q=original;
    q->ne[3]=2; reject(build); *q=original;
    reject([&] { G::build(c.ctx,q,k,q,gate,fine,geometry,inputs); }); // V layout
    reject([&] { G::build(c.ctx,q,k,v,q,fine,geometry,inputs); }); // gate layout
    reject([&] { G::build(c.ctx,q,k,v,gate,nullptr,geometry,inputs); });
    geometry.packed_tokens=G::max_tokens+1; reject(build);
    geometry.packed_tokens=1; geometry.variable_block_sizes.resize(G::max_tiles+1,1); reject(build);
    geometry.variable_block_sizes.resize(G::max_tiles,1);
    q->ne[0]=G::max_head_dim; q->ne[2]=G::max_heads; reject(build); // ggml int32 stride limit
    q->ne[0]=1; q->ne[2]=G::max_heads; reject(build); // stale/inconsistent metadata
    reject([&] { G::detail::mul(std::numeric_limits<size_t>::max(),2); });
    reject([&] { G::detail::add(std::numeric_limits<size_t>::max(),1); });
    check(inputs.buffers.empty(),"invalid builds never bind constants");
}
int main() {
    try {
        auto one=R::make_geometry({R::Segment::video({1,1,1})},64);
        auto mixed=R::make_geometry({R::Segment::video({2,3,5}),R::Segment::text(3),
            R::Segment::audio(65),R::Segment::reference(7),R::Segment::video({5,6,7})},64);
        auto odd=R::make_geometry({R::Segment::text(1),R::Segment::audio(2),R::Segment::video({3,2,4})},64);
        check(odd.tiles()%2==1,"odd logical tiles: no transport dummy in scores");
        parity(one,1,1); parity(one,128,2);
        parity(mixed,7,3); parity(mixed,4,5,0,false,true);
        parity(mixed,3,2,1); parity(mixed,3,2,2); parity(mixed,3,2,3);
        parity(odd,5,3,0,true); parity(odd,5,3,0,false,false,true);
        // Local 864x480/124f probe: latent [37,30,54], patch [1,2,2],
        // deterministic text64/audio207 fixture (not an asserted prompt length).
        planning(R::make_geometry({R::Segment::text(64),R::Segment::audio(207),
            R::Segment::video({37,15,27})},64),128,56);
        planning(R::make_geometry({R::Segment::video({1,1,8192})},64),1,64);
        planning(R::make_geometry({R::Segment::video({16,64,64})},64),1,1);
        bounds();
        std::cout << "PASS h3_vsa_graph: " << assertions << " checks, " << graphs
                  << " graphs, " << fixed_nodes << " fixed nodes, max error " << max_error << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL h3_vsa_graph: " << e.what() << '\n'; return 1; }
}
