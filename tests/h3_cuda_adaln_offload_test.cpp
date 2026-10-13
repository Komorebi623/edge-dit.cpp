// Standalone, OPT-IN real CUDA helper/graph-path test, NOT production offload enablement.
// No CTest registration. Build-only script never runs this executable.
// Run ONLY via scripts/test_h3_gpu_parity.sh --run --device GPU-... --binary ... .
// Synthetic F32 tiny H3; no downloaded/checkpoint weights, no quality/speed claim.
// Serial graph-cut traversal below uses the actual planner, binding, segment
// builder and GGMLRunner executor, with additional per-segment validation hooks.
// It does not test the production compute() dispatch, max-VRAM planner or async prefetch.
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include "ggml-cuda.h"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <memory>
#include <unordered_set>
#if defined(GGML_BACKEND_DL) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#error "Requires directly linked NVIDIA CUDA and CPU backends"
#endif
static_assert(GGML_MAX_NAME == 128, "candidate ggml ABI mismatch");
// Avoid shared-runner CPU discovery wrappers. Register exactly these two backends.
extern "C" ggml_backend_reg_t h3_offload_cpu_reg() __asm__("ggml_backend_cpu_reg");
extern "C" ggml_backend_t h3_offload_cpu_init() __asm__("ggml_backend_cpu_init");
namespace {
namespace EA = MiniMaxH3::ExactAdaLN;
namespace Cut = sd::ggml_graph_cut;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
size_t comparisons=0, audited_graphs=0, audited_nodes=0, cut_segments=0;
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

void check(bool ok, const std::string& s) { require(ok,s); }
void expect(EA::Status got, EA::Status wanted, const char* s) {
    require(got==wanted,std::string(s)+": "+EA::status_name(got));
}
void compare(const sd::Tensor<float>& a,const sd::Tensor<float>& b,const std::string& label,
             float atol=2e-4f,float rtol=2e-4f) {
    require(!a.empty() && a.shape()==b.shape(),label+": shape");
    float maximum=0;
    for(int64_t i=0;i<a.numel();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),label+": nonfinite");
        const float d=std::fabs(a[i]-b[i]);maximum=std::max(maximum,d);
        require(d<=atol+rtol*std::fabs(a[i]),label+": numerical mismatch at "+std::to_string(i));
    }
    ++comparisons;
    std::cout<<"PARITY "<<label<<" elements="<<a.numel()<<" max_abs="<<maximum
             <<" atol="<<atol<<" rtol="<<rtol<<'\n';
}
String2TensorStorage tiny_weights(bool curves = false) {
    String2TensorStorage weights;
    auto add = [&](const std::string& suffix, std::initializer_list<int64_t> shape) {
        TensorStorage tensor;
        tensor.name = "model.diffusion_model." + suffix;
        tensor.n_dims = static_cast<int>(shape.size());
        int axis = 0;
        for (auto extent : shape) tensor.ne[axis++] = extent;
        weights.emplace(tensor.name, tensor);
    };
    add("video_patch_proj.weight", {4, 8});
    add("audio_patch_proj.weight", {1, 8});
    add("condition_proj.weight", {8, 8});
    if (curves) add("adaln_t_table", {4, 5});
    else {
        add("time_embedder.proj_in.weight", {4, 8});
        add("time_embedder.proj_out.weight", {8, 4});
    }
    add("rope.inv_freq", {1});
    for (int i = 0; i < 4; ++i) {
        const auto prefix = "blocks." + std::to_string(i) + ".";
        add(prefix + "attn.q_norm.weight", {128});
        add(prefix + "attn.qkv_proj.weight", {8, 384});
        add(prefix + "mlp.fc1.weight", {8, 32});
    }
    return weights;
}
struct Inputs {
    sd::Tensor<float> packed{{2, 2, 1, 2}, {.2f, -.3f, .1f, .4f, .5f, -.2f, 0.f, 0.f}};
    sd::Tensor<float> context{{8, 3}};
    sd::Tensor<float> timestep{{1}, {800.f}};
    sd::Tensor<int32_t> tags{{3}, {1, 2, 0}};
    sd::Tensor<int32_t> keyframes{{1}, {0}};
    std::vector<sd::Tensor<float>> videos, audios;
    std::vector<MiniMaxH3ReferenceBlock> references;
    Inputs() {
        for (int64_t i = 0; i < context.numel(); ++i) context[i] = .1f + .01f * static_cast<float>(i);
    }
    DiffusionParams params(bool with_refs = false) {
        DiffusionParams p;
        p.x = &packed; p.context = &context; p.timesteps = &timestep;
        p.minimax_audio_length = 1;
        p.minimax_video_sigma_shift = 12.f; p.minimax_audio_sigma_shift = 3.f;
        if (with_refs) {
            p.ref_latents = &videos; p.minimax_reference_audio_latents = &audios;
            p.minimax_reference_blocks = &references;
            p.minimax_text_token_tags = &tags; p.minimax_keyframe_indices = &keyframes;
        }
        return p;
    }
    void add_references() {
        videos.emplace_back(std::vector<int64_t>{2, 2, 1, 1, 1}, std::vector<float>{.3f, .1f, -.2f, .4f});
        videos.emplace_back(std::vector<int64_t>{2, 2, 1, 1, 1}, std::vector<float>{-.1f, .2f, .3f, .1f});
        audios.emplace_back(std::vector<int64_t>{1, 2, 1, 1}, std::vector<float>{.2f, -.1f});
        references = {{MiniMaxH3ReferenceKind::VIDEO_AUDIO, 0, 0}, {MiniMaxH3ReferenceKind::IMAGE, 1, -1}};
    }
};

const std::vector<float> schedule{.8f,.5f,.125f,0.f};
struct Harness : MiniMaxH3::MiniMaxH3Runner {
    struct Saved { ggml_tensor* t; ggml_backend_buffer_t buffer; void* data; void* extra; std::vector<float> values; };
    std::vector<Saved> saved;
    std::unordered_set<const ggml_tensor*> projection_weights;
    std::string weights_identity;
    size_t prefill_executions=0, fail_prefill_at=0, all_parameter_fallbacks=0;
    bool gpu;
    explicit Harness(ggml_backend_t backend,bool curves,bool cuda)
        : MiniMaxH3Runner(backend,tiny_weights(curves),"model.diffusion_model",cuda),gpu(cuda) {
        require(ggml_backend_is_cuda(runtime_backend)==gpu,"runtime mismatch");
        require(!gpu || params_backend!=runtime_backend,"offload must really keep parameters on CPU");
        require(alloc_params_buffer(),"parameter allocation");
        std::map<std::string,ggml_tensor*> named;get_param_tensors(named,"model.diffusion_model");
        MiniMaxH3::Sha256 hash;
        for(const auto& item:named) {
            auto* t=item.second;const auto& name=item.first;
            require(t->type==GGML_TYPE_F32 && t->buffer && ggml_backend_buffer_is_host(t->buffer),"CPU F32 source parameter required");
            uint32_t h=2166136261u;for(unsigned char c:name)h=(h^c)*16777619u;
            std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
            for(size_t i=0;i<v.size();++i)v[i]=name.find("norm")!=std::string::npos?1.f:
                name.find("rope.inv_freq")!=std::string::npos?.1f:.02f+.005f*std::sin(float(i)+float(h%101u)*.01f);
            ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));
            hash.update(name.data(),name.size());hash.update(t->ne,sizeof(t->ne));hash.update(v.data(),v.size()*sizeof(float));
            saved.push_back({t,t->buffer,t->data,t->extra,std::move(v)});
            if(name.find("adaln_proj.linear.weight")!=std::string::npos)projection_weights.insert(t);
        }
        size_t count=0;for(auto* t=ggml_get_first_tensor(params_ctx);t;t=ggml_get_next_tensor(params_ctx,t))++count;
        require(count==saved.size() && count>40 && projection_weights.size()==5,"fixture omitted real model weights");
        require(config.hidden_size==8 && config.num_layers==4 && config.attention_head_dim==128,"tiny architecture mismatch");
        weights_identity=hash.finish();
        // Dense native attention, no research attention/fusion toggles. Backend
        // native GEMMs may use their normal CUDA implementation; no CPU fallback.
        set_flash_attention_enabled(false);
        if(gpu) {
            MiniMaxH3::CudaOptimizationOptions o;o.adaln_cache=true;
            bool rejected=false;
            try { configure_cuda_optimizations(o); }
            catch(const std::invalid_argument& e) { rejected=std::string(e.what()).find("resident immutable")!=std::string::npos; }
            require(rejected && !model.cuda_optimizations.any_enabled(),"production offload guard changed");
        }
    }
    ~Harness() { cleanup(); }
    void cleanup() {
        if(runtime_backend)ggml_backend_synchronize(runtime_backend);
        model.adaln_cache=nullptr;model.adaln_graph_inputs=nullptr;model.adaln_token={};
        exact_release_projection_graph();reset_graph_cut_run_cache();adaln_inputs.reset();
        model.cuda_optimizations={};
    }
    void restored() {
        require(!params_on_runtime_backend && partial_offload_pairs.empty() && !partial_runtime_params_buffer,
                "parameter staging survived segment");
        for(const auto& s:saved) {
            require(s.t->buffer==s.buffer && s.t->data==s.data && s.t->extra==s.extra,"CPU weight pointers not restored");
            require(std::memcmp(s.t->data,s.values.data(),s.values.size()*sizeof(float))==0,"source weights mutated");
        }
    }
    void audit(ggml_cgraph* graph) {
        ++audited_graphs;
        auto storage=[&](ggml_tensor* t) {
            require(t && t->data,"unallocated tensor in executing graph");
            auto* buffer=Cut::tensor_buffer(t);require(buffer,"missing backend buffer");
            if(gpu) {
                auto* type=ggml_backend_buffer_get_type(buffer);
                require(!ggml_backend_buffer_is_host(buffer) && type==ggml_backend_cuda_buffer_type(0) &&
                        ggml_backend_buft_get_device(type)==ggml_backend_get_device(runtime_backend),
                        std::string("host/split/foreign-device tensor: ")+t->name);
            } else require(ggml_backend_buffer_is_host(buffer),"CPU reference has device data");
        };
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* t=ggml_graph_node(graph,i);++audited_nodes;
            require(ggml_backend_supports_op(runtime_backend,t),std::string(gpu?"CUDA":"CPU")+" unsupported "+ggml_op_name(t->op)+" / "+t->name);
            storage(t);for(auto* s:t->src)if(s)storage(s);
        }
        for(int i=0;i<Cut::leaf_count(graph);++i)storage(Cut::leaf_tensor(graph,i));
    }
    bool exact_execute_graph(ggml_cgraph* graph,int threads,const std::vector<ggml_tensor*>& params,
                             std::function<bool(ggml_cgraph*)> post) {
        ++prefill_executions;
        size_t projections=0;for(auto* p:params)projections+=projection_weights.count(p);
        require(projections==1,"prefill segment must stage exactly one real H3 projection");
        auto result=execute_graph<float>(graph,threads,true,params,false,true,nullptr,
            [&](auto* g) {
                const bool ok=post(g);
                if(prefill_executions==fail_prefill_at)throw Failure("injected prefill failure after readback",1);
                return ok;
            },[&](auto* g) {
                audit(g);
                if(gpu)for(const auto& s:saved) {
                    const bool selected=std::find(params.begin(),params.end(),s.t)!=params.end();
                    require((s.t->data!=s.data)==selected && (s.t->buffer!=s.buffer)==selected,
                            "prefill transferred wrong parameter set");
                }
                return true;
            });
        return result.has_value();
    }
    EA::RequestContract contract(const DiffusionParams& p,bool conditional,uint64_t generation) {
        auto split=split_av_latents(*p.x,p.minimax_audio_length);
        static const std::vector<sd::Tensor<float>> empty;
        static const std::vector<MiniMaxH3ReferenceBlock> refs;
        static const sd::Tensor<int32_t> ints;
        EA::RequestContract c;c.full_weights_identity=weights_identity;c.load_generation=generation;c.adapter_identity="none";
        c.cfg_identity=cuda_cache_identity(p)+(conditional?":conditional":":unconditional");
        c.math_identity=env("CUDA_VISIBLE_DEVICES")+":tiny-native-F32:serial-cuts:v1";
        c.schedule_identity="tiny-fixed-flow:v1";c.video_shift=p.minimax_video_sigma_shift;c.audio_shift=p.minimax_audio_sigma_shift;
        c.timestep_input_dim=config.timestep_input_dim;c.time_embed_hidden=config.time_embed_hidden_size;
        c.time_embed_dim=config.time_embed_dim;c.curve_grid=config.adaln_curve_grid;c.sigmas=schedule;
        c.projections=EA::h3_projections(config.hidden_size,config.num_layers);
        for(size_t i=0;i+1<c.sigmas.size();++i) {
            c.sigmas[i]=std::clamp((c.sigmas[i]*1000.f)/1000.f,1e-6f,1.f);
            auto layout=MiniMaxH3::build_layout(p.context->shape()[1],split.first.shape()[2],split.first.shape()[1],split.first.shape()[0],
                p.minimax_audio_length,p.ref_latents?*p.ref_latents:empty,p.minimax_reference_audio_latents?*p.minimax_reference_audio_latents:empty,
                p.minimax_keyframe_indices?*p.minimax_keyframe_indices:ints,p.minimax_reference_blocks?*p.minimax_reference_blocks:refs,
                p.minimax_text_token_tags?*p.minimax_text_token_tags:ints,1.f-c.sigmas[i],
                1.f-MiniMaxH3::time_shift_sigma(c.sigmas[i],c.video_shift,c.audio_shift));
            c.plans.push_back(EA::plan_from_layout(i,c.sigmas[i],conditional?EA::CfgRole::Conditional:EA::CfgRole::Unconditional,layout));
        }
        return c;
    }
    void prefill(EA::Cache& cache,const EA::RequestContract& c) {
        expect(cache.begin_request(c),EA::Status::Stored,"begin prefill");
        expect(EA::prefill_request_segmented(*this,cache,c,1,[&](size_t i,size_t j,auto t) {
            restored();return build_adaln_projection_graph(cache,t,c.plans[i],j,1);
        }),EA::Status::Stored,"segmented prefill");restored();
    }
    ggml_cgraph* denoiser_graph(const DiffusionParams& p) {
        static const std::vector<sd::Tensor<float>> empty;
        static const std::vector<MiniMaxH3ReferenceBlock> refs;
        static const sd::Tensor<int32_t> ints;
        return build_graph(*p.x,*p.timesteps,*p.context,p.ref_latents?*p.ref_latents:empty,
            p.minimax_reference_audio_latents?*p.minimax_reference_audio_latents:empty,
            p.minimax_text_token_tags?*p.minimax_text_token_tags:ints,p.minimax_keyframe_indices?*p.minimax_keyframe_indices:ints,
            p.minimax_reference_blocks?*p.minimax_reference_blocks:refs,p.minimax_audio_length,p.minimax_video_sigma_shift,p.minimax_audio_sigma_shift);
    }
    // This is a serial test traversal of the real graph-cut APIs. Unlike the
    // production traversal, it adds pre/post hooks to audit EVERY executing
    // segment and inject an in-flight failure. No fake cuts/no backend scheduler.
    sd::Tensor<float> run_cuts(ggml_cgraph* graph,int fail_segment=-1) {
        std::unordered_set<const ggml_tensor*> params;for(const auto& s:saved)params.insert(s.t);
        // Reject unsupported graphs before the planner measures device buffers.
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* node=ggml_graph_node(graph,i);
            require(ggml_backend_supports_op(runtime_backend,node),
                    std::string("unsupported graph before planning: ")+ggml_op_name(node->op)+" / "+node->name);
        }
        const auto plan=Cut::build_plan(runtime_backend,graph,params,"h3-offload-tiny");
        require(plan.available && plan.valid && plan.has_cuts && plan.segments.size()>1,"real multi-segment plan required");
        free_compute_buffer();reset_graph_cut_run_cache();
        sd::Tensor<float> result;
        for(size_t i=0;i<plan.segments.size();++i) {
            const auto& segment=plan.segments[i];require(segment.comm_ops.empty(),"single-device test only");
            reset_segment_runtime_tensors(segment,graph);
            require(bind_segment_cached_inputs(graph,segment),"bind previous graph-cut output");
            const bool last=i+1==plan.segments.size();
            if(!last)for(size_t j=0;j<segment.output_node_indices.size();++j) {
                auto* t=Cut::output_tensor(graph,segment,j);
                if(t && Cut::is_graph_cut_tensor(t) && segment.future_input_names.count(t->name))cache(t->name,t);
            }
            ggml_context* segment_ctx=nullptr;
            auto* g=Cut::build_segment_graph(graph,segment,&segment_ctx);
            std::unique_ptr<ggml_context,decltype(&ggml_free)> owner(segment_ctx,ggml_free);
            require(g!=nullptr,"real segment build failed");
            auto selected=Cut::runtime_param_tensors(graph,segment,"h3-offload-tiny");
            // Retain the real executor's empty-list behavior and report it. Tiny
            // tests do NOT claim a fix for its whole-model fallback or VRAM cap.
            if(gpu && selected.empty())++all_parameter_fallbacks;
            auto out=execute_graph<float>(g,1,true,selected,true,!last,&segment.future_input_names,
                [&](auto* executing) {
                    audit(executing);
                    if(int(i)==fail_segment)throw Failure("injected consumer segment failure",1);
                    return true;
                },[&](auto* executing){audit(executing);return true;});
            require(out.has_value(),"CUDA/CPU segment execution failed (no fallback)");
            result=std::move(*out);++cut_segments;restored();
        }
        return result;
    }
    sd::Tensor<float> evaluate(const DiffusionParams& p,EA::Cache* cached,const EA::RequestContract* c,size_t step,
                               bool reset_after_bind=false,int fail_segment=-1) {
        cleanup();struct Guard{Harness& r;~Guard(){r.cleanup();}}guard{*this};
        if(cached) {
            expect(cached->bind_step(*c,step,c->plans[step],&model.adaln_token),EA::Status::Hit,"bind exact consumer step");
            require(cached->complete(model.adaln_token),"incomplete cache");
            // Explicit test-only injection into the real graph builder; NEVER
            // invoke configure/compute with production offload guards bypassed.
            model.cuda_optimizations.adaln_cache=true;model.adaln_cache=cached;
        }
        ggml_cgraph* graph=nullptr;require(exact_prepare_graph([&]{return denoiser_graph(p);},&graph),"denoiser build");
        size_t gems=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* t=ggml_graph_node(graph,i);if(t->op==GGML_OP_MUL_MAT)gems+=projection_weights.count(t->src[0]);
        }
        require(gems==(cached?0u:5u),"uncached/cached graph did not contain/remove actual projection GEMMs");
        if(reset_after_bind)cached->reset(); // imports own their independent storage
        auto out=run_cuts(graph,fail_segment);restored();return out;
    }
};

void matrix(ggml_backend_t cuda,ggml_backend_t cpu,bool curves,bool mixed) {
    Harness candidate(cuda,curves,true),baseline(cuda,curves,true),reference(cpu,curves,false);
    require(candidate.weights_identity==baseline.weights_identity && candidate.weights_identity==reference.weights_identity,"control weights differ");
    Inputs inputs;if(mixed)inputs.add_references();auto p=inputs.params(mixed);
    // Distinct conditioning values for branches; no CFG combine/pipeline claim.
    auto conditional_context=inputs.context;
    std::array<std::unique_ptr<EA::Cache>,2> caches;
    std::array<EA::RequestContract,2> contracts;
    for(int branch=0;branch<2;++branch) {
        inputs.context=conditional_context;if(!branch)for(int64_t j=0;j<inputs.context.numel();++j)inputs.context[j]*=-.5f;
        contracts[branch]=candidate.contract(p,branch!=0,1);
        caches[branch]=std::make_unique<EA::Cache>(EA::Options{true,8u<<20,1024});
        candidate.prefill(*caches[branch],contracts[branch]);
        const auto executions=candidate.prefill_executions;
        const auto bytes=caches[branch]->used_bytes();
        expect(EA::prefill_request_segmented(candidate,*caches[branch],contracts[branch],1,
            [](size_t,size_t,EA::Cache::StepToken)->ggml_cgraph* {
                throw Failure("repeat prefill unexpectedly recomputed",1);
            }),EA::Status::Stored,"repeat prefill must be a storage-only hit");
        require(candidate.prefill_executions==executions && caches[branch]->used_bytes()==bytes,
                "repeat prefill performed execution/allocation");
    }
    // Wrong CFG contract must miss, without changing either branch's payload.
    EA::Cache::StepToken wrong;
    expect(caches[0]->bind_step(contracts[1],0,contracts[1].plans[0],&wrong),EA::Status::Miss,"branch identity isolation");
    for(int repeat=0;repeat<2;++repeat)for(int branch=0;branch<2;++branch) {
        inputs.context=conditional_context;if(!branch)for(int64_t j=0;j<inputs.context.numel();++j)inputs.context[j]*=-.5f;
        for(size_t step=0;step+1<schedule.size();++step) {
            inputs.timestep[0]=schedule[step]*1000.f;
            const auto label=std::string(curves?"curve":"embed")+"/"+(mixed?"mixed-keyframe":"t2va")+
                "/branch"+std::to_string(branch)+"/sigma"+std::to_string(step)+"/repeat"+std::to_string(repeat);
            const auto direct=baseline.evaluate(p,nullptr,nullptr,step);
            const auto host=reference.evaluate(p,nullptr,nullptr,step);
            const auto hit=candidate.evaluate(p,caches[branch].get(),&contracts[branch],step);
            compare(direct,hit,label+"/CUDA-uncached-vs-cached",2e-5f,2e-5f);
            compare(host,direct,label+"/CPU-vs-CUDA",2e-4f,2e-4f);
        }
    }
    inputs.context=conditional_context;inputs.timestep[0]=schedule[0]*1000.f;
    const auto direct=baseline.evaluate(p,nullptr,nullptr,0);
    const auto reset=candidate.evaluate(p,caches[1].get(),&contracts[1],0,true);
    compare(direct,reset,"cache-reset-after-import",2e-5f,2e-5f);
    require(caches[1]->used_bytes()==0,"reset/import storage leak");
    // Late prefill exception after real GPU readback: earlier slots discarded,
    // CUDA synchronized, CPU source pointers restored, next request succeeds.
    candidate.fail_prefill_at=candidate.prefill_executions+3;bool threw=false;
    try { candidate.prefill(*caches[1],contracts[1]); }
    catch(const Failure& e){threw=std::string(e.what())=="injected prefill failure after readback";}
    require(threw && caches[1]->used_bytes()==0,"injected prefill did not roll back");candidate.restored();
    candidate.fail_prefill_at=0;contracts[1]=candidate.contract(p,true,2);candidate.prefill(*caches[1],contracts[1]);
    threw=false;
    try { candidate.evaluate(p,caches[1].get(),&contracts[1],0,false,1); }
    catch(const Failure& e){threw=std::string(e.what())=="injected consumer segment failure";}
    require(threw,"consumer segment failure not exercised");candidate.restored();
    const auto recovered=candidate.evaluate(p,caches[1].get(),&contracts[1],0);
    compare(direct,recovered,"same-runner-recovery",2e-5f,2e-5f);
    std::cout<<"MATRIX PASS curve="<<curves<<" mixed_keyframe="<<mixed<<" prefill_graphs="<<candidate.prefill_executions
             <<" whole-param-empty-segment-fallbacks="<<candidate.all_parameter_fallbacks+baseline.all_parameter_fallbacks<<'\n';
}
} // namespace
int main(int argc,char** argv) {
    try {
        require_opt_in(argc,argv); // BEFORE backend registration/device APIs
        require(env("ED_ASYNC_OFFLOAD").empty() || env("ED_ASYNC_OFFLOAD")=="0","serial-only harness: ED_ASYNC_OFFLOAD must be unset/0",2);
        require(env("ED_MINIMAX_H3_REUSE_GRAPH").empty() || env("ED_MINIMAX_H3_REUSE_GRAPH")=="0","graph reuse must be off",2);
        require(std::getenv("ED_MINIMAX_H3_DEBUG_TARGET")==nullptr,"debug target must be unset",2);
        for(const char* key:{"ED_H3_FUSED_OPS","ED_H3_CACHE_DIT","ED_H3_ADALN_CACHE","ED_H3_SAGE_ATTN","ED_H3_SPARSE_ATTN","ED_H3_TURBO","ED_H3_VDN","ED_H3_VSA"})
            require(env(key).empty() || env(key)=="0",std::string("external research flag must be off: ")+key,2);
        ggml_backend_register(h3_offload_cpu_reg());
        auto reg=ggml_backend_cuda_reg();require(reg && std::string(ggml_backend_reg_name(reg))=="CUDA","CUDA backend missing",3);
        require(ggml_backend_reg_dev_count(reg)==1 && ggml_backend_cuda_get_device_count()==1,"exactly one actual CUDA device required",3);
        ggml_backend_register(reg);
        Backend cuda(ggml_backend_cuda_init(0),ggml_backend_free),cpu(h3_offload_cpu_init(),ggml_backend_free);
        require(cuda && cpu && ggml_backend_is_cuda(cuda.get()),"backend initialization failed",3);
        require(ggml_backend_get_device(cuda.get())==ggml_backend_reg_dev_get(reg,0),"CUDA identity mismatch",3);
        require(ggml_backend_dev_type(ggml_backend_get_device(cuda.get()))==GGML_BACKEND_DEVICE_TYPE_GPU,"not GPU",3);
        std::cout<<"TEST-ONLY helper/serial-graph-cut validation; production offload STILL REJECTED; device="<<env("CUDA_VISIBLE_DEVICES")<<'\n';
        for(bool curves:{false,true})for(bool mixed:{false,true})matrix(cuda.get(),cpu.get(),curves,mixed);
        require(comparisons==104 && cut_segments>0 && audited_graphs>0 && audited_nodes>0,"incomplete matrix");
        std::cout<<"PASS comparisons="<<comparisons<<" cut_segments="<<cut_segments<<" audited_graphs="<<audited_graphs<<" audited_nodes="<<audited_nodes<<'\n';
        return 0;
    } catch(const Failure& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return e.code;}
      catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
