// REAL 535-tensor mixed-Q4 H3, one host load / one production runner.
// Small native T2VA geometry: diagnostic only, NOT generation/performance proof.
// Never run automatically; use the existing guarded GPU wrapper after scheduling.
#define H3_ADALN_WEIGHT_PROBE_NO_MAIN
#include "h3_cuda_adaln_weight_probe.cpp"

namespace {
bool full_shape() { return env("H3_ADALN_PROBE_FULL_SHAPE") == "1"; }
size_t segment_budget() { return size_t(full_shape() ? 22 : 1) << 30; }
const std::vector<float> denoiser_sigmas{1.f, .6f, .2f, 0.f};
struct DenoiserProbe : Probe {
    using Probe::Probe;
    void clear_pass() {
        reset_cuda_optimization_state();
        invalidate_persistent_graph();
        reset_graph_cut_run_cache();
        exact_release_projection_graph();
        require(!model.adaln_cache && !model.adaln_graph_inputs && !adaln_inputs &&
                !adaln_branches[0].cache && !adaln_branches[1].cache && adaln_sigmas.empty(),
                "request state not reset");
        restored();
    }
    void configure_pass(bool cached) {
        clear_pass();
        set_max_graph_vram_bytes(segment_budget());
        set_flash_attention_enabled(true);
        require(serial_cuda_offload_mode() && can_attempt_graph_cut_segmented_compute(),
                "production single-GPU serial graph-cut mode required", 3);
        size_t bytes = 0; for (const auto& p : named) bytes += ggml_nbytes(p.second);
        require(bytes > (size_t(4) << 30), "full native model weight set required");
        MiniMaxH3::CudaOptimizationOptions o; o.adaln_cache = cached;
        configure_cuda_optimizations(o);
        require(model.cuda_optimizations.adaln_cache == cached &&
                !model.cuda_optimizations.fused_ops && !model.cuda_optimizations.cache_dit &&
                !model.cuda_optimizations.sage_attn && !model.cuda_optimizations.sparse_attn &&
                !model.cuda_optimizations.turbo && !model.cuda_optimizations.vdn && !model.cuda_optimizations.vsa,
                "unexpected optimization flag");
        prepare_cuda_schedule(denoiser_sigmas, "denoiser-probe-fixed3-start1-roundtrip1000:v2", MODEL_SHA);
        require(cached ? adaln_sigmas.size() == 4 : adaln_sigmas.empty(), "schedule configuration mismatch");
        std::cout << "PASS_SETUP adaln=" << cached << " runner=" << static_cast<const void*>(this)
                  << " native_host_bytes=" << bytes << " segment_budget_bytes=" << segment_budget()
                  << " flash_attention=1 async=0 reuse=0 synthetic_weights=0\n";
    }
    void verify_cache(bool cached) {
        if (!cached) {
            require(!adaln_branches[0].cache && !adaln_branches[1].cache && !model.adaln_cache,
                    "baseline unexpectedly populated AdaLN cache");
            return;
        }
        for (size_t branch = 0; branch < 2; ++branch) {
            const auto& state = adaln_branches[branch];
            require(state.cache && state.contract.plans.size() == 3 && state.contract.projections.size() == 51,
                    "production did not populate both CFG caches");
            require(state.cache->statistics().stores == 153, "incomplete/rebuilt production CFG cache");
            for (size_t i = 0; i < 3; ++i) {
                EA::Cache::StepToken token;
                status(state.cache->bind_step(state.contract, i, state.contract.plans[i], &token), EA::Status::Hit,
                       "production cache bind");
                require(state.cache->complete(token), "production cache incomplete");
            }
            std::cout << "PRODUCTION_CACHE branch=" << branch << " stores=" << state.cache->statistics().stores
                      << " complete_steps=3 projections=51\n";
        }
    }
};
struct NativeInputs {
    // Layout: 96 video elements (2*2*1*24) followed by 128 audio elements
    // (length2 * stereo2 * channels32). Divisor=4; no padding is needed.
    sd::Tensor<float> x;
    int audio_length = full_shape() ? 207 : 2;
    int width = full_shape() ? 54 : 2, height = full_shape() ? 30 : 2, frames = full_shape() ? 37 : 1;
    size_t video_elements = 0, audio_elements = 0;
    std::array<sd::Tensor<float>,2> contexts;
    NativeInputs(const MiniMaxH3::Config& config) {
        require(config.video_latent_channels == 24 && config.audio_latent_channels == 32 && config.text_dim == 5120,
                "unexpected native channels/text dimension");
        const int64_t spatial = int64_t(width) * height * frames;
        video_elements = size_t(spatial) * 24; audio_elements = size_t(audio_length) * 2 * 32;
        x = sd::Tensor<float>({width,height,frames,24 + (int64_t(audio_elements)+spatial-1)/spatial});
        for (int64_t i = 0; i < x.numel(); ++i)
            x[i] = .125f * std::sin(float(i+1)*.37f) + .025f * std::cos(float(i+3)*.11f);
        for (int role = 0; role < 2; ++role) {
            contexts[role] = sd::Tensor<float>({5120,full_shape()?64:4,1});
            for (int64_t i = 0; i < contexts[role].numel(); ++i)
                contexts[role][i] = .04f * std::sin(float((i+17*role)%1021)*.017f) + (role ? .02f : -.03f);
        }
    }
};
size_t denoiser_pairs = 0, denoiser_different = 0;
void metrics(const sd::Tensor<float>& a, const sd::Tensor<float>& b, size_t begin, size_t count,
             int step, int role, const char* part) {
    require(!a.empty() && a.shape() == b.shape() && begin+count <= size_t(a.numel()) && count > 0,
            "empty/mismatched/truncated denoiser output");
    double maxabs=0, err2=0, ref2=0; size_t bitdiff=0, worst=begin;
    for (size_t i = begin; i < begin+count; ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite denoiser output");
        const double d=double(a[i])-double(b[i]);
        if (std::abs(d)>maxabs) { maxabs=std::abs(d); worst=i; }
        err2+=d*d; ref2+=double(a[i])*double(a[i]);
        bitdiff+=EA::float_bits(a[i])!=EA::float_bits(b[i]);
    }
    const double rel=ref2>0 ? std::sqrt(err2/ref2) : (err2==0 ? 0 : std::numeric_limits<double>::infinity());
    std::cout << "DENOISER_COMPARE step=" << step << " conditional=" << role << " part=" << part
              << " sigma=" << (denoiser_sigmas[step]*1000.f)/1000.f << " elements=" << count
              << " maxabs=" << maxabs << " relL2=" << rel << " rmse=" << std::sqrt(err2/count)
              << " bitdiff=" << bitdiff << " worst_index=" << worst
              << " off_at_worst=" << a[worst] << " on_at_worst=" << b[worst] << " exact=" << (bitdiff==0) << '\n';
    if (std::string(part)=="packed") { ++denoiser_pairs; denoiser_different+=bitdiff!=0; }
}
void run_denoiser_matrix(DenoiserProbe& runner) {
    NativeInputs fixed(runner.config);
    // Validate packing using the actual native split, not a guessed tensor cast.
    const auto split = runner.split_av_latents(fixed.x, fixed.audio_length);
    require(split.first.shape()==std::vector<int64_t>({fixed.width,fixed.height,fixed.frames,24,1}) &&
            split.second.shape()==std::vector<int64_t>({fixed.audio_length,2,32,1}) &&
            size_t(split.first.numel())==fixed.video_elements && size_t(split.second.numel())==fixed.audio_elements,
            "invalid native packed input");
    std::array<sd::Tensor<float>,6> baseline;
    for (bool cached : {false, true}) {
        runner.configure_pass(cached);
        for (int step=0; step<3; ++step) for (int role=0; role<2; ++role) {
            // Every forward starts from the SAME pristine latent for both paths,
            // not the previous output or a solver update. CFG roles use distinct,
            // deterministic conditioning, identical between off and on passes.
            auto x=fixed.x; auto context=fixed.contexts[role];
            sd::Tensor<float> timestep({1}, {denoiser_sigmas[step]*1000.f});
            const auto xhash=digest(x.data(),size_t(x.numel())*sizeof(float));
            const auto chash=digest(context.data(),size_t(context.numel())*sizeof(float));
            DiffusionParams p; p.x=&x; p.context=&context; p.timesteps=&timestep;
            p.minimax_audio_length=fixed.audio_length; p.minimax_video_sigma_shift=12.f; p.minimax_audio_sigma_shift=3.f;
            runner.begin_cuda_step(step,3,role!=0);
            std::cout << "DENOISER_RUN adaln=" << cached << " step=" << step << " conditional=" << role
                      << " timestep=" << timestep[0] << " x_sha256=" << xhash << " context_sha256=" << chash
                      << " full_shape=" << full_shape() << " packed_elements=" << x.numel() << " audio_length=" << fixed.audio_length << '\n';
            // Actual production dispatch, including configure/schedule/branch
            // prefill and real graph-cut compute. No custom segment executor,
            // no model.adaln_cache injection, no altered projection builder.
            auto output=runner.MiniMaxH3::MiniMaxH3Runner::compute(1,p);
            ggml_backend_synchronize(runner.exact_runtime_backend());
            require(!output.empty() && output.shape()==fixed.x.shape(), "production output shape mismatch");
            for (int64_t i=0; i<output.numel(); ++i) require(std::isfinite(output[i]), "nonfinite production output");
            require(digest(x.data(),size_t(x.numel())*sizeof(float))==xhash &&
                    digest(context.data(),size_t(context.numel())*sizeof(float))==chash &&
                    timestep[0]==denoiser_sigmas[step]*1000.f, "production mutated its input");
            runner.restored();
            const size_t index=size_t(step*2+role);
            if (!cached) baseline[index]=std::move(output);
            else {
                metrics(baseline[index],output,0,size_t(output.numel()),step,role,"packed");
                metrics(baseline[index],output,0,fixed.video_elements,step,role,"video");
                metrics(baseline[index],output,fixed.video_elements,fixed.audio_elements,step,role,"audio");
            }
        }
        runner.verify_cache(cached);
    }
    runner.clear_pass();
    runner.restored(true); // source projection/time bytes unchanged after both passes
    require(denoiser_pairs==6, "incomplete production denoiser matrix");
}
}
int main(int argc,char** argv) {
    try {
        if (argc==2 && std::string(argv[1])=="--help") {
            std::cout << "Real locked Q4 H3 production denoiser probe; DEFAULT OFF.\n"
                         "Use H3_ADALN_PROBE_MODEL and test_h3_gpu_parity.sh --run --device GPU-... --binary ABSOLUTE_PATH.\n"
                         "One full 535-tensor host load, one runner, off then on; 3 sigma x 2 CFG roles = 6 pairs / 12 forwards.\n"
                         "Native T2VA latent 2x2x1x24, audio 2x2x32, text 5120x4; no TE/VAE or solver update.\n"
                         "Fixed 1-GiB graph planning budget, serial CUDA offload, flash on, only AdaLN toggled.\n"
                         "Exit 0=all packed outputs bit-exact; 4=complete with differences; 1=failure; 2=not opted in; 3=missing prerequisite.\n"
                         "No numerical tolerance relaxation; NOT an official performance/quality test.\n";
            return 0;
        }
        opt_in(argc,argv);
        const auto path=env("H3_ADALN_PROBE_MODEL");
        require(!path.empty() && std::filesystem::path(path).is_absolute(), "H3_ADALN_PROBE_MODEL absolute path required",3);
        FileIdentity file(path);
        for (const char* key : {"ED_ASYNC_OFFLOAD", "ED_MINIMAX_H3_REUSE_GRAPH"})
            require(env(key).empty() || env(key)=="0",std::string("disable conflicting option: ")+key,2);
        require(std::getenv("ED_MINIMAX_H3_DEBUG_TARGET")==nullptr,"debug target must be unset for production entry",2);
        for (const auto& kv : {std::pair<const char*,const char*>{"GGML_CUDA_DISABLE_GRAPHS","1"},
                              {"ED_CUDA_GRAPHS","0"},{"ED_ASYNC_OFFLOAD","0"},{"ED_MINIMAX_H3_REUSE_GRAPH","0"},
                              {"ED_SAGE_ATTN","0"},{"ED_PROFILE_GRAPH_CUTS","1"}})
            require(setenv(kv.first,kv.second,1)==0,"cannot set diagnostic process environment");
        ggml_backend_register(h3_probe_cpu_reg()); auto* reg=ggml_backend_cuda_reg();
        require(reg && ggml_backend_reg_dev_count(reg)==1 && ggml_backend_cuda_get_device_count()==1,
                "exactly one physical CUDA device required",3);
        ggml_backend_register(reg); Backend gpu(ggml_backend_cuda_init(0),ggml_backend_free);
        require(gpu && ggml_backend_is_cuda(gpu.get()),"no CUDA backend",3);
        std::cout << std::setprecision(17) << std::unitbuf;
        std::cout << "DENOISER_PROBE device=" << env("CUDA_VISIBLE_DEVICES") << " model=" << std::quoted(path)
                  << " expected_sha256=" << MODEL_SHA << " criterion=bit_exact threshold=0 full_host_loads=1\n";
        require(MiniMaxH3::sha256_file(path)==MODEL_SHA,"full model SHA256 mismatch",3);
        file.unchanged(path); std::cout << "FULL_SHA256_VERIFIED " << MODEL_SHA << '\n';
        ModelLoader loader;
        require(loader.init_from_file(path,std::string(PREFIX)+"."),"native ModelLoader init failed",3);
        require(loader.get_ld_version()==VERSION_MINIMAX_H3,"not native H3",3);
        loader.convert_tensors_name();
        std::map<ggml_type,size_t> types;
        for (const auto& item:loader.get_tensor_storage_map()) ++types[item.second.type];
        require(types==std::map<ggml_type,size_t>{{GGML_TYPE_F32,13},{GGML_TYPE_BF16,264},{GGML_TYPE_Q4_0,50},{GGML_TYPE_Q4_K,208}},
                "locked storage types mismatch");
        DenoiserProbe runner(gpu.get(),loader); file.unchanged(path);
        run_denoiser_matrix(runner); file.unchanged(path);
        std::cout << "DENOISER_SUMMARY completed=1 pairs=" << denoiser_pairs << " differing_pairs=" << denoiser_different
                  << " forwards=12 cfg_roles=2 sigma_count=3 production_compute=1 full_model_weights=1"
                  << " source_projection_time_bytes_unchanged=1 performance_test=0 full_resolution_test=" << full_shape() << " result="
                  << (denoiser_different ? "DIFFERENCES" : "EXACT") << '\n';
        return denoiser_different ? 4 : 0;
    } catch(const Failure& e) { std::cerr << "FAIL completed=0: " << e.what() << '\n'; return e.code; }
      catch(const std::exception& e) { std::cerr << "FAIL completed=0: " << e.what() << '\n'; return 1; }
}
