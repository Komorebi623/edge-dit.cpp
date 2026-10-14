// Opt-in REAL locked FL2VA checkpoint probe. Build only by default; no CTest.
// Compare the native all-projection/per-sigma graph with the native production
// layer-major helper. No denoiser/attention/VAEs, synthetic weights or dtype overrides.
#include "dit_models/diffusion_model.hpp"
#include "dit_models/models/minimax_h3_full.hpp"
#include "runtime/model_loader.h"
#include "ggml-cuda.h"
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sys/stat.h>

#if defined(GGML_BACKEND_DL) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#error "Requires directly linked NVIDIA CUDA and CPU backends"
#endif
static_assert(GGML_MAX_NAME == 128, "candidate ggml ABI mismatch");
extern "C" ggml_backend_reg_t h3_probe_cpu_reg() __asm__("ggml_backend_cpu_reg");
namespace {
namespace EA = MiniMaxH3::ExactAdaLN;
constexpr const char* MODEL_SHA = "a25ee3184dd56e7fc5c8db1508d2e48a7e3c92919b516935c6059e4228e16107";
constexpr uint64_t MODEL_BYTES = 18779848448ULL;
constexpr const char* PREFIX = "model.diffusion_model";
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
struct Failure : std::runtime_error {
    int code;
    Failure(const std::string& text, int status) : std::runtime_error(text), code(status) {}
};
void require(bool ok, const std::string& why, int code = 1) {
    if (!ok) throw Failure(why, code);
}
void status(EA::Status got, EA::Status wanted, const char* what) {
    require(got == wanted, std::string(what) + ": " + EA::status_name(got));
}
std::string env(const char* key) {
    const char* v = std::getenv(key); return v ? v : "";
}
bool uuid(const std::string& s) {
    if (s.size() != 40 || s.compare(0, 4, "GPU-") != 0) return false;
    for (size_t i = 4; i < s.size(); ++i) {
        if (i == 12 || i == 17 || i == 22 || i == 27) { if (s[i] != '-') return false; }
        else if (std::string("0123456789abcdefABCDEF").find(s[i]) == std::string::npos) return false;
    }
    return true;
}
void opt_in(int argc, char** argv) {
    require(argc == 2 && std::string(argv[1]) == "--run" && env("H3_GPU_PARITY_ALLOW") == "1",
            "NOT RUN: default off; use test_h3_gpu_parity.sh --run --device GPU-... --binary ABSOLUTE_PATH", 2);
    require(uuid(env("CUDA_VISIBLE_DEVICES")) && env("CUDA_VISIBLE_DEVICES") == env("H3_GPU_PARITY_PREFLIGHT_DEVICE"),
            "NOT RUN: single full UUID and matching preflight required", 2);
    const auto stamp = env("H3_GPU_PARITY_PREFLIGHT_EPOCH");
    require(!stamp.empty() && stamp.find_first_not_of("0123456789") == std::string::npos,
            "NOT RUN: missing preflight timestamp", 2);
    char* end = nullptr;
    const double age = std::difftime(std::time(nullptr), 0) - std::strtod(stamp.c_str(), &end);
    require(end && *end == '\0' && std::isfinite(age) && age >= 0 && age <= 30,
            "NOT RUN: stale preflight; rerun GPU guard", 2);
}
std::string shape(const int64_t* ne) {
    std::ostringstream out;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) { if (d) out << 'x'; out << ne[d]; }
    return out.str();
}
std::string digest(const void* data, size_t bytes) {
    MiniMaxH3::Sha256 h; h.update(data, bytes); return h.finish();
}
struct FileIdentity {
    struct stat st{};
    explicit FileIdentity(const std::string& p) {
        require(::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && uint64_t(st.st_size) == MODEL_BYTES,
                "missing/nonregular/wrong-size locked model", 3);
    }
    void unchanged(const std::string& p) const {
        FileIdentity now(p);
        require(st.st_dev == now.st.st_dev && st.st_ino == now.st.st_ino && st.st_size == now.st.st_size &&
                st.st_mtim.tv_sec == now.st.st_mtim.tv_sec && st.st_mtim.tv_nsec == now.st.st_mtim.tv_nsec &&
                st.st_ctim.tv_sec == now.st.st_ctim.tv_sec && st.st_ctim.tv_nsec == now.st.st_ctim.tv_nsec,
                "model file changed during probe");
    }
};
struct Probe : MiniMaxH3::MiniMaxH3Runner {
    struct Saved {
        ggml_tensor* t; ggml_backend_buffer_t buffer; void* data; void* extra;
        ggml_type type; std::string name, sha;
    };
    std::map<std::string, ggml_tensor*> named;
    std::vector<Saved> saved;
    size_t direct_graphs = 0;
    Probe(ggml_backend_t gpu, ModelLoader& loader) : MiniMaxH3Runner(gpu, loader.get_tensor_storage_map(), PREFIX, true) {
        require(ggml_backend_is_cuda(runtime_backend) && runtime_backend != params_backend &&
                ggml_backend_is_cpu(params_backend) && !params_on_runtime_backend,
                "real CUDA compute / CPU source offload required", 3);
        require(config.num_layers == 50 && config.hidden_size == 5376 && config.time_embed_dim == 2688 &&
                config.timestep_input_dim == 256 && config.time_embed_hidden_size == 5376 && !config.uses_adaln_curves(),
                "not the locked full H3 time-embedder architecture");
        // Identical bound for both paths; weights + workspace peak is NOT a hard 22-GiB cap.
        set_max_graph_vram_bytes(size_t(22) * 1024 * 1024 * 1024);
        require(alloc_params_buffer(), "full host parameter allocation failed", 3);
        get_param_tensors(named, PREFIX);
        const auto& storages = loader.get_tensor_storage_map();
        require(named.size() == storages.size() && named.size() == 535, "missing/extra native model parameters");
        size_t context_count = 0;
        for (auto* t = ggml_get_first_tensor(params_ctx); t; t = ggml_get_next_tensor(params_ctx, t)) ++context_count;
        require(context_count == named.size(), "unnamed/unloaded parameters in runner");
        for (const auto& item : named) {
            const auto found = storages.find(item.first);
            require(found != storages.end(), "parameter absent from file: " + item.first);
            const auto& s = found->second; auto* t = item.second;
            require(s.expected_type == GGML_TYPE_COUNT && !s.swap_scale_shift && !s.swap_swiglu_halves,
                    "unexpected dtype override or weight transformation");
            for (int d = 0; d < GGML_MAX_DIMS; ++d) require(t->ne[d] == s.ne[d], "native/file shape mismatch: " + item.first);
            require(t->buffer && t->data && ggml_backend_buffer_is_host(t->buffer), "non-host parameter");
            const bool projection = item.first.find("adaln_proj.linear.") != std::string::npos;
            const bool time = item.first.find(".time_embedder.") != std::string::npos;
            if ((projection || time) && item.first.find(".weight") != std::string::npos)
                require(t->type == s.type, "probe must not replace real projection/time weights with F32");
            // Native Linear/RMSNorm may widen BF16 biases/norms to F32. Do not
            // alter the model to avoid this; log storage AND native graph types.
            std::cout << "PARAM name=" << item.first << " storage_dtype=" << ggml_type_name(s.type)
                      << " native_dtype=" << ggml_type_name(t->type) << " shape=" << shape(t->ne)
                      << " native_bytes=" << ggml_nbytes(t) << " projection_or_time=" << (projection || time) << '\n';
        }
        // Map overload validates that EVERY destination was loaded. No ignore set,
        // no fill-with-zero fallback, no set_wtype_override/apply_dtype_policy.
        require(loader.load_tensors(named, {}, 8, false), "native ModelLoader::load_tensors failed: " + loader.get_last_error(), 3);
        size_t projection_weights = 0, time_params = 0;
        for (const auto& item : named) {
            auto* t = item.second;
            const bool projected = item.first.find("adaln_proj.linear.") != std::string::npos;
            const bool time = item.first.find(".time_embedder.") != std::string::npos;
            if (projected && item.first.find(".weight") != std::string::npos) {
                ++projection_weights;
                const bool final = item.first.find(".final_layer.") != std::string::npos;
                require(t->type == (final ? GGML_TYPE_BF16 : GGML_TYPE_Q4_0), "locked AdaLN matrix type mismatch");
            }
            time_params += time;
            const std::string sha = projected || time ? digest(t->data, ggml_nbytes(t)) : "";
            saved.push_back({t, t->buffer, t->data, t->extra, t->type, item.first, sha});
            if (!sha.empty()) std::cout << "LOADED name=" << item.first << " native_bytes_sha256=" << sha << '\n';
        }
        require(projection_weights == 51 && time_params == 4, "incomplete real AdaLN/time weights");
        require(!model.cuda_optimizations.any_enabled(), "unrequested research optimization enabled");
        std::cout << "LOAD_COMPLETE native_tensors=" << named.size() << " projection_weights=" << projection_weights
                  << " time_params=" << time_params << " full_host_load=1 native_loader_bias_widening_preserved=1\n";
    }
    ~Probe() { exact_release_projection_graph(); }
    void restored(bool check_bytes = false) {
        require(!params_on_runtime_backend, "whole model unexpectedly resident");
        for (const auto& s : saved) {
            require(s.t->buffer == s.buffer && s.t->data == s.data && s.t->extra == s.extra && s.t->type == s.type,
                    "parameter binding/type not restored: " + s.name);
            if (check_bytes && !s.sha.empty()) require(digest(s.t->data, ggml_nbytes(s.t)) == s.sha,
                                                     "projection/time source bytes changed: " + s.name);
        }
    }
    // Only the direct template dispatch is intercepted. Native layer-major uses
    // its unmodified MiniMaxH3Runner implementation and synchronous CUDA executor.
    bool exact_execute_graph(ggml_cgraph* graph, int threads, const std::vector<ggml_tensor*>& params,
                             std::function<bool(ggml_cgraph*)> post) {
        require(!params.empty(), "empty selection would transfer the whole model");
        size_t projections = 0;
        for (const auto& item : named)
            if (item.first.find("adaln_proj.linear.weight") != std::string::npos &&
                std::find(params.begin(), params.end(), item.second) != params.end()) ++projections;
        require(projections == 51, "direct graph must contain ALL 51 projections");
        auto pre = [&](ggml_cgraph* g) {
            auto storage = [&](ggml_tensor* t) {
                if (!t || !ggml_nelements(t)) return;
                require(t->data && t->buffer && !ggml_backend_buffer_is_host(t->buffer) &&
                        ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(0),
                        "host/missing/foreign CUDA tensor in direct graph");
            };
            for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
                auto* t = ggml_graph_node(g, i);
                require(ggml_backend_supports_op(runtime_backend, t), std::string("CUDA unsupported: ") + ggml_op_name(t->op));
                storage(t); for (auto* src : t->src) if (src) storage(src);
            }
            return true;
        };
        const bool ok = execute_graph<float>(graph, threads, true, params, false, true, nullptr, post, pre).has_value();
        require(ok, "direct CUDA graph compute/capture failed");
        ++direct_graphs; return true;
    }
    EA::RequestContract contract(bool mixed) {
        EA::RequestContract c;
        c.full_weights_identity = MODEL_SHA; c.adapter_identity = "none"; c.load_generation = 1;
        c.cfg_identity = mixed ? "probe:mixed:conditional:fixed-shapes" : "probe:T2VA:conditional:fixed-shapes";
        c.math_identity = env("CUDA_VISIBLE_DEVICES") + ":native-locked-Q4:direct-v-layer-major";
        c.schedule_identity = "probe-fixed-3sigma:0.95,0.6,0.2,terminal0:roundtrip1000";
        c.video_shift = 12.f; c.audio_shift = 3.f;
        c.timestep_input_dim = config.timestep_input_dim; c.time_embed_hidden = config.time_embed_hidden_size;
        c.time_embed_dim = config.time_embed_dim; c.curve_grid = 0;
        c.sigmas = {.95f, .6f, .2f, 0.f}; c.projections = EA::h3_projections(config.hidden_size, config.num_layers);
        std::vector<sd::Tensor<float>> videos, audios;
        std::vector<MiniMaxH3ReferenceBlock> refs;
        sd::Tensor<int32_t> keys, tags;
        if (mixed) {
            // Semantic shape fixtures, NOT VAE-encoded reference content. Only
            // layout/timestep rows are consumed by the projection-only graphs.
            videos.emplace_back(std::vector<int64_t>{54, 30, 2, config.video_latent_channels, 1});
            videos.emplace_back(std::vector<int64_t>{54, 30, 1, config.video_latent_channels, 1});
            audios.emplace_back(std::vector<int64_t>{12, 2, config.audio_latent_channels, 1});
            refs = {{MiniMaxH3ReferenceKind::VIDEO_AUDIO, 0, 0}, {MiniMaxH3ReferenceKind::IMAGE, 1, -1}};
            keys = sd::Tensor<int32_t>({1}, {0});
            std::vector<int32_t> values(64, 0); for (int i = 0; i < 16; ++i) values[i] = 1;
            for (int i = 16; i < 32; ++i) values[i] = 2;
            tags = sd::Tensor<int32_t>({64}, values);
        }
        // 864x480 / 124 frames at 24 fps: native latent geometry (54,30,37),
        // 207 audio time positions, 64 deterministic text layout tokens.
        for (size_t i = 0; i < 3; ++i) {
            const float original = c.sigmas[i];
            c.sigmas[i] = std::clamp((original * 1000.f) / 1000.f, 1e-6f, 1.f);
            const auto layout = MiniMaxH3::build_layout(64, 37, 30, 54, 207, videos, audios, keys, refs, tags,
                1.f-c.sigmas[i], 1.f-MiniMaxH3::time_shift_sigma(c.sigmas[i], c.video_shift, c.audio_shift));
            c.plans.push_back(EA::plan_from_layout(i, c.sigmas[i], EA::CfgRole::Conditional, layout));
            std::cout << "PLAN layout=" << (mixed ? "mixed" : "T2VA") << " step=" << i
                      << " sigma=" << c.sigmas[i] << " sigma_bits=" << EA::float_bits(c.sigmas[i])
                      << " pipeline_timestep=" << original * 1000.f << " rows=" << layout.timesteps.size()
                      << " spans=" << c.plans.back().spans.size() << " timesteps=";
            for (float t : layout.timesteps) std::cout << t << ':' << EA::float_bits(t) << ',';
            std::cout << '\n';
            for (const auto& s : c.plans.back().spans)
                std::cout << "SPAN layout=" << (mixed ? "mixed" : "T2VA") << " step=" << i << " begin=" << s.begin
                          << " end=" << s.end << " role=" << unsigned(s.role) << " source=" << s.source_index
                          << " row=" << s.timestep_row << " modality=" << s.modality << '\n';
        }
        return c;
    }
};
size_t compared = 0, different = 0;
void compare(EA::Cache& direct, EA::Cache& major, const EA::RequestContract& c, bool mixed, const Probe& runner) {
    for (size_t i = 0; i < c.plans.size(); ++i) {
        EA::Cache::StepToken dt, mt;
        status(direct.bind_step(c, i, c.plans[i], &dt), EA::Status::Hit, "direct bind");
        status(major.bind_step(c, i, c.plans[i], &mt), EA::Status::Hit, "major bind");
        require(direct.complete(dt) && major.complete(mt), "incomplete native CUDA prefill");
        for (size_t j = 0; j < c.projections.size(); ++j) {
            EA::Shape ds{}, ms{}; const float *a = nullptr, *b = nullptr;
            status(direct.expected_shape(dt, j, &ds), EA::Status::Hit, "direct shape");
            status(major.expected_shape(mt, j, &ms), EA::Status::Hit, "major shape");
            require(ds == ms, "direct/major output shape mismatch");
            status(direct.lookup(dt, j, ds, &a), EA::Status::Hit, "direct lookup");
            status(major.lookup(mt, j, ds, &b), EA::Status::Hit, "major lookup");
            size_t n = 1; for (auto d : ds) { require(d > 0, "bad shape"); n *= size_t(d); }
            double maxabs = 0, sqerr = 0, sqref = 0; size_t bitdiff = 0, worst = 0;
            for (size_t k = 0; k < n; ++k) {
                require(std::isfinite(a[k]) && std::isfinite(b[k]), "nonfinite projection output");
                const double d = double(a[k]) - double(b[k]);
                if (std::abs(d) > maxabs) { maxabs = std::abs(d); worst = k; }
                sqerr += d*d; sqref += double(a[k])*double(a[k]);
                bitdiff += EA::float_bits(a[k]) != EA::float_bits(b[k]);
            }
            const double rel = sqref > 0 ? std::sqrt(sqerr/sqref) : (sqerr == 0 ? 0 : std::numeric_limits<double>::infinity());
            const auto name = std::string(PREFIX) + "." + c.projections[j].name + ".weight";
            auto* w = runner.named.at(name);
            // This is an exact-cache diagnostic, not a quality tolerance gate.
            // Record every difference and finish the matrix before nonzero exit.
            ++compared; different += bitdiff != 0;
            std::cout << "COMPARE layout=" << (mixed ? "mixed" : "T2VA") << " step=" << i << " sigma=" << c.sigmas[i]
                      << " projection=" << j << " name=" << c.projections[j].name
                      << " weight_dtype=" << ggml_type_name(w->type) << " weight_shape=" << shape(w->ne)
                      << " output_dtype=f32 output_shape=" << shape(ds.data()) << " elements=" << n
                      << " maxabs=" << maxabs << " relL2=" << rel << " rmse=" << std::sqrt(sqerr/n)
                      << " bitdiff=" << bitdiff << " worst_index=" << worst << " direct_at_worst=" << a[worst]
                      << " major_at_worst=" << b[worst] << " exact=" << (bitdiff == 0) << '\n';
        }
    }
}
} // namespace
#ifndef H3_ADALN_WEIGHT_PROBE_NO_MAIN
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "REAL locked FL2VA AdaLN projection probe; NOT a tiny model or performance test.\n"
                         "Default OFF. H3_ADALN_PROBE_MODEL=/absolute/readonly/locked.gguf; run ONLY via\n"
                         "scripts/test_h3_gpu_parity.sh --run --device GPU-... --binary ABSOLUTE_PATH.\n"
                         "535 native host tensors; 51 projections x 3 sigmas x 2 layouts = 306 comparisons.\n"
                         "Full SHA256 rechecked; no precision override. Native loader bias/norm widening logged.\n"
                         "Exit 0=all bit-exact; 4=completed with numerical differences; 1=failure; 2=not opted in; 3=missing prerequisite.\n"
                         "No full denoiser/CFG branch execution, media quality, hard VRAM cap or speed claim.\n";
            return 0;
        }
        opt_in(argc, argv); // Must precede model IO and ANY CUDA/backend initialization.
        const auto path = env("H3_ADALN_PROBE_MODEL");
        require(!path.empty() && std::filesystem::path(path).is_absolute(), "H3_ADALN_PROBE_MODEL must be an absolute read-only model path", 3);
        FileIdentity file(path);
        // All settings are local to this process. Never touch user model bytes.
        require(setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1) == 0 && setenv("ED_CUDA_GRAPHS", "0", 1) == 0,
                "cannot disable CUDA capture");
        ggml_backend_register(h3_probe_cpu_reg());
        auto* reg = ggml_backend_cuda_reg();
        require(reg && ggml_backend_reg_dev_count(reg) == 1 && ggml_backend_cuda_get_device_count() == 1,
                "exactly one real CUDA device required", 3);
        ggml_backend_register(reg);
        Backend gpu(ggml_backend_cuda_init(0), ggml_backend_free);
        require(gpu && ggml_backend_is_cuda(gpu.get()), "no CUDA backend", 3);
        std::cout << std::setprecision(17) << std::unitbuf;
        std::cout << "PROBE device=" << env("CUDA_VISIBLE_DEVICES") << " model=" << std::quoted(path)
                  << " expected_sha256=" << MODEL_SHA << " CUDA_capture=0 criterion=bit_exact threshold=0\n";
        std::cout << "HASHING full model; device acquired after fresh preflight, no compute yet\n";
        require(MiniMaxH3::sha256_file(path) == MODEL_SHA, "full model SHA256 mismatch", 3);
        file.unchanged(path);
        std::cout << "FULL_SHA256_VERIFIED " << MODEL_SHA << '\n';
        ModelLoader loader;
        require(loader.init_from_file(path, std::string(PREFIX) + "."), "ModelLoader init_from_file failed", 3);
        require(loader.get_ld_version() == VERSION_MINIMAX_H3, "loader did not detect real H3", 3);
        loader.convert_tensors_name();
        std::map<ggml_type, size_t> types;
        for (const auto& item : loader.get_tensor_storage_map()) ++types[item.second.type];
        require(types == std::map<ggml_type, size_t>{{GGML_TYPE_F32,13},{GGML_TYPE_BF16,264},{GGML_TYPE_Q4_0,50},{GGML_TYPE_Q4_K,208}},
                "locked checkpoint mixed type counts mismatch");
        Probe runner(gpu.get(), loader); file.unchanged(path);
        for (bool mixed : {false, true}) {
            const auto c = runner.contract(mixed);
            EA::Cache direct({true, size_t(512)*1024*1024, 16384});
            EA::Cache major({true, size_t(512)*1024*1024, 16384});
            status(direct.begin_request(c), EA::Status::Stored, "direct begin");
            status(major.begin_request(c), EA::Status::Stored, "major begin");
            std::cout << "RUN_PATH layout=" << (mixed ? "mixed" : "T2VA") << " path=direct_all51_per_sigma\n";
            status(EA::prefill_request(runner, direct, c, 1, [&](size_t i, auto token) {
                runner.restored(); return runner.build_adaln_projection_graph(direct, token, c.plans[i]);
            }), EA::Status::Stored, "direct native CUDA prefill");
            runner.restored();
            std::cout << "RUN_PATH layout=" << (mixed ? "mixed" : "T2VA") << " path=production_layer_major_all_sigma\n";
            status(runner.prefill_adaln_layer_major(major, c, 1), EA::Status::Stored, "production layer-major CUDA prefill");
            runner.restored();
            require(direct.statistics().stores == 153 && major.statistics().stores == 153, "missing/duplicate projection stores");
            compare(direct, major, c, mixed, runner);
        }
        runner.restored(true); file.unchanged(path);
        require(compared == 306 && runner.direct_graphs == 6, "incomplete real-weight matrix");
        std::cout << "SUMMARY completed=1 comparisons=" << compared << " differing_projections=" << different
                  << " direct_cuda_graphs=" << runner.direct_graphs
                  << " native_layer_major_calls=2 all_source_projection_time_bytes_unchanged=1"
                  << " full_denoiser_executed=0 performance_test=0 result=" << (different ? "DIFFERENCES" : "EXACT") << '\n';
        return different ? 4 : 0;
    } catch (const Failure& e) { std::cerr << "FAIL completed=0: " << e.what() << '\n'; return e.code; }
      catch (const std::exception& e) { std::cerr << "FAIL completed=0: " << e.what() << '\n'; return 1; }
}
#endif // H3_ADALN_WEIGHT_PROBE_NO_MAIN
