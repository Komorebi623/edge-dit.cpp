"""GPU-free H3 request cleanup regression.

Run: python3 -m unittest discover -s tests -p 'test_h3_request_cleanup.py' -v
Source checks bind the cleanup to generate_video, not individual compute/CFG
calls. A tiny C++17 harness compiles the *actual* ScopeExit and guard declarations
from the pipeline with fake request buffers; it is not a full pipeline/GPU test.
No models, network, existing build directory or runtime tests are needed.
"""
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src/dit_models/pipelines/minimax_h3_pipeline.cpp"


def code_only(source):
    """Remove comments while retaining string literals and line boundaries."""
    return re.sub(
        r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|//[^\n]*|/\*.*?\*/',
        lambda match: match.group(1) or "\n" * match.group(0).count("\n"),
        source, flags=re.S,
    )


def braced_block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start:position + 1]
    raise AssertionError(f"Unclosed source block: {marker}")


SOURCE = code_only(PIPELINE.read_text())
GENERATION = SOURCE[SOURCE.index("ed_status_t MiniMaxH3Pipeline::generate_video("):
                    SOURCE.index("ed_scheduler_t MiniMaxH3Pipeline::default_scheduler(")]
BODY = braced_block(GENERATION, "std::string* error) try")
REQUEST_GUARD = braced_block(BODY, "ScopeExit reset_diffusion_request(") + ");"
RELEASE_GUARD = braced_block(BODY, "ScopeExit release_diffusion(") + ");"
ENTRY = BODY[BODY.index("ScopeExit reset_diffusion_request("):
             BODY.index("MiniMaxH3Profile profile;")]
LOOP = braced_block(BODY, "for (int step = 0; step < steps; ++step)")


class RequestCleanupSourceTests(unittest.TestCase):
    def test_guard_precedes_validation_and_every_early_return(self):
        self.assertLess(BODY.index(REQUEST_GUARD), BODY.index("if (out == nullptr"))
        self.assertLess(BODY.index(REQUEST_GUARD), BODY.index("return "))
        self.assertLess(BODY.index(REQUEST_GUARD), BODY.index("prepare_cuda_schedule("))
        self.assertLess(BODY.index(REQUEST_GUARD), BODY.index("stage_params_for_phase("))
        # Preserve the pre-existing start reset as well as cleanup on exit.
        self.assertEqual(ENTRY.count("reset_cuda_optimization_state();"), 2)

    def test_request_guard_owns_only_optimization_state(self):
        self.assertIn("[this]", REQUEST_GUARD)
        self.assertIn("diffusion_ != nullptr", REQUEST_GUARD)
        self.assertIn("model.cuda_optimizations.any_enabled()", REQUEST_GUARD)
        self.assertIn("[this]() noexcept", REQUEST_GUARD)
        calls = re.findall(r"diffusion_->(\w+)\(", REQUEST_GUARD)
        self.assertEqual(calls, ["reset_cuda_optimization_state"])
        self.assertNotIn("out", REQUEST_GUARD)

    def test_cleanup_precedes_existing_phase_release(self):
        self.assertIn("[this, &reset_diffusion_request]", RELEASE_GUARD)
        self.assertLess(RELEASE_GUARD.index("reset_diffusion_request.run_now();"),
                        RELEASE_GUARD.index("release_params_after_phase();"))
        self.assertEqual(re.findall(r"diffusion_->(\w+)\(", RELEASE_GUARD),
                         ["release_params_after_phase"])
        # Existing placement/staging policy is not changed by the new cleanup.
        self.assertIn("if (stage_diffusion_lifecycle_ && !diffusion_->stage_params_for_phase())", BODY)
        self.assertIn("if (diffusion_ != nullptr)", RELEASE_GUARD)

    def test_release_guard_lifetime_encloses_entire_denoising_loop(self):
        self.assertLess(BODY.index(RELEASE_GUARD), BODY.index(LOOP))
        self.assertIn("return ED_STATUS_CANCELLED;", LOOP)
        self.assertEqual(LOOP.count("return ED_STATUS_GENERATION_FAILED;"), 4)
        for cleanup in ("reset_cuda_optimization_state", "reset_diffusion_request",
                        "release_diffusion", "release_params_after_phase", "ScopeExit"):
            self.assertNotIn(cleanup, LOOP)

    def test_conditional_and_cfg_compute_keep_step_metadata_and_cache(self):
        conditional = "diffusion_->begin_cuda_step(step, steps, true);"
        unconditional = "diffusion_->begin_cuda_step(step, steps, false);"
        self.assertEqual(LOOP.count("diffusion_->compute("), 2)
        self.assertLess(LOOP.index(conditional), LOOP.index("diffusion_->compute("))
        cfg = braced_block(LOOP, "if (use_cfg)")
        self.assertLess(cfg.index(unconditional), cfg.index("diffusion_->compute("))
        self.assertIn("velocity = uncond_velocity + (velocity - uncond_velocity) * cfg_scale;", cfg)
        self.assertIn("control->step_done();", LOOP)

    def test_normal_cleanup_is_before_latent_split_and_both_decoders(self):
        after_loop = BODY[BODY.index(LOOP) + len(LOOP):]
        release = "release_diffusion.run_now();"
        self.assertTrue(after_loop.lstrip().startswith(release))
        self.assertEqual(BODY.count(release), 1)
        self.assertEqual(BODY.count("reset_diffusion_request.run_now();"), 1)
        for operation in ("diffusion_->split_av_latents(packed, audio_length)",
                          "decode_video_latent(", "decode_audio_latent("):
            self.assertLess(after_loop.index(release), after_loop.index(operation))
        self.assertIn("if (status != ED_STATUS_OK)", after_loop)
        self.assertIn("ed_free_video(out);", after_loop)

    def test_exception_handlers_keep_existing_output_and_reset_fallbacks(self):
        handlers = GENERATION[GENERATION.index("catch (const std::exception& e)"):]
        self.assertEqual(handlers.count("if (out != nullptr) ed_free_video(out);"), 2)
        self.assertEqual(handlers.count("if (diffusion_ != nullptr) diffusion_->reset_cuda_optimization_state();"), 2)
        self.assertEqual(handlers.count("return ED_STATUS_GENERATION_FAILED;"), 2)

    def test_new_cleanup_is_opt_in_and_h3_request_local(self):
        # New exit cleanup is opt-in; retain the existing default-off entry and
        # exception resets. Do not introduce backend/weight placement policy.
        self.assertNotRegex(REQUEST_GUARD + RELEASE_GUARD,
                            r"backend|free_param|reset\(\)|ed_free")
        self.assertEqual(SOURCE.count("ScopeExit reset_diffusion_request("), 1)
        self.assertEqual(SOURCE.count("reset_diffusion_request.run_now();"), 1)


HARNESS = r'''
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

@@SCOPE_EXIT@@;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct Runner {
    bool optimized;
    bool pinned;
    const int weights = 12345;
    const char* backend;
    struct Options { bool enabled; bool any_enabled() const { return enabled; } };
    struct Model { Options cuda_optimizations; } model{{optimized}};
    int reset_calls = 0;
    int phase_calls = 0;
    int weight_moves = 0;
    int request_buffers = 0;
    int computes = 0;
    int last_step = -1;
    std::vector<std::string> events;

    void reset_cuda_optimization_state() noexcept {
        ++reset_calls;
        request_buffers = 0;
        last_step = -1;
        events.push_back("reset");
    }
    void release_params_after_phase() {
        require(request_buffers == 0, "request buffers must precede phase release");
        ++phase_calls;
        if (pinned) ++weight_moves;  // Only the pre-existing offload lifecycle.
        events.push_back("phase");
    }
    void prepare() {
        if (optimized) request_buffers = 4;  // Cache-DiT/AdaLN/attention stand-ins.
        events.push_back("prepare");
    }
    void compute(int step, bool conditional) {
        require(!optimized || request_buffers == 4, "cache was reset inside denoising/CFG");
        require(step >= last_step, "step metadata regressed");
        if (!conditional) require(last_step == step, "CFG lost conditional state");
        last_step = step;
        ++computes;
        events.push_back(conditional ? "cond" : "uncond");
    }
};

struct Pipeline {
    Runner* diffusion_;
    int packed = 0;
    void decode() {
        require(diffusion_->request_buffers == 0, "cache survived into VAE decode");
        require(diffusion_->events.back() == "phase", "decode preceded diffusion release");
    }
    int generate(const std::string& mode, bool cfg) try {
        @@ENTRY@@
        if (mode == "invalid" || diffusion_ == nullptr) return 1;
        if (mode == "conditioning_return") return 2;
        if (mode == "conditioning_throw") throw std::runtime_error("conditioning");
        diffusion_->prepare();
        if (mode == "schedule_return" || mode == "stage_return") return 3;
        if (mode == "schedule_throw" || mode == "stage_throw") throw std::runtime_error("schedule/stage");
        @@RELEASE_GUARD@@
        packed = 0;
        for (int step = 0; step < 3; ++step) {
            if (mode == "cancel_first" || (mode == "cancel_later" && step == 1)) return 4;
            diffusion_->compute(step, true);
            if (mode == "cond_return" || mode == "nonfinite_return" || mode == "debug_return") return 5;
            if (mode == "cond_throw") throw std::runtime_error("conditional");
            if (mode == "unknown_throw") throw 17;
            if (cfg) {
                diffusion_->compute(step, false);
                if (mode == "cfg_return") return 6;
                if (mode == "cfg_throw") throw std::runtime_error("CFG");
            }
            packed += (step + 1) * (cfg ? 2 : 1);
        }
        release_diffusion.run_now();
        decode();
        if (mode == "video_return") return 7;
        if (mode == "video_throw") throw std::runtime_error("video decode");
        decode();
        if (mode == "audio_return") return 8;
        if (mode == "audio_throw") throw std::runtime_error("audio decode");
        return 0;
    } catch (const std::exception&) {
        if (diffusion_ != nullptr) diffusion_->reset_cuda_optimization_state();
        return 9;
    } catch (...) {
        if (diffusion_ != nullptr) diffusion_->reset_cuda_optimization_state();
        return 10;
    }
};

void check_scope_exit() {
    static_assert(noexcept(std::declval<ScopeExit&>().~ScopeExit()), "destructor must be noexcept");
    int calls = 0;
    { ScopeExit scope([&] { ++calls; }); scope.run_now(); scope.run_now(); }
    require(calls == 1, "explicit cleanup retried by destructor");
    { ScopeExit scope([&] { ++calls; }); }
    require(calls == 2, "early-return scope did not clean");
    try { ScopeExit scope([&] { ++calls; }); throw std::runtime_error("generation"); }
    catch (const std::runtime_error&) {}
    require(calls == 3, "exception scope did not clean");
    { ScopeExit scope([&] { ++calls; throw 42; }); }
    require(calls == 4, "throwing destructor callback escaped");
    { ScopeExit scope([&] { ++calls; throw 42; });
      try { scope.run_now(); } catch (int) {} }
    require(calls == 5, "throwing explicit callback retried by destructor");
    try { ScopeExit scope([&] { ++calls; throw 42; }); throw std::runtime_error("original"); }
    catch (const std::runtime_error& e) { require(std::string(e.what()) == "original", "exception replaced"); }
    require(calls == 6, "throwing unwind callback failed");
}

int main(int argc, char** argv) {
    check_scope_exit();
    require(argc == 2, "one scenario required");
    const std::string mode(argv[1]);
    const bool early = mode == "invalid" || mode == "conditioning_return" ||
        mode == "conditioning_throw" || mode == "schedule_return" ||
        mode == "schedule_throw" || mode == "stage_return" || mode == "stage_throw";
    const bool exception = mode.find("throw") != std::string::npos;
    for (const char* backend : {"CPU", "CUDA", "Vulkan", "Metal"}) {
        for (bool optimized : {false, true}) {
            for (bool pinned : {false, true}) {
                for (bool cfg : {false, true}) {
                    if (!cfg && (mode == "cfg_return" || mode == "cfg_throw")) continue;
                    Runner runner{optimized, pinned, 12345, backend};
                    Pipeline pipeline{&runner};
                    for (int request = 0; request < 2; ++request) {
                        runner.request_buffers = optimized ? 99 : 0;  // Stale prior-request state.
                        const int resets = runner.reset_calls;
                        const int phases = runner.phase_calls;
                        const int moves = runner.weight_moves;
                        const int status = pipeline.generate(mode, cfg);
                        require((status == 0) == (mode == "success"), "status changed");
                        require(runner.request_buffers == 0, "request leaked on exit");
                        require(runner.reset_calls - resets == 1 + int(optimized) + int(exception), "reset was missing or repeated");
                        require(runner.phase_calls - phases == (early ? 0 : 1), "phase lifecycle changed");
                        require(runner.weight_moves - moves == (!early && pinned ? 1 : 0), "weight placement changed");
                        require(runner.weights == 12345, "model weights changed");
                        // The next request must work after *every* failure/cancel.
                        const int calls = runner.computes;
                        require(pipeline.generate("success", cfg) == 0, "repeat failed");
                        require(runner.computes - calls == (cfg ? 6 : 3), "denoiser calls changed");
                        require(pipeline.packed == (cfg ? 12 : 6), "output changed");
                        require(runner.request_buffers == 0, "successful repeat leaked");
                    }
                }
            }
        }
    }
    Pipeline missing{nullptr};
    require(missing.generate("invalid", false) == 1, "null diffusion not safe");
}
'''


class RequestCleanupCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep artifacts for diagnosis; never remove files or use GPU builds.
        cls.scratch = Path(tempfile.mkdtemp(prefix="h3-request-cleanup-"))
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise RuntimeError("CPU lifecycle harness requires a C++17 compiler")
        scope_exit = braced_block(SOURCE, "class ScopeExit")
        harness = (HARNESS.replace("@@SCOPE_EXIT@@", scope_exit)
                   .replace("@@ENTRY@@", ENTRY)
                   .replace("@@RELEASE_GUARD@@", RELEASE_GUARD))
        harness_path = cls.scratch / "request_cleanup.cpp"
        harness_path.write_text(harness)
        cls.binary = cls.scratch / "request_cleanup"
        result = subprocess.run(compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                                            "-Wno-missing-field-initializers", str(harness_path),
                                            "-o", str(cls.binary)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(f"CPU harness compilation failed ({harness_path}):\n{result.stderr}")

    def check_scenario(self, scenario):
        result = subprocess.run([str(self.binary), scenario], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0,
                         f"{scenario} ({self.scratch}):\n{result.stdout}{result.stderr}")


# Each scenario exercises repeated requests, CFG on/off, default-off/enabled,
# resident/offload lifecycle and fake CPU/CUDA/Vulkan/Metal identities. Those
# identities verify guard policy only, not actual backend execution/compatibility.
for _scenario in (
    "success", "invalid", "conditioning_return", "conditioning_throw",
    "schedule_return", "schedule_throw", "stage_return", "stage_throw",
    "cancel_first", "cancel_later", "cond_return", "cond_throw",
    "nonfinite_return", "debug_return", "cfg_return", "cfg_throw",
    "unknown_throw", "video_return", "video_throw", "audio_return", "audio_throw",
):
    def _test(self, scenario=_scenario):
        self.check_scenario(scenario)
    setattr(RequestCleanupCpuTests, "test_" + _scenario, _test)


if __name__ == "__main__":
    unittest.main()
