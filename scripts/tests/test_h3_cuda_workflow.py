"""No-GPU H3 workflow tests; only optional integration tests execute CPU ffmpeg."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("h3_cuda_workflow", ROOT / "scripts/h3_cuda_workflow.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)
# Tests substitute tiny fake model bytes. Isolate only the repository's real
# full-weight identity lock; do not loosen production manifest validation.
_REAL_READ_JSON = h3.read_json
def fixture_read_json(path):
    result = _REAL_READ_JSON(path)
    if Path(path).resolve() == (ROOT / "docs/research/h3-precision-lock.json").resolve():
        result = copy.deepcopy(result)
        for component in result["components"]: component["full_sha256"] = None
    return result

PROFILE = ("[info] minimax-h3 profile: total=100 ms | conditioning=10 ms | "
           "diffusion=70 ms (20 steps, 20 calls) | decode=15 ms | noise=5 ms\n"
           "[info] minimax-h3 profile diffusion: conditional=70 ms unconditional=0 ms cfg-combine=0 ms | "
           "decode video-vae=14 ms video-copy=0 ms audio-vae=1 ms audio-copy=0 ms\n")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        # Intentionally retain scratch artifacts/logs: workflow never deletes.
        self.tmp = Path(tempfile.mkdtemp(prefix="h3-workflow-unit-"))
        identity_patch = patch.object(h3, "read_json", side_effect=fixture_read_json)
        identity_patch.start(); self.addCleanup(identity_patch.stop)
        self.manifest = h3.resolve_manifest(ROOT / "benchmark/h3_cuda/example.json")

    def test_example_precision_and_mode_order(self):
        h3.validate_manifest(self.manifest)
        self.assertEqual(h3.selected_modes(self.manifest, None), list(h3.MODE_ORDER))
        lock = json.loads((ROOT / "docs/research/h3-precision-lock.json").read_text())
        for name, source in zip(h3.COMPONENTS, lock["components"]):
            model = self.manifest["models"][name]
            self.assertEqual(Path(model["path"]).name, source["component"])
            self.assertEqual(model["bytes"], source["bytes"])
            self.assertEqual(model["tensor_types"], source["tensor_types"])
        self.assertEqual(self.manifest["sampling"]["steps"], 20)
        self.assertEqual(self.manifest["sampling"]["seed"], 42)

    def test_real_repository_full_hash_cannot_be_bypassed(self):
        data = copy.deepcopy(self.manifest)
        data["models"]["fl2va"]["sha256"] = "1" * 64
        with patch.object(h3, "read_json", side_effect=_REAL_READ_JSON), self.assertRaisesRegex(h3.WorkflowError, "pinned full SHA256"):
            h3.validate_manifest(data)

    def test_legal_frames(self):
        for n in (22, 39, 56, 73, 90, 107, 124, 362):
            self.manifest["sampling"]["frames"] = n
            h3.validate_manifest(self.manifest)
        for n in (0, 5, 21, 23, 120, True, 22.0, "22"):
            self.manifest["sampling"]["frames"] = n
            with self.subTest(n=n), self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(self.manifest)

    def test_bad_sampling(self):
        cases = (("width", 96), ("height", 513), ("fps", 16), ("seed", -1),
                 ("seed", 2**63), ("steps", 0), ("sampler", "auto"),
                 ("scheduler", "ltx2"), ("cfg_scale", math.nan),
                 ("audio_sample_rate", 44100), ("audio_channels", 1))
        for key, value in cases:
            data = copy.deepcopy(self.manifest)
            data["sampling"][key] = value
            with self.subTest(key=key), self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(data)

    def test_input_mode_checkpoint_and_pairing(self):
        for mutate in (
            lambda d: d["modes"]["ref_mixed"].update(dit="fl2va"),
            lambda d: d["modes"]["first_frame"].update(dit="ref2va"),
            lambda d: d["modes"]["ref_mixed"]["inputs"].update(image=["first"]),
            lambda d: d["modes"]["ref_mixed"]["inputs"].update(**{"ref-video-audio": ["paired_audio", "extra_audio"]}),
            lambda d: d["modes"]["first_frame"]["inputs"].update(image=["paired_audio"]),
            lambda d: d["modes"]["ref_image"]["inputs"].update(**{"ref-image": ["missing"]}),
            lambda d: d["modes"]["t2va"].update(prompt=""),
        ):
            data = copy.deepcopy(self.manifest)
            mutate(data)
            with self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(data)

    def test_invalid_run_arguments_and_cli(self):
        for key, value in (("warmup", -1), ("warmup", True), ("warmup", 0.0), ("repeats", 0), ("timeout_seconds", 0),
                           ("gpu_index", -1), ("min_free_mib", 0), ("max_utilization_percent", 100)):
            data = copy.deepcopy(self.manifest)
            data["run"][key] = value
            with self.subTest(key=key), self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(data)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            h3.parser().parse_args(["candidate", "--manifest", "x", "--output-dir", "y"])
        with patch("sys.stderr"):
            self.assertEqual(h3.main(["baseline", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                "--dry-run", "--warmup", "-1", "--output-dir", str(self.tmp / "bad")]), 2)

    def test_host_container_path_overrides(self):
        data = h3.resolve_manifest(ROOT / "benchmark/h3_cuda/example.json", "container")
        self.assertTrue(data["binaries"]["baseline"]["path"].startswith("/workspace/edge-dit.cpp/build-h3-baseline-97cc585/"))
        self.assertTrue(data["models"]["fl2va"]["path"].startswith("/models/MiniMax-H3-GGUF/"))
        data = h3.resolve_manifest(ROOT / "benchmark/h3_cuda/example.json", "host", {"repo_root": self.tmp, "model_root": self.tmp / "weights"})
        self.assertTrue(data["models"]["ref2va"]["path"].startswith(str(self.tmp / "weights")))
        self.assertTrue(data["binaries"]["candidate"]["path"].startswith(str(self.tmp)))

    def test_baseline_environment_isolation(self):
        inherited = {"PATH": "/bin", "HOME": "/home/test", "ED_H3_FUSED_OPS": "1",
            "CUDA_VISIBLE_DEVICES": "an-inherited-gpu", "NVIDIA_VISIBLE_DEVICES": "all",
            "ED_H3_CACHE_THRESHOLD": "nan", "ED_H3_UNKNOWN_FUTURE_OPT": "1",
            "ED_MINIMAX_H3_FUSED_SEGMENT_MODULATION": "1", "ED_MINIMAX_H3_REUSE_GRAPH": "1",
            "ED_MINIMAX_H3_DEBUG_TARGET": "exit", "ED_RUNTIME_CONST_CACHE": "1",
            "GGML_CUDA_FORCE_MMQ": "1", "GGML_CPU_PROFILE": "1", "SD_CACHE": "on"}
        env, removed = h3.isolated_env(self.manifest, "baseline", inherited)
        self.assertEqual({k: env[k] for k in h3.OPTIONS}, dict.fromkeys(h3.OPTIONS, "0"))
        self.assertEqual(env["ED_MINIMAX_H3_PROFILE"], "1")
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(inherited["ED_H3_FUSED_OPS"], "1")
        self.assertNotIn("ED_MINIMAX_H3_REUSE_GRAPH", env)
        self.assertNotIn("GGML_CUDA_FORCE_MMQ", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", env)
        self.assertIn("ED_H3_UNKNOWN_FUTURE_OPT", removed)
        candidate, _ = h3.isolated_env(self.manifest, "candidate", inherited)
        self.assertEqual(candidate["ED_H3_FUSED_OPS"], "1")
        self.assertEqual(candidate["ED_H3_CACHE_DIT"], "0")

    def test_nonzero_baseline_option_rejected(self):
        self.manifest["binaries"]["baseline"]["optimization_env"]["ED_H3_FUSED_OPS"] = "1"
        with self.assertRaisesRegex(h3.WorkflowError, "baseline optimizations"):
            h3.validate_manifest(self.manifest)

    def test_native_commands_do_not_change_precision_steps(self):
        for mode in h3.MODE_ORDER:
            command = h3.command_for(self.manifest, "baseline", mode, self.tmp / "out.mp4")
            self.assertEqual(command[command.index("--steps") + 1], "20")
            self.assertEqual(command[command.index("--type") + 1], "preserve")
            self.assertEqual(command[command.index("--cache") + 1], "off")
            self.assertNotIn("--auto-fit", command)
            expected = "ref2va" if mode.startswith("ref_") else "fl2va"
            self.assertEqual(command[command.index("--diffusion-model") + 1], self.manifest["models"][expected]["path"])
        self.manifest["modes"]["t2va"]["prompt"] = "$(touch /tmp/never); a prompt"
        command = h3.command_for(self.manifest, "baseline", "t2va", self.tmp / "out.mp4")
        self.assertEqual(command[command.index("--prompt") + 1], "$(touch /tmp/never); a prompt")

    def test_no_shell_and_timeout(self):
        with patch.object(h3.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ok", "")) as run:
            h3.invoke(["test", "a; b"], 17)
            self.assertFalse(run.call_args.kwargs["shell"])
            self.assertEqual(run.call_args.kwargs["timeout"], 17)
        with patch.object(h3.subprocess, "run", side_effect=subprocess.TimeoutExpired(["test"], 1)):
            with self.assertRaisesRegex(h3.WorkflowError, "timeout"):
                h3.invoke(["test"], 1)
        with patch.object(h3.subprocess, "run", return_value=subprocess.CompletedProcess([], 7, "", "error")):
            with self.assertRaisesRegex(h3.WorkflowError, "nonzero exit 7"):
                h3.invoke(["test"], 1)

    def test_dry_run_no_gpu_no_hash_no_performance(self):
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("GPU must not be queried")), \
             patch.object(h3, "verify_inputs", side_effect=AssertionError("weights must not be hashed")), \
             patch.object(h3, "git_provenance", return_value={"commit": "mock"}):
            path = h3.run_phase(self.manifest, "baseline", self.tmp / "dry", list(h3.MODE_ORDER), dry_run=True)
        report = h3.read_json(path)
        self.assertEqual(report["mode_order"], list(h3.MODE_ORDER))
        self.assertEqual(report["status"], "dry_run")
        self.assertIsNone(report["statistics"])
        self.assertIsNone(report["gpu"])
        self.assertFalse(report["performance_measured"])
        self.assertEqual(list(report["modes"]), sorted(h3.MODE_ORDER))
        for records in report["modes"].values():
            self.assertEqual(len(records), 4)
            self.assertTrue(records[0]["warmup"])
            self.assertFalse(records[1]["warmup"])
            self.assertNotIn("wall_ms", records[0])
        with self.assertRaisesRegex(h3.WorkflowError, "dry-run/failed"):
            h3.frozen_report(path)

    def test_one_click_dry_workflow(self):
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), \
             patch.object(h3, "git_provenance", return_value={}):
            result = h3.main(["workflow", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                              "--dry-run", "--output-dir", str(self.tmp / "workflow")])
        self.assertEqual(result, 0)
        self.assertFalse(h3.read_json(self.tmp / "workflow/workflow.json")["performance_measured"])
        candidate = h3.read_json(self.tmp / "workflow/candidate/report.json")
        self.assertEqual(candidate["environment"]["optimization_env"]["ED_H3_FUSED_OPS"], "1")


    def test_zero_warmup_cli_baseline(self):
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), \
             patch.object(h3, "git_provenance", return_value={}):
            result = h3.main(["baseline", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                              "--dry-run", "--mode", "t2va", "--warmup", "0", "--repeats", "1",
                              "--output-dir", str(self.tmp / "zero-warmup")])
        self.assertEqual(result, 0)
        report = h3.read_json(self.tmp / "zero-warmup/report.json")
        self.assertEqual(len(report["modes"]["t2va"]), 1)
        self.assertFalse(report["modes"]["t2va"][0]["warmup"])
        self.assertIsNone(report["statistics"])
        self.assertFalse(report["performance_measured"])

    def test_workflow_baseline_only_binary_override_and_no_candidate(self):
        binary = self.tmp / "baseline-ed-cli"
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), \
             patch.object(h3, "verify_inputs", side_effect=AssertionError("no hashing")), \
             patch.object(h3, "compare", side_effect=AssertionError("no comparison")), \
             patch.object(h3, "run_phase", wraps=h3.run_phase) as run, \
             patch.object(h3, "git_provenance", return_value={}):
            result = h3.main(["workflow", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                              "--phase", "baseline-only", "--dry-run", "--mode", "t2va",
                              "--warmup", "0", "--repeats", "1", "--binary", str(binary),
                              "--output-dir", str(self.tmp / "baseline-only")])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[1], "baseline")
        summary = h3.read_json(self.tmp / "baseline-only/workflow.json")
        self.assertEqual(summary["phase"], "baseline-only")
        self.assertIsNone(summary["candidate"])
        self.assertIsNone(summary["compare"])
        self.assertFalse(summary["performance_measured"])
        self.assertFalse(summary["baseline_performance_measured"])
        self.assertFalse((self.tmp / "baseline-only/candidate").exists())
        report = h3.read_json(self.tmp / "baseline-only/baseline/report.json")
        self.assertEqual(report["modes"]["t2va"][0]["command"][0], str(binary))
        self.assertEqual(set(report["environment"]["optimization_env"].values()), {"0"})
        self.assertEqual(report["manifest"]["binaries"]["candidate"]["path"],
                         self.manifest["binaries"]["candidate"]["path"])

    def test_baseline_only_refuses_ambiguous_reuse(self):
        with patch.object(h3, "run_phase", side_effect=AssertionError("no launch")), patch("sys.stderr"):
            result = h3.main(["workflow", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                              "--phase", "baseline-only", "--baseline", str(self.tmp / "previous.json"),
                              "--output-dir", str(self.tmp / "ambiguous")])
        self.assertEqual(result, 2)
        self.assertFalse((self.tmp / "ambiguous").exists())

    def test_baseline_only_failure_never_starts_candidate(self):
        with patch.object(h3, "run_phase", side_effect=h3.WorkflowError("baseline failed")) as run, \
             patch.object(h3, "compare", side_effect=AssertionError("no comparison")), patch("sys.stderr"):
            result = h3.main(["workflow", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                              "--phase", "baseline-only", "--output-dir", str(self.tmp / "failed-only")])
        self.assertEqual(result, 2)
        self.assertEqual(run.call_count, 1)
        summary = h3.read_json(self.tmp / "failed-only/workflow.json")
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["performance_measured"])
        self.assertFalse(summary["baseline_performance_measured"])
        self.assertIsNone(summary["candidate"])
        self.assertIsNone(summary["compare"])

    def test_locked_manifest_paths_remap_without_hash_changes(self):
        manifest = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for item in manifest[group].values():
                item["sha256"] = "2" * 64
        fixed = self.tmp / "portable.json"
        h3.write_json(fixed, manifest)
        remapped = h3.resolve_manifest(fixed, "container", {"asset_root": Path("/remote-fixtures")})
        self.assertTrue(remapped["models"]["ref2va"]["path"].startswith("/models/MiniMax-H3-GGUF/"))
        self.assertTrue(remapped["assets"]["reference_image"]["path"].startswith("/remote-fixtures/"))
        self.assertEqual(h3.contract(manifest, list(h3.MODE_ORDER)), h3.contract(remapped, list(h3.MODE_ORDER)))

    def test_workflow_failure_summary_persisted(self):
        with patch.object(h3, "run_phase", side_effect=h3.WorkflowError("failure without GPU")), patch("sys.stderr"):
            result = h3.main(["workflow", "--manifest", str(ROOT / "benchmark/h3_cuda/example.json"),
                              "--dry-run", "--output-dir", str(self.tmp / "workflow-failure")])
        self.assertEqual(result, 2)
        summary = h3.read_json(self.tmp / "workflow-failure/workflow.json")
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["performance_measured"])

    def test_pending_methods_single_option_dryrun_no_gpu(self):
        for option in ("ED_H3_CACHE_DIT", "ED_H3_ADALN_CACHE", "ED_H3_SAGE_ATTN", "ED_H3_SPARSE_ATTN"):
            data = copy.deepcopy(self.manifest)
            data["binaries"]["candidate"]["optimization_env"] = dict.fromkeys(h3.OPTIONS, "0")
            data["binaries"]["candidate"]["optimization_env"][option] = "1"
            with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), patch.object(h3, "git_provenance", return_value={}):
                path = h3.run_phase(data, "candidate", self.tmp / option, ["ref_mixed"], dry_run=True)
            report = h3.read_json(path)
            self.assertEqual(report["status"], "dry_run")
            self.assertEqual(report["environment"]["optimization_env"][option], "1")
            self.assertEqual(report["optimization_status"][option], "planned_build_and_runtime_not_verified")
            self.assertIsNone(report["statistics"])

    def test_implemented_methods_provenance_and_catalog_gate(self):
        data = copy.deepcopy(self.manifest)
        data["binaries"]["candidate"]["optimization_env"] = dict.fromkeys(h3.OPTIONS, "0")
        data["binaries"]["candidate"]["optimization_env"]["ED_H3_CACHE_DIT"] = "1"
        data["method_catalog"]["ED_H3_CACHE_DIT"]["status"] = "pending"
        binary = self.tmp / "candidate-ed-cli"
        binary.write_bytes(b"binary policy ED_H3_CACHE_DIT")
        values = {"source_revision": "candidate", "cuda_sage_attention": "OFF",
                  "ggml_revision": data["submodules"]["ggml"], "onednn_revision": data["submodules"]["onednn"]}
        provenance = {"binary": str(binary), "binary_sha256": h3.sha256(binary), "files": {
            "h3-build-provenance.txt": {"values": values},
            "CMakeCache.txt": {"values": {"ED_GGML_CUDA": "ON", **data["required_build"]}}}}
        with self.assertRaisesRegex(h3.WorkflowError, "no implemented"):
            h3.validate_build(data, "candidate", provenance)
        values["implemented_methods"] = "ED_H3_CACHE_DIT,ED_H3_ADALN_CACHE"
        self.assertIn("runtime_not_attested", h3.validate_build(data, "candidate", provenance)["options"]["ED_H3_CACHE_DIT"])
        values.pop("implemented_methods")
        data["method_catalog"]["ED_H3_CACHE_DIT"]["status"] = "implemented"
        self.assertIn("runtime_not_attested", h3.validate_build(data, "candidate", provenance)["options"]["ED_H3_CACHE_DIT"])
        self.assertEqual(h3.implemented_methods({"implemented_methods": '["ED_H3_FUSED_OPS"]'}), {"ED_H3_FUSED_OPS"})

    def test_unsupported_option_even_in_dry_run(self):
        self.manifest["binaries"]["candidate"]["optimization_env"]["ED_H3_TURBO"] = "1"
        with self.assertRaisesRegex(h3.WorkflowError, "unsupported"):
            h3.run_phase(self.manifest, "candidate", self.tmp / "unsupported", ["ref_mixed"], dry_run=True)
        report = h3.read_json(self.tmp / "unsupported/report.json")
        self.assertEqual(report["status"], "unsupported")
        self.assertFalse(report["performance_measured"])
        self.assertIsNone(report["statistics"])

    def test_full_hash_and_asset_mismatch(self):
        path = self.tmp / "weights.gguf"
        path.write_bytes(b"one small CPU-test fixture, never a GPU model")
        digest = h3.fingerprint(path)
        data = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for value in data[group].values():
                value.update(path=str(path), **digest)
        with patch.object(h3, "verify_precision_header"):
            self.assertEqual(h3.verify_inputs(data)["models"]["fl2va"]["sha256"], digest["sha256"])
        path.write_bytes(b"changed fixture")
        with patch.object(h3, "verify_precision_header"), self.assertRaisesRegex(h3.WorkflowError, "hash mismatch"):
            h3.verify_inputs(data)
        data["models"]["fl2va"]["sha256"] = None
        with self.assertRaisesRegex(h3.WorkflowError, "full SHA256 required"):
            h3.verify_inputs(data)

    def test_frame_directory_hash_order_and_change(self):
        directory = self.tmp / "frames"
        directory.mkdir()
        (directory / "002.png").write_bytes(b"second")
        (directory / "001.png").write_bytes(b"first")
        before = h3.fingerprint(directory)
        self.assertEqual([r["name"] for r in before["files"]], ["001.png", "002.png"])
        (directory / "002.png").write_bytes(b"changed")
        self.assertNotEqual(before["sha256"], h3.fingerprint(directory)["sha256"])

    def test_empty_input_and_output(self):
        path = self.tmp / "empty.mp4"
        path.touch()
        with self.assertRaisesRegex(h3.WorkflowError, "empty input"):
            h3.fingerprint(path)
        with self.assertRaisesRegex(h3.WorkflowError, "empty media"):
            h3.probe(path, 1)
        with self.assertRaises(h3.WorkflowError):
            h3.probe(self.tmp / "missing.mp4", 1)

    def test_parse_profile_existing_markers(self):
        markers = "[[phase]] stage=encode event=begin t=1\n[[phase]] stage=encode event=end t=1.25\n"
        parsed = h3.parse_profiles(PROFILE + markers)
        self.assertEqual(parsed["summaries"][0]["diffusion_ms"], 70)
        self.assertEqual(parsed["summaries"][0]["steps"], 20)
        self.assertEqual(parsed["phase_component_ms"]["text_encoder"], 250)
        self.assertEqual(len(parsed["details"]), 1)
        parsed = h3.parse_profiles(PROFILE.replace("minimax-h3", "[MiniMax-H3]"))
        self.assertEqual(parsed["summaries"][0]["total_ms"], 100)

    def test_existing_cpu_stage_parser_reused(self):
        log = self.tmp / "cpu.log"
        log.write_text("=== edge-dit CPU op profile [stage=vae] (per graph_compute) ===\n  TOTAL 12.0\n")
        parsed = h3.parse_profiles(log.read_text(), log)
        self.assertEqual(parsed["cpu_stage_profiles"][0]["stage"], "vae")
        self.assertEqual(parsed["cpu_stage_profiles"][0]["total"], 12.0)

    def test_tracked_cpu_stage_parser_multiple_graphs(self):
        parser = h3.helper("h3_test_stage_parser", ROOT / "benchmark/scripts/parse_cpu_stage_profile.py")
        log = self.tmp / "cpu-multiple.log"
        log.write_text("=== edge-dit CPU op profile (per graph_compute) ===\n"
                       "MUL_MAT 8.00 2 4000.0 66.7% 0.100 12.5\nTOTAL 12.00 0.100\n"
                       "=== edge-dit CPU op profile [stage=diffusion] (per graph_compute) ===\n"
                       "MUL_MAT nan 1 0.0 0.0% 0.0 0.0\nTOTAL 24.00\n"
                       "CUSTOM 999 1 0 0% 0 0\n")
        rows = parser.parse(str(log))
        self.assertEqual([r["stage"] for r in rows], ["unlabelled", "diffusion"])
        self.assertEqual([r["total"] for r in rows], [12.0, 24.0])
        self.assertEqual(rows[0]["ops"], [{"op": "MUL_MAT", "total_ms": 8.0, "calls": 2}])
        self.assertEqual(rows[1]["ops"], [])

    def test_report_mean_median_excludes_warmup_failure(self):
        profile = h3.parse_profiles(PROFILE)
        records = [{"warmup": warmup, "status": status, "wall_ms": ms, "profile": profile}
                   for warmup, status, ms in ((True, "validated", 999), (False, "validated", 10),
                                              (False, "validated", 20), (False, "validated", 90), (False, "failed", 1000))]
        result = h3.summarize(records)
        self.assertEqual(result["cli_wall_ms"]["mean"], 40)
        self.assertEqual(result["cli_wall_ms"]["median"], 20)
        self.assertEqual(result["cli_wall_ms"]["count"], 3)
        self.assertIsNone(h3.summarize([]))
        self.assertIsNone(h3.statistics_for([]))
        with self.assertRaises(h3.WorkflowError):
            h3.statistics_for([float("nan")])

    def test_gpu_busy_fails_without_terminating(self):
        rows = subprocess.CompletedProcess([], 0, "0, GPU-mock, H200, 143000, 4000, 139000, 0\n", "")
        procs = subprocess.CompletedProcess([], 0, "GPU-mock, 12345, 139000, another-user\n", "")
        with patch.object(h3, "invoke", side_effect=[rows, procs]) as run:
            with self.assertRaisesRegex(h3.WorkflowError, "occupied"):
                h3.gpu_precheck(self.manifest["run"])
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in run.call_args_list))

    def test_gpu_low_memory_utilization_and_missing_telemetry(self):
        for line in ("0, GPU-mock, H200, 143000, 4000, 139000, 0\n",
                     "0, GPU-mock, H200, 143000, 143000, 0, 99\n",
                     "0, GPU-mock, H200, N/A, N/A, N/A, N/A\n"):
            with patch.object(h3, "invoke", side_effect=[subprocess.CompletedProcess([], 0, line, ""), subprocess.CompletedProcess([], 0, "", "")]):
                with self.assertRaises(h3.WorkflowError):
                    h3.gpu_precheck(self.manifest["run"])

    def test_gpu_idle_snapshot(self):
        with patch.object(h3, "invoke", side_effect=[subprocess.CompletedProcess([], 0, "0, GPU-mock, H200, 143000, 140000, 3000, 0\n", ""), subprocess.CompletedProcess([], 0, "", "")]):
            gpu = h3.gpu_precheck(self.manifest["run"])
        self.assertEqual(gpu["free_mib"], 140000)
        self.assertEqual(gpu["compute_processes"], [])

    def test_audio_metrics_finite_and_shape(self):
        a, b = self.tmp / "a.f32", self.tmp / "b.f32"
        a.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
        b.write_bytes(struct.pack("<4f", 1, 3, 3, 4))
        error = h3.audio_errors(a, b)
        self.assertEqual(error["rmse"], .5)
        self.assertAlmostEqual(error["relative_l2"], math.sqrt(1 / 30))
        b.write_bytes(struct.pack("<f", 1))
        with self.assertRaisesRegex(h3.WorkflowError, "shape mismatch"):
            h3.audio_errors(a, b)
        b.write_bytes(struct.pack("<4f", 1, float("nan"), 3, 4))
        with self.assertRaisesRegex(h3.WorkflowError, "nonfinite"):
            h3.audio_errors(a, b)
        a.write_bytes(struct.pack("<4f", 0, 0, 0, 0))
        b.write_bytes(struct.pack("<4f", 1, 0, 0, 0))
        self.assertIsNone(h3.audio_errors(a, b)["relative_l2"])

    def test_metadata_shape_mismatches_fail_before_decode(self):
        data = {"streams": [
            {"codec_type": "video", "width": 512, "height": 512, "nb_read_frames": "124", "avg_frame_rate": "24/1", "r_frame_rate": "24/1"},
            {"codec_type": "audio", "sample_rate": "32000", "channels": 2}]}
        for field, value, index in (("width", 256, 0), ("height", 256, 0), ("nb_read_frames", "123", 0),
                                    ("avg_frame_rate", "25/1", 0), ("r_frame_rate", "0/0", 0),
                                    ("sample_rate", "44100", 1), ("channels", 1, 1)):
            current = copy.deepcopy(data)
            current["streams"][index][field] = value
            with self.subTest(field=field), patch.object(h3, "probe", return_value=current), patch.object(h3, "invoke") as invoke:
                with self.assertRaises(h3.WorkflowError):
                    h3.validate_media(self.tmp / "out.mp4", self.manifest["sampling"], 1, self.tmp)
                invoke.assert_not_called()
        with patch.object(h3, "probe", return_value={"streams": [data["streams"][0]]}):
            with self.assertRaisesRegex(h3.WorkflowError, "one video and one audio"):
                h3.validate_media(self.tmp / "out.mp4", self.manifest["sampling"], 1, self.tmp)

    def test_reject_old_binary_unsupported_build(self):
        binary = self.tmp / "ed-cli"
        binary.write_bytes(b"old ed-cli without policy")
        provenance = {"binary": str(binary), "binary_sha256": h3.sha256(binary), "files": {
            "CMakeCache.txt": {"values": {"ED_GGML_CUDA": "ON", **self.manifest["required_build"], "ED_ENABLE_CUDA_MODULATION": "ON"}},
            "h3-build-provenance.txt": {"values": {"source_revision": "candidate-commit", "cuda_sage_attention": "OFF",
                "ggml_revision": self.manifest["submodules"]["ggml"], "onednn_revision": self.manifest["submodules"]["onednn"]}}}}
        with self.assertRaisesRegex(h3.WorkflowError, "policy parser absent"):
            h3.validate_build(self.manifest, "candidate", provenance)
        binary.write_bytes(b"new binary ED_H3_FUSED_OPS parser")
        provenance["files"]["CMakeCache.txt"]["values"]["ED_ENABLE_CUDA_MODULATION"] = "OFF"
        with self.assertRaisesRegex(h3.WorkflowError, "unsupported in this build"):
            h3.validate_build(self.manifest, "candidate", provenance)
        provenance["files"]["CMakeCache.txt"]["values"]["ED_ENABLE_CUDA_MODULATION"] = "ON"
        status = h3.validate_build(self.manifest, "candidate", provenance)
        self.assertIn("runtime_not_attested", status["options"]["ED_H3_FUSED_OPS"])
        with self.assertRaisesRegex(h3.WorkflowError, "not fixed 97cc585"):
            h3.validate_build(self.manifest, "baseline", provenance)

    def mocked_generation(self, result=None, exception=None, media_error=None):
        mocks = [patch.object(h3, "verify_inputs", return_value={}), patch.object(h3, "input_stat_signatures", return_value={}), patch.object(h3, "validate_assets"),
                 patch.object(h3, "build_provenance", return_value={}), patch.object(h3, "validate_build", return_value={}),
                 patch.object(h3, "git_provenance", return_value={}), patch.object(h3, "gpu_precheck", return_value={"uuid": "GPU-mock"}),
                 patch.object(h3, "invoke", return_value=subprocess.CompletedProcess([], 0, "mock media tool version", "")),
                 patch.object(h3, "validate_media", side_effect=media_error, return_value={"sha256": "mock"}),
                 patch.object(h3.subprocess, "run", side_effect=exception,
                              return_value=result or subprocess.CompletedProcess([], 0, PROFILE, ""))]
        return mocks

    def test_generation_nonzero_captured_no_timing_success(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for mock in self.mocked_generation(result=subprocess.CompletedProcess([], 9, "stdout", "failure")):
                stack.enter_context(mock)
            with self.assertRaisesRegex(h3.WorkflowError, "nonzero exit: 9"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "failed", ["t2va"])
        report = h3.read_json(self.tmp / "failed/report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["performance_measured"])
        self.assertIsNone(report["statistics"])
        self.assertEqual((self.tmp / "failed/t2va-000.stderr.log").read_text(), "failure")

    def test_generation_timeout_persists_partial_logs(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for mock in self.mocked_generation(exception=subprocess.TimeoutExpired(["ed-cli"], 1, output=b"partial", stderr=b"timed out")):
                stack.enter_context(mock)
            with self.assertRaisesRegex(h3.WorkflowError, "timeout"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "timeout", ["t2va"])
        self.assertEqual((self.tmp / "timeout/t2va-000.stdout.log").read_text(), "partial")
        self.assertEqual(h3.read_json(self.tmp / "timeout/report.json")["status"], "failed")

    def test_generation_empty_output_or_missing_profile_fails(self):
        from contextlib import ExitStack
        for name, result, error in (("empty", None, h3.WorkflowError("empty media output")),
                                     ("profile", subprocess.CompletedProcess([], 0, "", ""), None)):
            with ExitStack() as stack:
                for mock in self.mocked_generation(result=result, media_error=error):
                    stack.enter_context(mock)
                with self.assertRaises(h3.WorkflowError):
                    h3.run_phase(self.manifest, "baseline", self.tmp / name, ["t2va"])
            self.assertIsNone(h3.read_json(self.tmp / name / "report.json")["statistics"])

    def test_mocked_warmup_repeats_report(self):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for mock in self.mocked_generation():
                stack.enter_context(mock)
            report_path = h3.run_phase(self.manifest, "baseline", self.tmp / "success", ["t2va"])
        report = h3.read_json(report_path)
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["statistics"]["t2va"]["cli_wall_ms"]["count"], 3)
        self.assertTrue(report["modes"]["t2va"][0]["warmup"])

    def test_contract_ignores_paths_but_not_hashes_or_inputs(self):
        a = h3.contract(self.manifest, ["ref_mixed"])
        data = copy.deepcopy(self.manifest)
        data["binaries"]["candidate"]["path"] = "/different/build/ed-cli"
        data["models"]["ref2va"]["path"] = "/models/same-content.gguf"
        self.assertEqual(a, h3.contract(data, ["ref_mixed"]))
        data["models"]["ref2va"]["sha256"] = "0" * 64
        self.assertNotEqual(a, h3.contract(data, ["ref_mixed"]))
        data = copy.deepcopy(self.manifest)
        data["modes"]["ref_mixed"]["inputs"]["ref-audio"] = ["paired_audio"]
        self.assertNotEqual(a, h3.contract(data, ["ref_mixed"]))

    def test_compare_contract_mismatch_writes_failure(self):
        base = {"phase": "baseline", "status": "validated", "contract": {"seed": 42}}
        cand = {"phase": "candidate", "status": "validated", "contract": {"seed": 43}}
        with patch.object(h3, "frozen_report", side_effect=[base, cand]):
            with self.assertRaisesRegex(h3.WorkflowError, "contract mismatch"):
                h3.compare(self.tmp / "base.json", self.tmp / "cand.json", self.tmp / "compare")
        self.assertEqual(h3.read_json(self.tmp / "compare/report.json")["status"], "failed")


    def test_asset_frames_and_precision_mismatches(self):
        for key, field, value in (("ref2va", "tensor_types", {"Q8_0": 100}),
                                  ("fl2va", "bytes", 123), ("llm", "source_header_sha256", "0" * 64)):
            data = copy.deepcopy(self.manifest)
            data["models"][key][field] = value
            with self.assertRaisesRegex(h3.WorkflowError, "precision differs"):
                h3.validate_manifest(data)
        path = self.tmp / "badmodel.gguf"
        path.write_bytes(b"wrong model")
        with self.assertRaisesRegex(h3.WorkflowError, "size mismatch"):
            h3.verify_precision_header("ref2va", {"path": str(path)})
        frames = self.tmp / "refs"
        frames.mkdir()
        (frames / "001.png").write_bytes(b"fake")
        (frames / "002.png").write_bytes(b"fake")
        self.manifest["assets"]["reference_frames"]["path"] = str(frames)
        with patch.object(h3, "probe", side_effect=[
            {"streams": [{"codec_type": "video", "width": 256, "height": 256}]},
            {"streams": [{"codec_type": "video", "width": 512, "height": 256}]},
            {"streams": [{"codec_type": "audio", "sample_rate": "32000", "channels": 2}]}]):
            # Asset set traversal is intentionally unordered; isolate the frame
            # entry so no mock responses depend on hash randomization.
            self.manifest["modes"]["ref_video"]["inputs"]["ref-video"] = ["reference_frames"]
            with self.assertRaisesRegex(h3.WorkflowError, "frame dimensions mismatch"):
                h3.validate_assets(self.manifest, ["ref_video"])

    def test_lock_full_hashes_no_gpu_no_overwrite(self):
        data = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for key, item in data[group].items():
                path = self.tmp / (group + "-" + key)
                path.write_bytes(b"small CPU fixture")
                item.update(path=str(path), bytes=path.stat().st_size)
        target = self.tmp / "locked.json"
        precision = json.loads((ROOT / "docs/research/h3-precision-lock.json").read_text())
        for component in precision["components"]:
            component["full_sha256"] = h3.sha256(self.tmp / "models-fl2va")
        def tiny_precision(path):
            if Path(path).resolve() == (ROOT / "docs/research/h3-precision-lock.json").resolve():
                return copy.deepcopy(precision)
            return _REAL_READ_JSON(path)
        with patch.object(h3, "verify_precision_header"), patch.object(h3, "read_json", side_effect=tiny_precision), \
             patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")):
            h3.lock_manifest(data, target)
            locked = h3.read_json(target)
            self.assertTrue(all(h3.SHA_RE.fullmatch(item["sha256"]) for item in locked["models"].values()))
            with self.assertRaises(FileExistsError):
                h3.lock_manifest(data, target)

    def test_report_binding_and_changed_artifact(self):
        manifest = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for item in manifest[group].values():
                item["sha256"] = "1" * 64
        output = self.tmp / "fixed.mp4"
        output.write_bytes(b"fixed artifact")
        report = {"phase": "baseline", "status": "imported", "manifest": manifest,
            "contract": h3.contract(manifest, ["t2va"]), "performance_measured": False,
            "modes": {"t2va": [{"warmup": False, "status": "validated", "output": str(output),
                                  "media": {"sha256": h3.sha256(output)}}]}}
        report_path = self.tmp / "fixed-report.json"
        h3.write_json(report_path, report)
        self.assertEqual(h3.frozen_report(report_path)["status"], "imported")
        output.write_bytes(b"changed")
        with self.assertRaisesRegex(h3.WorkflowError, "fixed artifact changed"):
            h3.frozen_report(report_path)
        output.write_bytes(b"fixed artifact")
        report["manifest"] = copy.deepcopy(manifest)
        report["manifest"]["sampling"]["seed"] = 123
        h3.write_json(report_path, report)
        with self.assertRaisesRegex(h3.WorkflowError, "manifest/report contract"):
            h3.frozen_report(report_path)

    def test_refuse_overwriting_outputs(self):
        self.tmp.joinpath("existing").mkdir()
        with self.assertRaises(FileExistsError):
            h3.run_phase(self.manifest, "baseline", self.tmp / "existing", ["t2va"], dry_run=True)
        json_path = self.tmp / "fixed.json"
        h3.write_json(json_path, {}, exclusive=True)
        with self.assertRaises(FileExistsError):
            h3.write_json(json_path, {"changed": True}, exclusive=True)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "system ffmpeg/ffprobe required for CPU media integration")
class CPUMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.identity_patch = patch.object(h3, "read_json", side_effect=fixture_read_json)
        cls.identity_patch.start(); cls.addClassCleanup(cls.identity_patch.stop)
        cls.tmp = Path(tempfile.mkdtemp(prefix="h3-workflow-media-"))
        cls.manifest = h3.resolve_manifest(ROOT / "benchmark/h3_cuda/example.json")
        cls.manifest["sampling"].update(width=256, height=256, frames=22)
        cls.timeout = 60
        cls.video = cls.tmp / "cpu-only.mp4"
        h3.invoke(["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000", "-t", str(22 / 24), "-c:v", "libx264", "-threads", "1",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", str(cls.video)], cls.timeout)

    def test_cpu_fixtures_hashes_and_full_matrix_assets(self):
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")):
            path = h3.fixtures(self.tmp / "fixtures", copy.deepcopy(self.manifest))
            manifest = h3.read_json(path)
            h3.validate_assets(manifest, list(h3.MODE_ORDER))
        self.assertTrue(all(h3.SHA_RE.fullmatch(v["sha256"]) for v in manifest["assets"].values()))
        self.assertEqual(len(manifest["assets"]["reference_frames"]["files"]), 24)

    def test_decode_and_identical_per_frame_ssim_psnr_audio(self):
        work = self.tmp / "validate"
        work.mkdir()
        result = h3.validate_media(self.video, self.manifest["sampling"], self.timeout, work)
        self.assertTrue(result["finite"])
        self.assertEqual(result["shape"]["frames"], 22)
        self.assertEqual(result["shape"]["fps"], "24")
        metrics_dir = self.tmp / "metrics"
        metrics_dir.mkdir()
        metrics = h3.video_metrics(self.video, self.video, 22, self.timeout, metrics_dir)
        self.assertEqual(metrics["exact_frame_count"], 22)
        self.assertEqual(len(metrics["per_frame"]), 22)
        self.assertEqual(metrics["mean_ssim"], 1.0)
        self.assertIsNone(metrics["mean_psnr_db"])
        pcm = Path(result["audio_pcm"])
        self.assertEqual(h3.audio_errors(pcm, pcm)["rmse"], 0)


    def import_cpu_baseline(self, suffix):
        data = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for item in data[group].values():
                item["sha256"] = "1" * 64
        with patch.object(h3, "verify_inputs", return_value={}), patch.object(h3, "input_stat_signatures", return_value={}), patch.object(h3, "validate_assets"), \
             patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU import")), patch.object(h3, "git_provenance", return_value={}):
            base = h3.run_phase(data, "baseline", self.tmp / (suffix + "-baseline"), ["t2va"], artifacts={"t2va": str(self.video)})
        return data, base

    def test_import_repeated_artifact_basename_all_modes(self):
        data = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for item in data[group].values():
                item["sha256"] = "1" * 64
        modes = ["t2va", "first_frame"]
        with patch.object(h3, "verify_inputs", return_value={}), patch.object(h3, "input_stat_signatures", return_value={}), patch.object(h3, "validate_assets"), \
             patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), patch.object(h3, "git_provenance", return_value={}):
            path = h3.run_phase(data, "baseline", self.tmp / "repeat-artifact", modes,
                                artifacts={mode: str(self.video) for mode in modes})
        report = h3.frozen_report(path)
        self.assertEqual(set(report["modes"]), set(modes))
        self.assertNotEqual(report["modes"]["t2va"][0]["media"]["audio_pcm"],
                            report["modes"]["first_frame"][0]["media"]["audio_pcm"])

    def test_import_artifact_no_gpu_no_performance(self):
        data, path = self.import_cpu_baseline("import")
        report = h3.frozen_report(path)
        self.assertEqual(report["status"], "imported")
        self.assertIsNone(report["gpu"])
        self.assertIsNone(report["statistics"])
        self.assertFalse(report["performance_measured"])
        self.assertNotIn("wall_ms", report["modes"]["t2va"][0])

    def test_compare_cpu_import_complete_and_bound(self):
        data, base_path = self.import_cpu_baseline("compare")
        # Candidate here is a CPU-produced test artifact, not a claimed H3 run.
        # Both reports explicitly carry no performance measurements.
        candidate = copy.deepcopy(h3.read_json(base_path))
        candidate["phase"] = "candidate"
        candidate["baseline_report_sha256"] = h3.sha256(base_path)
        cand_path = self.tmp / "cpu-candidate.json"
        h3.write_json(cand_path, candidate)
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU compare")):
            result = h3.compare(base_path, cand_path, self.tmp / "cpu-compare")
        report = h3.read_json(result)
        self.assertEqual(report["status"], "validated_numerical_comparison_not_semantic_acceptance")
        self.assertFalse(report["performance_comparable"])
        self.assertNotIn("timings", report)
        self.assertEqual(len(report["modes"]["t2va"][0]["video"]["per_frame"]), 22)
        self.assertEqual(report["modes"]["t2va"][0]["audio"]["rmse"], 0)
        candidate["baseline_report_sha256"] = "0" * 64
        h3.write_json(cand_path, candidate)
        with self.assertRaisesRegex(h3.WorkflowError, "not bound"):
            h3.compare(base_path, cand_path, self.tmp / "cpu-wrong-binding")

    def test_compare_audio_sample_count_mismatch_fails(self):
        data, base_path = self.import_cpu_baseline("sample-shape")
        longer = self.tmp / "longer-audio.mp4"
        h3.invoke(["ffmpeg", "-nostdin", "-v", "error", "-i", str(self.video),
                   "-filter_complex_threads", "1", "-af", "apad=pad_dur=0.1",
                   "-c:v", "copy", "-c:a", "aac", str(longer)], self.timeout)
        candidate = copy.deepcopy(h3.read_json(base_path))
        candidate["phase"] = "candidate"
        candidate["baseline_report_sha256"] = h3.sha256(base_path)
        record = candidate["modes"]["t2va"][0]
        record["output"] = str(longer)
        record["media"]["sha256"] = h3.sha256(longer)
        cand_path = self.tmp / "longer-candidate.json"
        h3.write_json(cand_path, candidate)
        with self.assertRaisesRegex(h3.WorkflowError, "shape mismatch"):
            h3.compare(base_path, cand_path, self.tmp / "cpu-sample-mismatch")
        self.assertEqual(h3.read_json(self.tmp / "cpu-sample-mismatch/report.json")["status"], "failed")

    def test_corrupt_media_fails(self):
        corrupt = self.tmp / "corrupt.mp4"
        corrupt.write_bytes(b"not a media file")
        with self.assertRaisesRegex(h3.WorkflowError, "nonzero exit"):
            h3.validate_media(corrupt, self.manifest["sampling"], self.timeout, self.tmp)

    def test_changed_frame_metadata_fails(self):
        sampling = copy.deepcopy(self.manifest["sampling"])
        sampling["frames"] = 39
        with self.assertRaisesRegex(h3.WorkflowError, "frames"):
            h3.validate_media(self.video, sampling, self.timeout, self.tmp)


if __name__ == "__main__":
    unittest.main()
