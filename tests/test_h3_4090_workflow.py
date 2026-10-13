"""4090 offloaded experiment contracts. No models, GPUs or media tools are used."""
import copy
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_4090_workflow", ROOT / "scripts/h3_cuda_workflow.py")
h3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h3)
T2VA = ROOT / "benchmark/h3_cuda/rtx4090_t2va.json"
MIXED = ROOT / "benchmark/h3_cuda/rtx4090_ref_mixed.json"
UUID = "GPU-12345678-1234-5678-90ab-1234567890ab"
PROFILE = ("[info] minimax-h3 profile: total=100 ms | conditioning=10 ms | "
           "diffusion=70 ms (20 steps, 20 calls) | decode=15 ms | noise=5 ms\n")


class Workflow4090Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3-4090-unit-"))
        self.manifest = h3.resolve_manifest(T2VA)

    def test_separate_modes_fixed_sampling_and_original_precision_pins(self):
        pins = dict(zip(h3.COMPONENTS, h3.read_json(ROOT / "docs/research/h3-precision-lock.json")["components"]))
        for path, mode, dit in ((T2VA, "t2va", "fl2va"), (MIXED, "ref_mixed", "ref2va")):
            data = h3.resolve_manifest(path)
            self.assertEqual(h3.selected_modes(data, None), [mode])
            self.assertEqual(set(data["models"]), {dit, "llm", "video_vae", "audio_vae"})
            self.assertEqual(data["sampling"], {
                "width": 864, "height": 480, "frames": 124, "fps": 24, "steps": 20, "seed": 42,
                "cfg_scale": 1, "sampler": "res_multistep", "scheduler": "simple", "ref_image_size": "match",
                "audio_sample_rate": 32000, "audio_channels": 2})
            self.assertEqual(data["baseline_revision"], h3.BASELINE_REVISION)
            self.assertEqual(data["required_build"]["CMAKE_CUDA_ARCHITECTURES"], "89")
            self.assertEqual(data["placement"], {"dit_offload": True, "text_encoder_offload": True,
                                                  "vae_offload": True, "max_vram_gib": 22, "cpu_threads": 6})
            self.assertEqual(data["run"]["phase"], "baseline-only")
            self.assertEqual((data["run"]["warmup"], data["run"]["repeats"]), (0, 1))
            self.assertEqual(data["run"]["min_free_mib"], 22 * 1024)
            self.assertEqual(data["run"]["max_utilization_percent"], 0)
            self.assertTrue(h3.GPU_UUID_RE.fullmatch(data["run"]["gpu_uuid"]))
            self.assertIsNone(data["performance_results"])
            self.assertEqual(data["hash_status"], "expected_full_sha256_pins_not_locally_verified")
            for key, item in data["models"].items():
                pinned = pins[key]
                self.assertEqual(item["sha256"], pinned["full_sha256"])
                self.assertEqual(item["tensor_types"], pinned["tensor_types"])
                self.assertEqual(item["source_header_sha256"], pinned["header_sha256"])
                self.assertEqual(item["bytes"], pinned["bytes"])
                folder = "diffusion_models" if key == dit else "text_encoders" if key == "llm" else "vae"
                self.assertEqual(item["path"], "/home/public/models/minimax-h3/" + folder + "/" + pinned["component"])
            for phase in ("baseline", "candidate"):
                self.assertTrue(data["binaries"][phase]["path"].startswith(str(ROOT / "_scratch") + "/"))
        self.assertEqual(self.manifest["assets"], {})
        self.assertNotEqual(h3.contract(self.manifest, ["t2va"]), h3.contract(h3.resolve_manifest(MIXED), ["ref_mixed"]))

    def test_native_cli_offload_budget_and_fusion_only_switches(self):
        for path in (T2VA, MIXED):
            data = h3.resolve_manifest(path)
            mode = next(iter(data["modes"]))
            for phase in ("baseline", "candidate"):
                cmd = h3.command_for(data, phase, mode, self.tmp / "output.mp4")
                for flag, value in (("--max-vram", "22"), ("--threads", "6"), ("--type", "preserve"), ("--backend", "cuda"),
                                    ("--cache", "off"), ("--video-frames", "124"), ("--steps", "20")):
                    self.assertEqual(cmd[cmd.index(flag) + 1], value)
                for flag in ("--dit-offload", "--text-encoder-offload", "--vae-offload"):
                    self.assertEqual(cmd.count(flag), 1)
                self.assertNotIn("--auto-fit", cmd)
                self.assertNotIn("--max-vram-gib", cmd)
                env, _ = h3.isolated_env(data, phase, {key: "1" for key in h3.OPTIONS})
                enabled = {key for key in h3.OPTIONS if env[key] == "1"}
                self.assertEqual(enabled, {"ED_H3_FUSED_OPS"} if phase == "candidate" else set())
            if mode == "ref_mixed":
                for flag, keys in data["modes"][mode]["inputs"].items():
                    self.assertEqual(cmd[cmd.index("--" + flag) + 1], data["assets"][keys[0]]["path"])

    def test_cpu_threads_are_positive_and_part_of_contract(self):
        data = h3.resolve_manifest(T2VA)
        before = copy.deepcopy(h3.contract(data, ["t2va"]))
        data["placement"]["cpu_threads"] = 8
        self.assertNotEqual(before, h3.contract(data, ["t2va"]))
        for bad in (0, -1, True, 1.5):
            data["placement"]["cpu_threads"] = bad
            with self.subTest(bad=bad), self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(data)

    def test_other_research_methods_remain_rejected_in_offloaded_plan(self):
        for path in (T2VA, MIXED):
            for option in set(h3.OPTIONS) - {"ED_H3_FUSED_OPS"}:
                data = h3.resolve_manifest(path)
                data["binaries"]["candidate"]["optimization_env"] = dict.fromkeys(h3.OPTIONS, "0")
                data["binaries"]["candidate"]["optimization_env"][option] = "1"
                with self.subTest(path=path.name, option=option), self.assertRaisesRegex(h3.WorkflowError, "unsupported"):
                    h3.method_plan(data, "candidate", list(data["modes"]), dry_run=True)

    def test_missing_required_or_unknown_models_and_wrong_full_hash_rejected(self):
        for key in self.manifest["models"]:
            data = copy.deepcopy(self.manifest)
            del data["models"][key]
            with self.subTest(missing=key), self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(data)
            data = copy.deepcopy(self.manifest)
            data["models"][key]["sha256"] = "a" * 64
            with self.subTest(hash=key), self.assertRaisesRegex(h3.WorkflowError, "pinned full SHA256"):
                h3.validate_manifest(data)
        data = copy.deepcopy(self.manifest)
        data["models"]["unknown"] = copy.deepcopy(data["models"]["fl2va"])
        with self.assertRaises(h3.WorkflowError):
            h3.validate_manifest(data)
        data = copy.deepcopy(self.manifest)
        data["modes"].update(h3.resolve_manifest(MIXED)["modes"])
        with self.assertRaises(h3.WorkflowError):
            h3.validate_manifest(data)

    def test_t2va_hash_verification_and_lock_need_only_four_declared_components(self):
        # Tiny mocked fingerprints; this is not local model verification evidence.
        items = {item["path"]: item for item in self.manifest["models"].values()}
        def fingerprint(path):
            item = items[str(path)]
            return {k: item[k] for k in ("sha256", "bytes")}
        with patch.object(h3, "fingerprint", side_effect=fingerprint) as hashes, \
             patch.object(h3, "verify_precision_header") as headers, \
             patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")):
            identities = h3.verify_inputs(self.manifest)
            self.assertEqual(hashes.call_count, 4)
            self.assertEqual(headers.call_count, 4)
            self.assertNotIn("ref2va", identities["models"])
            locked = h3.lock_manifest(copy.deepcopy(self.manifest), self.tmp / "locked.json")
            h3.validate_manifest(h3.read_json(locked))
            self.assertEqual(hashes.call_count, 8)

    def test_missing_ref2va_fails_before_hash_or_gpu_and_has_no_performance(self):
        data = h3.resolve_manifest(MIXED)
        data["models"]["ref2va"]["path"] = str(self.tmp / "missing" / Path(data["models"]["ref2va"]["path"]).name)
        with patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), \
             patch.object(h3, "verify_inputs", side_effect=AssertionError("no hash")), \
             patch.object(h3, "git_provenance", return_value={}):
            with self.assertRaisesRegex(h3.WorkflowError, "missing"):
                h3.run_phase(data, "baseline", self.tmp / "missing-ref", ["ref_mixed"])
        report = h3.read_json(self.tmp / "missing-ref/report.json")
        self.assertFalse(report["performance_measured"])
        self.assertIsNone(report["statistics"])

    def test_offload_budget_is_part_of_comparison_contract(self):
        for key, value in (("max_vram_gib", 21), ("dit_offload", False), ("vae_offload", False)):
            data = copy.deepcopy(self.manifest)
            data["placement"][key] = value
            self.assertNotEqual(h3.contract(data, ["t2va"]), h3.contract(self.manifest, ["t2va"]))

    def test_invalid_budget_uuid_and_phase_fail_closed(self):
        cases = [("placement", "max_vram_gib", v) for v in (0, -22, True, "22GiB", float("nan"), float("inf"))]
        cases += [("run", "gpu_uuid", v) for v in (None, "0", "GPU-short", UUID + "," + UUID)]
        cases += [("run", "phase", "candidate"), ("placement", "auto_fit", True)]
        for group, key, value in cases:
            data = copy.deepcopy(self.manifest)
            data[group][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(h3.WorkflowError):
                h3.validate_manifest(data)

    def telemetry(self, *, processes="", xml=None, uuid=UUID, free=24000, used=564, utilization=0):
        return [subprocess.CompletedProcess([], 0, f"0, {uuid}, NVIDIA GeForce RTX 4090, 24564, {free}, {used}, {utilization}\n", ""),
                subprocess.CompletedProcess([], 0, processes, ""),
                subprocess.CompletedProcess([], 0, xml if xml is not None else
                    f"<nvidia_smi_log><gpu><uuid>{UUID}</uuid><processes/></gpu></nvidia_smi_log>", "")]

    def test_pinned_idle_gpu_no_compute_or_graphics_processes(self):
        settings = dict(self.manifest["run"], gpu_uuid=UUID)
        with patch.object(h3, "invoke", side_effect=self.telemetry()) as query:
            gpu = h3.gpu_precheck(settings)
        self.assertEqual(gpu["uuid"], UUID)
        self.assertEqual(gpu["compute_processes"], [])
        self.assertEqual(gpu["all_processes"], [])
        self.assertEqual(query.call_args.args[0], ["nvidia-smi", "-q", "-x", "-i", UUID])
        self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in query.call_args_list))

    def test_uuid_placeholder_wrong_gpu_and_occupied_gpu_rejected(self):
        for settings, telemetry in (
            (self.manifest["run"], self.telemetry()),
            (dict(self.manifest["run"], gpu_uuid=UUID, gpu_index=1), self.telemetry()),
            (dict(self.manifest["run"], gpu_uuid=UUID), self.telemetry(processes=f"{UUID}, 12345, 1, other-user\n")),
            (dict(self.manifest["run"], gpu_uuid=UUID), self.telemetry(free=22000, used=2564)),
            (dict(self.manifest["run"], gpu_uuid=UUID), self.telemetry(utilization=1)),
        ):
            with patch.object(h3, "invoke", side_effect=telemetry), self.assertRaises(h3.WorkflowError):
                h3.gpu_precheck(settings)

    def test_graphics_process_missing_or_malformed_telemetry_rejected(self):
        settings = dict(self.manifest["run"], gpu_uuid=UUID)
        for section in ("<processes><process_info><pid>42</pid><type>G</type></process_info></processes>",
                        "", "<processes>N/A</processes>", "<processes><unexpected/></processes>"):
            xml = f"<nvidia_smi_log><gpu><uuid>{UUID}</uuid>{section}</gpu></nvidia_smi_log>"
            with self.subTest(section=section), patch.object(h3, "invoke", side_effect=self.telemetry(xml=xml)), \
                 self.assertRaises(h3.WorkflowError):
                h3.gpu_precheck(settings)
        for xml in ("not XML", "<nvidia_smi_log/>", "<nvidia_smi_log><gpu><uuid>GPU-wrong</uuid><processes/></gpu></nvidia_smi_log>"):
            with patch.object(h3, "invoke", side_effect=self.telemetry(xml=xml)), self.assertRaises(h3.WorkflowError):
                h3.gpu_precheck(settings)

    def test_baseline_only_default_and_explicit_full_workflow_later_dry_run(self):
        for suffix, args, phases, count in (("first", [], ["baseline"], 1),
                ("later", ["--phase", "all", "--warmup", "1", "--repeats", "3"], ["baseline", "candidate"], 4)):
            output = self.tmp / suffix
            with patch.object(h3, "run_phase", wraps=h3.run_phase) as run, \
                 patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU")), \
                 patch.object(h3, "verify_inputs", side_effect=AssertionError("no hash")), \
                 patch.object(h3, "compare", side_effect=AssertionError("no compare")), \
                 patch.object(h3, "git_provenance", return_value={}):
                rc = h3.main(["workflow", "--manifest", str(T2VA), "--dry-run", "--output-dir", str(output), *args])
            self.assertEqual(rc, 0)
            self.assertEqual([call.args[1] for call in run.call_args_list], phases)
            for phase in phases:
                report = h3.read_json(output / phase / "report.json")
                self.assertEqual(len(report["modes"]["t2va"]), count)
                self.assertEqual(sum(r["warmup"] for r in report["modes"]["t2va"]), 0 if count == 1 else 1)
                self.assertFalse(report["performance_measured"])
                self.assertIsNone(report["statistics"])
            self.assertFalse(h3.read_json(output / "workflow.json")["performance_measured"])

    def test_mocked_baseline_measures_only_baseline_and_checks_before_each_launch(self):
        # All external activity is mocked; timings here are test data only.
        for warmup, repeats in ((0, 1), (1, 3)):
            output = self.tmp / f"mocked-{warmup}-{repeats}"
            with ExitStack() as stack:
                for name, result in (("verify_inputs", {}), ("input_stat_signatures", {}), ("validate_assets", None),
                                     ("build_provenance", {}), ("validate_build", {}), ("git_provenance", {}),
                                     ("validate_media", {"sha256": "mock"})):
                    stack.enter_context(patch.object(h3, name, return_value=result))
                stack.enter_context(patch.object(h3, "compare", side_effect=AssertionError("no compare")))
                stack.enter_context(patch.object(h3, "invoke", return_value=subprocess.CompletedProcess([], 0, "mock tools", "")))
                gpu = stack.enter_context(patch.object(h3, "gpu_precheck", return_value={"uuid": UUID}))
                child = stack.enter_context(patch.object(h3.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, PROFILE, "")))
                rc = h3.main(["workflow", "--manifest", str(T2VA), "--warmup", str(warmup), "--repeats", str(repeats),
                              "--output-dir", str(output)])
            self.assertEqual(rc, 0)
            self.assertEqual(child.call_count, warmup + repeats)
            self.assertEqual(gpu.call_count, 1 + warmup + repeats)
            for call in child.call_args_list:
                self.assertFalse(call.kwargs["shell"])
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], UUID)
                self.assertTrue(all(call.kwargs["env"][key] == "0" for key in h3.OPTIONS))
                self.assertEqual(call.args[0][0], self.manifest["binaries"]["baseline"]["path"])
            summary = h3.read_json(output / "workflow.json")
            self.assertEqual(summary["status"], "baseline_only_validated")
            self.assertTrue(summary["baseline_performance_measured"])
            self.assertFalse(summary["performance_measured"])
            self.assertIsNone(summary["candidate"])
            self.assertIsNone(summary["compare"])
            self.assertFalse((output / "candidate").exists())
            baseline = h3.read_json(output / "baseline/report.json")
            self.assertTrue(baseline["performance_measured"])
            self.assertEqual(len(baseline["modes"]["t2va"]), warmup + repeats)
            self.assertEqual(baseline["statistics"]["t2va"]["cli_wall_ms"]["count"], repeats)


if __name__ == "__main__":
    unittest.main()
