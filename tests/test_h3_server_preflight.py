"""Targeted, offline server-preflight audit of the existing H3 workflow.

Run: python3 -B -m unittest discover -s tests -p test_h3_server_preflight.py -v

Only tiny synthetic weights/assets are opened. The *real* full-hash/header
verification functions run against a synthetic precision lock retaining all
five component names/types; unlike the older suite, full_sha256 is never
removed from that lock. External commands are forbidden except mocked
telemetry and mocked generation. Temporary artifacts are retained, not deleted.

Regression assertions cover fail-closed lock, telemetry, report serialization
and per-launch input stat checks. No expected-failure markers or test-side GPU
snapshot normalization hide production behavior. Boundary tests distinguish
A/B binding from original-fixture/primary-recipe continuity and snapshot checks from
exclusive reservation or a proven peak-VRAM budget. No new production workflow
or preflight implementation is introduced here.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_preflight_audit_workflow", ROOT / "scripts/h3_cuda_workflow.py")
h3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h3)
READ_JSON = h3.read_json
PRECISION_PATH = ROOT / "docs/research/h3-precision-lock.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(uuid="GPU-test-A", *, free=140000, used=3000, total=143000, utilization=0, processes=""):
    return (f"0, {uuid}, H200, {total}, {free}, {used}, {utilization}\n", processes)


class ServerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="h3-server-preflight-tiny-"))
        self.repo_manifest = h3.resolve_manifest(ROOT / "benchmark/h3_cuda/example.json")
        self.manifest = copy.deepcopy(self.repo_manifest)
        self.precision = copy.deepcopy(READ_JSON(PRECISION_PATH))
        self.model_root = self.tmp / "models"
        self.asset_root = self.tmp / "assets"
        self.model_root.mkdir()
        self.asset_root.mkdir()
        for key, pinned in zip(h3.COMPONENTS, self.precision["components"]):
            header = ("tiny-header-" + key).encode()
            if "header_bytes" in pinned:
                pinned["header_bytes"] = len(header)
                body = header
            else:
                body = struct.pack("<Q", len(header)) + header
            body += ("::tiny-payload-" + key).encode() * 9
            path = self.model_root / pinned["component"]
            with path.open("xb") as stream:
                stream.write(body)
            pinned.update(bytes=len(body), header_sha256=digest(header), full_sha256=digest(body))
            self.manifest["models"][key].update(
                path=str(path), bytes=len(body), sha256=digest(body),
                source_header_sha256=digest(header))
        for key, item in self.manifest["assets"].items():
            path = self.asset_root / key
            if item["kind"] == "frames":
                path.mkdir()
                for name in ("001.png", "002.png"):
                    with (path / name).open("xb") as stream:
                        stream.write(("tiny-frame-" + name).encode())
            else:
                with path.open("xb") as stream:
                    stream.write(("tiny-asset-" + key).encode())
            item.update(path=str(path), **h3.fingerprint(path))
        self.manifest["path_variables"].update(model_root=str(self.model_root), asset_root=str(self.asset_root))
        self.manifest["locations"]["host"].update(model_root=str(self.model_root), asset_root=str(self.asset_root))

        def read_json(path):
            return copy.deepcopy(self.precision) if Path(path).resolve() == PRECISION_PATH.resolve() else READ_JSON(path)

        # Guard even against accidental opening of a production weight by a
        # future test/refactor. No weight-header/hash function is mocked out.
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.suffix in (".gguf", ".safetensors"):
                self.assertTrue(path.resolve().is_relative_to(self.tmp), "production weight access forbidden")
            return original_open(path, *args, **kwargs)

        for guard in (
            patch.object(h3, "read_json", side_effect=read_json),
            patch.object(Path, "open", new=guarded_open),
            patch.object(h3, "invoke", side_effect=AssertionError("external commands forbidden")),
            patch.object(h3.subprocess, "run", side_effect=AssertionError("real subprocess/GPU forbidden")),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        h3.validate_manifest(self.manifest)

    def corrupt_tail(self, key="ref2va"):
        path = Path(self.manifest["models"][key]["path"])
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
        return path

    def check_snapshot(self, data):
        rows, processes = data
        with patch.object(h3, "invoke", side_effect=[
            subprocess.CompletedProcess([], 0, rows, ""),
            subprocess.CompletedProcess([], 0, processes, ""),
        ]) as invoke:
            result = h3.gpu_precheck(self.manifest["run"])
            self.assertEqual(invoke.call_count, 2)
            for call in invoke.call_args_list:
                self.assertEqual(call.args[0][0], "nvidia-smi")
                self.assertTrue(any(arg.startswith("--query-") for arg in call.args[0]))
            return result

    @contextmanager
    def phase_mocks(self, snapshots, *, fixed=None, after_launch=None, after_check=None):
        # Exercise the real GPU parser/checker and raw report structures. Only
        # external tools, build/media operations and generation are mocked.
        telemetry = iter(snapshots)
        state = {"events": [], "launches": [], "processes": None}

        def invoke(command, *_args, **_kwargs):
            if command[0] in ("ffmpeg", "ffprobe"):
                self.assertEqual(command[1:], ["-version"])
                return subprocess.CompletedProcess(command, 0, "mock tool version", "")
            self.assertEqual(command[0], "nvidia-smi")
            if command[1].startswith("--query-gpu="):
                rows, state["processes"] = next(telemetry)
                state["events"].append("check")
                if after_check:
                    after_check(state["events"].count("check"))
                return subprocess.CompletedProcess(command, 0, rows, "")
            self.assertTrue(command[1].startswith("--query-compute-apps="))
            self.assertIsNotNone(state["processes"])
            processes, state["processes"] = state["processes"], None
            return subprocess.CompletedProcess(command, 0, processes, "")

        def fake_launch(command, **kwargs):
            self.assertIn(command[0], [v["path"] for v in self.manifest["binaries"].values()])
            self.assertFalse(kwargs["shell"])
            state["events"].append("launch")
            state["launches"].append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            if after_launch:
                after_launch(len(state["launches"]))
            return subprocess.CompletedProcess(command, 0, "mock generation only", "")

        with ExitStack() as stack:
            for mock in (
                patch.object(h3, "invoke", side_effect=invoke),
                patch.object(h3.subprocess, "run", side_effect=fake_launch),
                patch.object(h3, "git_provenance", return_value={}),
                patch.object(h3, "build_provenance", return_value={}),
                patch.object(h3, "validate_build", return_value={}),
                patch.object(h3, "validate_assets"),
                patch.object(h3, "parse_profiles", return_value={"summaries": [{"steps": 20, "total_ms": 100}]}),
                patch.object(h3, "validate_media", return_value={"sha256": "a" * 64}),
            ):
                stack.enter_context(mock)
            if fixed is not None:
                stack.enter_context(patch.object(h3, "frozen_report", return_value=fixed))
            yield state

    def baseline_stub(self, uuid="GPU-test-A"):
        env, _ = h3.isolated_env(self.manifest, "baseline")
        return {"phase": "baseline", "contract": h3.contract(self.manifest, ["ref_mixed"]),
                "gpu": {"uuid": uuid, "name": "H200", "total_mib": 143000},
                "environment": {"other_runtime_env": {k: v for k, v in env.items()
                    if k.startswith(("CUDA_", "CUBLAS_", "CUDNN_", "NVIDIA_", "NCCL_", "OMP_", "MKL_"))
                    and k != "CUDA_VISIBLE_DEVICES"}}, "performance_measured": False}

    def test_all_five_tiny_models_bind_full_precision_lock(self):
        identities = h3.verify_inputs(self.manifest)
        for key, pinned in zip(h3.COMPONENTS, self.precision["components"]):
            self.assertEqual(identities["models"][key]["sha256"], pinned["full_sha256"])
            self.assertLess(identities["models"][key]["bytes"], 1024)

    def test_real_lock_rejects_wrong_full_hash_for_each_component_without_opening_weights(self):
        with patch.object(h3, "read_json", side_effect=READ_JSON):
            for key in h3.COMPONENTS:
                data = copy.deepcopy(self.repo_manifest)
                data["models"][key]["sha256"] = "1" * 64
                with self.subTest(key=key), self.assertRaisesRegex(h3.WorkflowError, "pinned full SHA256"):
                    h3.validate_manifest(data)

    def test_null_hash_wrong_payload_never_emits_a_locked_manifest(self):
        self.corrupt_tail()
        self.manifest["models"]["ref2va"]["sha256"] = None
        target = self.tmp / "inconsistent-lock.json"
        with self.assertRaises(h3.WorkflowError):
            h3.lock_manifest(self.manifest, target)
        self.assertFalse(target.exists())
        self.assertNotEqual(self.manifest.get("hash_status"), "full_file_sha256_locked")

    def test_missing_model_or_asset_full_hash_blocks_verification(self):
        for group, key in (("models", "ref2va"), ("assets", "reference_image")):
            data = copy.deepcopy(self.manifest)
            data[group][key]["sha256"] = None
            with self.subTest(group=group), self.assertRaisesRegex(h3.WorkflowError, "full SHA256 required"):
                h3.verify_inputs(data)

    def test_same_size_same_header_payload_corruption_is_rejected(self):
        self.corrupt_tail()
        h3.verify_precision_header("ref2va", self.manifest["models"]["ref2va"])
        with self.assertRaisesRegex(h3.WorkflowError, "full-file asset/weight hash mismatch"):
            h3.verify_inputs(self.manifest)

    def test_rehashed_wrong_payload_still_rejected_by_precision_lock(self):
        path = self.corrupt_tail()
        self.manifest["models"]["ref2va"].update(h3.fingerprint(path))
        with self.assertRaisesRegex(h3.WorkflowError, "pinned full SHA256"):
            h3.validate_manifest(self.manifest)
        with self.assertRaisesRegex(h3.WorkflowError, "pinned full-file SHA256"):
            h3.verify_inputs(self.manifest)

    def test_lock_rejects_corruption_when_manifest_digest_is_already_pinned(self):
        self.corrupt_tail()
        target = self.tmp / "must-not-lock.json"
        with self.assertRaisesRegex(h3.WorkflowError, "existing full hash mismatch"):
            h3.lock_manifest(self.manifest, target)
        self.assertFalse(target.exists())

    def test_lock_with_null_manifest_hash_rejects_wrong_locked_payload_before_writing(self):
        # A matching header never substitutes for the pinned full-file hash,
        # even when the caller has not populated the manifest digest yet.
        self.corrupt_tail()
        self.manifest["models"]["ref2va"]["sha256"] = None
        target = self.tmp / "must-not-lock-null.json"
        with self.assertRaises(h3.WorkflowError):
            h3.lock_manifest(self.manifest, target)
        self.assertFalse(target.exists())

    def test_lock_hashes_every_tiny_file_and_refuses_existing_output(self):
        data = copy.deepcopy(self.manifest)
        for group in ("models", "assets"):
            for item in data[group].values():
                item["sha256"] = None
        target = self.tmp / "locked.json"
        h3.lock_manifest(data, target)
        locked = READ_JSON(target)
        self.assertEqual(h3.verify_inputs(locked), h3.verify_inputs(self.manifest))
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            h3.lock_manifest(copy.deepcopy(self.manifest), target)
        self.assertEqual(target.read_bytes(), original)

    def test_full_hash_contract_survives_model_and_asset_root_remapping(self):
        new_models, new_assets = self.tmp / "moved-models", self.tmp / "moved-assets"
        for old_root, new_root in ((self.model_root, new_models), (self.asset_root, new_assets)):
            new_root.mkdir()
            for original in sorted(old_root.rglob("*")):
                destination = new_root / original.relative_to(old_root)
                if original.is_dir():
                    destination.mkdir()
                else:
                    with destination.open("xb") as stream:
                        stream.write(original.read_bytes())
        source = self.tmp / "portable.json"
        h3.write_json(source, self.manifest, exclusive=True)
        moved = h3.resolve_manifest(source, overrides={"model_root": new_models, "asset_root": new_assets})
        self.assertEqual(h3.contract(moved, list(h3.MODE_ORDER)), h3.contract(self.manifest, list(h3.MODE_ORDER)))
        self.assertEqual(h3.verify_inputs(moved), h3.verify_inputs(self.manifest))
        self.assertTrue(all(Path(v["path"]).is_relative_to(new_models) for v in moved["models"].values()))

    def test_fixed_contract_binds_seed_recipe_precision_placement_and_input_order(self):
        original = h3.contract(self.manifest, ["ref_mixed"])
        mutations = (
            lambda d: d["sampling"].update(seed=43),
            lambda d: d["sampling"].update(steps=21),
            lambda d: d["sampling"].update(sampler="euler"),
            lambda d: d["placement"].update(vae_offload=True),
            lambda d: d["models"]["llm"].update(tensor_types={"F16": 1}),
            lambda d: d["assets"]["reference_image"].update(sha256="1" * 64),
            lambda d: d["modes"]["ref_mixed"]["inputs"].update(**{"ref-audio": ["paired_audio", "extra_audio"]}),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            self.assertNotEqual(original, h3.contract(changed, ["ref_mixed"]))
        two = copy.deepcopy(self.manifest)
        two["modes"]["ref_mixed"]["inputs"]["ref-audio"] = ["extra_audio", "paired_audio"]
        before = copy.deepcopy(h3.contract(two, ["ref_mixed"]))
        two["modes"]["ref_mixed"]["inputs"]["ref-audio"].reverse()
        self.assertNotEqual(before, h3.contract(two, ["ref_mixed"]))

    def test_primary_recipe_is_configurable_not_implicitly_original_baseline(self):
        # Legal alternate recipes are accepted, but cannot share the old A/B
        # contract. This is not proof of primary_task continuation.
        changed = copy.deepcopy(self.manifest)
        changed["sampling"].update(width=256, height=256, frames=22, steps=21, seed=43, sampler="euler")
        h3.validate_manifest(changed)
        self.assertNotEqual(h3.contract(changed, ["ref_mixed"]), h3.contract(self.manifest, ["ref_mixed"]))

    def test_original_fixture_manifest_digest_is_not_a_runtime_gate(self):
        self.precision["fixture_manifest_sha256"] = "f" * 64
        source = self.tmp / "not-original-fixture.json"
        h3.write_json(source, self.manifest, exclusive=True)
        self.assertNotEqual(h3.sha256(source), self.precision["fixture_manifest_sha256"])
        loaded = h3.resolve_manifest(source)
        h3.verify_inputs(loaded)  # binds this manifest's assets, not original manifest bytes

    def test_changed_candidate_contract_blocks_before_gpu_or_launch(self):
        fixed = self.baseline_stub()
        fixed["contract"] = copy.deepcopy(fixed["contract"])
        fixed["contract"]["sampling"]["seed"] = 43
        with self.phase_mocks([], fixed=fixed) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "contract differs"):
                h3.run_phase(self.manifest, "candidate", self.tmp / "candidate", ["ref_mixed"], baseline_path=self.tmp / "base.json")
            self.assertEqual(state["events"], [])

    def test_precheck_is_two_readonly_queries_and_writes_nothing(self):
        before = sorted(self.tmp.rglob("*"))
        gpu = self.check_snapshot(snapshot())
        self.assertEqual(gpu["uuid"], "GPU-test-A")
        self.assertEqual(sorted(self.tmp.rglob("*")), before)

    def test_raw_gpu_snapshot_is_serializable_for_preflight_reports(self):
        gpu = self.check_snapshot(snapshot())
        encoded = json.dumps(gpu, allow_nan=False)
        self.assertEqual(json.loads(encoded)["uuid"], "GPU-test-A")
        self.assertTrue(all(item is not gpu for item in gpu["all_gpus"]))

    def test_raw_precheck_can_be_recorded_before_a_mock_launch(self):
        count = self.manifest["run"]["warmup"] + self.manifest["run"]["repeats"]
        with self.phase_mocks([snapshot()] * (count + 1)) as state:
            path = h3.run_phase(self.manifest, "baseline", self.tmp / "raw-report", ["ref_mixed"])
            self.assertEqual(len(state["launches"]), count)
            self.assertEqual(READ_JSON(path)["status"], "validated")

    def test_raw_busy_gpu_failure_details_are_serializable(self):
        try:
            self.check_snapshot(snapshot(processes="GPU-test-A, 123, 1, another-user\n"))
        except h3.WorkflowError as exc:
            json.dumps(exc.details, allow_nan=False)
        else:
            self.fail("busy GPU must be rejected")

    def test_raw_busy_snapshot_blocks_launch_and_preserves_valid_failure_report(self):
        busy = snapshot(processes="GPU-test-A, 123, 1, another-user\n")
        with self.phase_mocks([busy]) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "occupied"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "raw-failure", ["ref_mixed"])
        self.assertEqual(state["launches"], [])
        report = READ_JSON(self.tmp / "raw-failure/report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["performance_measured"])
        self.assertTrue(report["gpu_precheck_failure"]["compute_processes"])

    def test_compute_process_blocks_even_with_enough_free_memory(self):
        with self.assertRaisesRegex(h3.WorkflowError, "occupied"):
            self.check_snapshot(snapshot(processes="GPU-test-A, 123, 1, another-user\n"))

    def test_utilization_blocks_without_visible_compute_processes(self):
        with self.assertRaisesRegex(h3.WorkflowError, "not idle"):
            self.check_snapshot(snapshot(utilization=1))

    def test_reported_new_server_busy_h200s_are_rejected_despite_103_gib_free(self):
        # User-provided snapshot values only; never queries these actual GPUs.
        for utilization in (99, 100):
            with self.subTest(utilization=utilization), self.assertRaisesRegex(h3.WorkflowError, "not idle"):
                self.check_snapshot(snapshot(free=103 * 1024, used=143000 - 103 * 1024,
                                             utilization=utilization))

    def test_free_memory_floor_is_enforced(self):
        with self.assertRaisesRegex(h3.WorkflowError, "free memory"):
            self.check_snapshot(snapshot(free=65535, used=77465))

    def test_unavailable_and_malformed_telemetry_fail_closed(self):
        for rows, processes in (
            ("0, GPU-test-A, H200, N/A, N/A, N/A, N/A\n", ""),
            ("0, GPU-test-A, H200, 143000\n", ""),
            (snapshot()[0], "unparseable process row\n"),
            ("", ""),
        ):
            with self.subTest(rows=rows), self.assertRaises(h3.WorkflowError):
                self.check_snapshot((rows, processes))

    def test_query_failure_is_not_treated_as_idle(self):
        with patch.object(h3, "invoke", side_effect=h3.WorkflowError("telemetry unavailable")):
            with self.assertRaisesRegex(h3.WorkflowError, "telemetry unavailable"):
                h3.gpu_precheck(self.manifest["run"])

    def test_unattributable_compute_process_telemetry_is_not_silently_idle(self):
        with self.assertRaises(h3.WorkflowError):
            self.check_snapshot(snapshot(processes="N/A, 123, N/A, hidden-user\n"))

    def test_configured_free_floor_is_not_a_model_peak_vram_budget(self):
        # Valid telemetry alone does not prove model capacity or exclusivity.
        # Large unoccupied-by-compute-apps usage is not assigned a new arbitrary
        # ceiling: the floor is a configured threshold, not a peak-VRAM budget.
        gpu = self.check_snapshot(snapshot(free=66000, used=77000))
        self.assertEqual(gpu["used_mib"], 77000)
        self.assertEqual(gpu["compute_processes"], [])
        self.manifest["run"]["min_free_mib"] = 1
        h3.validate_manifest(self.manifest)
        gpu = self.check_snapshot(snapshot(total=2, free=2, used=0))
        self.assertEqual(gpu["total_mib"], 2)

    def test_impossible_free_greater_than_total_is_rejected(self):
        with self.assertRaises(h3.WorkflowError):
            self.check_snapshot(snapshot(total=100, free=140000, used=0))

    def test_negative_used_memory_is_rejected(self):
        with self.assertRaises(h3.WorkflowError):
            self.check_snapshot(snapshot(used=-1))

    def test_negative_utilization_is_rejected(self):
        with self.assertRaises(h3.WorkflowError):
            self.check_snapshot(snapshot(utilization=-1))

    def test_missing_gpu_uuid_is_rejected(self):
        with self.assertRaises(h3.WorkflowError):
            self.check_snapshot(snapshot(uuid=""))

    def test_gpu_identity_and_telemetry_range_matrix_is_rejected(self):
        valid = snapshot()[0]
        rows = (
            valid.replace("0, GPU-test-A", "-1, GPU-test-A"),
            snapshot(uuid="N/A")[0],
            snapshot(uuid="GPU-")[0],
            valid + valid.replace("0, GPU-test-A", "1, GPU-test-A"),
            valid + valid.replace("GPU-test-A", "GPU-test-B"),
            snapshot(total=0, free=0, used=0)[0],
            snapshot(total=-1, free=0, used=0)[0],
            snapshot(free=-1)[0],
            snapshot(used=143001)[0],
            snapshot(utilization=101)[0],
        )
        for row in rows:
            with self.subTest(row=row), self.assertRaises(h3.WorkflowError):
                self.check_snapshot((row, ""))

    def test_process_telemetry_requires_known_uuid_positive_pid_and_valid_memory(self):
        for uuid, pid, memory in (
            ("GPU-unknown", "123", "1"),
            ("N/A", "123", "N/A"),
            ("GPU-test-A", "0", "1"),
            ("GPU-test-A", "-1", "1"),
            ("GPU-test-A", "not-a-pid", "1"),
            ("GPU-test-A", "1.5", "1"),
            ("GPU-test-A", "123", "-1"),
            ("GPU-test-A", "123", "invalid"),
        ):
            row = f"{uuid}, {pid}, {memory}, another-user\n"
            with self.subTest(row=row), self.assertRaises(h3.WorkflowError):
                self.check_snapshot(snapshot(processes=row))

    def test_known_compute_process_with_na_memory_still_blocks(self):
        for memory in ("N/A", "[N/A]", "0"):
            with self.subTest(memory=memory), self.assertRaisesRegex(h3.WorkflowError, "occupied"):
                self.check_snapshot(snapshot(processes=f"GPU-test-A, 123, {memory}, another-user\n"))

    def test_other_known_gpu_process_does_not_occupy_selected_gpu(self):
        rows = snapshot()[0] + snapshot(uuid="GPU-test-B")[0].replace("0, GPU-test-B", "1, GPU-test-B")
        gpu = self.check_snapshot((rows, "GPU-test-B, 123, N/A, another-user\n"))
        self.assertEqual(gpu["uuid"], "GPU-test-A")
        self.assertEqual(gpu["compute_processes"], [])

    def test_malformed_process_on_other_gpu_is_not_ignored(self):
        rows = snapshot()[0] + snapshot(uuid="GPU-test-B")[0].replace("0, GPU-test-B", "1, GPU-test-B")
        with self.assertRaises(h3.WorkflowError):
            self.check_snapshot((rows, "GPU-test-B, -1, 1, another-user\n"))

    def test_initial_selection_is_by_index_not_a_predeclared_uuid(self):
        # UUID gets pinned by the first snapshot/report, not by precision lock.
        gpu = self.check_snapshot(snapshot(uuid="GPU-new-server"))
        self.assertEqual(gpu["uuid"], "GPU-new-server")
        self.assertNotIn("gpu_uuid", self.manifest["run"])

    def test_rechecks_before_every_warmup_and_repeat_and_launches_by_uuid(self):
        count = self.manifest["run"]["warmup"] + self.manifest["run"]["repeats"]
        with self.phase_mocks([snapshot()] * (count + 1)) as state:
            path = h3.run_phase(self.manifest, "baseline", self.tmp / "baseline", ["ref_mixed"])
        self.assertEqual(state["events"], ["check"] + ["check", "launch"] * count)
        self.assertEqual(state["launches"], ["GPU-test-A"] * count)
        report = READ_JSON(path)
        self.assertIn("not an exclusive reservation", report["gpu_safety_note"])
        self.assertIn("scheduler ownership still required", report["gpu_safety_note"])
        self.assertTrue(all(r["gpu_precheck"]["uuid"] == "GPU-test-A" for r in report["modes"]["ref_mixed"]))

    def test_uuid_change_before_first_launch_blocks(self):
        with self.phase_mocks([snapshot(), snapshot(uuid="GPU-test-B")]) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "GPU identity changed"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "baseline", ["ref_mixed"])
        self.assertEqual(state["launches"], [])
        self.assertFalse(READ_JSON(self.tmp / "baseline/report.json")["performance_measured"])

    def test_uuid_change_before_repeat_blocks_remaining_launches(self):
        with self.phase_mocks([snapshot(), snapshot(), snapshot(uuid="GPU-test-B")]) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "GPU identity changed"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "baseline", ["ref_mixed"])
        self.assertEqual(state["launches"], ["GPU-test-A"])

    def test_gpu_becoming_busy_before_repeat_blocks_and_preserves_telemetry(self):
        busy = snapshot(processes="GPU-test-A, 321, 1, another-user\n")
        with self.phase_mocks([snapshot(), snapshot(), busy]) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "occupied"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "baseline", ["ref_mixed"])
        self.assertEqual(state["launches"], ["GPU-test-A"])
        report = READ_JSON(self.tmp / "baseline/report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["performance_measured"])
        self.assertIsNone(report["statistics"])
        self.assertTrue(report["gpu_precheck_failure"]["compute_processes"])

    def test_free_memory_drop_before_first_launch_blocks(self):
        with self.phase_mocks([snapshot(), snapshot(free=65535, used=77465)]) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "free memory"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "baseline", ["ref_mixed"])
        self.assertEqual(state["launches"], [])

    def test_candidate_requires_baseline_gpu_uuid_name_and_total(self):
        for field, value in (("uuid", "GPU-other"), ("name", "other-card"), ("total_mib", 1)):
            fixed = self.baseline_stub()
            fixed["gpu"][field] = value
            base = self.tmp / (field + "-base.json")
            h3.write_json(base, {"mock": True}, exclusive=True)
            with self.subTest(field=field), self.phase_mocks([snapshot()], fixed=fixed) as state:
                with self.assertRaisesRegex(h3.WorkflowError, "candidate GPU differs"):
                    h3.run_phase(self.manifest, "candidate", self.tmp / (field + "-candidate"),
                                 ["ref_mixed"], baseline_path=base)
                self.assertEqual(state["launches"], [])

    def test_changed_model_after_initial_full_hash_blocks_next_launch(self):
        # The next launch must compare against the post-hash stat snapshot;
        # no per-launch full-weight rehash is needed to detect this mutation.
        def mutate_after_first(count):
            if count == 1:
                self.corrupt_tail()

        count = self.manifest["run"]["warmup"] + self.manifest["run"]["repeats"]
        with self.phase_mocks([snapshot()] * (count + 1), after_launch=mutate_after_first) as state:
            with self.assertRaises(h3.WorkflowError):
                h3.run_phase(self.manifest, "baseline", self.tmp / "baseline", ["ref_mixed"])
            self.assertEqual(state["launches"], ["GPU-test-A"])

    def assert_input_change_blocks_next_launch(self, mutate, output_name):
        count = self.manifest["run"]["warmup"] + self.manifest["run"]["repeats"]

        def after_first(launches):
            if launches == 1:
                mutate()

        with self.phase_mocks([snapshot()] * (count + 1), after_launch=after_first) as state:
            with self.assertRaises(h3.WorkflowError):
                h3.run_phase(self.manifest, "baseline", self.tmp / output_name, ["ref_mixed"])
        self.assertEqual(state["launches"], ["GPU-test-A"])
        report = READ_JSON(self.tmp / output_name / "report.json")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["performance_measured"])
        self.assertIsNone(report["statistics"])

    def test_same_size_model_change_with_restored_mtime_is_detected_by_ctime(self):
        path = Path(self.manifest["models"]["ref2va"]["path"])
        before = path.stat()

        def mutate():
            self.corrupt_tail()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = path.stat()
            self.assertEqual((after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns))
            self.assertNotEqual(after.st_ctime_ns, before.st_ctime_ns)

        self.assert_input_change_blocks_next_launch(mutate, "ctime-change")

    def test_identical_model_replacement_with_restored_mtime_is_detected_by_inode(self):
        path = Path(self.manifest["models"]["ref2va"]["path"])
        before = path.stat()
        original = path.read_bytes()

        def mutate():
            path.rename(self.model_root / "retained-original.gguf")
            with path.open("xb") as stream:
                stream.write(original)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            self.assertNotEqual(path.stat().st_ino, before.st_ino)

        self.assert_input_change_blocks_next_launch(mutate, "inode-change")

    def test_device_signature_change_blocks_even_with_other_stat_fields_unchanged(self):
        from types import SimpleNamespace
        path = Path(self.manifest["models"]["ref2va"]["path"])
        stat = Path.stat
        changed = {"value": False}

        def fake_stat(current, *args, **kwargs):
            result = stat(current, *args, **kwargs)
            if current == path and changed["value"]:
                values = {name: getattr(result, name) for name in dir(result) if name.startswith("st_")}
                values["st_dev"] += 1
                return SimpleNamespace(**values)
            return result

        with patch.object(Path, "stat", new=fake_stat):
            self.assert_input_change_blocks_next_launch(lambda: changed.update(value=True), "device-change")

    def test_asset_payload_change_blocks_next_launch(self):
        path = Path(self.manifest["assets"]["paired_audio"]["path"])

        def mutate():
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                old = stream.read(1)
                stream.seek(-1, 2)
                stream.write(bytes([old[0] ^ 1]))

        self.assert_input_change_blocks_next_launch(mutate, "asset-change")

    def test_frame_payload_change_blocks_next_launch(self):
        path = Path(self.manifest["assets"]["reference_frames"]["path"]) / "001.png"

        def mutate():
            with path.open("r+b") as stream:
                stream.write(b"X")

        self.assert_input_change_blocks_next_launch(mutate, "frame-change")

    def test_added_frame_blocks_next_launch(self):
        path = Path(self.manifest["assets"]["reference_frames"]["path"]) / "003.png"

        def mutate():
            with path.open("xb") as stream:
                stream.write(b"tiny-added-frame")

        self.assert_input_change_blocks_next_launch(mutate, "added-frame")

    def test_missing_frame_blocks_next_launch_without_deleting_file(self):
        path = Path(self.manifest["assets"]["reference_frames"]["path"]) / "001.png"
        self.assert_input_change_blocks_next_launch(
            lambda: path.rename(self.tmp / "retained-frame.png"), "missing-frame")

    def test_model_size_change_blocks_next_launch(self):
        path = Path(self.manifest["models"]["ref2va"]["path"])

        def mutate():
            with path.open("ab") as stream:
                stream.write(b"tiny-extra-byte")

        self.assert_input_change_blocks_next_launch(mutate, "size-change")

    def test_input_change_after_hash_before_first_launch_is_rejected(self):
        def mutate_at_first_snapshot(checks):
            if checks == 1:
                self.corrupt_tail()

        with self.phase_mocks([snapshot(), snapshot()], after_check=mutate_at_first_snapshot) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "input changed"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "before-first-launch", ["ref_mixed"])
        self.assertEqual(state["launches"], [])
        self.assertFalse(READ_JSON(self.tmp / "before-first-launch/report.json")["performance_measured"])

    def test_input_change_at_end_of_full_verification_is_rejected_before_gpu(self):
        verify = h3.verify_inputs

        def verify_then_mutate(manifest):
            identities = verify(manifest)
            self.corrupt_tail()
            return identities

        with patch.object(h3, "verify_inputs", side_effect=verify_then_mutate), self.phase_mocks([]) as state:
            with self.assertRaisesRegex(h3.WorkflowError, "input changed"):
                h3.run_phase(self.manifest, "baseline", self.tmp / "hash-window", ["ref_mixed"])
        self.assertEqual(state["events"], [])
        self.assertEqual(READ_JSON(self.tmp / "hash-window/report.json")["status"], "failed")

    def test_unmodified_inputs_are_not_fully_rehashed_before_each_launch(self):
        count = self.manifest["run"]["warmup"] + self.manifest["run"]["repeats"]
        real_sha256 = h3.sha256
        with patch.object(h3, "sha256", wraps=real_sha256) as hashes, \
             self.phase_mocks([snapshot()] * (count + 1)) as state:
            h3.run_phase(self.manifest, "baseline", self.tmp / "no-rehash", ["ref_mixed"])
        self.assertEqual(len(state["launches"]), count)
        for model in self.manifest["models"].values():
            path = Path(model["path"])
            self.assertEqual(sum(call.args[0] == path for call in hashes.call_args_list), 1)
        for asset in self.manifest["assets"].values():
            path = Path(asset["path"])
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in files:
                if file.is_file():
                    self.assertEqual(sum(call.args[0] == file for call in hashes.call_args_list), 1)

    def test_dry_run_reports_unverified_hashes_without_gpu_or_full_weight_reads(self):
        with patch.object(h3, "verify_inputs", side_effect=AssertionError("no hashing in dry-run")), \
             patch.object(h3, "gpu_precheck", side_effect=AssertionError("no GPU in dry-run")), \
             patch.object(h3, "git_provenance", return_value={}):
            path = h3.run_phase(self.manifest, "baseline", self.tmp / "dry", ["ref_mixed"], dry_run=True)
        report = READ_JSON(path)
        self.assertEqual(report["hash_status"], "not_verified_dry_run")
        self.assertIsNone(report["gpu"])
        self.assertFalse(report["performance_measured"])
        self.assertIsNone(report["statistics"])


if __name__ == "__main__":
    unittest.main()
