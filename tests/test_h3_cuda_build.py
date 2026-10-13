"""Standard-library-only offline tests: no CMake/nvcc/GPU/download/build writes.

Run: python3 -B tests/test_h3_cuda_build.py
Filesystem/build failure paths are mocked; command-line preview is exercised
against the real shell wrapper without requiring initialized dependencies.
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_build", ROOT / "scripts/h3_cuda_build_config.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ConfigTest(unittest.TestCase):
    def config(self, **overrides):
        env = {"H3_CUDA_ARCHITECTURES": "90", "H3_BUILD_ROOT": "_scratch/h3-offline-test-never-created",
               "H3_CUDA_ROOT": "/offline/cuda13", "H3_CUDNN_ROOT": "/offline/cudnn",
               "H3_CUDNN_FRONTEND_SOURCE": "/offline/frontend"}
        env.update(overrides)
        return m.config_from_env(ROOT, "all", env)

    def test_no_implicit_or_host_detected_architecture(self):
        for arch in ("", "native", "all", "all-major", "OFF", "0", "90;", "90 100", "90\n", "90;-DOTHER=ON"):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                self.config(H3_CUDA_ARCHITECTURES=arch)

    def test_explicit_architectures_not_restricted_to_h200(self):
        for arch in ("80", "89", "90", "100", "90a", "100f-real", "90-real;90-virtual", "80;90"):
            with self.subTest(arch=arch):
                cfg = self.config(H3_CUDA_ARCHITECTURES=arch)
                self.assertEqual(cfg["cuda_architectures"], arch)
                for phase in cfg["phases"]:
                    self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=" + arch, phase["configure"])

    def test_phase_common_options_are_identical(self):
        cfg = self.config()
        baseline, candidate = cfg["phases"]
        baseline_d = [a for a in baseline["configure"] if a.startswith("-D")]
        candidate_d = [a for a in candidate["configure"] if a.startswith("-D")]
        self.assertEqual(baseline_d[:-1], candidate_d[:-1])
        self.assertEqual(baseline_d[-1], "-DED_ENABLE_CUDA_SAGE_ATTN=OFF")
        self.assertEqual(candidate_d[-1], "-DED_ENABLE_CUDA_SAGE_ATTN=ON")
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", baseline_d)
        self.assertIn("-DFETCHCONTENT_UPDATES_DISCONNECTED=ON", baseline_d)
        self.assertEqual(cfg["dependency_pins"], m.PINS)

    def test_toolchain_and_cudnn_are_explicit_and_shared(self):
        cfg = self.config(H3_C_COMPILER="/offline/gcc12", H3_CXX_COMPILER="/offline/g++12",
                          H3_CUDNN_RUNTIME_LIBRARY_DIRS="/offline/nvrtc;/offline/runtime")
        options = cfg["common_cmake_args"]
        for expected in ("-DCMAKE_C_COMPILER=/offline/gcc12", "-DCMAKE_CXX_COMPILER=/offline/g++12",
                         "-DCMAKE_CUDA_HOST_COMPILER=/offline/g++12",
                         "-DCMAKE_CUDA_COMPILER=/offline/cuda13/bin/nvcc",
                         "-DCUDNN_INCLUDE_DIR=/offline/cudnn/include",
                         "-Dcudnn_LIBRARY=/offline/cudnn/lib/libcudnn.so",
                         "-DCUDNN_RUNTIME_LIBRARY_DIRS=/offline/nvrtc;/offline/runtime"):
            self.assertIn(expected, options)

    def test_make_generator_requires_no_ninja(self):
        cfg = self.config(H3_CMAKE_GENERATOR="Unix Makefiles")
        self.assertEqual(cfg["cmake_generator"], "Unix Makefiles")
        self.assertEqual(Path(cfg["make_program"]).name, "make")
        for phase in cfg["phases"]:
            self.assertEqual(phase["configure"][5:7], ["-G", "Unix Makefiles"])
            self.assertIn("-DCMAKE_MAKE_PROGRAM=" + cfg["make_program"], phase["configure"])
        self.assertEqual(self.config()["cmake_generator"], "Ninja")
        with self.assertRaises(ValueError):
            self.config(H3_CMAKE_GENERATOR="Other")

    def test_cudnn_component_cache_names_match_fixed_frontend(self):
        cfg = self.config()
        self.assertIn("-Dcudnn_LIBRARY=" + cfg["cudnn_library"], cfg["common_cmake_args"])
        self.assertFalse(any(a.startswith("-DCUDNN_LIBRARY=") for a in cfg["common_cmake_args"]))
        self.assertIn("-DGGML_CUDA_CUB_3DOT2=OFF", cfg["common_cmake_args"])

    def test_spaces_are_single_command_arguments(self):
        cfg = self.config(H3_CUDNN_FRONTEND_SOURCE="/offline/with spaces/frontend")
        self.assertIn("-DFETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND=/offline/with spaces/frontend",
                      cfg["phases"][0]["configure"])

    def test_temp_and_cache_are_in_new_project_scratch(self):
        cfg = self.config()
        env = cfg["environment"]
        self.assertEqual(env["TMPDIR"], str(ROOT / "_scratch/h3-offline-test-never-created/tmp"))
        self.assertEqual(env["TMP"], env["TMPDIR"])
        self.assertEqual(env["TEMP"], env["TMPDIR"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(env["CUDA_CACHE_PATH"].startswith(env["TMPDIR"] + "/"))

    def test_default_output_is_unique_not_old_build(self):
        env = {"H3_CUDA_ARCHITECTURES": "90"}
        one = m.config_from_env(ROOT, "all", env)
        two = m.config_from_env(ROOT, "all", env)
        self.assertNotEqual(one["build_root"], two["build_root"])
        self.assertIn(ROOT / "_scratch", Path(one["build_root"]).parents)

    def test_refuse_outputs_outside_project_scratch(self):
        for path in ("/tmp/h3-build", "build-h3-candidate", "_scratch", "_scratch/../outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.config(H3_BUILD_ROOT=path)

    def test_refuse_symlink_output_even_if_dangling(self):
        with patch.object(Path, "is_symlink", return_value=True), self.assertRaisesRegex(ValueError, "symlink"):
            self.config()

    def test_no_old_frontend_default_missing_is_explicit(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        self.assertEqual(set(cfg["missing_settings"]),
                         {"H3_CUDA_ROOT", "H3_CUDNN_ROOT", "H3_CUDNN_FRONTEND_SOURCE"})
        self.assertNotIn("build-h200-current-20260814", json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "Missing explicit local settings"):
            m.validate(cfg)

    def test_fixed_baseline_not_overrideable(self):
        with self.assertRaisesRegex(ValueError, "fixed research baseline"):
            self.config(H3_BASELINE_REF="HEAD")
        self.assertEqual(self.config(H3_BASELINE_REF=m.BASELINE)["baseline_revision"], m.BASELINE)

    def test_invalid_jobs_or_path_list_refused(self):
        for jobs in ("0", "-1", "2;4", "many"):
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                self.config(H3_BUILD_JOBS=jobs)
        with self.assertRaisesRegex(ValueError, "single path"):
            self.config(H3_CUDNN_FRONTEND_SOURCE="/one;/two")

    def test_existing_empty_output_refused_before_commands(self):
        cfg = self.config()
        with patch.object(Path, "exists", return_value=True), patch.object(m.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "existing build/output"):
                m.validate(cfg)
            run.assert_not_called()

    def test_cli_preview_without_dependencies_is_read_only(self):
        cfg = self.config()
        with patch.object(m, "config_from_env", return_value=cfg), patch.object(m, "validate") as validate, \
                patch.object(m, "build") as build, patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(m.main(["all", "--dry-run"]), 0)
        plan = json.loads(out.getvalue())
        self.assertFalse(plan["validation_performed"])
        validate.assert_not_called()
        build.assert_not_called()

    def test_cli_validate_does_not_build(self):
        cfg = self.config()
        with patch.object(m, "config_from_env", return_value=cfg), patch.object(m, "validate", return_value={}), \
                patch.object(m, "build") as build, patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(m.main(["--validate"]), 0)
        self.assertTrue(json.loads(out.getvalue())["validation_performed"])
        build.assert_not_called()

    def test_shell_preview_and_help_work_offline(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("H3_")}
        env.update(H3_CUDA_ARCHITECTURES="90", PYTHONDONTWRITEBYTECODE="1")
        command = ["bash", str(ROOT / "scripts/build_h3_cuda.sh")]
        preview = subprocess.run(command + ["all", "--preview"], env=env, text=True, capture_output=True, check=True)
        plan = json.loads(preview.stdout)
        self.assertEqual([p["phase"] for p in plan["phases"]], ["baseline", "candidate"])
        self.assertFalse(plan["validation_performed"])
        self.assertFalse(Path(plan["build_root"]).exists())
        help_result = subprocess.run(command + ["--help"], env=env, text=True, capture_output=True, check=True)
        self.assertIn("H3_CUDA_ARCHITECTURES", help_result.stdout)
        refused = subprocess.run(command + ["--validate"], env=env, text=True, capture_output=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("Missing explicit local settings", refused.stderr)


class IntegrityTest(unittest.TestCase):
    def test_uninitialized_submodule_cannot_inherit_parent_head(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(m, "git", return_value=str(ROOT)):
            with self.assertRaisesRegex(ValueError, "own Git checkout"):
                m.clean_pin(ROOT / "third_party/ggml", m.PINS["ggml"])

    def test_clean_git_pin_and_nested_status(self):
        path = ROOT / "third_party/ggml"
        for status in ("", " " + "a" * 40 + " nested (heads/main)"):
            replies = [str(path), m.PINS["ggml"], "", status]
            with patch.object(Path, "is_file", return_value=True), patch.object(m, "git", side_effect=replies):
                self.assertEqual(m.clean_pin(path, m.PINS["ggml"]), m.PINS["ggml"])

    def test_wrong_pin_dirty_tree_or_nested_submodule_refused(self):
        path = ROOT / "third_party/ggml"
        cases = [[str(path), "0" * 40], [str(path), m.PINS["ggml"], "?? local-file"],
                 [str(path), m.PINS["ggml"], "", "-" + "a" * 40 + " nested"]]
        for replies in cases:
            with self.subTest(replies=replies), patch.object(Path, "is_file", return_value=True), \
                    patch.object(m, "git", side_effect=replies), self.assertRaises(ValueError):
                m.clean_pin(path, m.PINS["ggml"])

    def test_git_status_leading_space_is_preserved(self):
        line = " " + "a" * 40 + " nested\n"
        with patch.object(m.subprocess, "check_output", return_value=line):
            self.assertEqual(m.git(ROOT, "submodule", "status"), line.rstrip("\n"))

    def test_changed_gitlink_is_refused(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        with patch.object(m, "git", return_value="160000 commit " + "0" * 40 + " third_party/ggml"):
            with self.assertRaisesRegex(ValueError, "gitlink changed"):
                m.dependency_state(cfg)

    def test_cmake_cache_target_or_dependency_drift_is_refused(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        expected = dict(arg[2:].split("=", 1) for arg in cfg["common_cmake_args"] if arg.startswith("-D"))
        expected["ED_ENABLE_CUDA_SAGE_ATTN"] = "OFF"
        for changed in (None, "CMAKE_CUDA_ARCHITECTURES", "cudnn_LIBRARY", "FETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND"):
            values = dict(expected)
            if changed:
                values[changed] = "wrong"
            cache = "\n".join(k + ":STRING=" + v for k, v in values.items())
            with self.subTest(changed=changed), patch.object(Path, "read_text", return_value=cache):
                if changed:
                    with self.assertRaisesRegex(ValueError, "configuration drift"):
                        m.check_cache(cfg, ROOT / "unused", "OFF")
                else:
                    m.check_cache(cfg, ROOT / "unused", "OFF")

    def test_validation_is_read_only_and_checks_real_local_files(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90", "H3_CUDA_ROOT": "/offline/cuda",
                         "H3_CUDNN_ROOT": "/offline/cudnn", "H3_CUDNN_FRONTEND_SOURCE": "/offline/frontend",
                         "H3_CMAKE_GENERATOR": "Unix Makefiles"})
        cfg["cudnn_libraries"].update(cudnn_graph_LIBRARY="/offline/cudnn/lib/libcudnn_graph.so.9",
                          cudnn_engines_runtime_compiled_LIBRARY="/offline/cudnn/lib/libcudnn_engines_runtime_compiled.so.9")
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_text", return_value="#define CUDNN_MAJOR 9"), \
                patch.object(os, "access", return_value=True), patch.object(m, "dependency_state", return_value={"fixed": True}), \
                patch.object(m.subprocess, "run") as run, patch.object(Path, "mkdir") as mkdir:
            self.assertEqual(m.validate(cfg), {"fixed": True})
        run.assert_not_called()
        mkdir.assert_not_called()

    def test_provenance_shared_fingerprint_excludes_output_paths(self):
        # Separate baseline/candidate invocations use different NEW roots, but
        # must still have the same configuration identity when all pins match.
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        streams = []
        def opened(*args, **kwargs):
            stream = MagicMock()
            stream.__enter__.return_value = stream
            streams.append(stream)
            return stream
        hashes = []
        for phase in cfg["phases"]:
            cfg["environment"]["TMPDIR"] = phase["destination"] + "/tmp"
            with patch.object(Path, "read_text", return_value="git_commit=old\nggml_commit=old\n"), \
                    patch.object(Path, "write_text"), patch.object(Path, "rglob", return_value=[]), \
                    patch.object(Path, "open", side_effect=opened), patch.object(m, "file_hash", return_value="a" * 64), \
                    patch.object(m.subprocess, "check_output", return_value=b""):
                m.write_provenance(cfg, {"candidate_revision": "b" * 40}, phase)
            provenance = streams[-2].write.call_args.args[0]
            hashes.append(next(line for line in provenance.splitlines() if line.startswith("shared_config_sha256=")))
        self.assertEqual(hashes[0], hashes[1])

    def test_build_stops_if_dependencies_change_before_phase(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        fake_stream = MagicMock()
        with patch.object(Path, "mkdir"), patch.object(Path, "open", return_value=fake_stream), \
                patch.object(m, "dependency_state", return_value={"changed": True}), \
                patch.object(m.subprocess, "run") as run, patch.object(m, "snapshot_baseline") as snapshot:
            with self.assertRaisesRegex(ValueError, "changed after validation"):
                m.build(cfg, {})
        run.assert_not_called()
        snapshot.assert_not_called()

    def test_mocked_build_uses_only_shared_config_no_gpu_and_project_tmp(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        fake_stream = MagicMock()
        with patch.object(Path, "mkdir"), patch.object(Path, "open", return_value=fake_stream), \
                patch.object(m, "dependency_state", return_value={}), patch.object(m, "snapshot_baseline"), \
                patch.object(m, "check_cache"), patch.object(m, "write_provenance") as provenance, \
                patch.object(m.subprocess, "run") as run:
            m.build(cfg, {})
        self.assertEqual(run.call_count, 4)
        self.assertEqual(provenance.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[0], cfg["phases"][0]["configure"][0])
            self.assertEqual(call.kwargs["env"]["TMPDIR"], cfg["environment"]["TMPDIR"])
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_build_existing_output_race_stops_before_cmake(self):
        cfg = m.config_from_env(ROOT, "all", {"H3_CUDA_ARCHITECTURES": "90"})
        with patch.object(Path, "mkdir", side_effect=[None, FileExistsError("existing output")]), \
                patch.object(m.subprocess, "run") as run:
            with self.assertRaises(FileExistsError):
                m.build(cfg, {})
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
