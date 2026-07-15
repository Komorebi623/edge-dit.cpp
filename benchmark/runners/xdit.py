"""xDiT benchmark runner."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
from typing import Any

from .base import BenchmarkRunner, PreflightResult


class XditRunner(BenchmarkRunner):
    def preflight(self) -> PreflightResult:
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        messages: list[str] = []
        metadata: dict[str, Any] = {
            "force_update_policy": self.system_config.get("repo", {}).get("commit_policy")
        }

        if repo is None or not repo.exists():
            messages.append(f"missing xDiT repository: {repo}")
        else:
            metadata["commit"] = self.git_commit(repo)
            metadata["dirty"] = self.git_dirty(repo)
            metadata["entrypoints"] = [
                str(path.relative_to(repo))
                for path in self.find_flux_entrypoints(repo)
            ]
            if not metadata["entrypoints"]:
                messages.append("no candidate xDiT CLI or FLUX entrypoint found")
            for module_name in self.system_config.get("preflight", {}).get("require_imports", []):
                import_error = self.repo_import_error(repo, module_name)
                if import_error:
                    messages.append(f"cannot import xDiT module {module_name}: {import_error}")

        return PreflightResult(
            system_id=self.system_id,
            ok=not messages,
            messages=messages,
            metadata=metadata,
        )

    def prepare_for_execution(self, force_external_update: bool = False) -> list[str]:
        policy = self.system_config.get("repo", {}).get("commit_policy")
        if policy != "force_latest_origin_main":
            return []
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        if repo is None:
            raise RuntimeError("xDiT repo path is not resolved")
        if not force_external_update:
            raise RuntimeError("xDiT requires --force-external-update before execution")
        commands = self.system_config.get("preflight", {}).get("force_update_commands", [])
        executed = []
        for command in commands:
            subprocess.run(command.split(), cwd=repo, check=True)
            executed.append(command)
        return executed

    def requires_runner_metrics(self) -> bool:
        return True

    def extra_env(self, gpu_count: int) -> dict[str, str]:
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        if repo is None:
            return {}
        existing = os.environ.get("PYTHONPATH", "")
        return {
            "PYTHONPATH": f"{repo}:{existing}" if existing else str(repo),
        }

    def repo_import_error(self, repo: Path, module_name: str) -> str | None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo}:{existing}" if existing else str(repo)
        completed = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode == 0:
            return None
        lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        return lines[-1] if lines else f"import exited with code {completed.returncode}"

    def find_flux_entrypoints(self, repo: Path) -> list[Path]:
        candidates = [
            repo / "xfuser" / "cli.py",
            repo / "examples" / "flux_example.py",
            repo / "benchmark" / "fid" / "flux_generate.py",
            repo / "tests" / "context_parallel" / "debug_flux_usp_example.py",
        ]
        return [path for path in candidates if path.exists()]

    def build_execution_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        run_options: dict[str, Any],
        output_dir: Path,
        warmup_runs: int,
        measured_runs: int,
    ) -> list[str]:
        if workload["task"] != "text-to-image" or workload["model_family"] != "FLUX.1":
            raise NotImplementedError("xDiT e2e adapter currently supports FLUX.1 text-to-image")
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        model_path = self.resolve_path(workload["model"]["local_path_ref"])
        if repo is None or model_path is None:
            raise NotImplementedError("xDiT path references are not resolved")
        if not model_path.exists():
            raise NotImplementedError(f"missing xDiT model path: {model_path}")

        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = self.prompt_text(workload, run_options)
        ulysses_degree = gpu_count if parallel_mode in (None, "ulysses") else 1
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={gpu_count}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29501",
            str(self.repo_root / "benchmark" / "scripts" / "run_xdit_e2e.py"),
            "--benchmark-output-dir",
            str(output_dir.resolve()),
            "--benchmark-warmup-runs",
            str(warmup_runs),
            "--benchmark-measured-runs",
            str(measured_runs),
            "--benchmark-seed",
            str(generation["seed"]),
            "--model",
            str(model_path),
            "--height",
            str(generation["height"]),
            "--width",
            str(generation["width"]),
            "--num_inference_steps",
            str(generation["steps"]),
            "--max_sequence_length",
            str(generation.get("max_sequence_length", 256)),
            "--guidance_scale",
            str(generation["guidance"]),
            "--seed",
            str(generation["seed"]),
            "--output_type",
            "pil",
            "--attention_backend",
            str(run_options.get("attention_backend", "sdpa_flash")),
            "--no_use_resolution_binning",
            "--warmup_steps",
            "0",
            "--ulysses_degree",
            str(ulysses_degree),
            "--ring_degree",
            "1",
            "--prompt",
            prompt,
        ]
        if run_options.get("use_parallel_vae"):
            command.append("--use_parallel_vae")
        return command

    def build_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> list[str]:
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        model_path = self.resolve_path(workload["model"]["local_path_ref"])
        if repo is None or model_path is None:
            raise NotImplementedError("xDiT path references are not resolved")
        if not model_path.exists():
            raise NotImplementedError(f"missing xDiT model path: {model_path}")

        cli = repo / "xfuser" / "cli.py"
        if not cli.exists():
            raise NotImplementedError("xDiT has no xfuser CLI entrypoint in this checkout")

        run_options = run_options or {}
        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = self.prompt_text(workload, run_options)
        ulysses_degree = gpu_count if parallel_mode in (None, "ulysses") else 1
        return [
            "python3",
            "-m",
            "xfuser.cli",
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--height",
            str(generation["height"]),
            "--width",
            str(generation["width"]),
            "--num_inference_steps",
            str(generation["steps"]),
            "--ulysses_degree",
            str(ulysses_degree),
            "--ring_degree",
            "1",
            "--nproc_per_node",
            str(gpu_count),
        ]
