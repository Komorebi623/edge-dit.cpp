"""stable-diffusion.cpp benchmark runner."""

from __future__ import annotations

import subprocess
from typing import Any

from .base import BenchmarkRunner, PreflightResult


class StableDiffusionCppRunner(BenchmarkRunner):
    def preflight(self) -> PreflightResult:
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        messages: list[str] = []
        metadata: dict[str, Any] = {
            "force_update_policy": self.system_config.get("repo", {}).get("commit_policy")
        }

        if repo is None or not repo.exists():
            messages.append(f"missing stable-diffusion.cpp repository: {repo}")
        else:
            metadata["commit"] = self.git_commit(repo)
            metadata["dirty"] = self.git_dirty(repo)
            if (
                self.system_config.get("repo", {}).get("commit_policy")
                == "force_latest_origin_main"
                and metadata["dirty"]
            ):
                messages.append(
                    "stable-diffusion.cpp worktree is dirty; refusing destructive force update"
                )

        if binary is None or not binary.exists():
            messages.append(f"missing sd-cli binary: {binary}")
        else:
            metadata["binary"] = str(binary)

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
            raise RuntimeError("stable-diffusion.cpp repo path is not resolved")
        if not force_external_update:
            raise RuntimeError(
                "stable-diffusion.cpp requires --force-external-update before execution"
            )
        commands = self.system_config.get("preflight", {}).get("force_update_commands", [])
        executed = []
        for command in commands:
            subprocess.run(command.split(), cwd=repo, check=True)
            executed.append(command)
        return executed

    def requires_runner_metrics(self) -> bool:
        return True

    def build_execution_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None,
        run_options: dict[str, Any],
        output_dir,
        warmup_runs: int,
        measured_runs: int,
    ) -> list[str]:
        raise NotImplementedError(
            "stable-diffusion.cpp load-once e2e wrapper is not implemented yet; "
            "do not report process-level CLI timing"
        )

    def build_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> list[str]:
        if gpu_count != 1:
            raise NotImplementedError("stable-diffusion.cpp is a single-GPU baseline")
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        model_path = self.resolve_path(workload["model"]["local_path_ref"])
        if binary is None:
            raise NotImplementedError("stable-diffusion.cpp path references are not resolved")
        if model_path is None or not model_path.exists():
            raise NotImplementedError(
                f"missing stable-diffusion.cpp model path: {model_path}"
            )
        run_options = run_options or {}
        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = workload.get("resolved_prompt", {}).get("prompt", "")
        command = [
            str(binary),
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--width",
            str(generation["width"]),
            "--height",
            str(generation["height"]),
            "--steps",
            str(generation["steps"]),
            "--seed",
            str(generation["seed"]),
            "--guidance",
            str(generation["guidance"]),
            "--output",
            "samples/output.avi" if workload["task"] == "text-to-video" else "samples/output.png",
        ]
        if workload["task"] == "text-to-video":
            command.extend(["--mode", "vid_gen", "--video-frames", str(generation.get("frames", 1))])
            if "fps" in generation:
                command.extend(["--fps", str(generation["fps"])])
        if workload["task"] == "image-editing":
            input_path = self.resolve_path(workload.get("input_image_ref"))
            if input_path is None:
                raise NotImplementedError("image editing input path reference is not resolved")
            command.extend(["--init-img", str(input_path)])
        if (
            generation.get("precision")
            and generation["precision"] != "auto"
            and run_options.get("precision") is None
        ):
            command.extend(["--type", str(generation["precision"])])
        if run_options.get("vae_tiling"):
            command.append("--vae-tiling")
        if run_options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if run_options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(run_options["max_vram_gib"])])
        if run_options.get("flash_attention") is True:
            command.append("--fa")
        cache = run_options.get("cache")
        if cache is not None and cache is not False and cache != "off":
            command.extend(["--cache-mode", str(cache)])
        return command
