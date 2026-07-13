"""edge-dit.cpp benchmark runner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BenchmarkRunner, PreflightResult


class EdgeDitRunner(BenchmarkRunner):
    def preflight(self) -> PreflightResult:
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        sample_binary = self.edge_sample_binary()
        repo = self.resolve_path(self.system_config.get("repo", {}).get("path_ref"))
        messages: list[str] = []
        metadata: dict[str, Any] = {}

        if binary is None or not binary.exists():
            messages.append(f"missing ed-cli binary: {binary}")
        else:
            metadata["binary"] = str(binary)
        if sample_binary is None or not sample_binary.exists():
            messages.append(f"missing ed-sample binary: {sample_binary}")
        else:
            metadata["sample_binary"] = str(sample_binary)

        if repo is None or not repo.exists():
            messages.append(f"missing edge-dit.cpp repository: {repo}")
        else:
            metadata["commit"] = self.git_commit(repo)
            metadata["dirty"] = self.git_dirty(repo)
            ggml = repo / "third_party" / "ggml"
            if ggml.exists():
                metadata["ggml_commit"] = self.git_commit(ggml)

        return PreflightResult(
            system_id=self.system_id,
            ok=not messages,
            messages=messages,
            metadata=metadata,
        )

    def requires_runner_metrics(self) -> bool:
        return True

    def edge_sample_binary(self) -> Path | None:
        sample_ref = self.system_config.get("sample_binary", {}).get("path_ref")
        sample_binary = self.resolve_path(sample_ref)
        if sample_binary is not None:
            return sample_binary
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        if binary is None:
            return None
        return binary.parent / "ed-sample"

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
        if workload["task"] != "text-to-image":
            raise NotImplementedError("edge e2e benchmark adapter currently supports text-to-image")
        sample_binary = self.edge_sample_binary()
        model_ref = workload["model"]["local_path_ref"]
        model_path = self.resolve_path(model_ref)
        if sample_binary is None:
            raise NotImplementedError("edge-dit path references are not resolved")
        if model_path is None or not model_path.exists():
            raise NotImplementedError(f"missing model path for {model_ref}: {model_path}")

        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = workload_prompt(workload)
        command = [
            "python3",
            str(self.repo_root / "benchmark" / "scripts" / "run_edge_e2e.py"),
            "--binary",
            str(sample_binary),
            "--model",
            str(model_path),
            "--prompt",
            prompt,
            "--output-dir",
            str(output_dir.resolve()),
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
            "--cfg-scale",
            str(generation.get("cfg_scale", 1.0)),
            "--dtype",
            str(generation.get("precision", "auto")),
            "--backend",
            "cuda",
            "--warmup-runs",
            str(warmup_runs),
            "--measured-runs",
            str(measured_runs),
        ]
        self.apply_edge_wrapper_options(command, workload.get("model_options", {}))
        self.apply_edge_wrapper_options(command, run_options)
        if gpu_count > 1:
            devices = edge_device_csv(gpu_count)
            command.extend(["--devices", devices])
            if parallel_mode == "sequence":
                command.extend(["--sp-size", str(gpu_count)])
            elif parallel_mode == "cfg":
                command.extend(["--cfg-parallel-size", str(gpu_count)])
        return command

    def build_command(
        self,
        workload: dict[str, Any],
        gpu_count: int,
        parallel_mode: str | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> list[str]:
        binary = self.resolve_path(self.system_config.get("binary", {}).get("path_ref"))
        model_ref = workload["model"]["local_path_ref"]
        model_path = self.resolve_path(model_ref)
        if binary is None:
            raise NotImplementedError("edge-dit path references are not resolved")
        if model_path is None or not model_path.exists():
            raise NotImplementedError(f"missing model path for {model_ref}: {model_path}")

        run_options = run_options or {}
        generation = dict(workload["generation"])
        generation.update({k: v for k, v in run_options.items() if k in generation})
        prompt = workload_prompt(workload)
        command = [
            str(binary),
            "--backend",
            "cuda",
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
            command.extend(["--video", "--frames", str(generation.get("frames", 1))])
            if "fps" in generation:
                command.extend(["--fps", str(generation["fps"])])
        if workload["task"] == "image-editing":
            input_ref = workload.get("input_image_ref")
            input_path = self.resolve_path(input_ref)
            if input_path is None:
                raise NotImplementedError("image editing input path reference is not resolved")
            command.extend(["--image", str(input_path)])
        if (
            generation.get("precision")
            and generation["precision"] != "auto"
            and run_options.get("precision") is None
        ):
            command.extend(["--type", str(generation["precision"])])
        self.apply_edge_options(command, workload.get("model_options", {}))
        self.apply_edge_options(command, run_options)
        if gpu_count > 1:
            devices = edge_device_csv(gpu_count)
            command.extend(["--devices", devices])
            if parallel_mode == "sequence":
                command.extend(["--sp-size", str(gpu_count)])
            elif parallel_mode == "cfg":
                command.extend(["--cfg-parallel-size", str(gpu_count)])
        return command

    def apply_edge_wrapper_options(self, command: list[str], options: dict[str, Any]) -> None:
        if options.get("qwen_image_zero_cond_t"):
            command.append("--qwen-image-zero-cond-t")
        if options.get("vae_tiling"):
            command.append("--vae-tiling")
        if options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if options.get("keep_text_encoder_on_cpu"):
            command.append("--keep-text-encoder-on-cpu")
        if options.get("keep_vae_on_cpu"):
            command.append("--keep-vae-on-cpu")
        if options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(options["max_vram_gib"])])
        if options.get("flash_attention") is False:
            command.append("--no-flash-attention")
        if options.get("profile_graph_cuts"):
            command.append("--profile-graph-cuts")
        cache = options.get("cache")
        if cache is not None and cache is not False:
            command.extend(["--cache", str(cache)])

    def apply_edge_options(self, command: list[str], options: dict[str, Any]) -> None:
        if options.get("qwen_image_zero_cond_t"):
            command.append("--qwen-image-zero-cond-t")
        if options.get("vae_tiling"):
            command.append("--vae-tiling")
        if options.get("offload_to_cpu"):
            command.append("--offload-to-cpu")
        if options.get("keep_text_encoder_on_cpu"):
            command.append("--keep-text-encoder-on-cpu")
        if options.get("keep_vae_on_cpu"):
            command.append("--keep-vae-on-cpu")
        if options.get("max_vram_gib") is not None:
            command.extend(["--max-vram", str(options["max_vram_gib"])])
        if options.get("profile_graph_cuts"):
            command.append("--profile-graph-cuts")
        if options.get("flash_attention") is False:
            command.append("--no-flash-attention")
        cache = options.get("cache")
        if cache is not None and cache is not False:
            command.extend(["--cache", str(cache)])
        if options.get("precision") is not None:
            command.extend(["--type", str(options["precision"])])


def workload_prompt(workload: dict[str, Any]) -> str:
    # The orchestrator resolves prompts before command construction.
    return workload.get("resolved_prompt", {}).get("prompt", "")


def edge_device_csv(gpu_count: int) -> str:
    """Return physical device ids for edge's MPI launcher."""
    value = os.environ.get("BENCHMARK_CUDA_VISIBLE_DEVICES")
    if value:
        devices = [item.strip() for item in value.split(",") if item.strip()]
        if len(devices) < gpu_count:
            raise ValueError(
                "BENCHMARK_CUDA_VISIBLE_DEVICES must list at least "
                f"{gpu_count} devices for edge parallel execution"
            )
        return ",".join(devices[:gpu_count])
    return ",".join(str(i) for i in range(gpu_count))
