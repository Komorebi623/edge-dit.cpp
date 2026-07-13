#!/usr/bin/env python3
"""Run Diffusers load-once e2e generation timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "f16", "fp32", "f32"], default="bf16")
    parser.add_argument("--task", default="text-to-image")
    parser.add_argument("--model-family", default="FLUX.1")
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--measured-runs", type=int, required=True)
    args = parser.parse_args()

    if args.task != "text-to-image":
        raise SystemExit("Diffusers e2e runner currently supports text-to-image only")

    import torch
    from diffusers import DiffusionPipeline, FluxPipeline

    output_dir = Path(args.output_dir).resolve()
    sample_dir = output_dir / "samples" / "diffusers"
    sample_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "fp32": torch.float32,
        "f32": torch.float32,
    }[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_cls = FluxPipeline if args.model_family == "FLUX.1" else DiffusionPipeline

    load_t0 = time.perf_counter()
    pipe = pipeline_cls.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    synchronize(device)
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    warmup_ms: list[float] = []
    measured_ms: list[float] = []
    metadata: list[dict[str, Any]] = []
    total_runs = args.warmup_runs + args.measured_runs
    for index in range(total_runs):
        phase = "warmup" if index < args.warmup_runs else "measured"
        phase_index = index if phase == "warmup" else index - args.warmup_runs
        elapsed_ms, result = run_generation(args, pipe, device)
        if phase == "warmup":
            warmup_ms.append(elapsed_ms)
        else:
            measured_ms.append(elapsed_ms)
            save_result_image(result, sample_dir / f"output_{phase_index:03d}.png")
        metadata.append(
            {
                "phase": phase,
                "index": phase_index,
                "elapsed_ms": elapsed_ms,
                "seed": args.seed,
            }
        )
        print(f"[diffusers-e2e] {phase} {phase_index} {elapsed_ms / 1000.0:.3f}s", flush=True)

    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "schema_version": 1,
        "metric_source": "diffusers_pipeline",
        "measurement_boundary": "load_once_e2e_generation_no_output_encoding",
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "measured_ms": measured_ms,
        "component_ms": {
            "text_encoder": None,
            "dit": None,
            "vae": None,
            "per_step_avg": (statistics.mean(measured_ms) / args.steps) if measured_ms else None,
        },
        "sample_output_dir": str(sample_dir),
    }
    (output_dir / "runner_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def run_generation(args: argparse.Namespace, pipe: Any, device: str) -> tuple[float, Any]:
    import torch

    generator = torch.Generator(device=device).manual_seed(args.seed) if device == "cuda" else None
    synchronize(device)
    start = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, result


def save_result_image(result: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "images"):
        result.images[0].save(output)
        return
    if hasattr(result, "frames"):
        frames = result.frames[0] if result.frames and isinstance(result.frames[0], list) else result.frames
        frames[0].save(output)
        return
    raise RuntimeError("Diffusers pipeline result has neither images nor frames")


def synchronize(device: str) -> None:
    if device != "cuda":
        return
    import torch

    torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
