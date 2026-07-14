#!/usr/bin/env python3
"""Run xDiT FLUX load-once e2e generation timing."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any


def main() -> int:
    import torch
    from xfuser import xFuserArgs, xFuserFluxPipeline
    from xfuser.config import FlexibleArgumentParser
    from xfuser.core.distributed import get_runtime_state, get_world_group

    benchmark_args, remaining_argv = parse_benchmark_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining_argv]

    parser = FlexibleArgumentParser(description=__doc__)
    args = xFuserArgs.add_cli_args(parser).parse_args()

    output_dir = Path(benchmark_args.benchmark_output_dir).resolve()
    sample_dir = output_dir / "samples" / "xdit"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16

    local_rank = get_world_group().local_rank
    device = f"cuda:{local_rank}"

    barrier()
    load_start = time.perf_counter()
    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    else:
        pipe = pipe.to(device)
    pipe.prepare_run(input_config, steps=1)
    synchronize()
    barrier()
    load_ms = (time.perf_counter() - load_start) * 1000.0

    warmup_ms: list[float] = []
    measured_ms: list[float] = []
    total_runs = benchmark_args.benchmark_warmup_runs + benchmark_args.benchmark_measured_runs
    last_output: Any = None
    for index in range(total_runs):
        phase = "warmup" if index < benchmark_args.benchmark_warmup_runs else "measured"
        phase_index = index if phase == "warmup" else index - benchmark_args.benchmark_warmup_runs
        elapsed_ms, output = run_generation(args, benchmark_args, pipe, input_config, device)
        if phase == "warmup":
            warmup_ms.append(elapsed_ms)
        else:
            measured_ms.append(elapsed_ms)
            last_output = output
        if local_rank == 0:
            print(f"[xdit-e2e] {phase} {phase_index} {elapsed_ms / 1000.0:.3f}s", flush=True)

    if benchmark_args.benchmark_save_samples and local_rank == 0 and last_output is not None:
        save_output(last_output, sample_dir / "output_000.png")

    if local_rank == 0:
        metrics = {
            "schema_version": 1,
            "metric_source": "xdit_xfuser_flux_pipeline",
            "measurement_boundary": "load_once_e2e_generation_no_output_encoding",
            "load_ms": load_ms,
            "warmup_ms": warmup_ms,
            "measured_ms": measured_ms,
            "component_ms": {
                "text_encoder": None,
                "dit": None,
                "vae": None,
                "per_step_avg": (
                    sum(measured_ms) / len(measured_ms) / args.num_inference_steps
                    if measured_ms
                    else None
                ),
            },
            "sample_output_dir": str(sample_dir),
        }
        (output_dir / "runner_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    barrier()
    get_runtime_state().destroy_distributed_env()
    return 0


def parse_benchmark_args(argv: list[str]) -> tuple[Any, list[str]]:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark-output-dir", required=True)
    parser.add_argument("--benchmark-warmup-runs", type=int, required=True)
    parser.add_argument("--benchmark-measured-runs", type=int, required=True)
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--benchmark-save-samples", action="store_true")
    return parser.parse_known_args(argv)


def run_generation(
    args: Any,
    benchmark_args: Any,
    pipe: Any,
    input_config: Any,
    device: str,
) -> tuple[float, Any]:
    import torch

    barrier()
    synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = pipe(
        height=args.height,
        width=args.width,
        prompt=args.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(benchmark_args.benchmark_seed),
    )
    end.record()
    synchronize()
    barrier()
    return float(start.elapsed_time(end)), output


def save_output(output: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = getattr(output, "images", None)
    if images:
        images[0].save(path)


def synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    raise SystemExit(main())
