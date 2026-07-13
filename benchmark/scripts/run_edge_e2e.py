#!/usr/bin/env python3
"""Run edge-dit.cpp load-once e2e generation timing via ed-sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


PASS_RE = re.compile(
    r"\[ed-sample\]\s+pass\s+\d+/\d+\s+\d+/\d+\s+"
    r"seed=.*?\s+([0-9]+(?:\.[0-9]+)?)s(?P<warmup>\s+\(warmup\))?"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--measured-runs", type=int, required=True)
    parser.add_argument("--devices")
    parser.add_argument("--sp-size", type=int)
    parser.add_argument("--cfg-parallel-size", type=int)
    parser.add_argument("--cache")
    parser.add_argument("--qwen-image-zero-cond-t", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--keep-text-encoder-on-cpu", action="store_true")
    parser.add_argument("--keep-vae-on-cpu", action="store_true")
    parser.add_argument("--max-vram")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--profile-graph-cuts", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = output_dir / "inputs"
    prompt_dir.mkdir(exist_ok=True)
    prompt_file = prompt_dir / "prompt.txt"
    prompt_file.write_text(args.prompt + "\n", encoding="utf-8")

    sample_dir = output_dir / "samples" / "edge"
    repeat = args.warmup_runs + args.measured_runs
    command = [
        args.binary,
        "--model",
        args.model,
        "--prompt_file",
        str(prompt_file),
        "--output_dir",
        str(sample_dir),
        "--backend",
        args.backend,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--num_steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--guidance_scale",
        str(args.guidance),
        "--cfg_scale",
        str(args.cfg_scale),
        "--warmup",
        str(args.warmup_runs),
        "--repeat",
        str(repeat),
        "--start_index",
        "0",
        "--end_index",
        "1",
    ]
    if args.dtype and args.dtype != "auto":
        command.extend(["--type", args.dtype])
    if args.devices:
        command.extend(["--devices", args.devices])
    if args.sp_size:
        command.extend(["--sp-size", str(args.sp_size)])
    if args.cfg_parallel_size:
        command.extend(["--cfg-parallel-size", str(args.cfg_parallel_size)])
    if args.cache:
        command.extend(["--cache_method", args.cache])
    if args.qwen_image_zero_cond_t:
        command.append("--qwen-image-zero-cond-t")
    if args.vae_tiling:
        command.append("--vae-tiling")
    if args.offload_to_cpu:
        command.append("--offload-to-cpu")
    if args.keep_text_encoder_on_cpu:
        command.append("--keep-text-encoder-on-cpu")
    if args.keep_vae_on_cpu:
        command.append("--keep-vae-on-cpu")
    if args.max_vram:
        command.extend(["--max-vram", args.max_vram])
    if args.no_flash_attention:
        command.append("--no-flash-attention")
    if args.profile_graph_cuts:
        command.append("--profile-graph-cuts")

    (output_dir / "adapter_command.txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode

    warmup_ms, measured_ms = parse_ed_sample_stdout(completed.stdout)
    if len(measured_ms) != args.measured_runs:
        print(
            f"expected {args.measured_runs} measured samples, got {len(measured_ms)}",
            file=sys.stderr,
        )
        return 3

    load_ms = load_ms_from_ed_sample(sample_dir / "timing.json")
    metrics = {
        "schema_version": 1,
        "metric_source": "edge_dit_ed_sample",
        "measurement_boundary": "load_once_e2e_generation_no_output_encoding",
        "load_ms": load_ms,
        "warmup_ms": warmup_ms,
        "measured_ms": measured_ms,
        "component_ms": {
            "text_encoder": None,
            "dit": None,
            "vae": None,
            "per_step_avg": (sum(measured_ms) / len(measured_ms) / args.steps) if measured_ms else None,
        },
        "sample_output_dir": str(sample_dir),
    }
    (output_dir / "runner_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def parse_ed_sample_stdout(stdout: str) -> tuple[list[float], list[float]]:
    warmup_ms: list[float] = []
    measured_ms: list[float] = []
    for match in PASS_RE.finditer(stdout):
        value_ms = float(match.group(1)) * 1000.0
        if match.group("warmup"):
            warmup_ms.append(value_ms)
        else:
            measured_ms.append(value_ms)
    return warmup_ms, measured_ms


def load_ms_from_ed_sample(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    seconds = data.get("model_load_seconds")
    if isinstance(seconds, (int, float)):
        return float(seconds) * 1000.0
    return None


if __name__ == "__main__":
    raise SystemExit(main())
