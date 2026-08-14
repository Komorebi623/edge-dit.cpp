#!/usr/bin/env python3
"""Run the official Diffusers MiniMax-H3 FL2VA pipeline on one CUDA GPU."""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video


DEFAULT_PROMPT = (
    "Create a smooth cinematic transition between the supplied first and last frame, "
    "preserving the subject, lighting, and composition with natural coherent motion "
    "and synchronized ambient audio."
)


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--last-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1376)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 Diffusers benchmark requires CUDA")
    if arguments.height % 32 or arguments.width % 32:
        raise ValueError("height and width must be divisible by 32")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=arguments.device)

    started = time.perf_counter()
    pipeline = ModularPipeline.from_pretrained(
        arguments.model,
        workflow="fl2va",
        components_manager=manager,
        local_files_only=True,
    )
    pipeline.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=str(arguments.model),
        local_files_only=True,
    )
    loaded = time.perf_counter()

    first_image = load_image(str(arguments.image))
    last_image = load_image(str(arguments.last_image))
    generator = torch.Generator(device="cpu").manual_seed(arguments.seed)

    sync_time()
    generated = pipeline(
        prompt=arguments.prompt,
        image=first_image,
        last_image=last_image,
        height=arguments.height,
        width=arguments.width,
        num_frames=arguments.num_frames,
        num_inference_steps=arguments.steps,
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )
    generated_at = sync_time()

    encode_video(
        generated["videos"][0],
        fps=24,
        output_path=str(arguments.output),
        audio=generated["audio"][0],
        audio_sample_rate=generated["sampling_rate"],
    )
    finished = time.perf_counter()

    results = {
        "framework": "diffusers-main",
        "workflow": "fl2va",
        "model": str(arguments.model),
        "device": arguments.device,
        "dtype": "bfloat16",
        "offload": "official ComponentsManager auto_cpu_offload",
        "prompt": arguments.prompt,
        "seed": arguments.seed,
        "height": arguments.height,
        "width": arguments.width,
        "requested_num_frames": arguments.num_frames,
        "num_inference_steps": arguments.steps,
        "output": str(arguments.output),
        "seconds": {
            "component_load": loaded - started,
            "generate_cuda": generated_at - loaded,
            "mux": finished - generated_at,
            "end_to_end": finished - started,
        },
    }
    result_path = arguments.output.with_suffix(".json")
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
