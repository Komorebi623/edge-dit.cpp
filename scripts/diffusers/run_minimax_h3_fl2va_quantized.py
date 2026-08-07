#!/usr/bin/env python3
"""Run official Diffusers MiniMax-H3 FL2VA with TorchAO weight-only quantization."""

import argparse
import inspect
import json
import time
from pathlib import Path

import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.hooks import apply_group_offloading
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig


DEFAULT_PROMPT = (
    "Create a smooth cinematic transition between the supplied first and last frame, "
    "preserving the subject, lighting, and composition with natural coherent motion "
    "and synchronized ambient audio."
)

TRANSFORMER_NOT_QUANTIZED = [
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_embedder",
    "time_proj",
    "token_refiner",
    "norm_out",
    "proj_out",
    "audio_proj_out",
]

TEXT_ENCODER_NOT_QUANTIZED = [
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
]


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def make_quant_config(bits: int):
    if bits == 8:
        if "version" in inspect.signature(Int8WeightOnlyConfig).parameters:
            return Int8WeightOnlyConfig(version=2)
        return Int8WeightOnlyConfig()
    if bits == 4:
        return Int4WeightOnlyConfig(group_size=128)
    raise ValueError(f"unsupported quantization bits: {bits}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--last-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-stream-offload", action="store_true")
    parser.add_argument("--resident", action="store_true", help="keep quantized components resident on the CUDA device")
    parser.add_argument("--profile-transformer", action="store_true", help="synchronize and time every transformer forward")
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 Diffusers benchmark requires CUDA")
    if arguments.height % 32 or arguments.width % 32:
        raise ValueError("height and width must be divisible by 32")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    quant_config = make_quant_config(arguments.bits)

    started = time.perf_counter()
    pipeline = ModularPipeline.from_pretrained(
        arguments.model,
        workflow="fl2va",
        local_files_only=True,
    )
    initialized = time.perf_counter()

    pipeline.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            arguments.model,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                quant_config,
                modules_to_not_convert=TRANSFORMER_NOT_QUANTIZED,
            ),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            arguments.model,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                quant_config,
                modules_to_not_convert=TEXT_ENCODER_NOT_QUANTIZED,
            ),
            local_files_only=True,
        ),
    )
    quantized_loaded = time.perf_counter()

    pipeline.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(arguments.model), local_files_only=True)
    pipeline.transformer.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    device = torch.device(arguments.device)
    if arguments.resident:
        pipeline.transformer.to(device)
        pipeline.text_encoder.to(device)
        offload_summary = {
            "transformer": str(device),
            "text_encoder": str(device),
            "vae": str(device),
            "audio_vae": str(device),
            "use_stream": False,
        }
    else:
        offload = {
            "onload_device": device,
            "offload_device": torch.device("cpu"),
            "use_stream": not arguments.no_stream_offload,
            "low_cpu_mem_usage": True,
        }
        pipeline.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(pipeline.text_encoder.model, offload_type="leaf_level", **offload)
        offload_summary = {
            "transformer": "block_level group offload",
            "text_encoder": "leaf_level group offload",
            "vae": str(device),
            "audio_vae": str(device),
            "use_stream": not arguments.no_stream_offload,
        }
    pipeline.vae.to(device)
    pipeline.audio_vae.to(device)
    components_ready = time.perf_counter()

    transformer_profile = {"calls": 0, "seconds": 0.0}
    transformer_profile_stack = []
    if arguments.profile_transformer:
        def profile_forward_pre_hook(module, args, kwargs):
            torch.cuda.synchronize()
            transformer_profile_stack.append(time.perf_counter())

        def profile_forward_hook(module, args, kwargs, output):
            torch.cuda.synchronize()
            forward_started = transformer_profile_stack.pop()
            transformer_profile["calls"] += 1
            transformer_profile["seconds"] += time.perf_counter() - forward_started

        pipeline.transformer.register_forward_pre_hook(profile_forward_pre_hook, with_kwargs=True)
        pipeline.transformer.register_forward_hook(profile_forward_hook, with_kwargs=True)

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
        "quantization": {
            "method": "TorchAO weight-only",
            "bits": arguments.bits,
            "transformer_quantized": True,
            "text_encoder_quantized": True,
            "vae_quantized": False,
            "audio_vae_quantized": False,
            "scheduler_quantized": False,
            "transformer_modules_to_not_convert": TRANSFORMER_NOT_QUANTIZED,
            "text_encoder_modules_to_not_convert": TEXT_ENCODER_NOT_QUANTIZED,
        },
        "offload": offload_summary,
        "prompt": arguments.prompt,
        "seed": arguments.seed,
        "height": arguments.height,
        "width": arguments.width,
        "requested_num_frames": arguments.num_frames,
        "num_inference_steps": arguments.steps,
        "output": str(arguments.output),
        "seconds": {
            "pipeline_init": initialized - started,
            "quantized_component_load": quantized_loaded - initialized,
            "component_finalize": components_ready - quantized_loaded,
            "generate_cuda": generated_at - components_ready,
            "transformer_forward_cuda": transformer_profile["seconds"],
            "mux": finished - generated_at,
            "end_to_end": finished - started,
        },
        "transformer_forward_calls": transformer_profile["calls"],
    }
    result_path = arguments.output.with_suffix(".json")
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
