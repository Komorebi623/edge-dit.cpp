#!/usr/bin/env python3
"""Run official Diffusers MiniMax-H3 FL2VA/T2VA/I2VA/L2VA with optional keyframes and TorchAO quantization."""

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

TRANSFORMER_NOT_QUANTIZED = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
TEXT_ENCODER_NOT_QUANTIZED = [
    "model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head",
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
    raise ValueError(bits)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--last-image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--no-stream-offload", action="store_true")
    parser.add_argument("--profile-transformer", action="store_true")
    parser.add_argument("--profile-components", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    workflow = "fl2va" if (args.image is not None or args.last_image is not None) else "t2va"

    started = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(args.model, workflow=workflow, local_files_only=True)
    initialized = time.perf_counter()
    qconf = make_quant_config(args.bits)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(qconf, modules_to_not_convert=TRANSFORMER_NOT_QUANTIZED),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            args.model,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(qconf, modules_to_not_convert=TEXT_ENCODER_NOT_QUANTIZED),
            local_files_only=True,
        ),
    )
    quantized_loaded = time.perf_counter()
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(args.model), local_files_only=True)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    if args.resident:
        pipe.transformer.to(device)
        pipe.text_encoder.to(device)
        pipe.vae.to(device)
        pipe.audio_vae.to(device)
        offload_summary = {"transformer": str(device), "text_encoder": str(device), "vae": str(device), "audio_vae": str(device)}
    else:
        offload = {"onload_device": device, "offload_device": torch.device("cpu"), "use_stream": not args.no_stream_offload, "low_cpu_mem_usage": True}
        pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
        pipe.vae.to(device)
        pipe.audio_vae.to(device)
        offload_summary = {"transformer": "block_level group offload", "text_encoder": "leaf_level group offload", "vae": str(device), "audio_vae": str(device), "use_stream": not args.no_stream_offload}
    components_ready = time.perf_counter()

    transformer_profile = {"calls": 0, "seconds": 0.0}
    stack = []
    if args.profile_transformer:
        def pre(_module, _args, _kwargs):
            torch.cuda.synchronize(device)
            stack.append(time.perf_counter())
        def post(_module, _args, _kwargs, _output):
            torch.cuda.synchronize(device)
            transformer_profile["calls"] += 1
            transformer_profile["seconds"] += time.perf_counter() - stack.pop()
        pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True)
        pipe.transformer.register_forward_hook(post, with_kwargs=True)

    component_profile = {k: {"calls": 0, "seconds": 0.0} for k in ["text_encoder", "video_vae_encode", "video_vae_decode", "audio_vae_encode", "audio_vae_decode"]}
    if args.profile_components:
        def text_pre(_module, _args, _kwargs):
            torch.cuda.synchronize(device)
            component_profile["text_encoder"]["started_at"] = time.perf_counter()
        def text_post(_module, _args, _kwargs, _output):
            torch.cuda.synchronize(device)
            started_at = component_profile["text_encoder"].pop("started_at")
            component_profile["text_encoder"]["calls"] += 1
            component_profile["text_encoder"]["seconds"] += time.perf_counter() - started_at
        def wrap(name, method):
            def wrapped(*wargs, **wkwargs):
                torch.cuda.synchronize(device)
                begin = time.perf_counter()
                result = method(*wargs, **wkwargs)
                torch.cuda.synchronize(device)
                component_profile[name]["calls"] += 1
                component_profile[name]["seconds"] += time.perf_counter() - begin
                return result
            return wrapped
        pipe.text_encoder.model.register_forward_pre_hook(text_pre, with_kwargs=True)
        pipe.text_encoder.model.register_forward_hook(text_post, with_kwargs=True)
        pipe.vae.encode = wrap("video_vae_encode", pipe.vae.encode)
        pipe.vae.decode = wrap("video_vae_decode", pipe.vae.decode)
        pipe.audio_vae.encode = wrap("audio_vae_encode", pipe.audio_vae.encode)
        pipe.audio_vae.decode = wrap("audio_vae_decode", pipe.audio_vae.decode)

    first_image = load_image(str(args.image)) if args.image is not None else None
    last_image = load_image(str(args.last_image)) if args.last_image is not None else None
    call_kwargs = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )
    if workflow == "fl2va":
        call_kwargs["image"] = first_image
        call_kwargs["last_image"] = last_image

    torch.cuda.reset_peak_memory_stats(device)
    sync_time()
    out = pipe(**call_kwargs)
    generated_at = sync_time()
    encode_video(out["videos"][0], fps=24, output_path=str(args.output), audio=out["audio"][0], audio_sample_rate=out["sampling_rate"])
    finished = time.perf_counter()

    result = {
        "framework": "diffusers-main",
        "workflow": workflow,
        "model": str(args.model),
        "device": args.device,
        "dtype": "bfloat16",
        "quantization": {"method": "TorchAO weight-only", "bits": args.bits, "transformer_quantized": True, "text_encoder_quantized": True, "vae_quantized": False, "audio_vae_quantized": False},
        "offload": offload_summary,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "requested_num_frames": args.num_frames,
        "num_inference_steps": args.steps,
        "image": str(args.image) if args.image else None,
        "last_image": str(args.last_image) if args.last_image else None,
        "output": str(args.output),
        "seconds": {"pipeline_init": initialized - started, "quantized_component_load": quantized_loaded - initialized, "component_finalize": components_ready - quantized_loaded, "generate_cuda": generated_at - components_ready, "transformer_forward_cuda": transformer_profile["seconds"], "mux": finished - generated_at, "end_to_end": finished - started},
        "transformer_forward_calls": transformer_profile["calls"],
        "component_profile": component_profile if args.profile_components else None,
        "memory_bytes": {"max_allocated_during_generate": torch.cuda.max_memory_allocated(device), "max_reserved_during_generate": torch.cuda.max_memory_reserved(device)},
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
