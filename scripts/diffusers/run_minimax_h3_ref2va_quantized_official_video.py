#!/usr/bin/env python3
import argparse, inspect, json, subprocess, time, wave
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.hooks import apply_group_offloading
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.utils.export_utils import encode_video
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig

TRANSFORMER_NOT_QUANTIZED = ["proj_in","audio_proj_in","context_embedder","time_embedder","time_proj","token_refiner","norm_out","proj_out","audio_proj_out"]
TEXT_ENCODER_NOT_QUANTIZED = ["model.visual","model.language_model.embed_tokens","model.language_model.norm","lm_head"]

def make_quant_config(bits:int):
    if bits == 8:
        return Int8WeightOnlyConfig(version=2) if "version" in inspect.signature(Int8WeightOnlyConfig).parameters else Int8WeightOnlyConfig()
    if bits == 4:
        return Int4WeightOnlyConfig(group_size=128)
    raise ValueError(bits)

def sync_time():
    torch.cuda.synchronize(); return time.perf_counter()

def read_wav(path: Path):
    try:
        with wave.open(str(path), "rb") as wf:
            channels=wf.getnchannels(); rate=wf.getframerate(); width=wf.getsampwidth(); frames=wf.getnframes(); data=wf.readframes(frames)
        if width == 2:
            arr=np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            arr=np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        elif width == 1:
            arr=(np.frombuffer(data, dtype=np.uint8).astype(np.float32)-128.0)/128.0
        else:
            raise ValueError(f"unsupported wav sample width {width}")
        arr=arr.reshape(-1, channels).T.copy()
        return torch.from_numpy(arr), rate
    except wave.Error:
        rate=32000
        cmd=["ffmpeg","-v","error","-i",str(path),"-f","f32le","-acodec","pcm_f32le","-ac","2","-ar",str(rate),"-"]
        raw=subprocess.check_output(cmd)
        arr=np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy()
        return torch.from_numpy(arr), rate

def load_frame_dir(path: Path):
    files=sorted([p for p in path.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}])
    if not files: raise ValueError(f"no frames in {path}")
    return [Image.open(p).convert("RGB") for p in files]

def build_video_reference(path: Path, audio=None, sample_rate=None):
    if path.is_dir():
        return MiniMaxH3VideoReference(frames=load_frame_dir(path), fps=24.0, audio=audio, sample_rate=sample_rate)
    ref = MiniMaxH3VideoReference.from_file(str(path))
    if audio is not None:
        ref.audio = audio
        ref.sample_rate = sample_rate
    return ref

def build_refs(args):
    refs=[]
    for p in args.ref_image:
        refs.append(MiniMaxH3ImageReference.from_file(str(p)))
    for i,p in enumerate(args.ref_video):
        audio=None; sample_rate=None
        if i < len(args.ref_video_audio):
            audio, sample_rate = read_wav(args.ref_video_audio[i])
        refs.append(build_video_reference(p, audio=audio, sample_rate=sample_rate))
    for p in args.ref_audio:
        audio, sample_rate = read_wav(p)
        refs.append(MiniMaxH3AudioReference(audio=audio, sample_rate=sample_rate))
    return refs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metrics", type=Path)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--num-frames", type=int, default=124)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--bits", type=int, choices=(4,8), default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--profile-transformer", action="store_true")
    ap.add_argument("--profile-components", action="store_true")
    ap.add_argument("--ref-image", type=Path, action="append", default=[])
    ap.add_argument("--ref-video", type=Path, action="append", default=[])
    ap.add_argument("--ref-video-audio", type=Path, action="append", default=[])
    ap.add_argument("--ref-audio", type=Path, action="append", default=[])
    args=ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device(args.device)
    refs=build_refs(args)
    started=time.perf_counter()
    pipe=ModularPipeline.from_pretrained(args.model, workflow="ref2va", local_files_only=True)
    if args.fps == 24.0 and args.num_frames == 362:
        type(pipe).max_duration = property(lambda self: args.num_frames / args.fps)
    initialized=time.perf_counter()
    qconf=make_quant_config(args.bits)
    pipe.update_components(
        transformer_ref=MiniMaxH3Transformer3DModel.from_pretrained(args.model, subfolder="transformer_ref", dtype=torch.bfloat16, quantization_config=TorchAoConfig(qconf, modules_to_not_convert=TRANSFORMER_NOT_QUANTIZED), low_cpu_mem_usage=True, local_files_only=True),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(args.model, subfolder="text_encoder", dtype=torch.bfloat16, quantization_config=TransformersTorchAoConfig(qconf, modules_to_not_convert=TEXT_ENCODER_NOT_QUANTIZED), local_files_only=True),
    )
    quant_loaded=time.perf_counter()
    pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(args.model), local_files_only=True)
    pipe.transformer_ref.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    if args.resident:
        pipe.transformer_ref.to(device); pipe.text_encoder.to(device)
    else:
        off={"onload_device":device,"offload_device":torch.device("cpu"),"use_stream":True,"low_cpu_mem_usage":True}
        pipe.transformer_ref.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **off)
        apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **off)
    pipe.vae.to(device); pipe.audio_vae.to(device)
    ready=time.perf_counter()
    transformer_profile={"calls":0,"seconds":0.0}; stack=[]
    if args.profile_transformer:
        def pre(m,a,k): torch.cuda.synchronize(device); stack.append(time.perf_counter())
        def post(m,a,k,o): torch.cuda.synchronize(device); transformer_profile["calls"]+=1; transformer_profile["seconds"]+=time.perf_counter()-stack.pop()
        pipe.transformer_ref.register_forward_pre_hook(pre, with_kwargs=True); pipe.transformer_ref.register_forward_hook(post, with_kwargs=True)
    comp={k:{"calls":0,"seconds":0.0} for k in ["text_encoder","video_vae_encode","video_vae_decode","audio_vae_encode","audio_vae_decode"]}
    if args.profile_components:
        def wrap(name, fn):
            def inner(*a, **k):
                torch.cuda.synchronize(device); t=time.perf_counter(); out=fn(*a, **k); torch.cuda.synchronize(device); comp[name]["calls"]+=1; comp[name]["seconds"]+=time.perf_counter()-t; return out
            return inner
        pipe.vae.encode=wrap("video_vae_encode", pipe.vae.encode); pipe.vae.decode=wrap("video_vae_decode", pipe.vae.decode)
        pipe.audio_vae.encode=wrap("audio_vae_encode", pipe.audio_vae.encode); pipe.audio_vae.decode=wrap("audio_vae_decode", pipe.audio_vae.decode)
        pipe.text_encoder.model.forward=wrap("text_encoder", pipe.text_encoder.model.forward)
    gen=torch.Generator(device="cpu").manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device); sync_time()
    out=pipe(prompt=args.prompt, references=refs, height=args.height, width=args.width, num_frames=args.num_frames, num_inference_steps=args.steps, generator=gen, output=["videos","audio","sampling_rate"])
    generated_at=sync_time()
    encode_video(out["videos"][0], fps=args.fps, output_path=str(args.output), audio=out["audio"][0], audio_sample_rate=out["sampling_rate"])
    finished=time.perf_counter()
    metrics={"framework":"diffusers-main","workflow":"ref2va","model":str(args.model),"dtype":"bfloat16","quantization":{"method":"TorchAO weight-only","bits":args.bits,"transformer_quantized":True,"text_encoder_quantized":True,"vae_quantized":False,"audio_vae_quantized":False},"prompt":args.prompt,"references":[{"type":type(r).__name__,"kind":getattr(r,"kind",None),"fps":getattr(r,"fps",None),"frames":(len(getattr(r,"frames",[])) if not hasattr(getattr(r,"frames",None),"shape") else int(getattr(r,"frames").shape[0])) if hasattr(r,"frames") else None,"has_audio":bool(getattr(r,"has_audio",False)),"sample_rate":getattr(r,"sample_rate",None)} for r in refs],"height":args.height,"width":args.width,"requested_num_frames":args.num_frames,"fps":args.fps,"num_inference_steps":args.steps,"seconds":{"pipeline_init":initialized-started,"quantized_component_load":quant_loaded-initialized,"component_finalize":ready-quant_loaded,"generate_cuda":generated_at-ready,"transformer_forward_cuda":transformer_profile["seconds"],"mux":finished-generated_at,"end_to_end":finished-started},"transformer_forward_calls":transformer_profile["calls"],"component_profile":comp,"memory_bytes":{"max_allocated_during_generate":torch.cuda.max_memory_allocated(device),"max_reserved_during_generate":torch.cuda.max_memory_reserved(device)}}
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
