#!/usr/bin/env python3
"""Diffusers Flux-dev CPU profiler (op timing, for edge-dit / sd.cpp comparison).

Runs Flux-dev on CPU in bf16 at a small config (a cat / 512 / 4 steps / guidance=3.5 / seed 42),
profiles CPU aten ops with torch.profiler, and buckets them into the SAME categories
edge-dit / sd.cpp use (GEMM / FLASH_ATTN / LAYOUT / NORM / ELEMENTWISE / OTHER) so all three
engines can be compared class-by-class.

Run inside container wty-edgedit-dev (torch CPU):
  python3 /workspace/edge-dit.cpp/scripts/diffusers_profile_flux.py
Outputs go to /tmp/diffusers_flux_profile.txt (container /tmp).
"""
import os, time
import torch

MODEL = os.environ.get("FLUX_MODEL", "/models/flux-dev")
OUT = os.environ.get("WTY_OUTFILE", "/tmp/diffusers_flux_profile.txt")
IMG = os.environ.get("WTY_OUTIMG", "/tmp/diffusers_flux.png")
STEPS = int(os.environ.get("STEPS", "4"))
W = int(os.environ.get("W", "512"))
H = int(os.environ.get("H", "512"))
THREADS = int(os.environ.get("THREADS", "0"))  # 0 = leave torch default
PROMPT = "a cat"

if THREADS > 0:
    torch.set_num_threads(THREADS)

def log(m): print(f"[diffusers-flux-prof] {m}", flush=True)

# --- op name -> category. CUDA-kernel keywords PLUS CPU aten dispatcher names. ---
def categorize(name):
    n = name.lower()
    if any(k in n for k in ["flash", "fmha", "attention", "sdpa", "mha", "scaled_dot_product"]):
        return "FLASH_ATTN"
    # GEMM: cpu aten mm/addmm/bmm/baddbmm/linear + cuda gemm families
    if any(k in n for k in ["addmm", "baddbmm", "bmm", "::mm", "aten::mm", "linear",
                            "gemm", "cutlass", "cublas", "matmul", "mkldnn_linear",
                            "convolution", "conv2d", "conv"]):
        return "GEMM"
    if any(k in n for k in ["layer_norm", "layernorm", "rms_norm", "rmsnorm", "group_norm",
                            "batch_norm", "native_layer_norm", "welford", "norm"]):
        return "NORM"
    if any(k in n for k in ["elementwise", "gelu", "silu", "softmax", "sigmoid",
                            "add", "mul", "div", "sub", "neg", "scale", "activation",
                            "binary", "unary", "pointwise", "exp", "rsqrt"]):
        return "ELEMENTWISE"
    if any(k in n for k in ["copy", "cat", "permute", "transpose", "contiguous", "reshape",
                            "clone", "view", "expand", "index", "slice", "pad", "roll",
                            "stack", "::to", "narrow", "select"]):
        return "LAYOUT"
    return "OTHER"

def main():
    from diffusers import FluxPipeline
    log(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    log(f"loading {MODEL} (bf16, CPU) ...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cpu")
    log(f"loaded in {time.time()-t0:.1f}s")

    # Flux-dev is guidance-distilled: single forward/step, guidance_scale=3.5, no true CFG.
    common = dict(prompt=PROMPT, width=W, height=H, num_inference_steps=STEPS,
                  guidance_scale=3.5, max_sequence_length=512)

    # warmup (1 step) to trigger oneDNN/MKL dispatch
    log("warmup (1 step) ...")
    g = torch.Generator(device="cpu").manual_seed(42)
    _ = pipe(prompt=PROMPT, width=W, height=H, num_inference_steps=1,
             guidance_scale=3.5, max_sequence_length=512, generator=g)

    # --- baseline wall (no profiler) ---
    g = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    img = pipe(generator=g, **common).images[0]
    wall = time.time() - t0
    img.save(IMG)
    log(f"baseline wall (full pipe, {STEPS} steps): {wall:.2f}s")

    # --- stage timing via hooks (same caliber as edge-dit's generate breakdown:
    #     text_encode / sampling(transformer) / vae_decode; excludes model load) ---
    stage = {"text": 0.0, "dit": 0.0, "vae": 0.0}
    def wrap(mod, key):
        orig = mod.forward
        def timed(*a, **k):
            t = time.time(); r = orig(*a, **k); stage[key] += time.time() - t
            return r
        mod.forward = timed
        return orig
    o_te  = wrap(pipe.text_encoder,  "text")
    o_te2 = wrap(pipe.text_encoder_2,"text")
    o_tf  = wrap(pipe.transformer,   "dit")
    o_vae = wrap(pipe.vae.decoder,   "vae")
    g = torch.Generator(device="cpu").manual_seed(42)
    tS = time.time()
    _ = pipe(generator=g, **common)
    stage_wall = time.time() - tS
    pipe.text_encoder.forward=o_te; pipe.text_encoder_2.forward=o_te2
    pipe.transformer.forward=o_tf; pipe.vae.decoder.forward=o_vae
    log(f"stage breakdown (excl. load): text_encode={stage['text']:.2f}s "
        f"sampling={stage['dit']:.2f}s vae_decode={stage['vae']:.2f}s "
        f"| stage_wall={stage_wall:.2f}s")

    # --- profiled run ---
    log("profiled run (torch.profiler CPU) ...")
    from torch.profiler import profile, ProfilerActivity
    g = torch.Generator(device="cpu").manual_seed(42)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        _ = pipe(generator=g, **common)

    cats = {}
    ops = []
    for e in prof.key_averages():
        cpu_us = getattr(e, "self_cpu_time_total", 0)
        if cpu_us <= 0:
            continue
        c = categorize(e.key)
        d = cats.setdefault(c, {"us": 0.0, "calls": 0})
        d["us"] += cpu_us
        d["calls"] += e.count
        ops.append((e.key, cpu_us, e.count))

    total_us = sum(d["us"] for d in cats.values())
    lines = []
    lines.append(f"=== diffusers Flux-dev CPU profiling ({STEPS} steps, guidance=3.5, {W}x{H}, bf16) ===")
    lines.append(f"torch threads: {torch.get_num_threads()}")
    lines.append(f"baseline wall (full pipe): {wall:.2f}s")
    lines.append(f"total CPU self-time (profiled, all {STEPS} steps): {total_us/1e6:.3f}s")
    lines.append("")
    lines.append(f"{'category':<14}{'calls':>10}{'total_ms':>14}{'pct':>8}")
    for c in ["GEMM","FLASH_ATTN","LAYOUT","NORM","ELEMENTWISE","OTHER"]:
        if c in cats:
            d = cats[c]
            lines.append(f"{c:<14}{d['calls']:>10}{d['us']/1e3:>14.1f}{100*d['us']/total_us:>7.1f}%")
    lines.append("")
    lines.append("=== top 25 ops (self cpu time) ===")
    for name, us, cnt in sorted(ops, key=lambda x:-x[1])[:25]:
        lines.append(f"{us/1e3:>10.1f}ms  x{cnt:<6} [{categorize(name)}] {name[:80]}")

    txt = "\n".join(lines)
    print(txt)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    log(f"written {OUT}")

if __name__ == "__main__":
    main()
