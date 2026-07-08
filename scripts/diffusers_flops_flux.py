#!/usr/bin/env python3
"""Diffusers Flux-dev CPU: exact FLOP count + per-op time, to settle the
'edge-dit does ~4x the matmul work' question against edge-dit's MUL_MAT FLOPs.

FLOPs are hardware/clock independent, so counting them on both engines cleanly
separates 'redundant compute' from 'AMX-vs-AVX512 per-FLOP speed'.

Uses torch.utils.flop_counter.FlopCounterMode (exact analytic count: mm/addmm/
bmm/baddbmm 2*M*N*K, conv, sdpa) — NOT the profiler's approximate with_flops.

Run inside container wty-edgedit-dev:
  python3 /workspace/edge-dit.cpp/scripts/diffusers_flops_flux.py
"""
import os, time
import torch

MODEL = os.environ.get("FLUX_MODEL", "/models/flux-dev")
OUT = os.environ.get("WTY_OUTFILE", "/tmp/diffusers_flux_flops.txt")
STEPS = int(os.environ.get("STEPS", "4"))
W = int(os.environ.get("W", "512"))
H = int(os.environ.get("H", "512"))
THREADS = int(os.environ.get("THREADS", "0"))
PROMPT = "a cat"

if THREADS > 0:
    torch.set_num_threads(THREADS)

def log(m): print(f"[diffusers-flux-flops] {m}", flush=True)

def main():
    from diffusers import FluxPipeline
    from torch.utils.flop_counter import FlopCounterMode
    from torch.profiler import profile, ProfilerActivity

    log(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    log(f"loading {MODEL} (bf16, CPU) ...")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cpu")
    log(f"loaded in {time.time()-t0:.1f}s")

    common = dict(prompt=PROMPT, width=W, height=H, num_inference_steps=STEPS,
                  guidance_scale=3.5, max_sequence_length=512)

    # warmup
    log("warmup (1 step) ...")
    g = torch.Generator(device="cpu").manual_seed(42)
    _ = pipe(prompt=PROMPT, width=W, height=H, num_inference_steps=1,
             guidance_scale=3.5, max_sequence_length=512, generator=g)

    # baseline wall
    g = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    _ = pipe(generator=g, **common)
    wall = time.time() - t0
    log(f"baseline wall (full pipe, {STEPS} steps): {wall:.2f}s")

    # --- exact FLOP count for the WHOLE pipeline ---
    log("FLOP-counted run (FlopCounterMode, whole pipe) ...")
    g = torch.Generator(device="cpu").manual_seed(42)
    fc = FlopCounterMode(display=False, depth=None)
    with fc:
        _ = pipe(generator=g, **common)
    total_flops = fc.get_total_flops()

    # FlopCounterMode.flop_counts is HIERARCHICAL: key "Global" holds the flat
    # per-op total across everything; every other key is a module subtree that is
    # ALREADY included in Global. So aggregate ONLY "Global" to avoid double count.
    from collections import Counter
    global_counts = fc.flop_counts.get("Global", {})
    op_flops = Counter({str(op): f for op, f in global_counts.items()})

    # DiT-only: find the transformer subtree key (its own module class name).
    dit_key = None
    for k in fc.flop_counts.keys():
        if k != "Global" and ("transformer" in k.lower() or "flux" in k.lower()):
            dit_key = k
            break
    dit_counts = fc.flop_counts.get(dit_key, {}) if dit_key else {}
    dit_flops_by_op = Counter({str(op): f for op, f in dit_counts.items()})

    def bucket(opname):
        n = opname.lower()
        if "convolution" in n or "conv" in n:
            return "CONV"
        if any(k in n for k in ["scaled_dot_product", "sdpa", "flash", "efficient_attention", "attention"]):
            return "ATTN"
        if any(k in n for k in ["addmm", "baddbmm", "bmm", "mm", "linear", "matmul"]):
            return "GEMM"
        return "OTHER"

    buckets = Counter()
    for op, f in op_flops.items():
        buckets[bucket(op)] += f

    lines = []
    lines.append(f"=== diffusers Flux-dev EXACT FLOPs ({STEPS} steps, {W}x{H}, bf16, whole pipe incl VAE+TE) ===")
    lines.append(f"torch threads: {torch.get_num_threads()}")
    lines.append(f"baseline wall (full pipe): {wall:.2f}s")
    lines.append(f"TOTAL FLOPs (get_total_flops, whole pipe): {total_flops/1e12:.3f} TFLOP")
    lines.append(f"  (sum of Global per-op below should match this)")
    lines.append("")
    lines.append(f"{'bucket':<10}{'TFLOP':>12}{'pct':>8}   (whole pipe)")
    gsum = sum(op_flops.values())
    for b in ["GEMM", "ATTN", "CONV", "OTHER"]:
        if b in buckets:
            lines.append(f"{b:<10}{buckets[b]/1e12:>12.3f}{100*buckets[b]/gsum:>7.1f}%")
    lines.append("")
    lines.append("  GEMM+CONV+ATTN (matmul-class, whole pipe): "
                 f"{(buckets['GEMM']+buckets.get('CONV',0)+buckets.get('ATTN',0))/1e12:.3f} TFLOP")
    if dit_key:
        dit_total = sum(dit_flops_by_op.values())
        dit_mm = sum(f for op, f in dit_flops_by_op.items() if bucket(op) in ("GEMM", "ATTN"))
        lines.append("")
        lines.append(f"  >>> DiT-only (subtree '{dit_key}', all {STEPS} steps): {dit_total/1e12:.3f} TFLOP total, "
                     f"{dit_mm/1e12:.3f} TFLOP matmul-class (GEMM+ATTN)")
        lines.append(f"  >>> DiT-only per step: {dit_total/1e12/STEPS:.3f} TFLOP")
    lines.append("")
    lines.append("=== per-aten-op FLOPs, whole pipe (desc) ===")
    for op, f in sorted(op_flops.items(), key=lambda x: -x[1]):
        if f <= 0:
            continue
        lines.append(f"{f/1e12:>10.4f} TFLOP  [{bucket(op)}] {op}")

    # --- per-op TIME (torch profiler self_cpu_time) for reference ---
    lines.append("")
    lines.append("=== per-op TIME (torch.profiler self_cpu_time, whole pipe) ===")
    g = torch.Generator(device="cpu").manual_seed(42)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _ = pipe(generator=g, **common)
    tcat = Counter()
    for e in prof.key_averages():
        us = getattr(e, "self_cpu_time_total", 0)
        if us <= 0:
            continue
        tcat[bucket(e.key)] += us
    ttot = sum(tcat.values())
    lines.append(f"total self_cpu_time: {ttot/1e6:.2f}s")
    for b in ["GEMM", "ATTN", "CONV", "OTHER"]:
        if b in tcat:
            lines.append(f"{b:<10}{tcat[b]/1e6:>10.2f}s{100*tcat[b]/ttot:>7.1f}%")

    txt = "\n".join(lines)
    print(txt)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    log(f"written {OUT}")

if __name__ == "__main__":
    main()
