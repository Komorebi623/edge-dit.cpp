#!/usr/bin/env python3
"""Diffusers Flux-dev DiT GEMM shape profiler — for matmul-for-matmul comparison
with edge-dit. Records aten op input shapes, buckets GEMM ops by shape, and
reports self_cpu_time + call count per shape. Also times the transformer forward
in isolation (wall clock) to get a load-robust per-step DiT number.

Run in container wty-edgedit-dev:
  STEPS=4 W=512 H=512 python3 scripts/diffusers_dit_shapes.py
"""
import os, time
import torch

MODEL = os.environ.get("FLUX_MODEL", "/models/flux-dev")
OUT = os.environ.get("WTY_OUTFILE", "/tmp/diff_dit_shapes.txt")
STEPS = int(os.environ.get("STEPS", "4"))
W = int(os.environ.get("W", "512"))
H = int(os.environ.get("H", "512"))
PROMPT = "a cat"

def log(m): print(f"[diff-dit-shapes] {m}", flush=True)

def main():
    from diffusers import FluxPipeline
    log(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cpu")
    log("loaded")

    common = dict(prompt=PROMPT, width=W, height=H, num_inference_steps=STEPS,
                  guidance_scale=3.5, max_sequence_length=512)

    # warmup
    g = torch.Generator(device="cpu").manual_seed(42)
    _ = pipe(prompt=PROMPT, width=W, height=H, num_inference_steps=1,
             guidance_scale=3.5, max_sequence_length=512, generator=g)

    # --- time transformer forward in isolation via forward hook (wall) ---
    tf = pipe.transformer
    fwd_times = []
    _orig = tf.forward
    def timed_forward(*a, **k):
        t = time.time()
        r = _orig(*a, **k)
        fwd_times.append(time.time() - t)
        return r
    tf.forward = timed_forward
    g = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    _ = pipe(generator=g, **common)
    wall = time.time() - t0
    tf.forward = _orig
    dit_total = sum(fwd_times)
    log(f"full wall {wall:.2f}s | transformer fwd total {dit_total:.3f}s over {len(fwd_times)} calls "
        f"| per-step {dit_total/max(1,len(fwd_times)):.3f}s")

    # --- profiled run with shapes ---
    from torch.profiler import profile, ProfilerActivity
    g = torch.Generator(device="cpu").manual_seed(42)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        _ = pipe(generator=g, **common)

    GEMM = ("addmm","baddbmm","bmm","aten::mm","linear","matmul")
    def is_gemm(n):
        n=n.lower(); return any(k in n for k in GEMM)

    # group by (op, input shapes)
    shapes = {}
    for e in prof.key_averages(group_by_input_shape=True):
        us = getattr(e, "self_cpu_time_total", 0)
        if us <= 0 or not is_gemm(e.key): continue
        key = (e.key, str(getattr(e, "input_shapes", "")))
        d = shapes.setdefault(key, {"us":0.0,"calls":0})
        d["us"] += us; d["calls"] += e.count

    lines = [f"=== diffusers DiT GEMM by shape ({STEPS} steps, {W}x{H}) ===",
             f"full wall {wall:.2f}s | transformer per-step {dit_total/max(1,len(fwd_times)):.3f}s "
             f"({len(fwd_times)} fwd calls)",
             f"{'op':<16}{'calls':>7}{'self_ms':>10}  input_shapes"]
    for (op,shp),d in sorted(shapes.items(), key=lambda x:-x[1]["us"])[:30]:
        lines.append(f"{op[:15]:<16}{d['calls']:>7}{d['us']/1e3:>10.1f}  {shp[:90]}")
    txt="\n".join(lines)
    print(txt)
    with open(OUT,"w") as f: f.write(txt+"\n")
    log(f"written {OUT}")

if __name__ == "__main__":
    main()
