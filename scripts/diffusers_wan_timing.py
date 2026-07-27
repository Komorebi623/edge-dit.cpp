#!/usr/bin/env python3
"""Diffusers Wan2.1-T2V-1.3B CPU 视频生成 端到端+各阶段(text/DiT/VAE)墙钟计时。
口径与 edge-dit 的 wan 三段打点(text condition/sampling/vae decode)对齐。
  - 各阶段独立计时, 不含模型加载/import/warmup
用法: STEPS=4 W=256 H=256 FRAMES=9 python3 scripts/diffusers_wan_timing.py
默认 CPU + bf16。串行跑, 自己看 load。
"""
import os, sys, time
import torch

MODEL = os.environ.get("WAN_MODEL", "/export/home/wangtianyang.21/wan_models/Wan2.1-T2V-1.3B-Diffusers")
STEPS = int(os.environ.get("STEPS", "4"))
W = int(os.environ.get("W", "256"))
H = int(os.environ.get("H", "256"))
FRAMES = int(os.environ.get("FRAMES", "9"))
SEED = int(os.environ.get("SEED", "42"))
PROMPT = os.environ.get("PROMPT", "a cat walking on grass")
OUTDIR = os.environ.get("OUTDIR", "bench_results/diffusers")
WARMUP = os.environ.get("NOWARMUP", "") == ""

def log(m): print(f"[cmp] {m}", flush=True)
def loadavg():
    with open("/proc/loadavg") as f: return f.read().split()[0]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    from diffusers import WanPipeline
    log(f"=== wan-t2v-1.3b | {W}x{H} frames={FRAMES} {STEPS}step seed={SEED} ===")
    log(f"torch {torch.__version__}, threads={torch.get_num_threads()}, load(1m)={loadavg()}")

    t0 = time.time()
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cpu")
    log(f"model loaded in {time.time()-t0:.1f}s (不计入端到端)")

    common = dict(prompt=PROMPT, negative_prompt="", width=W, height=H,
                  num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0,
                  generator=torch.Generator(device="cpu").manual_seed(SEED))

    if WARMUP:
        log("warmup 1 step ...")
        wu = dict(common); wu["num_inference_steps"] = 1
        wu["generator"] = torch.Generator(device="cpu").manual_seed(SEED)
        _ = pipe(**wu)

    timers = {"text": 0.0, "dit": 0.0, "vae": 0.0}
    def wrap(mod, key, meth="forward"):
        if mod is None: return
        fn = getattr(mod, meth)
        def timed(*a, **kw):
            s = time.time(); r = fn(*a, **kw); timers[key] += time.time()-s
            return r
        setattr(mod, meth, timed)
    wrap(getattr(pipe, "transformer", None), "dit")
    wrap(getattr(pipe, "text_encoder", None), "text")
    # Wan VAE 是 AutoencoderKLWan, decode 方法计时
    wrap(getattr(pipe, "vae", None), "vae", "decode")

    common["generator"] = torch.Generator(device="cpu").manual_seed(SEED)
    log(f"load(1m) before generate = {loadavg()}")
    t0 = time.time()
    out = pipe(**common)
    wall = time.time() - t0

    import numpy as np
    frames = out.frames[0]  # (F,H,W,C) np
    arr = np.asarray(frames).astype(float)
    log(f"END-TO-END WALL (generate, {STEPS} steps, {FRAMES} frames): {wall:.2f}s   [load after={loadavg()}]")
    log(f"  text_encode : {timers['text']:.2f}s")
    log(f"  sampling(DiT): {timers['dit']:.2f}s")
    log(f"  vae_decode  : {timers['vae']:.2f}s")
    log(f"  other       : {wall - sum(timers.values()):.2f}s")
    log(f"  out shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")

if __name__ == "__main__":
    main()
