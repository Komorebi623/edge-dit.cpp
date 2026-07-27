#!/usr/bin/env python3
"""Diffusers Wan2.1-T2V 分段直接实测: import / load(+bf16) / generate(text+DiT+vae) / teardown.
每段用 time.time() 直接测量, 各段之和 + 残差 = 进程内墙钟, 与 edge 的 [ED_WALL] 四段同粒度对齐。
所有数字均为实测, 不做减法推算。
用法: W=256 H=256 FRAMES=9 STEPS=4 SEED=42 WAN_MODEL=<path> python3 diffusers_wan_segwall.py
"""
import os, time, gc

_proc_t0 = time.time()

MODEL = os.environ["WAN_MODEL"]
W = int(os.environ.get("W", "256"))
H = int(os.environ.get("H", "256"))
FRAMES = int(os.environ.get("FRAMES", "9"))
STEPS = int(os.environ.get("STEPS", "4"))
SEED = int(os.environ.get("SEED", "42"))
PROMPT = os.environ.get("PROMPT", "a cat walking on grass")


def log(m):
    print(f"[segwall] {m}", flush=True)


def main():
    # 段1: import (含 torch/diffusers 加载, 从进程启动算起)
    t_imp0 = time.time()
    import torch
    from diffusers import WanPipeline
    import numpy as np
    seg_import = time.time() - t_imp0
    # 进程启动到 import 开始的解释器开销也算进 import 段
    seg_import_full = time.time() - _proc_t0

    # 段2: load + f32->bf16 转换
    t_load0 = time.time()
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cpu")
    seg_load = time.time() - t_load0

    # 段2.5: 各阶段打点 (text/DiT/vae), 不计入总墙钟拆分, 仅用于定位
    timers = {"text": 0.0, "dit": 0.0, "vae": 0.0}

    def wrap(mod, key, meth="forward"):
        if mod is None:
            return
        fn = getattr(mod, meth)

        def timed(*a, **kw):
            s = time.time()
            r = fn(*a, **kw)
            timers[key] += time.time() - s
            return r
        setattr(mod, meth, timed)
    wrap(getattr(pipe, "transformer", None), "dit")
    wrap(getattr(pipe, "text_encoder", None), "text")
    wrap(getattr(pipe, "vae", None), "vae", "decode")

    common = dict(prompt=PROMPT, negative_prompt="", width=W, height=H,
                  num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0,
                  generator=torch.Generator(device="cpu").manual_seed(SEED))

    # 段3: generate (含 text+DiT+vae+调度+后处理, 全部在这一段内)
    t_gen0 = time.time()
    out = pipe(**common)
    seg_gen = time.time() - t_gen0

    frames = out.frames[0]
    arr = np.asarray(frames).astype(float)

    # 段4: teardown (释放 pipe, 与 edge 的 free_context 对齐)
    t_free0 = time.time()
    del out, pipe
    gc.collect()
    seg_free = time.time() - t_free0

    proc_wall = time.time() - _proc_t0
    residual = proc_wall - (seg_import_full + seg_load + seg_gen + seg_free)

    log(f"=== wan-t2v-1.3b | {W}x{H} frames={FRAMES} {STEPS}step seed={SEED} out={arr.shape} ===")
    log(f"[SEG] import={seg_import_full*1000:.0f}ms load={seg_load*1000:.0f}ms "
        f"gen={seg_gen*1000:.0f}ms free={seg_free*1000:.0f}ms residual={residual*1000:.0f}ms")
    log(f"[SEG] PROC_WALL(in-python)={proc_wall*1000:.0f}ms")
    log(f"  gen breakdown: text={timers['text']:.2f}s DiT={timers['dit']:.2f}s vae={timers['vae']:.2f}s")
    log(f"  out mean={arr.mean():.4f} std={arr.std():.4f}")


if __name__ == "__main__":
    main()
