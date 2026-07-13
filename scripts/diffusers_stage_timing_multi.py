#!/usr/bin/env python3
"""Diffusers CPU 多模型 端到端 + 各阶段(text/DiT/VAE) 墙钟计时。

两种口径(用户 2026-07-11 明确要求以"命令→拿图"完整墙钟为准):
  - 进程墙钟 = 从 python 启动到存图退出的全过程,含 import + 模型加载 + 生成。
    由外层 `/usr/bin/time -v` 抓取,与 edge-dit 的 `/usr/bin/time` wall 严格同口径。
    NOWARMUP=1 关掉 warmup(单次冷出图才符合"命令→拿图"),默认已关。
  - 分阶段(text/DiT/VAE) = 纯生成拆解,**不含加载**,仅用于定位瓶颈落在哪个板块。
    这段与 edge-dit `generate breakdown` 对齐。
用法:
  MODEL_KEY=sd3|flux|qwen-image|qwen-image-edit STEPS=4 W=512 H=512 python3 diffusers_stage_timing_multi.py
默认 CPU + bf16 + 无 warmup。串行跑,跑前后自己看 load。
"""
import os, sys, time
_T_PROC_START = time.time()  # 尽量早地打点; 外层 /usr/bin/time 是权威进程墙钟, 这个用于自洽核对
import torch

MODELS = {
    "sd3":             ("/mnt/cfs/9n-das-admin/llm_models/stable-diffusion-3-medium-diffusers", "StableDiffusion3Pipeline"),
    "flux":            ("/mnt/cfs/9n-das-admin/llm_models/flux-dev",                            "FluxPipeline"),
    "qwen-image":      ("/mnt/cfs/9n-das-admin/llm_models/qwen-image",                          "QwenImagePipeline"),
    "qwen-image-edit": ("/mnt/cfs/9n-das-admin/llm_models/Qwen-Image-Edit-2509",                "QwenImageEditPipeline"),
}

MODEL_KEY = os.environ.get("MODEL_KEY", "flux")
STEPS = int(os.environ.get("STEPS", "4"))
W = int(os.environ.get("W", "512"))
H = int(os.environ.get("H", "512"))
SEED = int(os.environ.get("SEED", "42"))
PROMPT = os.environ.get("PROMPT", "a cat")
OUTDIR = os.environ.get("OUTDIR", "bench_results/diffusers")
WARMUP = os.environ.get("WARMUP", "") == "1"  # 默认不 warmup(单次冷出图,符合"命令→拿图"口径); WARMUP=1 开

def log(m): print(f"[cmp] {m}", flush=True)

def loadavg():
    with open("/proc/loadavg") as f: return f.read().split()[0]

def build_pipe(cls_name, path):
    import diffusers
    cls = getattr(diffusers, cls_name)
    return cls.from_pretrained(path, torch_dtype=torch.bfloat16).to("cpu")

def main():
    if MODEL_KEY not in MODELS:
        log(f"未知 MODEL_KEY={MODEL_KEY}, 可选: {list(MODELS)}"); sys.exit(1)
    path, cls_name = MODELS[MODEL_KEY]
    os.makedirs(OUTDIR, exist_ok=True)
    outimg = os.path.join(OUTDIR, f"{MODEL_KEY}_{W}x{H}_{STEPS}step.png")

    log(f"=== {MODEL_KEY} | {cls_name} | {W}x{H} {STEPS}step seed={SEED} ===")
    log(f"torch {torch.__version__}, threads={torch.get_num_threads()}, load(1m)={loadavg()}")

    t0 = time.time()
    pipe = build_pipe(cls_name, path)
    log(f"model loaded in {time.time()-t0:.1f}s (不计入端到端)")

    # 各模型 pipe 调用参数(仅生成必需项，避免口径漂移)
    common = dict(prompt=PROMPT, width=W, height=H, num_inference_steps=STEPS,
                  generator=torch.Generator(device="cpu").manual_seed(SEED))
    # 模型特定参数
    if MODEL_KEY == "flux":
        common.update(guidance_scale=3.5, max_sequence_length=512)
    elif MODEL_KEY == "sd3":
        common.update(guidance_scale=7.0)
    elif MODEL_KEY in ("qwen-image", "qwen-image-edit"):
        common.update(true_cfg_scale=4.0)
    # qwen-image-edit 需要输入图, 用与 edge 相同的占位图文件(口径一致)
    if MODEL_KEY == "qwen-image-edit":
        from PIL import Image
        common.pop("width", None); common.pop("height", None)
        edit_input = os.environ.get("EDIT_INPUT", "bench_results/edit_input.png")
        common["image"] = Image.open(edit_input).convert("RGB") if os.path.exists(edit_input) \
                          else Image.new("RGB", (W, H), (127, 127, 127))

    if WARMUP:
        log("warmup 1 step ...")
        wu = dict(common); wu["num_inference_steps"] = 1
        wu["generator"] = torch.Generator(device="cpu").manual_seed(SEED)
        _ = pipe(**wu)

    # 各阶段 wall hook
    timers = {"text": 0.0, "dit": 0.0, "vae": 0.0}
    def wrap(mod, key):
        if mod is None: return
        fwd = mod.forward
        def timed(*a, **kw):
            s = time.time(); r = fwd(*a, **kw); timers[key] += time.time()-s
            return r
        mod.forward = timed
    wrap(getattr(pipe, "transformer", None), "dit")
    wrap(getattr(getattr(pipe, "vae", None), "decoder", None), "vae")
    for te in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        wrap(getattr(pipe, te, None), "text")

    common["generator"] = torch.Generator(device="cpu").manual_seed(SEED)
    log(f"load(1m) before generate = {loadavg()}")
    t0 = time.time()
    out = pipe(**common)
    wall = time.time() - t0
    img = out.images[0]
    img.save(outimg)

    import numpy as np
    arr = np.asarray(img.convert("RGB")).astype(float)
    proc_wall = time.time() - _T_PROC_START  # import→save 的进程内墙钟(≈外层 /usr/bin/time)
    log(f"PROCESS WALL (import→save, 含加载): {proc_wall:.2f}s   <- 与 edge /usr/bin/time 同口径")
    log(f"END-TO-END WALL (generate only, {STEPS} steps): {wall:.2f}s   [load(1m) after={loadavg()}]")
    log(f"  text_encode : {timers['text']:.2f}s")
    log(f"  sampling(DiT): {timers['dit']:.2f}s")
    log(f"  vae_decode  : {timers['vae']:.2f}s")
    log(f"  other       : {wall - sum(timers.values()):.2f}s")
    log(f"  out mean={arr.mean():.2f} std={arr.std():.2f}  saved {outimg}")

if __name__ == "__main__":
    main()
