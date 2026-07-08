#!/usr/bin/env python3
"""Diffusers Flux-dev CPU: end-to-end + per-stage (text/DiT/VAE) wall timing,
matching edge-dit's stage breakdown for a like-for-like comparison.
Config: a cat / 512 / 4 steps / guidance 3.5 / seed 42, bf16, CPU.
Run in container wty-edgedit-dev.
"""
import os, time
import torch

MODEL = os.environ.get("FLUX_MODEL", "/models/flux-dev")
OUTIMG = os.environ.get("WTY_OUTIMG", "/workspace/edge-dit.cpp/.wty_gap_probe/compare/diffusers_cat.png")
STEPS = int(os.environ.get("STEPS", "4"))
W = int(os.environ.get("W", "512"))
H = int(os.environ.get("H", "512"))
PROMPT = "a cat"

def log(m): print(f"[cmp] {m}", flush=True)

def main():
    from diffusers import FluxPipeline
    log(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cpu")
    log(f"model loaded in {time.time()-t0:.1f}s")

    # stage timers via hooks
    stages = {}
    # time text encoders
    import types
    common = dict(prompt=PROMPT, width=W, height=H, num_inference_steps=STEPS,
                  guidance_scale=3.5, max_sequence_length=512)

    # warmup
    log("warmup 1 step ...")
    g = torch.Generator(device="cpu").manual_seed(42)
    _ = pipe(prompt=PROMPT, width=W, height=H, num_inference_steps=1,
             guidance_scale=3.5, max_sequence_length=512, generator=g)

    # --- instrument: wrap transformer + vae + text_encoder(s) to sum their wall time ---
    timers = {"text": 0.0, "dit": 0.0, "vae": 0.0}
    def wrap(mod, key):
        fwd = mod.forward
        def timed(*a, **kw):
            s = time.time(); r = fwd(*a, **kw); timers[key] += time.time()-s
            return r
        mod.forward = timed
    wrap(pipe.transformer, "dit")
    wrap(pipe.vae.decoder, "vae")
    wrap(pipe.text_encoder, "text")
    wrap(pipe.text_encoder_2, "text")

    g = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.time()
    img = pipe(generator=g, **common).images[0]
    wall = time.time() - t0
    img.save(OUTIMG)

    log(f"END-TO-END WALL (generate, {STEPS} steps): {wall:.2f}s")
    log(f"  text encode (T5+CLIP): {timers['text']:.2f}s")
    log(f"  DiT sampling ({STEPS} steps): {timers['dit']:.2f}s")
    log(f"  VAE decode: {timers['vae']:.2f}s")
    log(f"  (other/overhead: {wall - sum(timers.values()):.2f}s)")
    log(f"saved {OUTIMG}")

if __name__ == "__main__":
    main()
