"""imatrix offline calibration tool (pure Python, does not modify C++).

Collects the per-layer, per-input-channel mean-square activation E[x^2] as the quantization
importance (imatrix), feeding it into the weighted rounding of k-quants such as q4_K. The importance
metric borrows AWQ's idea (large activation = important weight column), but here we
only produce imatrix weighting and do not implement AWQ's per-channel scaling.

Load SD3-medium with diffusers, register a forward hook on every nn.Linear of the transformer (DiT),
run the denoising forward on a number of diverse prompts, and collect per-input-channel statistics of
each layer's input activations:
    - sum_x2[c] = sum over all tokens of x[:,c]^2   (-> E[x^2] mean-square activation)
This E[x^2] metric borrows AWQ's saliency idea (large activation = important weight column), and is also the diagonal approximation of the GPTQ Hessian (X X^T).
The resulting per-input-channel importance vector has length = in_features = n_per_row in edge,
which feeds exactly into edge's ggml_quantize_chunk (imatrix interface), with zero runtime changes.

At the same time, keep a small batch of real activation samples per layer (reservoir), used during offline verification to compute the "output-domain error"
(||W X - Wq X|| / ||W X||) -- the most direct proxy for image quality.

Outputs are written to --outdir (default ./imatrix-out/ under the current directory):
    imatrix.npz   per-layer imatrix vectors (float32) + count
    imatrix.gguf  canonical gguf format (for future C++ integration)
    acts.npz      per-layer activation samples (float16, <=SAMPLE_ROWS rows per layer)
"""
import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch

def log(*a):
    print(*a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to the model (diffusers dir or safetensors), e.g. /path/to/sd3-medium")
    ap.add_argument("--outdir", default="imatrix-out",
                    help="calibration outputs (imatrix.gguf etc), default ./imatrix-out")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--nprompts", type=int, default=16)
    ap.add_argument("--sample-rows", type=int, default=256, help="number of activation sample rows kept per layer")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.nprompts, args.sample_rows = 2, 2, 64
    os.makedirs(args.outdir, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline

    dev = "cuda"
    log(f"[load] {args.model} steps={args.steps} nprompts={args.nprompts}")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to(dev)
    tf = pipe.transformer
    tf.eval()

    # ---- Register hook: collect input activation statistics for each nn.Linear ----
    stats = {}   # name -> dict(sum_x2 float64[in], count int, sample float16[R,in])
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            if x is None:
                return
            xf = x.detach().reshape(-1, x.shape[-1]).float()  # [N, in]
            s = stats.get(name)
            sx2 = (xf * xf).sum(dim=0).double().cpu().numpy()  # [in]
            n = xf.shape[0]
            if s is None:
                # Initialize + take the activation sample from the first batch (first SAMPLE_ROWS rows)
                r = min(args.sample_rows, xf.shape[0])
                stats[name] = {
                    "sum_x2": sx2,
                    "count": int(n),
                    "in": int(xf.shape[1]),
                    "sample": xf[:r].half().cpu().numpy(),
                }
            else:
                s["sum_x2"] += sx2
                s["count"] += int(n)
        return hook

    n_lin = 0
    for name, mod in tf.named_modules():
        if isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))
            n_lin += 1
    log(f"[hook] registered {n_lin} nn.Linear")

    # ---- Diverse calibration prompts ----
    base_prompts = [
        "a photograph of an astronaut riding a horse on the moon",
        "a bustling medieval marketplace at golden hour, highly detailed",
        "close-up portrait of an elderly fisherman, weathered skin, studio light",
        "a serene japanese zen garden with cherry blossoms, misty morning",
        "a futuristic cyberpunk city street at night, neon reflections, rain",
        "a bowl of ramen with steam, food photography, shallow depth of field",
        "an oil painting of a stormy sea with a lighthouse",
        "a cute corgi puppy playing in autumn leaves, bokeh",
        "architectural render of a modern glass museum, blue sky",
        "a fantasy dragon perched on a snowy mountain peak, epic lighting",
        "macro shot of a dewdrop on a spider web at dawn",
        "a vintage red sports car parked on a coastal road",
        "abstract geometric pattern, vibrant colors, bauhaus style",
        "a wizard casting a glowing spell in a dark forest, cinematic",
        "flat lay of watercolor art supplies on a wooden desk",
        "a majestic tiger walking through tall grass, wildlife photography",
    ]
    prompts = base_prompts[:args.nprompts]

    gen = torch.Generator(device=dev).manual_seed(1234)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            _ = pipe(
                p,
                num_inference_steps=args.steps,
                guidance_scale=7.0,
                height=1024, width=1024,
                generator=gen,
                output_type="latent",
            )
            log(f"[calib] {i+1}/{len(prompts)} done: {p[:40]}")

    for h in handles:
        h.remove()

    # ---- Produce imatrix: E[x^2] = sum_x2 / count ----
    imatrix = {}
    acts = {}
    meta = {}
    for name, s in stats.items():
        ex2 = (s["sum_x2"] / max(1, s["count"])).astype(np.float32)  # [in]
        imatrix[name] = ex2
        acts[name] = s["sample"]
        meta[name] = np.array([s["count"], s["in"]], dtype=np.int64)

    np.savez(os.path.join(args.outdir, "imatrix.npz"), **imatrix,
             **{f"__meta__{k}": v for k, v in meta.items()})
    np.savez(os.path.join(args.outdir, "acts.npz"), **acts)
    log(f"[save] imatrix.npz / acts.npz : {len(imatrix)} layers")

    # ---- Canonical gguf (for future C++ integration), one float32 tensor per layer ----
    try:
        import gguf
        w = gguf.GGUFWriter(os.path.join(args.outdir, "imatrix.gguf"), "sd3-dit-imatrix")
        w.add_uint32("imatrix.n_tensors", len(imatrix))
        w.add_string("imatrix.source", "sd3-medium Ex2 activation calibration")
        for name, v in imatrix.items():
            # gguf tensor name: use the diffusers weight name + .weight (mapped to the edge tensor name during C++ integration)
            w.add_tensor(name + ".weight", v.reshape(1, -1))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        log("[save] imatrix.gguf")
    except Exception as e:
        log(f"[warn] gguf export failed (does not affect verification): {e}")

    # Print the imatrix distribution of a few representative layers, to visually inspect outlier channels
    for name in list(imatrix.keys())[:3] + [k for k in imatrix if "attn.to_q" in k][:1]:
        v = imatrix[name]
        log(f"[imatrix] {name}: in={v.size} max/mean={v.max()/v.mean():.1f} "
            f"top1%/median={np.percentile(v,99)/ (np.median(v)+1e-12):.1f}")
    log("[done] calibrate")

if __name__ == "__main__":
    main()
