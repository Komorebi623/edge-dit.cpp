#!/usr/bin/env python
"""Merge a PEFT-style flux LoRA (.lora_A/.lora_B, transformer. prefix, no .alpha)
into base flux/kontext transformer diffusers shards.
W' = W + scale * (lora_B @ lora_A). scale defaults to 1.0 (no .alpha in this LoRA).
Outputs a mirror /transformer/ dir. Verifies the merge actually moved weights.
"""
import json, os, sys, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main(base_dir, lora_path, out_dir, scale=1.0):
    scale = float(scale)
    idx_path = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]

    shards = {}
    for k, shard in weight_map.items():
        shards.setdefault(shard, []).append(k)

    lf = safe_open(lora_path, "pt")
    lkeys = list(lf.keys())
    # group into modules by .lora_A/.lora_B ; strip leading "transformer." to match base layer names
    mods = {}
    for k in lkeys:
        for suf in (".lora_A.weight", ".lora_B.weight"):
            if k.endswith(suf):
                base = k[:-len(suf)]
                if base.startswith("transformer."):
                    base = base[len("transformer."):]
                mods.setdefault(base, {})[suf] = k
                break
    print(f"lora modules: {len(mods)}  scale={scale}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(base_dir, "config.json"), os.path.join(out_dir, "config.json"))
    shutil.copy(idx_path, os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json"))

    merged_count = 0
    max_reldiff = 0.0
    for shard, keys in sorted(shards.items()):
        sp = os.path.join(base_dir, shard)
        tensors = {}
        with safe_open(sp, "pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for k in list(tensors.keys()):
            if not k.endswith(".weight"):
                continue
            mod = k[:-len(".weight")]
            if mod not in mods:
                continue
            m = mods[mod]
            if ".lora_A.weight" not in m or ".lora_B.weight" not in m:
                print(f"  incomplete lora pair for {mod}", flush=True); continue
            A = lf.get_tensor(m[".lora_A.weight"]).float()   # (rank, in)
            B = lf.get_tensor(m[".lora_B.weight"]).float()   # (out, rank)
            delta = scale * (B @ A)                          # (out, in)
            W = tensors[k]
            orig_dtype = W.dtype
            if W.shape != delta.shape:
                print(f"  SHAPE MISMATCH {k}: W{tuple(W.shape)} delta{tuple(delta.shape)}", flush=True)
                continue
            Wf = W.float()
            newW = Wf + delta
            rd = (delta.abs().mean() / (Wf.abs().mean() + 1e-9)).item()
            max_reldiff = max(max_reldiff, rd)
            tensors[k] = newW.to(orig_dtype)
            merged_count += 1
        save_file(tensors, os.path.join(out_dir, shard), metadata={"format": "pt"})
        print(f"  wrote {shard} ({len(keys)} keys)", flush=True)
    print(f"merged {merged_count}/{len(mods)} weights into {out_dir}", flush=True)
    print(f"max per-layer mean|delta|/mean|W| = {max_reldiff:.4f} (should be >> 0.001 for a real distill)", flush=True)
    if merged_count != len(mods):
        print(f"  WARNING: {len(mods)-merged_count} lora modules did not match a base weight", flush=True)

if __name__ == "__main__":
    scale = sys.argv[4] if len(sys.argv) > 4 else 1.0
    main(sys.argv[1], sys.argv[2], sys.argv[3], scale)
