#!/usr/bin/env python
"""Merge a Qwen-Image-Lightning LoRA into base qwen transformer diffusers shards.
W' = W + (alpha/rank) * (lora_up @ lora_down). Outputs a mirror /transformer/ dir.
"""
import json, os, sys, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def main(base_dir, lora_path, out_dir):
    idx_path = os.path.join(base_dir, "diffusion_pytorch_model.safetensors.index.json")
    idx = json.load(open(idx_path))
    weight_map = idx["weight_map"]

    # group base keys by shard
    shards = {}
    for k, shard in weight_map.items():
        shards.setdefault(shard, []).append(k)

    # load lora into module dict
    lf = safe_open(lora_path, "pt")
    lkeys = list(lf.keys())
    mods = {}
    for k in lkeys:
        for suf in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if k.endswith(suf):
                base = k[:-len(suf)]
                mods.setdefault(base, {})[suf] = k
                break
    print(f"lora modules: {len(mods)}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    # copy config + index
    shutil.copy(os.path.join(base_dir, "config.json"), os.path.join(out_dir, "config.json"))
    shutil.copy(idx_path, os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json"))

    merged_count = 0
    for shard, keys in sorted(shards.items()):
        sp = os.path.join(base_dir, shard)
        tensors = {}
        with safe_open(sp, "pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        # merge any weight key with a matching lora module
        for k in list(tensors.keys()):
            if not k.endswith(".weight"):
                continue
            mod = k[:-len(".weight")]
            if mod not in mods:
                continue
            m = mods[mod]
            down = lf.get_tensor(m[".lora_down.weight"]).float()
            up = lf.get_tensor(m[".lora_up.weight"]).float()
            alpha = float(lf.get_tensor(m[".alpha"]).item())
            rank = down.shape[0]
            scale = alpha / rank
            delta = scale * (up @ down)
            W = tensors[k]
            orig_dtype = W.dtype
            if W.shape != delta.shape:
                print(f"  SHAPE MISMATCH {k}: W{tuple(W.shape)} delta{tuple(delta.shape)}", flush=True)
                continue
            tensors[k] = (W.float() + delta).to(orig_dtype)
            merged_count += 1
        save_file(tensors, os.path.join(out_dir, shard), metadata={"format": "pt"})
        print(f"  wrote {shard} ({len(keys)} keys)", flush=True)
    print(f"merged {merged_count} weights into {out_dir}", flush=True)
    assert merged_count == len(mods), f"expected {len(mods)} merges, got {merged_count}"

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
