#!/usr/bin/env python
"""Merge a Qwen-Image-Lightning LoRA into base qwen transformer diffusers shards.
W' = W + (alpha/rank) * (lora_up @ lora_down). Outputs a mirror /transformer/ dir.

The output dir name is tagged with the LoRA's step count (e.g. `...-4steps`) when
the source filename carries one, so edge-dit's distilled-step detection reads the
right few-step default from the path. Without the tag it would fall back to 8.
"""
import json, os, re, sys, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Same marker the C++ detector scans for (detect_distilled_default_steps).
_STEP_RE = re.compile(r"([0-9]{1,2})[ _-]?steps?", re.IGNORECASE)
_COMPONENT_LEAVES = {"transformer", "dit"}


def step_tagged_out_dir(out_dir, lora_path):
    """Return out_dir with an `Nsteps` marker (taken from the LoRA filename) so
    edge-dit's distilled-step detection reads the correct few-step default from
    the path. Without a marker the runtime falls back to 8 steps.

    The marker is applied to the model directory rather than a `transformer`/`dit`
    component subdirectory, keeping the component name unchanged. If out_dir (or
    the LoRA filename) already carries a step marker, out_dir is returned as-is.
    """
    norm = out_dir.rstrip("/")
    if _STEP_RE.search(norm):
        return out_dir  # already tagged somewhere in the path
    m = _STEP_RE.search(os.path.basename(lora_path))
    if not m:
        print(f"note: the LoRA filename '{os.path.basename(lora_path)}' has no step "
              f"count, so the output path is left untagged and the runtime will "
              f"default to 8 steps. Include an 'Nsteps' marker in the output "
              f"directory, or pass --steps at run time.", flush=True)
        return out_dir
    tag = f"{m.group(1)}steps"
    head, leaf = os.path.split(norm)
    if leaf.lower() in _COMPONENT_LEAVES and head:
        parent, model_dir = os.path.split(head)
        tagged = os.path.join(parent, f"{model_dir}-{tag}", leaf)
    else:
        tagged = f"{norm}-{tag}"
    print(f"tagging output directory with step count from LoRA: {tagged}", flush=True)
    return tagged


def main(base_dir, lora_path, out_dir):
    out_dir = step_tagged_out_dir(out_dir, lora_path)
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
