#!/usr/bin/env python3
"""Rename Diffusers SD3 transformer tensors into stable-diffusion.cpp names."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from safetensors import safe_open
from safetensors.torch import save_file
import torch


DIRECT_MAP = {
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "pos_embed.pos_embed": "pos_embed",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "proj_out.weight": "final_layer.linear.weight",
    "proj_out.bias": "final_layer.linear.bias",
    "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
    "norm_out.linear.bias": "final_layer.adaLN_modulation.1.bias",
    "context_embedder.weight": "context_embedder.weight",
    "context_embedder.bias": "context_embedder.bias",
}

BLOCK_MAP = {
    "norm1.linear.weight": "x_block.adaLN_modulation.1.weight",
    "norm1.linear.bias": "x_block.adaLN_modulation.1.bias",
    "norm1_context.linear.weight": "context_block.adaLN_modulation.1.weight",
    "norm1_context.linear.bias": "context_block.adaLN_modulation.1.bias",
    "attn.to_q.weight": "x_block.attn.qkv.weight",
    "attn.to_q.bias": "x_block.attn.qkv.bias",
    "attn.to_k.weight": "x_block.attn.qkv.weight.1",
    "attn.to_k.bias": "x_block.attn.qkv.bias.1",
    "attn.to_v.weight": "x_block.attn.qkv.weight.2",
    "attn.to_v.bias": "x_block.attn.qkv.bias.2",
    "attn.add_q_proj.weight": "context_block.attn.qkv.weight",
    "attn.add_q_proj.bias": "context_block.attn.qkv.bias",
    "attn.add_k_proj.weight": "context_block.attn.qkv.weight.1",
    "attn.add_k_proj.bias": "context_block.attn.qkv.bias.1",
    "attn.add_v_proj.weight": "context_block.attn.qkv.weight.2",
    "attn.add_v_proj.bias": "context_block.attn.qkv.bias.2",
    "attn2.to_q.weight": "x_block.attn2.qkv.weight",
    "attn2.to_q.bias": "x_block.attn2.qkv.bias",
    "attn2.to_k.weight": "x_block.attn2.qkv.weight.1",
    "attn2.to_k.bias": "x_block.attn2.qkv.bias.1",
    "attn2.to_v.weight": "x_block.attn2.qkv.weight.2",
    "attn2.to_v.bias": "x_block.attn2.qkv.bias.2",
    "attn2.add_q_proj.weight": "context_block.attn2.qkv.weight",
    "attn2.add_q_proj.bias": "context_block.attn2.qkv.bias",
    "attn2.add_k_proj.weight": "context_block.attn2.qkv.weight.1",
    "attn2.add_k_proj.bias": "context_block.attn2.qkv.bias.1",
    "attn2.add_v_proj.weight": "context_block.attn2.qkv.weight.2",
    "attn2.add_v_proj.bias": "context_block.attn2.qkv.bias.2",
    "attn.norm_q.weight": "x_block.attn.ln_q.weight",
    "attn.norm_k.weight": "x_block.attn.ln_k.weight",
    "attn.norm_added_q.weight": "context_block.attn.ln_q.weight",
    "attn.norm_added_k.weight": "context_block.attn.ln_k.weight",
    "attn2.norm_q.weight": "x_block.attn2.ln_q.weight",
    "attn2.norm_k.weight": "x_block.attn2.ln_k.weight",
    "ff.net.0.proj.weight": "x_block.mlp.fc1.weight",
    "ff.net.0.proj.bias": "x_block.mlp.fc1.bias",
    "ff.net.2.weight": "x_block.mlp.fc2.weight",
    "ff.net.2.bias": "x_block.mlp.fc2.bias",
    "ff_context.net.0.proj.weight": "context_block.mlp.fc1.weight",
    "ff_context.net.0.proj.bias": "context_block.mlp.fc1.bias",
    "ff_context.net.2.weight": "context_block.mlp.fc2.weight",
    "ff_context.net.2.bias": "context_block.mlp.fc2.bias",
    "attn.to_out.0.weight": "x_block.attn.proj.weight",
    "attn.to_out.0.bias": "x_block.attn.proj.bias",
    "attn.to_add_out.weight": "context_block.attn.proj.weight",
    "attn.to_add_out.bias": "context_block.attn.proj.bias",
    "attn2.to_out.0.weight": "x_block.attn2.proj.weight",
    "attn2.to_out.0.bias": "x_block.attn2.proj.bias",
    "attn2.to_add_out.weight": "context_block.attn2.proj.weight",
    "attn2.to_add_out.bias": "context_block.attn2.proj.bias",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    intermediate = {}
    with safe_open(args.input, framework="pt", device="cpu") as source:
        for key in source.keys():
            mapped = map_key(key)
            if mapped is None:
                raise SystemExit(f"no SD3 stable-diffusion.cpp mapping for tensor: {key}")
            intermediate[f"model.diffusion_model.{mapped}"] = source.get_tensor(key)

    tensors = merge_qkv_tensors(intermediate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={"format": "pt"})
    print(f"wrote {len(tensors)} tensors to {args.output}")
    return 0


def map_key(key: str) -> str | None:
    direct = DIRECT_MAP.get(key)
    if direct is not None:
        return direct
    match = re.match(r"transformer_blocks\.(\d+)\.(.*)", key)
    if not match:
        return None
    block, suffix = match.groups()
    mapped = BLOCK_MAP.get(suffix)
    if mapped is None:
        return None
    return f"joint_blocks.{block}.{mapped}"


def merge_qkv_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for key, tensor in tensors.items():
        if key in consumed or key.endswith(".1") or key.endswith(".2"):
            continue
        part_1 = f"{key}.1"
        part_2 = f"{key}.2"
        if part_1 in tensors and part_2 in tensors:
            merged[key] = torch.cat([tensor, tensors[part_1], tensors[part_2]], dim=0)
            consumed.update({key, part_1, part_2})
        else:
            merged[key] = tensor
            consumed.add(key)

    leftovers = sorted(set(tensors) - consumed)
    if leftovers:
        raise SystemExit(
            "unmerged SD3 tensor fragments remain: " + ", ".join(leftovers[:10])
        )
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
