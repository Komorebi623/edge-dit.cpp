#!/usr/bin/env python3
"""Merge a MiniMax-H3 Turbo PEFT LoRA into Diffusers transformer shards."""

import argparse
import json
import hashlib
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"
STEP_RE = re.compile(r"([0-9]{1,2})[ _-]?steps?", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Diffusers transformer directory or shard index")
    parser.add_argument("lora", type=Path, help="MiniMax-H3 Turbo Diffusers LoRA safetensors")
    parser.add_argument("output", type=Path, help="Output transformer directory")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="CPU-only offline merge; never allocates CUDA memory")
    parser.add_argument("--alpha", type=float, help="Override the checkpoint alpha metadata")
    parser.add_argument("--scale", type=float, default=1.0, help="Additional LoRA multiplier")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing shards")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-verified completed shards")
    parser.add_argument("--steps", type=int, choices=[4, 8], required=True, help="Published adapter NFE; not inferred from filenames")
    parser.add_argument("--video-shift", type=float, required=True, help="Published adapter video flow shift")
    parser.add_argument("--audio-shift", type=float, default=3.0, help="Published adapter audio flow shift")
    return parser.parse_args()


def resolve_index(base):
    index = base if base.is_file() else base / INDEX_NAME
    if not index.is_file():
        raise FileNotFoundError(f"MiniMax-H3 transformer index does not exist: {index}")
    return index.resolve()


def tagged_output_path(output, lora):
    output_text = str(output).rstrip("/")
    source_name = lora.name.lower()
    tags = []
    if not STEP_RE.search(output_text):
        match = STEP_RE.search(source_name)
        if match:
            tags.append(f"{match.group(1)}steps")
    if "768p" in source_name and "768p" not in output_text.lower():
        tags.append("768p")
    if not tags:
        return output
    tagged = Path(f"{output_text}-{'-'.join(tags)}")
    print(f"tagging output for runtime defaults: {tagged}", flush=True)
    return tagged


def resolve_device(requested):
    if requested != "cpu":
        raise ValueError("H3 research merger is CPU-only; GPU merges are disabled")
    return torch.device("cpu")

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()

def safe_shard_path(directory, name):
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
        raise ValueError("Shard name must be a local safetensors basename")
    path = directory / name
    if path.resolve().parent != directory.resolve():
        raise ValueError("Shard path escapes its directory")
    return path


def load_adapter(lora_path, alpha_override):
    if not lora_path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 Turbo LoRA does not exist: {lora_path}")
    adapter = safe_open(lora_path, framework="pt", device="cpu")
    metadata = adapter.metadata() or {}
    key_format = metadata.get("key_format")
    if key_format not in (None, "minimax-h3-diffusers"):
        raise ValueError(f"unsupported MiniMax-H3 LoRA key_format={key_format!r}")
    modules = {}
    unsupported = []
    for key in adapter.keys():
        if key.endswith(LORA_A_SUFFIX):
            modules.setdefault(key[: -len(LORA_A_SUFFIX)], {})["a"] = key
        elif key.endswith(LORA_B_SUFFIX):
            modules.setdefault(key[: -len(LORA_B_SUFFIX)], {})["b"] = key
        else:
            unsupported.append(key)
    if unsupported:
        raise ValueError(f"unsupported LoRA tensor keys: {unsupported[:3]}")
    incomplete = [name for name, tensors in modules.items() if set(tensors) != {"a", "b"}]
    if incomplete:
        raise ValueError(f"unpaired LoRA tensors for modules: {incomplete[:3]}")
    if not modules:
        raise ValueError(f"no PEFT LoRA tensors found in {lora_path}")
    alpha_text = alpha_override if alpha_override is not None else metadata.get("alpha")
    if alpha_text is None:
        raise ValueError("LoRA alpha is absent; pass --alpha explicitly")
    alpha = float(alpha_text)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"LoRA alpha must be finite and positive, got {alpha}")
    return adapter, modules, metadata, alpha


def validate_adapter(index, adapter, modules):
    index_data = json.loads(index.read_text())
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {index}")
    missing = sorted(f"{name}.weight" for name in modules if f"{name}.weight" not in weight_map)
    if missing:
        raise ValueError(f"LoRA modules missing from the base transformer: {missing[:3]}")
    shard_modules = {}
    for module_name, tensor_names in modules.items():
        shard_name = weight_map[f"{module_name}.weight"]
        shard_modules.setdefault(shard_name, []).append((module_name, tensor_names))
    for shard_name, entries in shard_modules.items():
        shard_path = safe_shard_path(index.parent, shard_name)
        if not shard_path.is_file():
            raise FileNotFoundError(f"base transformer shard does not exist: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            for module_name, tensor_names in entries:
                weight_name = f"{module_name}.weight"
                if weight_name not in shard_keys:
                    raise ValueError(f"{weight_name} is mapped to {shard_name} but is absent")
                weight_shape = tuple(shard.get_slice(weight_name).get_shape())
                a_shape = tuple(adapter.get_slice(tensor_names["a"]).get_shape())
                b_shape = tuple(adapter.get_slice(tensor_names["b"]).get_shape())
                if len(weight_shape) != 2 or len(a_shape) != 2 or len(b_shape) != 2:
                    raise ValueError(f"LoRA merge requires matrices for {module_name}")
                if a_shape[0] != b_shape[1] or (b_shape[0], a_shape[1]) != weight_shape:
                    raise ValueError(
                        f"shape mismatch for {module_name}: W{weight_shape}, A{a_shape}, B{b_shape}"
                    )
    return index_data, shard_modules


def output_shard_complete(path, expected_keys):
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as shard:
            return set(shard.keys()) == expected_keys
    except Exception:
        return False


def merge_weight(weight, lora_a, lora_b, alpha, scale, device):
    rank = lora_a.shape[0]
    if rank <= 0 or not all(bool(torch.isfinite(t).all()) for t in (weight, lora_a, lora_b)):
        raise ValueError("LoRA inputs must be finite with positive rank")
    multiplier = scale * alpha / rank
    if not math.isfinite(multiplier): raise ValueError("LoRA multiplier overflow")
    weight_device = weight.to(device=device, dtype=torch.float32)
    lora_a_device = lora_a.to(device=device, dtype=torch.float32)
    lora_b_device = lora_b.to(device=device, dtype=torch.float32)
    weight_device.addmm_(lora_b_device, lora_a_device, beta=1.0, alpha=multiplier)
    if not bool(torch.isfinite(weight_device).all()):
        raise ValueError("LoRA merged FP32 weights are non-finite")
    merged = weight_device.to(dtype=weight.dtype, device="cpu")
    if not bool(torch.isfinite(merged).all()):
        raise ValueError("LoRA weights overflowed after output dtype conversion")
    del weight_device, lora_a_device, lora_b_device
    return merged


def merge(args):
    index = resolve_index(args.base)
    output = args.output.resolve()
    if output == index.parent or index.parent in output.parents:
        raise ValueError("Output must be separate from the base model directory")
    if not math.isfinite(args.scale) or args.scale <= 0:
        raise ValueError("LoRA scale must be finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.video_shift, args.audio_shift)):
        raise ValueError("Published flow shifts must be finite and positive")
    device = resolve_device(args.device)
    adapter, modules, metadata, alpha = load_adapter(args.lora.resolve(), args.alpha)
    index_data, shard_modules = validate_adapter(index, adapter, modules)
    ranks = {adapter.get_slice(tensors["a"]).get_shape()[0] for tensors in modules.values()}
    print(
        f"validated {len(modules)} modules; alpha={alpha:g}; ranks={sorted(ranks)}; "
        f"scale={args.scale:g}; device={device}",
        flush=True,
    )
    if args.dry_run:
        return output
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty; pass --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = index.parent / "config.json"

    identity = {"base_index_sha256": sha256_file(index), "lora_sha256": sha256_file(args.lora),
                "alpha": alpha, "scale": args.scale, "steps": args.steps,
                "video_shift": args.video_shift, "audio_shift": args.audio_shift}
    progress_path = output / "edge-dit-h3-merge-progress.json"
    progress = {"identity": identity, "shards": {}}
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("identity") != identity:
            raise ValueError("Resume identity differs from base/adapter/recipe")
    elif args.resume and any(output.iterdir()):
        raise ValueError("Resume requires a matching progress manifest")
    if config.is_file():
        target_config = output / config.name
        if target_config.exists() and sha256_file(target_config) != sha256_file(config):
            raise ValueError("Output config differs; refusing overwrite")
        if not target_config.exists(): shutil.copy2(config, target_config)
    shard_names = sorted(set(index_data["weight_map"].values()))
    merged_count = 0
    for shard_number, shard_name in enumerate(shard_names, 1):
        source_path = safe_shard_path(index.parent, shard_name)
        output_path = safe_shard_path(output, shard_name)
        with safe_open(source_path, framework="pt", device="cpu") as source:
            source_keys = set(source.keys())
            known = progress["shards"].get(shard_name)
            if (args.resume and known and output_shard_complete(output_path, source_keys)
                    and sha256_file(output_path) == known.get("output_sha256")
                    and sha256_file(source_path) == known.get("base_sha256")):
                merged_count += len(shard_modules.get(shard_name, []))
                print(f"[{shard_number}/{len(shard_names)}] reuse {shard_name}", flush=True)
                continue
            tensors = {key: source.get_tensor(key) for key in source.keys()}
            shard_metadata = source.metadata() or {"format": "pt"}
        for module_name, tensor_names in shard_modules.get(shard_name, []):
            weight_name = f"{module_name}.weight"
            tensors[weight_name] = merge_weight(
                tensors[weight_name],
                adapter.get_tensor(tensor_names["a"]),
                adapter.get_tensor(tensor_names["b"]),
                alpha,
                args.scale,
                device,
            )
            merged_count += 1
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite an unverified output shard: {output_path}")
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if temporary_path.exists():
            raise FileExistsError(f"Unfinished temporary shard exists; retained for inspection: {temporary_path}")
        save_file(tensors, temporary_path, metadata=shard_metadata)
        temporary_path.replace(output_path)
        progress["shards"][shard_name] = {"output_sha256": sha256_file(output_path),
                                          "base_sha256": sha256_file(source_path)}
        progress_temp = progress_path.with_suffix(".json.tmp")
        progress_temp.write_text(json.dumps(progress, indent=2) + "\n")
        progress_temp.replace(progress_path)
        print(f"[{shard_number}/{len(shard_names)}] wrote {shard_name}", flush=True)

    if merged_count != len(modules):
        raise RuntimeError(f"merged {merged_count} modules, expected {len(modules)}")
    shutil.copy2(index, output / index.name)
    manifest = {
        "schema_version": 1,
        "kind": "minimax-h3-turbo",
        "steps": args.steps,
        "video_shift": args.video_shift,
        "audio_shift": args.audio_shift,
        "sampler": "euler", "scheduler": "simple", "cfg_scale": 1.0,
        "output_index": str(output / index.name),
        "shards": progress["shards"],
        **identity,
        "base": str(index),
        "lora": str(args.lora.resolve()),
        "lora_metadata": metadata,
        "alpha": alpha,
        "scale": args.scale,
        "merged_modules": merged_count,
        "device": str(device),
    }
    (output / "edge-dit-h3-turbo.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"merged {merged_count} modules into {output}", flush=True)
    return output


if __name__ == "__main__":
    merge(parse_args())
