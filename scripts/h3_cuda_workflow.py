#!/usr/bin/env python3
"""MiniMax-H3 CUDA: freeze inputs, baseline -> candidate -> compare.

Stdlib only; ed-cli (not ed-sample) is invoked once per warmup/repeat. Wall
latency includes process startup, model load and mux, but excludes validation.
No shell, downloads, GPU-process termination, precision conversion or deletion.
Use fixtures/lock/dry-run before scheduling a real run on an idle GPU.
"""
from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2"
OPTIONS = tuple("ED_H3_" + name for name in (
    "FUSED_OPS", "CACHE_DIT", "ADALN_CACHE", "SAGE_ATTN", "SPARSE_ATTN", "VDN", "VSA", "TURBO"))
MODE_ORDER = ("ref_mixed", "ref_paired_audio", "ref_video", "ref_image",
              "first_last", "first_frame", "last_frame", "t2va")
INPUTS = {
    "t2va": {}, "first_frame": {"image": "image"},
    "last_frame": {"end-img": "image"},
    "first_last": {"image": "image", "end-img": "image"},
    "ref_image": {"ref-image": "image"}, "ref_video": {"ref-video": "video"},
    "ref_paired_audio": {"ref-video": "video", "ref-video-audio": "audio"},
    "ref_mixed": {"ref-image": "image", "ref-video": "video",
                  "ref-video-audio": "audio", "ref-audio": "audio"},
}
COMPONENTS = ("fl2va", "ref2va", "llm", "video_vae", "audio_vae")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PROFILE_RE = re.compile(r"(?:\[MiniMax-H3\]|minimax-h3)\s+profile(?P<section>[^:\n]*):(?P<body>[^\n]*)", re.I)
SUMMARY_RE = re.compile(
    r"total=(\d+)\s*ms\s*\|\s*conditioning=(\d+)\s*ms\s*\|\s*"
    r"diffusion=(\d+)\s*ms\s*\((\d+) steps,\s*(\d+) calls\)\s*\|\s*"
    r"decode=(\d+)\s*ms\s*\|\s*noise=(\d+)\s*ms")
WARNING = ("SSIM/PSNR and waveform errors measure decoded numerical differences, "
           "not prompt fidelity, motion, reference identity, synchronization or semantic quality. "
           "Finite checks cover decoded artifacts, not pre-encode model tensors.")


class WorkflowError(RuntimeError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowError(message)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"cannot read JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json(path: Path, data: Any, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    """Full-file SHA256, or ordered relative-path + full-file hashes for frames."""
    require(path.exists(), f"missing input: {path}")
    if path.is_file():
        before = path.stat()
        require(before.st_size > 0, f"empty input: {path}")
        digest = sha256(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                (after.st_size, after.st_mtime_ns, after.st_ino), f"input changed while hashing: {path}")
        return {"sha256": digest, "bytes": after.st_size}
    require(path.is_dir(), f"not a regular file/directory: {path}")
    files = sorted(path.rglob("*"))
    require(not any(p.is_symlink() for p in files), f"symlinks in frame directory: {path}")
    files = [p for p in files if p.is_file()]
    require(bool(files), f"empty frame directory: {path}")
    entries = [{"name": p.relative_to(path).as_posix(), **fingerprint(p)} for p in files]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(),
            "bytes": sum(p["bytes"] for p in entries), "files": entries}


def input_stat_signatures(manifest: dict) -> dict:
    """Cheap per-launch mutation guard after full hashing; not a storage lock.

    Includes ctime/device/inode as well as size/mtime and frame membership. A
    read-only guard cannot protect against concurrent writes after the check;
    weights and assets must remain immutable for the whole scheduled run.
    """
    result = {}
    for group in ("models", "assets"):
        for key, item in manifest[group].items():
            root = Path(item["path"])
            require(root.exists(), f"input changed/missing before launch: {root}")
            paths = [root] + sorted(root.rglob("*")) if root.is_dir() else [root]
            rows = []
            for path in paths:
                require(not path.is_symlink() or path == root, f"input frame symlink changed: {path}")
                stat = path.stat()
                rows.append((str(path), stat.st_dev, stat.st_ino, stat.st_size,
                             stat.st_mtime_ns, stat.st_ctime_ns))
            result[group + "." + key] = rows
    return result

def integer(value: Any, name: str, minimum: int = 1) -> int:
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def positive(value: Any, name: str, allow_zero: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and
            (value >= 0 if allow_zero else value > 0), f"invalid {name}")
    return float(value)


def validate_manifest(manifest: dict) -> None:
    require(manifest.get("schema_version") == 1, "schema_version must be 1")
    precision = read_json(ROOT / "docs/research/h3-precision-lock.json")
    require(manifest.get("baseline_revision") == BASELINE_REVISION, "baseline must be fixed at 97cc585")
    require(manifest.get("weight_variant") == "full_h3_existing_persistent_Q4_GGUF_not_pruned_not_distilled",
            "only the locked existing full Q4 weights are eligible; no pruned/distilled/conversion A/B")
    for group in ("models", "assets", "binaries", "sampling", "run", "modes", "locations", "placement"):
        require(isinstance(manifest.get(group), dict), f"missing object: {group}")
    require(set(manifest["models"]) == set(COMPONENTS), "provide FL2VA, Ref2VA, Qwen and both VAEs")
    for group in ("models", "assets"):
        for name, item in manifest[group].items():
            require(isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"],
                    f"{group}.{name}: explicit path required")
            digest = item.get("sha256")
            require(digest is None or isinstance(digest, str) and bool(SHA_RE.fullmatch(digest)),
                    f"{group}.{name}: sha256 must be a full 64-digit digest or null before lock")
            if "bytes" in item:
                integer(item["bytes"], f"{group}.{name}.bytes")
    for key, pinned in zip(COMPONENTS, precision["components"]):
        item = manifest["models"][key]
        require(Path(item["path"]).name == pinned["component"] and
                item.get("bytes") == pinned["bytes"] and item.get("tensor_types") == pinned["tensor_types"] and
                item.get("source_header_sha256") == pinned["header_sha256"],
                f"{key}: model identity/precision differs from docs/research/h3-precision-lock.json")
        if pinned.get("full_sha256") is not None:
            require(item.get("sha256") in (None, pinned["full_sha256"]), f"{key}: pinned full SHA256 differs")
    require(manifest.get("submodules") == {"ggml": precision["ggml_revision"], "onednn": precision["onednn_revision"]},
            "pinned submodule revisions differ")
    sampling = manifest["sampling"]
    require(set(sampling) == {"width", "height", "frames", "fps", "steps", "seed", "cfg_scale",
                              "sampler", "scheduler", "ref_image_size", "audio_sample_rate", "audio_channels"},
            "sampling has missing/unknown fields (no implicit sampling defaults)")
    width, height = (integer(sampling[k], k) for k in ("width", "height"))
    require(width % 32 == height % 32 == 0 and width * height >= 65536,
            "H3 canvas must be multiples of 32 and at least 65536 pixels")
    frames = integer(sampling["frames"], "frames", 22)
    require((frames - 5) % 17 == 0, "H3 frames must satisfy 17k+5, >=22")
    require(sampling["fps"] == 24, "H3 fps must be 24")
    integer(sampling["steps"], "steps")
    integer(sampling["seed"], "fixed nonnegative seed", 0)
    require(sampling["seed"] <= 2**63 - 1, "seed exceeds int64")
    positive(sampling["cfg_scale"], "cfg_scale", True)
    require(sampling["sampler"] in ("euler", "res_multistep"), "unsupported H3 sampler")
    require(sampling["scheduler"] in ("discrete", "simple"), "unsupported H3 scheduler")
    require(sampling["ref_image_size"] in ("match", "max"), "invalid ref_image_size")
    require(sampling["audio_sample_rate"] == 32000 and sampling["audio_channels"] == 2,
            "H3 generated audio must be 32000 Hz stereo")
    settings = manifest["run"]
    for key in ("warmup", "repeats", "min_free_mib"):
        integer(settings.get(key), key)
    integer(settings.get("gpu_index"), "gpu_index", 0)
    positive(settings.get("timeout_seconds"), "timeout_seconds")
    positive(settings.get("media_timeout_seconds"), "media_timeout_seconds")
    positive(settings.get("max_utilization_percent"), "max_utilization_percent", True)
    require(settings["max_utilization_percent"] <= 5, "idle check utilization ceiling must be <=5%")
    require(set(manifest["placement"]) == {"dit_offload", "text_encoder_offload", "vae_offload"},
            "explicit fixed placement required; auto-fit/quantization changes are not allowed")
    require(all(type(v) is bool for v in manifest["placement"].values()), "placement values must be boolean")
    require(set(manifest["binaries"]) == {"baseline", "candidate"}, "exactly baseline/candidate binaries required")
    for phase in ("baseline", "candidate"):
        binary = manifest["binaries"].get(phase, {})
        require(isinstance(binary, dict), f"{phase} binary must be an object")
        require(isinstance(binary.get("path"), str) and binary["path"], f"missing {phase} binary path")
        env = binary.get("optimization_env")
        require(isinstance(env, dict) and set(env) == set(OPTIONS) and
                all(v in ("0", "1") for v in env.values()), "all eight optimization env values must be explicit 0/1 strings")
        if phase == "baseline":
            require(all(v == "0" for v in env.values()), "baseline optimizations must all be zero")
    catalog = manifest.get("method_catalog")
    require(isinstance(catalog, dict) and set(catalog) <= set(OPTIONS), "explicit method_catalog required")
    for option, entry in catalog.items():
        require(isinstance(entry, dict) and entry.get("status") in ("implemented", "pending", "unsupported"),
                f"{option}: invalid catalog status")
        require(isinstance(entry.get("required_build", {}), dict), f"{option}: required_build must be an object")
        require(isinstance(entry.get("modes", []), list) and set(entry.get("modes", [])) <= set(MODE_ORDER),
                f"{option}: invalid catalog modes")
    require(bool(manifest["modes"]), "empty modes matrix")
    for name, mode in manifest["modes"].items():
        require(name in INPUTS and isinstance(mode, dict), f"unknown mode: {name}")
        expected = "ref2va" if name.startswith("ref_") else "fl2va"
        require(mode.get("dit") == expected, f"{name} requires {expected} checkpoint")
        require(isinstance(mode.get("prompt"), str) and mode["prompt"].strip(), f"{name}: empty prompt")
        inputs = mode.get("inputs")
        require(isinstance(inputs, dict) and set(inputs) == set(INPUTS[name]), f"{name}: incorrect conditioning flags")
        for flag, keys in inputs.items():
            require(isinstance(keys, list) and bool(keys), f"{name}: {flag} needs ordered asset list")
            for key in keys:
                require(key in manifest["assets"], f"unknown asset: {key}")
                kind = manifest["assets"][key].get("kind")
                wanted = INPUTS[name][flag]
                require(kind == wanted or wanted == "video" and kind == "frames", f"{key}: asset kind mismatch")
        for flag in ("image", "end-img"):
            require(len(inputs.get(flag, [])) <= 1, f"{flag} supports exactly one asset")
        if "ref-video-audio" in inputs:
            require(len(inputs["ref-video-audio"]) == len(inputs["ref-video"]), "paired audio/video count mismatch")


def resolve_manifest(path: Path, profile: str = "host", overrides: dict | None = None) -> dict:
    data = read_json(path)
    validate_manifest(data)
    require(profile in data["locations"], f"unknown location profile: {profile}")
    variables = {"repo_root": str(ROOT), **data["locations"][profile]}
    variables.update({k: str(v) for k, v in (overrides or {}).items() if v is not None})
    for _ in range(3):
        try:
            variables = {k: str(v).format_map(variables) for k, v in variables.items()}
        except KeyError as exc:
            raise WorkflowError(f"unknown path variable: {exc}") from exc
    previous_variables = data.get("path_variables", {})
    for group in ("models", "assets", "binaries"):
        prefix_key = {"models": "model_root", "assets": "asset_root", "binaries": "repo_root"}[group]
        old_prefix = previous_variables.get(prefix_key)
        for item in data[group].values():
            try:
                expanded = item["path"].format_map(variables)
            except KeyError as exc:
                raise WorkflowError(f"unknown path variable: {exc}") from exc
            p = Path(expanded).expanduser()
            if p.is_absolute() and old_prefix and prefix_key in variables:
                try:
                    suffix = p.relative_to(old_prefix)
                except ValueError:
                    pass
                else:
                    p = Path(variables[prefix_key]) / suffix
            item["path"] = str((Path(variables["repo_root"]) / p).resolve() if not p.is_absolute() else p)
    data["path_variables"] = variables
    data["resolved_location"] = profile
    data["resolved_repo_root"] = variables["repo_root"]
    data["source_manifest"] = str(path.resolve())
    return data


def selected_modes(manifest: dict, modes: list[str] | None) -> list[str]:
    chosen = set(modes or manifest["modes"])
    require(chosen <= set(manifest["modes"]), "selected mode absent from manifest")
    return [mode for mode in MODE_ORDER if mode in chosen]


def isolated_env(manifest: dict, phase: str, inherited: dict | None = None) -> tuple[dict, list[str]]:
    original = dict(os.environ if inherited is None else inherited)
    # Includes legacy H3, generic CUDA fusion/cache/Sage knobs and CPU profiler
    # knobs. Keep OS/loader/tool paths. Both arms start from the same defaults.
    removed = sorted(k for k in original if k.startswith(("ED_", "GGML_", "SD_")) or
                     k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"))
    env = {k: v for k, v in original.items() if k not in removed}
    env.update({key: "0" for key in OPTIONS})
    if phase == "candidate":
        env.update(manifest["binaries"][phase]["optimization_env"])
    env["ED_MINIMAX_H3_PROFILE"] = "1"
    return env, removed


def invoke(command: list[str], timeout: float, *, env: dict | None = None,
           cwd: Path | None = None, binary: bool = False) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, shell=False, capture_output=True,
                                text=not binary, timeout=timeout, env=env, cwd=cwd)
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(f"timeout after {timeout}s: {command[0]}") from exc
    except OSError as exc:
        raise WorkflowError(f"cannot execute {command[0]}: {exc}") from exc
    if result.returncode:
        error = result.stderr.decode(errors="replace") if binary else result.stderr
        raise WorkflowError(f"nonzero exit {result.returncode}: {command[0]}: {error[-2000:]}")
    return result


def helper(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_profiles(text: str, log_path: Path | None = None) -> dict:
    summaries, details = [], []
    for match in PROFILE_RE.finditer(text):
        body, section = match["body"], match["section"].strip()
        if not section:
            row = SUMMARY_RE.search(body)
            if row:
                keys = ("total_ms", "conditioning_ms", "diffusion_ms", "steps", "calls", "decode_ms", "noise_ms")
                summaries.append(dict(zip(keys, map(int, row.groups()))))
        else:
            details.append({"section": section, "raw": body.strip(), "fields_ms": {
                key.strip(): int(value) for key, value in re.findall(r"([^=|]+)=(\d+)\s*ms", body)}})
    existing = helper("h3_existing_e2e", ROOT / "benchmark/scripts/run_edge_e2e.py")
    components, boundaries = existing.parse_phase_markers(text)
    result = {"source": "native_minimax_h3_profile", "summaries": summaries, "details": details,
              "phase_component_ms": components, "stage_boundaries": boundaries,
              "actual_placement": existing.parse_auto_placement(text)}
    if log_path is not None and "CPU op profile" in text:
        parser = helper("h3_existing_stage_parser", ROOT / "benchmark/scripts/parse_cpu_stage_profile.py")
        result["cpu_stage_profiles"] = parser.parse(str(log_path))
    return result


def statistics_for(values: list[float]) -> dict | None:
    if not values:
        return None
    require(all(math.isfinite(v) and v >= 0 for v in values), "nonfinite/negative timing")
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def summarize(records: list[dict]) -> dict | None:
    measured = [r for r in records if not r["warmup"] and r.get("status") == "validated"]
    if not measured or any("wall_ms" not in r for r in measured):
        return None
    fields: dict[str, list[float]] = {}
    for record in measured:
        fields.setdefault("cli_wall_ms", []).append(record["wall_ms"])
        for key, value in record["profile"]["summaries"][-1].items():
            if key.endswith("_ms"):
                fields.setdefault(key, []).append(value)
    return {key: statistics_for(value) for key, value in fields.items()}


def key_values(path: Path) -> dict:
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line and not line.startswith(("//", "#")):
            key, value = line.split("=", 1)
            result[key.split(":", 1)[0]] = value
    return result


def build_provenance(binary: Path) -> dict:
    build = binary.parent.parent
    files = [build / name for name in ("CMakeCache.txt", "build-config.txt", "h3-build-provenance.txt")]
    return {"binary": str(binary), "binary_sha256": sha256(binary) if binary.is_file() else None,
            "files": {p.name: {"path": str(p), "sha256": sha256(p), "values": key_values(p)}
                      for p in files if p.is_file()}}


def implemented_methods(values: dict) -> set[str]:
    """Build-owned declaration: comma/space-separated env names or JSON list."""
    raw = values.get("implemented_methods", "")
    if raw.startswith("["):
        try:
            methods = json.loads(raw)
        except ValueError as exc:
            raise WorkflowError("invalid implemented_methods provenance") from exc
        require(isinstance(methods, list) and all(isinstance(v, str) for v in methods),
                "implemented_methods must be a list of ED_H3 names")
        methods = set(methods)
    else:
        methods = set(filter(None, re.split(r"[,;\s]+", raw)))
    require(methods <= set(OPTIONS), "unknown method in build provenance")
    return methods


def method_plan(manifest: dict, phase: str, modes: list[str], dry_run: bool) -> None:
    for option, value in manifest["binaries"][phase]["optimization_env"].items():
        if value == "0":
            continue
        entry = manifest["method_catalog"].get(option, {})
        # Turbo/distilled/adapters are separate experiments, never an env-only
        # exception to the fixed full-Q4 model contract.
        require(option != "ED_H3_TURBO" and not entry.get("requires_separate_manifest", False),
                f"{option} unsupported in this precision lock: requires a separate manifest/baseline")
        require(entry.get("status") in ("implemented", "pending"), f"{option} unsupported in method catalog")
        require(not entry.get("modes") or set(modes) <= set(entry["modes"]), f"{option}: selected modes not in method catalog")
        # Pending can plan; validate_build must prove implementation before execution.


def validate_build(manifest: dict, phase: str, provenance: dict) -> dict:
    require(provenance["binary_sha256"] is not None, f"missing {phase} binary")
    values = provenance["files"].get("h3-build-provenance.txt", {}).get("values", {})
    cache = provenance["files"].get("CMakeCache.txt", {}).get("values", {})
    config = provenance["files"].get("build-config.txt", {}).get("values", {})
    require(cache.get("ED_GGML_CUDA", cache.get("GGML_CUDA")) == "ON", "CUDA build provenance missing")
    require(values.get("ggml_revision") == manifest["submodules"]["ggml"] and
            values.get("onednn_revision") == manifest["submodules"]["onednn"], "build submodule provenance mismatch")
    require(bool(values.get("source_revision")), "build source provenance missing")
    sage_requested = manifest["binaries"][phase]["optimization_env"]["ED_H3_SAGE_ATTN"] == "1"
    require(values.get("cuda_sage_attention") in ("ON", "OFF"), "Sage build capability missing")
    require(not sage_requested or values.get("cuda_sage_attention") == "ON",
            "Sage request requires a compiled Sage capability")
    if phase == "baseline":
        require(values.get("source_revision") == manifest["baseline_revision"], "baseline source provenance is not fixed 97cc585")
        require(values.get("ggml_revision") == manifest["submodules"]["ggml"] and
                values.get("onednn_revision") == manifest["submodules"]["onednn"], "baseline submodule provenance mismatch")
    build_requirements = dict(manifest.get("required_build", {}))
    if phase == "candidate" and not sage_requested:
        build_requirements.pop("ED_ENABLE_CUDA_SAGE_ATTN", None)
    for option, enabled in manifest["binaries"][phase]["optimization_env"].items():
        if enabled == "1":
            build_requirements.update(manifest["method_catalog"].get(option, {}).get("required_build", {}))
    for key, value in build_requirements.items():
        require(cache.get(key) == value, f"required build setting mismatch (unsupported in this build): {key}={value}")
    declared = implemented_methods(values)
    statuses = {}
    for option, value in manifest["binaries"][phase]["optimization_env"].items():
        if value == "0":
            statuses[option] = "disabled"
        else:
            entry = manifest["method_catalog"].get(option, {})
            require(option != "ED_H3_TURBO" and not entry.get("requires_separate_manifest", False),
                    f"{option} unsupported in fixed main baseline")
            require(entry.get("status") == "implemented" or option in declared,
                    f"{option} unsupported: no implemented catalog/build declaration")
            require(all(cache.get(key) == value for key, value in entry.get("required_build", {}).items()),
                    f"{option} unsupported in this build")
            # Build proof alone is not a claim of runtime activation. Old binaries
            # must at least include the new policy parser, not ignore an env var.
            marker = option.encode()
            with Path(provenance["binary"]).open("rb") as stream:
                found, tail = False, b""
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    if marker in tail + block:
                        found = True
                        break
                    tail = block[-len(marker):]
            require(found, f"{option} unsupported: policy parser absent from binary")
            statuses[option] = "build_supported_requested_runtime_not_attested"
    return {"options": statuses, "source_revision": values.get("source_revision", config.get("git_commit"))}


def git_provenance(repo: str) -> dict:
    result = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"]),
                      ("submodules", ["submodule", "status"])):
        try:
            result[key] = invoke(["git", "-C", repo, *args], 30).stdout.strip()
        except WorkflowError as exc:
            result[key] = None
            result[key + "_error"] = str(exc)
    return result


def gpu_precheck(settings: dict) -> dict:
    """Read-only, fail-closed check. Never kill/evict/attach to another process."""
    raw = invoke(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
                  "--format=csv,noheader,nounits"], 15).stdout
    gpus = []
    for line in raw.strip().splitlines():
        fields = [v.strip() for v in line.split(",")]
        require(len(fields) == 7, "unparseable GPU availability")
        try:
            gpus.append(dict(zip(("index", "uuid", "name", "total_mib", "free_mib", "used_mib", "utilization_percent"),
                                 (int(fields[0]), fields[1], fields[2], *map(int, fields[3:])))))
        except ValueError as exc:
            raise WorkflowError("unavailable GPU memory/utilization telemetry") from exc
    require(bool(gpus), "empty GPU telemetry")
    require(len({g["index"] for g in gpus}) == len(gpus) and
            len({g["uuid"] for g in gpus}) == len(gpus), "duplicate GPU identity telemetry")
    for gpu in gpus:
        require(gpu["index"] >= 0 and gpu["uuid"].startswith("GPU-") and
                len(gpu["uuid"]) > 4 and bool(gpu["name"]), "invalid GPU identity telemetry")
        require(gpu["total_mib"] > 0 and 0 <= gpu["free_mib"] <= gpu["total_mib"] and
                0 <= gpu["used_mib"] <= gpu["total_mib"] and
                0 <= gpu["utilization_percent"] <= 100, "invalid GPU memory/utilization telemetry")
    selected = next((gpu for gpu in gpus if gpu["index"] == settings["gpu_index"]), None)
    chosen = dict(selected) if selected is not None else None
    require(chosen is not None, "selected GPU not found")
    processes = invoke(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
                        "--format=csv,noheader,nounits"], 15).stdout
    for line in processes.splitlines():
        if not line.strip():
            continue
        fields = [v.strip() for v in line.split(",")]
        require(len(fields) == 4, "unparseable compute-process telemetry")
        require(fields[0] in {gpu["uuid"] for gpu in gpus}, "unattributable compute-process telemetry")
        require(fields[1].isdigit() and int(fields[1]) > 0 and bool(fields[3]),
                "invalid compute-process telemetry")
        require(fields[2] in ("N/A", "[N/A]") or fields[2].isdigit(),
                "invalid compute-process memory telemetry")
    chosen["compute_processes"] = [line.strip() for line in processes.splitlines()
                                    if line.split(",", 1)[0].strip() == chosen["uuid"]]
    chosen["all_gpus"] = gpus
    for condition, message in (
        (not chosen["compute_processes"], f"GPU {chosen['index']} is occupied; no process will be terminated"),
        (chosen["free_mib"] >= settings["min_free_mib"], "GPU free memory below safe minimum"),
        (chosen["utilization_percent"] <= settings["max_utilization_percent"], "GPU is not idle")):
        if not condition:
            raise WorkflowError(message, details=chosen)
    return chosen


def command_for(manifest: dict, phase: str, mode: str, output: Path) -> list[str]:
    sampling = manifest["sampling"]
    task = manifest["modes"][mode]
    models = manifest["models"]
    command = [manifest["binaries"][phase]["path"], "--video", "--backend", "cuda",
               "--diffusion-model", models[task["dit"]]["path"], "--llm", models["llm"]["path"],
               "--vae", models["video_vae"]["path"], "--audio-vae", models["audio_vae"]["path"],
               "--type", "preserve", "--cache", "off", "--prompt", task["prompt"],
               "--video-frames", str(sampling["frames"]), "--fps", "24",
               "-W", str(sampling["width"]), "-H", str(sampling["height"]),
               "--steps", str(sampling["steps"]), "--seed", str(sampling["seed"]),
               "--cfg-scale", str(sampling["cfg_scale"]), "--sampler", sampling["sampler"],
               "--scheduler", sampling["scheduler"], "--ref-image-size", sampling["ref_image_size"],
               "--video-format", "mp4", "--output", str(output)]
    for key, enabled in manifest["placement"].items():
        if enabled:
            command.append("--" + key.replace("_", "-"))
    for flag, keys in task["inputs"].items():
        for key in keys:
            command += ["--" + flag, manifest["assets"][key]["path"]]
    return command


def probe(path: Path, timeout: float) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty media output: {path}")
    raw = invoke(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], timeout).stdout
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise WorkflowError("invalid/empty ffprobe output") from exc
    require(isinstance(data, dict) and isinstance(data.get("streams"), list), "ffprobe streams missing")
    return data


def fraction(value: Any) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise WorkflowError(f"invalid media fps: {value}") from exc
    require(result > 0, "nonpositive fps")
    return result


def pcm_values(path: Path):
    with path.open("rb") as stream:
        while True:
            raw = stream.read(256 * 1024)
            if not raw:
                break
            require(len(raw) % 4 == 0, "truncated float32 audio")
            values = array("f")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
            for value in values:
                require(math.isfinite(value), "nonfinite decoded audio")
            yield values


def validate_media(path: Path, sampling: dict, timeout: float, work: Path) -> dict:
    data = probe(path, timeout)
    videos = [s for s in data["streams"] if s.get("codec_type") == "video"]
    audios = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    require(len(videos) == len(audios) == 1, "exactly one video and one audio stream required (no missing mux/sidecar fallback)")
    video, audio = videos[0], audios[0]
    try:
        shape = {"width": int(video["width"]), "height": int(video["height"]),
                 "frames": int(video["nb_read_frames"]), "fps": str(fraction(video["avg_frame_rate"])),
                 "audio_sample_rate": int(audio["sample_rate"]), "audio_channels": int(audio["channels"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowError("incomplete media metadata") from exc
    for key in ("width", "height", "frames", "audio_sample_rate", "audio_channels"):
        require(shape[key] == sampling[key], f"media metadata mismatch: {key}: {shape[key]} != {sampling[key]}")
    require(fraction(video["r_frame_rate"]) == fraction(video["avg_frame_rate"]) == sampling["fps"], "media fps mismatch")
    # RGB24 integer pixels are finite by representation; full decode with -xerror
    # catches corruption. framemd5 also verifies every frame's decoded byte size.
    decoded = invoke(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
                      "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-threads", "1", "-vsync", "0",
                      "-f", "framemd5", "pipe:1"], timeout).stdout
    rows = [line.split(",") for line in decoded.splitlines() if line and not line.startswith("#")]
    require(len(rows) == sampling["frames"], "full decode frame count mismatch")
    try:
        require(all(int(row[4]) == sampling["width"] * sampling["height"] * 3 for row in rows), "decoded frame dimensions mismatch")
    except (IndexError, ValueError) as exc:
        raise WorkflowError("malformed framemd5 output") from exc
    md5_path = work / (path.stem + ".framemd5")
    md5_path.write_text(decoded, encoding="utf-8")
    pcm_path = work / (path.stem + ".audio.f32le")
    require(not pcm_path.exists(), "refuse overwriting decoded PCM")
    invoke(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
            "-map", "0:a:0", "-vn", "-c:a", "pcm_f32le", "-f", "f32le", str(pcm_path)], timeout)
    count = sum(len(values) for values in pcm_values(pcm_path))
    require(count > 0 and count % shape["audio_channels"] == 0, "empty/misaligned audio samples")
    shape["audio_samples_per_channel"] = count // shape["audio_channels"]
    return {"shape": shape, "finite": True, "finite_scope": "decoded_rgb24_and_float32_audio_only",
            "sha256": sha256(path), "bytes": path.stat().st_size, "audio_pcm": str(pcm_path),
            "audio_pcm_sha256": sha256(pcm_path), "framemd5": str(md5_path)}


def validate_assets(manifest: dict, modes: list[str]) -> None:
    timeout = manifest["run"]["media_timeout_seconds"]
    used = {key for mode in modes for keys in manifest["modes"][mode]["inputs"].values() for key in keys}
    for key in used:
        asset = manifest["assets"][key]
        path, kind = Path(asset["path"]), asset["kind"]
        if kind == "frames":
            require(path.is_dir(), f"{key}: frame directory required")
            frames = sorted(p for p in path.iterdir() if p.is_file())
            require(bool(frames) and all(p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp") for p in frames), "invalid frame directory assets")
            shapes = set()
            for frame in frames:
                streams = probe(frame, timeout)["streams"]
                require(len(streams) == 1 and streams[0].get("codec_type") == "video", "invalid reference frame")
                shapes.add((streams[0]["width"], streams[0]["height"]))
            require(len(shapes) == 1, "reference frame dimensions mismatch")
            continue
        streams = probe(path, timeout)["streams"]
        video = [s for s in streams if s.get("codec_type") == "video"]
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        if kind in ("image", "video"):
            require(len(video) == 1 and video[0].get("width", 0) > 0 and video[0].get("height", 0) > 0, f"{key}: missing visual stream")
            w, h = video[0]["width"], video[0]["height"]
            require(max(w, h) <= 4 * min(w, h), "H3 reference aspect ratio exceeds 4:1")
            if kind == "image":
                require(video[0].get("nb_read_frames") == "1", f"{key}: animated input is not a single image")
            else:
                require(int(video[0].get("nb_read_frames", 0)) > 0, "empty reference video")
        if kind == "audio":
            require(len(audio) == 1 and int(audio[0].get("sample_rate", 0)) > 0 and
                    int(audio[0].get("channels", 0)) in (1, 2), "invalid reference audio")
            require(path.suffix.lower() == ".wav", "ed-cli explicit reference audio must be WAV")
        if asset.get("embedded_audio_required"):
            require(len(audio) == 1, "reference video embedded audio missing")


def verify_precision_header(key: str, item: dict) -> None:
    """Header/tensor provenance gate, NOT a replacement for full-file SHA256."""
    precision = read_json(ROOT / "docs/research/h3-precision-lock.json")
    pinned = precision["components"][COMPONENTS.index(key)]
    path = Path(item["path"])
    require(path.is_file() and path.stat().st_size == pinned["bytes"], f"{key}: precision-lock size mismatch")
    with path.open("rb") as stream:
        if "header_bytes" in pinned:
            header = stream.read(pinned["header_bytes"])
        else:
            prefix = stream.read(8)
            require(len(prefix) == 8, "truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            require(0 < length <= 16 * 1024 * 1024, "invalid safetensors header size")
            header = stream.read(length)
    require(hashlib.sha256(header).hexdigest() == pinned["header_sha256"], f"{key}: pinned model tensor/header mismatch")
    if pinned.get("full_sha256") is not None:
        require(item.get("sha256") in (None, pinned["full_sha256"]), f"{key}: pinned full-file SHA256 mismatch")


def verify_inputs(manifest: dict) -> dict:
    identities = {}
    for group in ("models", "assets"):
        identities[group] = {}
        for key, item in manifest[group].items():
            require(item.get("sha256") is not None, f"{group}.{key}: full SHA256 required; use lock first")
            if group == "models":
                verify_precision_header(key, item)
            actual = fingerprint(Path(item["path"]))
            require(actual["sha256"] == item["sha256"], f"{group}.{key}: full-file asset/weight hash mismatch")
            require(item.get("bytes", actual["bytes"]) == actual["bytes"], f"{group}.{key}: size mismatch")
            identities[group][key] = actual
    return identities


def contract(manifest: dict, modes: list[str]) -> dict:
    # Paths/binaries may be remapped between machines; content/order/precision
    # may not. Do not compare requested dtype strings in place of tensor types.
    return {"sampling": manifest["sampling"], "placement": manifest["placement"],
            "weight_variant": manifest["weight_variant"],
            "models": {k: {field: v.get(field) for field in ("sha256", "bytes", "tensor_types", "source_header_sha256")}
                       for k, v in manifest["models"].items()},
            "assets": {k: {field: v.get(field) for field in ("sha256", "bytes", "kind", "embedded_audio_required")}
                       for k, v in manifest["assets"].items()},
            "modes": {k: manifest["modes"][k] for k in modes}}


def frozen_report(path: Path) -> dict:
    data = read_json(path)
    require(data.get("phase") in ("baseline", "candidate") and data.get("status") in ("validated", "imported"), "report is not a validated fixed run/artifact; dry-run/failed reports are not baselines")
    require(isinstance(data.get("contract"), dict) and bool(data.get("modes")), "incomplete fixed report")
    validate_manifest(data["manifest"])
    require(data["contract"] == contract(data["manifest"], list(data["modes"])), "fixed manifest/report contract differs")
    for group in ("models", "assets"):
        require(all(isinstance(item.get("sha256"), str) and SHA_RE.fullmatch(item["sha256"])
                    for item in data["contract"][group].values()), "fixed report lacks full input hashes")
    require(all(value == "0" for value in data["manifest"]["binaries"]["baseline"]["optimization_env"].values()),
            "fixed baseline has enabled optimizations")
    if data.get("performance_measured"):
        require(data["status"] == "validated" and isinstance(data.get("gpu"), dict) and
                isinstance(data.get("statistics"), dict), "unproven report performance values")
    for mode, records in data["modes"].items():
        if data.get("performance_measured"):
            require(len(records) == data["manifest"]["run"]["warmup"] + data["manifest"]["run"]["repeats"] and
                    sum(bool(r.get("warmup")) for r in records) == data["manifest"]["run"]["warmup"] and
                    all(r.get("status") == "validated" for r in records), "fixed measured report has incomplete warmup/repeats")
        measured = [r for r in records if not r.get("warmup")]
        require(bool(measured) and all(r.get("status") == "validated" for r in measured), f"{mode}: incomplete measured outputs")
        for record in measured:
            output = Path(record["output"])
            require(output.is_file() and sha256(output) == record["media"]["sha256"], "fixed artifact changed/missing")
    return data


def run_phase(manifest: dict, phase: str, output_dir: Path, modes: list[str], *, dry_run: bool = False,
              baseline_path: Path | None = None, artifacts: dict | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    validate_manifest(manifest)
    require(phase in ("baseline", "candidate"), "invalid generation phase")
    require(bool(modes) and len(modes) == len(set(modes)) and set(modes) <= set(manifest["modes"]),
            "empty, duplicate or unsupported selected modes")
    report = {"schema_version": 1, "phase": phase, "status": "pending", "created_utc": datetime.now(timezone.utc).isoformat(),
              "mode_order": modes, "manifest": manifest, "contract": contract(manifest, modes), "modes": {}, "statistics": None,
              "gpu": None, "quality_warning": WARNING,
              "measurement_boundary": "fresh_ed_cli_process_including_load_and_mux_excluding_media_validation",
              "warmup_boundary": "separate_process_not_load_once_steady_state", "performance_measured": False,
              "gpu_safety_note": "read-only precheck is not an exclusive reservation; scheduler ownership still required",
              "timeout_scope": "direct child only; stdlib subprocess timeout does not terminate unrelated processes or process groups"}
    write_json(report_path, report)
    try:
        method_plan(manifest, phase, modes, dry_run)
        if phase == "candidate" and baseline_path is not None:
            fixed = frozen_report(baseline_path)
            require(fixed["phase"] == "baseline", "candidate requires baseline report")
            require(fixed["contract"] == report["contract"], "candidate sampling/weight/assets/placement contract differs from baseline")
            report["baseline_report"] = str(baseline_path.resolve())
            report["baseline_report_sha256"] = sha256(baseline_path)
        else:
            require(phase == "baseline" or dry_run, "candidate requires --baseline fixed report")
        require(not (dry_run and artifacts), "--artifact cannot be combined with --dry-run")
        env, removed = isolated_env(manifest, phase)
        report["environment"] = {"optimization_env": {k: env[k] for k in OPTIONS},
                                 "profile_env": {"ED_MINIMAX_H3_PROFILE": "1"}, "removed_inherited_keys": removed,
                                 "other_runtime_env": {k: v for k, v in env.items() if k.startswith(("CUDA_", "CUBLAS_", "CUDNN_", "NVIDIA_", "NCCL_", "OMP_", "MKL_")) and k != "CUDA_VISIBLE_DEVICES"}}
        report["git"] = git_provenance(manifest["resolved_repo_root"])
        if dry_run:
            for mode in modes:
                report["modes"][mode] = [{"warmup": n < manifest["run"]["warmup"], "status": "planned",
                    "command": command_for(manifest, phase, mode, output_dir / f"{mode}-{n:03d}.mp4")}
                    for n in range(manifest["run"]["warmup"] + manifest["run"]["repeats"])]
            report["status"] = "dry_run"
            report["gpu_check"] = "skipped_no_generation_no_performance_values"
            report["missing_paths"] = [v["path"] for group in ("models", "assets", "binaries") for v in manifest[group].values() if not Path(v["path"]).exists()]
            report["hash_status"] = "not_verified_dry_run"
            report["optimization_status"] = {key: "disabled" if value == "0" else "planned_build_and_runtime_not_verified"
                for key, value in env.items() if key in OPTIONS}
            write_json(report_path, report)
            return report_path
        before_hash = input_stat_signatures(manifest)
        report["identities"] = verify_inputs(manifest)
        frozen_inputs = input_stat_signatures(manifest)
        require(before_hash == frozen_inputs, "input changed during full-file verification")
        report["media_tools"] = {tool: invoke([tool, "-version"], 15).stdout.splitlines()[0]
                                 for tool in ("ffmpeg", "ffprobe")}
        validate_assets(manifest, modes)
        if artifacts:
            require(phase == "baseline" and set(artifacts) == set(modes), "artifact import requires exactly one artifact per selected baseline mode")
            for mode in modes:
                path = Path(artifacts[mode]).resolve()
                artifact_work = output_dir / mode
                artifact_work.mkdir()
                media = validate_media(path, manifest["sampling"], manifest["run"]["media_timeout_seconds"], artifact_work)
                report["modes"][mode] = [{"warmup": False, "status": "validated", "output": str(path), "media": media}]
            report["status"] = "imported"
            report["artifact_generation_provenance"] = "user_supplied_manifest_not_runtime_verified"
            report["build"] = build_provenance(Path(manifest["binaries"][phase]["path"]))
            report["measurement_boundary"] = "imported_fixed_artifact_no_timing_or_gpu_claim"
            write_json(report_path, report)
            return report_path
        binary = Path(manifest["binaries"][phase]["path"])
        report["build"] = build_provenance(binary)
        report["optimization_status"] = validate_build(manifest, phase, report["build"])
        report["gpu"] = gpu_precheck(manifest["run"])
        if phase == "candidate":
            fixed_gpu = fixed.get("gpu")
            if fixed_gpu is not None:
                require(all(report["gpu"][k] == fixed_gpu[k] for k in ("uuid", "name", "total_mib")), "candidate GPU differs from baseline")
            require(report["environment"]["other_runtime_env"] == fixed["environment"]["other_runtime_env"], "runtime environment differs from baseline")
            if fixed.get("performance_measured"):
                require(all(manifest["run"][key] == fixed["manifest"]["run"][key] for key in ("warmup", "repeats")),
                        "candidate warmup/repeat schedule differs from measured baseline")
        env["CUDA_VISIBLE_DEVICES"] = report["gpu"]["uuid"]
        for mode in modes:
            report["modes"][mode] = []
            for n in range(manifest["run"]["warmup"] + manifest["run"]["repeats"]):
                # Recheck immediately before each launch. A read-only check is
                # not a reservation; operators still need exclusive scheduling.
                available = gpu_precheck(manifest["run"])
                require(available["uuid"] == report["gpu"]["uuid"], "GPU identity changed")
                require(input_stat_signatures(manifest) == frozen_inputs,
                        "input changed since full-file verification; refusing remaining launches")
                path = output_dir / f"{mode}-{n:03d}.mp4"
                require(not path.exists(), f"refuse overwriting existing output: {path}")
                record = {"warmup": n < manifest["run"]["warmup"], "status": "running", "output": str(path),
                          "command": command_for(manifest, phase, mode, path), "gpu_precheck": available}
                report["modes"][mode].append(record)
                write_json(report_path, report)
                start = time.monotonic()
                # Direct subprocess call retains nonzero stdout/stderr in logs.
                try:
                    completed = subprocess.run(record["command"], shell=False, capture_output=True, text=True,
                                               timeout=manifest["run"]["timeout_seconds"], env=env)
                except subprocess.TimeoutExpired as exc:
                    for label, text in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                        (output_dir / f"{path.stem}.{label}.log").write_bytes(text.encode() if isinstance(text, str) else text or b"")
                    record["status"] = "timeout"
                    raise WorkflowError("ed-cli timeout; no unrelated process is terminated") from exc
                elapsed_ms = (time.monotonic() - start) * 1000
                for label in ("stdout", "stderr"):
                    log = output_dir / f"{path.stem}.{label}.log"
                    log.write_text(getattr(completed, label), encoding="utf-8")
                    record[label + "_log"] = str(log)
                record["returncode"] = completed.returncode
                require(completed.returncode == 0, f"ed-cli nonzero exit: {completed.returncode}")
                combined = completed.stdout + "\n" + completed.stderr
                require(not re.search(r"ED_H3_\w+[^\n]*unsupported", combined, re.I), "runtime reported unsupported optimization")
                record["profile"] = parse_profiles(combined, Path(record["stderr_log"]))
                summaries = record["profile"]["summaries"]
                require(len(summaries) == 1 and summaries[0]["steps"] == manifest["sampling"]["steps"], "missing/invalid native H3 generation profile")
                # Use the subprocess boundary, not validation/profile parsing.
                record["wall_ms"] = elapsed_ms
                record["media"] = validate_media(path, manifest["sampling"], manifest["run"]["media_timeout_seconds"], output_dir)
                record["status"] = "validated"
                write_json(report_path, report)
        if phase == "candidate":
            require(sha256(baseline_path) == report["baseline_report_sha256"], "fixed baseline report changed during candidate")
        report["statistics"] = {mode: summarize(records) for mode, records in report["modes"].items()}
        report["performance_measured"] = True
        report["status"] = "validated"
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        report["status"] = "unsupported" if "unsupported" in str(exc) else "failed"
        report["error"] = str(exc)
        if isinstance(exc, WorkflowError) and exc.details is not None:
            report["gpu_precheck_failure"] = exc.details
        for records in report["modes"].values():
            for record in records:
                if record.get("status") == "running":
                    record["status"] = "failed"
        # Partial/failed runs are not valid measurements or reusable baselines.
        report["statistics"] = None
        report["performance_measured"] = False
        write_json(report_path, report)
        raise WorkflowError(f"{exc}; report: {report_path}") from exc
    # Rehash build provenance after generation; a rebuilt binary cannot
    # silently change a measured run's source identity.
    if report["performance_measured"]:
        current_build = build_provenance(Path(manifest["binaries"][phase]["path"]))
        if current_build != report["build"]:
            report.update(status="failed", statistics=None, performance_measured=False,
                          error="binary/build provenance changed during run")
            write_json(report_path, report)
            raise WorkflowError(f"binary/build provenance changed during run; report: {report_path}")
    write_json(report_path, report)
    return report_path


def audio_errors(baseline: Path, candidate: Path) -> dict:
    require(baseline.stat().st_size == candidate.stat().st_size and baseline.stat().st_size > 0, "audio sample shape mismatch")
    differences, energies, count = [], [], 0
    for a, b in zip(pcm_values(baseline), pcm_values(candidate)):
        require(len(a) == len(b), "audio sample shape mismatch")
        differences.append(math.fsum((x - y) ** 2 for x, y in zip(a, b)))
        energies.append(math.fsum(x * x for x in a))
        count += len(a)
    error, energy = math.fsum(differences), math.fsum(energies)
    return {"rmse": math.sqrt(error / count), "relative_l2": math.sqrt(error / energy) if energy else (0.0 if not error else None),
            "relative_l2_defined": bool(energy or not error), "scalar_samples": count,
            "reference_norm": math.sqrt(energy)}


def video_metrics(baseline: Path, candidate: Path, frames: int, timeout: float, output: Path) -> dict:
    # No scaling, rate conversion, padding, truncation or resampling. Shape
    # equality is mandatory before entering this function. Stats are per frame.
    filters = ("[0:v]format=yuv444p,settb=AVTB,setpts=PTS-STARTPTS,split[a][c];"
               "[1:v]format=yuv444p,settb=AVTB,setpts=PTS-STARTPTS,split[b][d];"
               "[a][b]ssim=stats_file=ssim.txt:repeatlast=0[s];"
               "[c][d]psnr=stats_file=psnr.txt:repeatlast=0[p]")
    invoke(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(baseline),
            "-threads", "1", "-i", str(candidate), "-filter_complex_threads", "1",
            "-filter_complex", filters, "-map", "[s]", "-map", "[p]", "-f", "null", "-"], timeout, cwd=output)
    ssim_text, psnr_text = (output / "ssim.txt").read_text(), (output / "psnr.txt").read_text()
    ssim = re.findall(r"n:(\d+).*?All:([\d.eE+\-]+)", ssim_text)
    psnr = re.findall(r"n:(\d+).*?psnr_avg:([\d.eE+\-]+|inf)", psnr_text)
    expected = list(range(1, frames + 1))
    require([int(n) for n, _ in ssim] == [int(n) for n, _ in psnr] == expected, "SSIM/PSNR missing/mismatched frame stats")
    per_frame = []
    for (n, s), (_, p) in zip(ssim, psnr):
        s_value = float(s)
        p_value = None if p == "inf" else float(p)
        require(math.isfinite(s_value) and (p_value is None or math.isfinite(p_value)), "nonfinite video metric")
        per_frame.append({"frame": int(n), "ssim": s_value, "psnr_db": p_value, "exact_match": p_value is None})
    finite_psnr = [r["psnr_db"] for r in per_frame if r["psnr_db"] is not None]
    return {"pixel_format": "yuv444p", "per_frame": per_frame,
            "mean_ssim": statistics.mean(r["ssim"] for r in per_frame),
            "mean_psnr_db": statistics.mean(finite_psnr) if len(finite_psnr) == frames else None,
            "exact_frame_count": frames - len(finite_psnr), "psnr_null_means": "positive_infinity_for_exact_match; no fictitious finite PSNR"}


def compare(baseline_path: Path, candidate_path: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    report = {"schema_version": 1, "phase": "compare", "status": "pending", "quality_warning": WARNING,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "baseline_report": str(baseline_path.resolve()), "candidate_report": str(candidate_path.resolve()),
              "modes": {}, "performance_comparable": False, "semantic_quality": "not_evaluated"}
    write_json(path, report)
    try:
        baseline, candidate = frozen_report(baseline_path), frozen_report(candidate_path)
        require(baseline["phase"] == "baseline" and candidate["phase"] == "candidate", "compare needs baseline then candidate")
        require(baseline["contract"] == candidate["contract"], "sampling/weights/assets/precision/placement contract mismatch")
        require(candidate.get("baseline_report_sha256") == sha256(baseline_path), "candidate was not bound to this frozen baseline")
        report["baseline_report_sha256"] = sha256(baseline_path)
        report["candidate_report_sha256"] = sha256(candidate_path)
        sampling = baseline["contract"]["sampling"]
        for mode in baseline["modes"]:
            a = [r for r in baseline["modes"][mode] if not r["warmup"]]
            b = [r for r in candidate["modes"][mode] if not r["warmup"]]
            require(len(a) == len(b) or baseline["status"] == "imported" and len(a) == 1, "baseline/candidate repeat count mismatch")
            report["modes"][mode] = []
            for n, right in enumerate(b):
                left = a[0] if baseline["status"] == "imported" else a[n]
                pair_dir = output / f"{mode}-{n:03d}"
                pair_dir.mkdir()
                # Revalidate original media instead of trusting cached metadata.
                lm = validate_media(Path(left["output"]), sampling, candidate["manifest"]["run"]["media_timeout_seconds"], pair_dir)
                rm_dir = pair_dir / "candidate"
                rm_dir.mkdir()
                rm = validate_media(Path(right["output"]), sampling, candidate["manifest"]["run"]["media_timeout_seconds"], rm_dir)
                require(lm["shape"] == rm["shape"], "decoded video/audio shape mismatch; no resizing/resampling allowed")
                require(left.get("profile", {}).get("actual_placement") == right.get("profile", {}).get("actual_placement"), "actual placement mismatch")
                metrics = {"shape": lm["shape"], "baseline": left["output"], "candidate": right["output"],
                           "video": video_metrics(Path(left["output"]), Path(right["output"]), sampling["frames"], candidate["manifest"]["run"]["media_timeout_seconds"], pair_dir),
                           "audio": audio_errors(Path(lm["audio_pcm"]), Path(rm["audio_pcm"]))}
                report["modes"][mode].append(metrics)
        if baseline.get("performance_measured") and candidate.get("performance_measured"):
            require(all(baseline["manifest"]["run"][key] == candidate["manifest"]["run"][key] for key in ("warmup", "repeats")),
                    "different warmup/repeat schedules cannot be compared for performance")
            require(all(baseline["gpu"][k] == candidate["gpu"][k] for k in ("uuid", "name", "total_mib")), "different GPUs cannot be compared for performance")
            require(baseline["environment"]["other_runtime_env"] == candidate["environment"]["other_runtime_env"], "runtime environment mismatch")
            report["performance_comparable"] = True
            report["timings"] = {"baseline": baseline["statistics"], "candidate": candidate["statistics"]}
        require(sha256(baseline_path) == report["baseline_report_sha256"] and
                sha256(candidate_path) == report["candidate_report_sha256"], "fixed report changed during compare")
        report["status"] = "validated_numerical_comparison_not_semantic_acceptance"
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        report["status"], report["error"] = "failed", str(exc)
        write_json(path, report)
        raise WorkflowError(f"{exc}; report: {path}") from exc
    write_json(path, report)
    return path


def lock_manifest(manifest: dict, output: Path) -> Path:
    # Lock records full hashes, never header/prefix hashes. This may take time
    # for the real weights, so it is explicit and never part of a dry-run.
    for group in ("models", "assets"):
        for key, item in manifest[group].items():
            if group == "models":
                verify_precision_header(key, item)
            actual = fingerprint(Path(item["path"]))
            require(item.get("bytes", actual["bytes"]) == actual["bytes"], f"{key}: precision-lock size mismatch")
            require(item.get("sha256") in (None, actual["sha256"]), f"{key}: existing full hash mismatch")
            if group == "models":
                precision = read_json(ROOT / "docs/research/h3-precision-lock.json")
                pinned = precision["components"][COMPONENTS.index(key)]
                require(actual["sha256"] == pinned["full_sha256"], f"{key}: pinned full-file SHA256 mismatch")
            item.update(actual)
    manifest["hash_status"] = "full_file_sha256_locked"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest, exclusive=True)
    return output


def fixtures(output: Path, manifest: dict) -> Path:
    """Small deterministic CPU-only synthetic fixtures, not model/quality outputs."""
    output.mkdir(parents=True, exist_ok=False)
    timeout = manifest["run"]["media_timeout_seconds"]
    prefix = ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1"]
    for name, color in (("first.png", "blue"), ("last.png", "red")):
        invoke(prefix + ["-f", "lavfi", "-i", f"color=c={color}:s=256x256:r=24", "-frames:v", "1", "-threads", "1", str(output / name)], timeout)
    invoke(prefix + ["-f", "lavfi", "-i", "testsrc2=size=256x256:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000",
                    "-t", "1", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "2", str(output / "reference.mp4")], timeout)
    for name, frequency in (("paired.wav", 440), ("extra.wav", 660)):
        invoke(prefix + ["-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=32000", "-t", "1", "-ac", "2", "-c:a", "pcm_s16le", str(output / name)], timeout)
    frames = output / "frames"
    frames.mkdir()
    invoke(prefix + ["-i", str(output / "reference.mp4"), "-an", "-threads", "1", str(frames / "%03d.png")], timeout)
    replacements = {"first": "first.png", "last": "last.png", "reference_image": "first.png",
                    "reference_video": "reference.mp4", "reference_frames": "frames", "paired_audio": "paired.wav", "extra_audio": "extra.wav"}
    require(set(manifest["assets"]) == set(replacements), "fixtures expect the supplied H3 asset keys")
    for key, name in replacements.items():
        manifest["assets"][key]["path"] = str(output / name)
        manifest["assets"][key].update(fingerprint(output / name))
    # Preserve root identities so a fully hashed fixture manifest can be moved
    # host -> container/remote with --asset-root without rehashing or editing.
    manifest.setdefault("path_variables", {})["asset_root"] = str(output)
    manifest["locations"][manifest.get("resolved_location", "host")]["asset_root"] = str(output)
    manifest["fixture_note"] = "CPU synthetic wiring fixtures, no downloaded resources; not semantic-quality evidence"
    path = output / "fixture-manifest.json"
    write_json(path, manifest, exclusive=True)
    return path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s workflow --manifest benchmark/h3_cuda/example.json --dry-run --output-dir /tmp/h3-plan\n"
               "  %(prog)s fixtures --manifest benchmark/h3_cuda/example.json --output-dir /tmp/h3-fixtures\n"
               "  %(prog)s lock --manifest /tmp/h3-fixtures/fixture-manifest.json --output-manifest /tmp/h3-locked.json\n"
               "  %(prog)s baseline --manifest /tmp/h3-locked.json --output-dir /tmp/h3-baseline\n"
               "  %(prog)s candidate --manifest /tmp/h3-locked.json --baseline /tmp/h3-baseline/report.json --output-dir /tmp/h3-candidate\n"
               "  %(prog)s compare --baseline /tmp/h3-baseline/report.json --candidate /tmp/h3-candidate/report.json --output-dir /tmp/h3-compare\n"
               "Use --profile container inside wty-edgedit-dev; no implicit docker execution.\n"
               "All output directories/files must be new. Dry-run never checks/uses GPUs or measures latency.")
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("fixtures", "lock", "baseline", "candidate", "workflow", "compare"):
        c = sub.add_parser(action, help={"fixtures": "create CPU-only local media fixtures", "lock": "freeze full weight/asset SHA256 (no GPU)",
            "baseline": "run/import an immutable all-options-off baseline", "candidate": "run against a fixed baseline", "workflow": "baseline (or reuse) -> candidate -> compare", "compare": "strict decoded comparison, no GPU"}[action])
        if action != "compare":
            c.add_argument("--manifest", type=Path, required=True)
            c.add_argument("--profile", choices=("host", "container"), default="host")
            for name in ("repo-root", "model-root", "asset-root"):
                c.add_argument("--" + name, type=Path, help="override manifest location path")
        if action == "lock":
            c.add_argument("--output-manifest", type=Path, required=True)
        else:
            c.add_argument("--output-dir", type=Path, required=True)
        if action in ("baseline", "candidate", "workflow"):
            c.add_argument("--dry-run", action="store_true", help="plan only; skip GPU/full hashing; no performance values")
            c.add_argument("--mode", action="append", choices=MODE_ORDER, help="repeatable; default full matrix, mixed first")
            c.add_argument("--warmup", type=int, help=">=1 separate-process warmups")
            c.add_argument("--repeats", type=int, help=">=1 measured repeats")
            c.add_argument("--binary", type=Path, help="override this phase binary (workflow: candidate)")
        if action in ("candidate", "workflow", "compare"):
            c.add_argument("--baseline", type=Path, required=action in ("candidate", "compare"), help="validated immutable baseline report, not a dry-run")
        if action == "baseline":
            c.add_argument("--artifact", action="append", metavar="MODE=PATH", help="import fixed output without GPU or performance claims")
        if action == "compare":
            c.add_argument("--candidate", type=Path, required=True)
    return p


def h3_performance_pair(baseline: Path, candidate: Path) -> bool:
    return bool(read_json(baseline).get("performance_measured") and read_json(candidate).get("performance_measured"))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "compare":
            result = compare(args.baseline.resolve(), args.candidate.resolve(), args.output_dir.resolve())
        else:
            manifest = resolve_manifest(args.manifest, args.profile, {name: getattr(args, name) for name in ("repo_root", "model_root", "asset_root")})
            if args.action == "lock":
                result = lock_manifest(manifest, args.output_manifest.resolve())
            elif args.action == "fixtures":
                result = fixtures(args.output_dir.resolve(), manifest)
            else:
                for key in ("warmup", "repeats"):
                    if getattr(args, key) is not None:
                        manifest["run"][key] = integer(getattr(args, key), key)
                if args.binary:
                    phase = "baseline" if args.action == "baseline" else "candidate"
                    manifest["binaries"][phase]["path"] = str(args.binary.resolve())
                modes = selected_modes(manifest, args.mode)
                if args.action == "workflow":
                    root = args.output_dir.resolve()
                    root.mkdir(parents=True, exist_ok=False)
                    result = root / "workflow.json"
                    workflow_report = {"status": "pending", "performance_measured": False, "quality_warning": WARNING}
                    write_json(result, workflow_report, exclusive=True)
                    try:
                        base = args.baseline.resolve() if args.baseline else run_phase(manifest, "baseline", root / "baseline", modes, dry_run=args.dry_run)
                        workflow_report["baseline"] = str(base)
                        cand = run_phase(manifest, "candidate", root / "candidate", modes, dry_run=args.dry_run,
                                         baseline_path=base if not args.dry_run or args.baseline else None)
                        workflow_report["candidate"] = str(cand)
                        cmp = None if args.dry_run else compare(base, cand, root / "compare")
                        workflow_report.update(status="dry_run" if args.dry_run else "validated_numerical_comparison_not_semantic_acceptance",
                            compare=str(cmp) if cmp else None,
                            performance_measured=not args.dry_run and h3_performance_pair(base, cand))
                    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
                        workflow_report.update(status="failed", error=str(exc), performance_measured=False)
                        write_json(result, workflow_report)
                        raise
                    write_json(result, workflow_report)
                else:
                    imported = None
                    if args.action == "baseline" and args.artifact:
                        imported = {}
                        for spec in args.artifact:
                            require("=" in spec, "--artifact requires MODE=PATH")
                            mode, path = spec.split("=", 1)
                            require(mode not in imported and mode in modes and bool(path), "duplicate/invalid artifact mode")
                            imported[mode] = path
                    result = run_phase(manifest, args.action, args.output_dir.resolve(), modes,
                                       dry_run=args.dry_run, baseline_path=getattr(args, "baseline", None), artifacts=imported)
        print(result)
        return 0
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"h3 workflow: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
