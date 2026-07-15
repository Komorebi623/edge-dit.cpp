#!/usr/bin/env python3
"""Generate Markdown tables from benchmark summary JSON."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


def fmt_ms(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f} ms"


def fmt_s(value: Any) -> str:
    return "-" if value is None else f"{float(value) / 1000.0:.3f} s"


def fmt_mib(value: Any) -> str:
    return "-" if value is None else f"{float(value):.0f} MiB"


def fmt_gib(value: Any) -> str:
    return "-" if value is None else f"{float(value) / 1024.0:.2f} GiB"


def fmt_x(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}x"


def fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100.0:.1f}%"


def fmt_num(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median(values: list[Any]) -> float | None:
    nums = numeric(values)
    return statistics.median(nums) if nums else None


def mean(values: list[Any]) -> float | None:
    nums = numeric(values)
    return statistics.fmean(nums) if nums else None


def single_gpu_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Scenario | System | Load | Median | P90 | Peak VRAM | Boundary |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    single_rows = [
        row
        for row in rows
        if row.get("status") == "success"
        if row.get("gpu_count", 1) == 1 and row.get("parallel_mode") is None
    ]
    if not single_rows:
        return ""
    for row in single_rows:
        lines.append(
            "| {workload} | {scenario} | {system} | {load} | {median} | {p90} | {vram} | {boundary} |".format(
                workload=row.get("workload", "unknown"),
                scenario=row.get("scenario", "default"),
                system=row.get("system", "unknown"),
                load=fmt_ms(row.get("load_ms")),
                median=fmt_ms(row.get("steady_state_median_ms")),
                p90=fmt_ms(row.get("steady_state_p90_ms")),
                vram=fmt_mib(row.get("peak_vram_mib")),
                boundary=row.get("measurement_boundary") or "unknown",
            )
        )
    return "\n".join(lines)


def parallel_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Scenario | System | Mode | GPUs | Median | Speedup | Efficiency | Peak VRAM | Boundary |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    parallel_rows = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("parallel_mode") is not None
    ]
    if not parallel_rows:
        return ""
    for row in parallel_rows:
        lines.append(
            "| {workload} | {scenario} | {system} | {mode} | {gpus} | {median} | {speedup} | {efficiency} | {vram} | {boundary} |".format(
                workload=row.get("workload", "unknown"),
                scenario=row.get("scenario", "default"),
                system=row.get("system", "unknown"),
                mode=row.get("parallel_mode") or "default",
                gpus=row.get("gpu_count", 1),
                median=fmt_ms(row.get("steady_state_median_ms")),
                speedup=fmt_x(row.get("speedup")),
                efficiency=fmt_pct(row.get("scaling_efficiency")),
                vram=fmt_mib(row.get("peak_vram_mib")),
                boundary=row.get("measurement_boundary") or "unknown",
            )
        )
    return "\n".join(lines)


def overall_single_gpu_table(rows: list[dict[str, Any]]) -> str:
    source = [row for row in rows if row.get("suite") == "readme-main-table"]
    if not source:
        return ""
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source:
        if row.get("status") != "success":
            continue
        model = row.get("model") or row.get("workload") or "unknown"
        by_model[str(model)][str(row.get("system"))] = row
    lines = [
        "| Model | edge-dit.cpp Median | Relative to Diffusers | Speedup over sd.cpp | Peak VRAM |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, systems in by_model.items():
        edge = systems.get("edge-dit.cpp")
        if edge is None:
            continue
        edge_median = edge.get("steady_state_median_ms")
        diffusers_median = systems.get("diffusers", {}).get("steady_state_median_ms")
        sdcpp_median = systems.get("stable-diffusion.cpp", {}).get("steady_state_median_ms")
        relative = (
            float(edge_median) / float(diffusers_median)
            if isinstance(edge_median, (int, float))
            and isinstance(diffusers_median, (int, float))
            and diffusers_median > 0
            else None
        )
        speedup = (
            float(sdcpp_median) / float(edge_median)
            if isinstance(edge_median, (int, float))
            and isinstance(sdcpp_median, (int, float))
            and edge_median > 0
            else None
        )
        lines.append(
            f"| {model} | {fmt_s(edge_median)} | {fmt_x(relative)} | {fmt_x(speedup)} | {fmt_mib(edge.get('peak_vram_mib'))} |"
        )
    return "\n".join(lines) if len(lines) > 2 else ""


def task_coverage_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == "task-coverage" and row.get("status") == "success"
    ]
    if not source:
        return ""
    by_workload: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source:
        by_workload[str(row.get("workload"))][str(row.get("system"))] = row
    lines = [
        "| Model | Task | Setting | Steps | edge-dit.cpp | Diffusers | Relative | Peak VRAM |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for workload, systems in by_workload.items():
        edge = systems.get("edge-dit.cpp")
        diffusers = systems.get("diffusers")
        row = edge or diffusers
        if row is None:
            continue
        edge_median = edge.get("steady_state_median_ms") if edge else None
        diffusers_median = diffusers.get("steady_state_median_ms") if diffusers else None
        relative = (
            float(edge_median) / float(diffusers_median)
            if isinstance(edge_median, (int, float))
            and isinstance(diffusers_median, (int, float))
            and diffusers_median > 0
            else None
        )
        lines.append(
            "| {model} | {task} | {setting} | {steps} | {edge} | {diffusers} | {relative} | {vram} |".format(
                model=row.get("model") or row.get("workload", "unknown"),
                task=row.get("task") or "-",
                setting=setting_label(row),
                steps=row.get("steps") if row.get("steps") is not None else "-",
                edge=fmt_s(edge_median),
                diffusers=fmt_s(diffusers_median),
                relative=fmt_x(relative),
                vram=fmt_mib(edge.get("peak_vram_mib") if edge else row.get("peak_vram_mib")),
            )
        )
    return "\n".join(lines)


def setting_label(row: dict[str, Any]) -> str:
    width = row.get("width")
    height = row.get("height")
    frames = row.get("frames")
    if isinstance(width, int) and isinstance(height, int):
        if isinstance(frames, int) and frames > 1:
            return f"{width}x{height}x{frames}"
        return f"{width}x{height}"
    return str(row.get("workload", "-"))


def cfg_parallel_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == "cfg-parallel" and row.get("status") == "success"
    ]
    if not source:
        return ""
    lines = [
        "| Mode | GPUs | Median | P90 | Speedup | Efficiency | Max VRAM / GPU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(source, key=lambda item: item.get("gpu_count", 0)):
        mode = "Single GPU" if row.get("gpu_count") == 1 else "CFG parallel"
        lines.append(
            f"| {mode} | {row.get('gpu_count', 1)} | {fmt_s(row.get('steady_state_median_ms'))} | "
            f"{fmt_s(row.get('steady_state_p90_ms'))} | {fmt_x(row.get('speedup'))} | "
            f"{fmt_pct(row.get('scaling_efficiency'))} | {fmt_mib(row.get('peak_vram_mib'))} |"
        )
    return "\n".join(lines)


def sequence_parallel_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") in {"sp-parallel", "parallel"}
        and row.get("status") == "success"
    ]
    if not source:
        return ""
    lines = [
        "| Workload | System | Mode | GPUs | Median | Speedup | Efficiency | Max VRAM / GPU | Comm. | Segments / Step |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source:
        lines.append(
            "| {workload} | {system} | {mode} | {gpus} | {median} | {speedup} | {eff} | {vram} | {comm} | {segments} |".format(
                workload=row.get("workload", "unknown"),
                system=row.get("system", "unknown"),
                mode=row.get("parallel_mode") or "single",
                gpus=row.get("gpu_count", 1),
                median=fmt_s(row.get("steady_state_median_ms")),
                speedup=fmt_x(row.get("speedup")),
                eff=fmt_pct(row.get("scaling_efficiency")),
                vram=fmt_mib(row.get("peak_vram_mib")),
                comm=fmt_ms(row.get("communication_ms")),
                segments=row.get("graph_segment_count") if row.get("graph_segment_count") is not None else "-",
            )
        )
    return "\n".join(lines)


def cache_speed_quality_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == "cache-quality" and row.get("status") == "success"
    ]
    if not source:
        return ""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        options = row.get("run_options", {})
        cache = options.get("cache", "off") if isinstance(options, dict) else "off"
        groups[str(cache)].append(row)
    lines = [
        "| Method | Granularity | Median | Speedup | Peak VRAM | Saved Steps | PSNR | LPIPS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = median([row.get("steady_state_median_ms") for row in groups.get("off", [])])
    cache_order = ["off", "easycache", "cache-dit", "magcache", "dicache", "sencache"]
    ordered_keys = [key for key in cache_order if key in groups]
    ordered_keys.extend(sorted(key for key in groups if key not in cache_order))
    for cache in ordered_keys:
        group = groups[cache]
        options = group[0].get("run_options", {}) if group else {}
        granularity = options.get("cache_granularity", "-") if isinstance(options, dict) else "-"
        med = median([row.get("steady_state_median_ms") for row in group])
        speedup = baseline / med if isinstance(baseline, (int, float)) and isinstance(med, (int, float)) and med > 0 else None
        reused, total = cache_reuse_summary(group)
        reuse_text = f"{reused}/{total}" if total is not None else "-"
        lines.append(
            f"| {cache_method_label(cache)} | {cache_granularity_label(granularity)} | {fmt_s(med)} | {fmt_x(speedup)} | "
            f"{fmt_mib(median([row.get('peak_vram_mib') for row in group]))} | "
            f"{reuse_text} | "
            f"{fmt_num(mean([row.get('psnr') for row in group]), 2)} | "
            f"{fmt_num(mean([row.get('lpips') for row in group]), 4)} |"
        )
    return "\n".join(lines)


def cache_method_label(value: str) -> str:
    return {
        "off": "Full compute",
        "easycache": "EasyCache",
        "cache-dit": "CacheDiT",
        "magcache": "MagCache",
        "dicache": "DiCache",
        "sencache": "SenCache",
    }.get(value, value)


def cache_granularity_label(value: Any) -> str:
    return {
        "full": "full",
        "output": "output",
        "block-output": "block/output",
        "feature": "feature",
        "probe": "probe",
    }.get(str(value), str(value))


def cache_reuse_summary(group: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    reused_counts = []
    totals = []
    saved_verbs = {"reused", "skipped"}
    for row in group:
        events = row.get("cache_events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("verb") not in saved_verbs:
                continue
            if isinstance(event.get("count"), int) and isinstance(event.get("total"), int):
                reused_counts.append(int(event["count"]))
                totals.append(int(event["total"]))
    if not totals:
        return None, None
    return int(round(statistics.fmean(reused_counts))), int(round(statistics.fmean(totals)))


def scenario_rows_table(rows: list[dict[str, Any]], suite: str, title_kind: str) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == suite and row.get("status") == "success"
    ]
    if not source:
        return ""
    lines = [
        f"| {title_kind} | Workload | Median | P90 | Peak VRAM | Host RAM | Slowdown / Speedup |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    baselines: dict[str, float] = {}
    for row in source:
        if row.get("scenario") in {"performance_bf16", "bf16", "untiled", "generic"}:
            med = row.get("steady_state_median_ms")
            if isinstance(med, (int, float)) and med > 0:
                baselines[str(row.get("workload"))] = float(med)
    for row in source:
        med = row.get("steady_state_median_ms")
        baseline = baselines.get(str(row.get("workload")))
        relative = (
            float(med) / baseline
            if isinstance(med, (int, float)) and isinstance(baseline, (int, float)) and baseline > 0
            else None
        )
        lines.append(
            "| {scenario} | {workload} | {median} | {p90} | {vram} | {host} | {relative} |".format(
                scenario=row.get("scenario", "default"),
                workload=row.get("workload", "unknown"),
                median=fmt_s(med),
                p90=fmt_s(row.get("steady_state_p90_ms")),
                vram=fmt_mib(row.get("peak_vram_mib")),
                host=fmt_mib(row.get("peak_host_rss_mib")),
                relative=fmt_x(relative),
            )
        )
    return "\n".join(lines)


def resource_profiles_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == "resource-profiles" and row.get("status") == "success"
    ]
    if not source:
        return ""
    baseline = median(
        [
            row.get("steady_state_median_ms")
            for row in source
            if row.get("scenario") == "performance_bf16"
        ]
    )
    lines = [
        "| Profile | Weight | CPU Placement / Offload | Graph Budget | VAE Tiling | Median | Slowdown | Peak VRAM | Host RAM |",
        "|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in source:
        options = row.get("run_options", {})
        if not isinstance(options, dict):
            options = {}
        med = row.get("steady_state_median_ms")
        slowdown = (
            float(med) / float(baseline)
            if isinstance(med, (int, float))
            and isinstance(baseline, (int, float))
            and baseline > 0
            else None
        )
        lines.append(
            "| {profile} | {weight} | {placement} | {budget} | {tiling} | {median} | {slowdown} | {vram} | {host} |".format(
                profile=options.get("profile_label", row.get("scenario", "default")),
                weight=str(options.get("precision", "-")).upper(),
                placement=placement_label(options),
                budget=graph_budget_label(options),
                tiling="on" if options.get("vae_tiling") else "off",
                median=fmt_s(med),
                slowdown=fmt_x(slowdown),
                vram=fmt_mib(row.get("peak_vram_mib")),
                host=fmt_mib(row.get("peak_host_rss_mib")),
            )
        )
    return "\n".join(lines)


def quantization_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == "quantization" and row.get("status") == "success"
    ]
    if not source:
        return ""
    baseline = median(
        [
            row.get("steady_state_median_ms")
            for row in source
            if row.get("scenario") == "bf16"
        ]
    )
    lines = [
        "| Weight Type | Load | Median | Slowdown | Peak VRAM | Host RAM | Policy |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in source:
        options = row.get("run_options", {})
        if not isinstance(options, dict):
            options = {}
        med = row.get("steady_state_median_ms")
        slowdown = (
            float(med) / float(baseline)
            if isinstance(med, (int, float))
            and isinstance(baseline, (int, float))
            and baseline > 0
            else None
        )
        lines.append(
            "| {weight} | {load} | {median} | {slowdown} | {vram} | {host} | {policy} |".format(
                weight=options.get("weight_type_label", row.get("scenario", "default")),
                load=fmt_s(row.get("load_ms")),
                median=fmt_s(med),
                slowdown=fmt_x(slowdown),
                vram=fmt_mib(row.get("peak_vram_mib")),
                host=fmt_mib(row.get("peak_host_rss_mib")),
                policy=options.get("tensor_type_rules", "-"),
            )
        )
    return "\n".join(lines)


def vae_tiling_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite") == "vae-tiling" and row.get("status") == "success"
    ]
    if not source:
        return ""
    baseline_latency = median(
        [
            row.get("steady_state_median_ms")
            for row in source
            if row.get("scenario") == "untiled"
        ]
    )
    baseline_vram = median(
        [
            row.get("peak_vram_mib")
            for row in source
            if row.get("scenario") == "untiled"
        ]
    )
    lines = [
        "| VAE Mode | Tile Layout | Median | Slowdown | Peak VRAM | VRAM Reduction | Host RAM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in source:
        options = row.get("run_options", {})
        if not isinstance(options, dict):
            options = {}
        med = row.get("steady_state_median_ms")
        vram = row.get("peak_vram_mib")
        slowdown = (
            float(med) / float(baseline_latency)
            if isinstance(med, (int, float))
            and isinstance(baseline_latency, (int, float))
            and baseline_latency > 0
            else None
        )
        reduction = (
            1.0 - float(vram) / float(baseline_vram)
            if isinstance(vram, (int, float))
            and isinstance(baseline_vram, (int, float))
            and baseline_vram > 0
            else None
        )
        layout = str(options.get("tile_layout", "full_image")).replace("_", " ")
        lines.append(
            "| {mode} | {layout} | {median} | {slowdown} | {vram} | {reduction} | {host} |".format(
                mode=options.get("vae_mode", row.get("scenario", "default")),
                layout=layout,
                median=fmt_s(med),
                slowdown=fmt_x(slowdown),
                vram=fmt_mib(vram),
                reduction=fmt_pct(reduction),
                host=fmt_mib(row.get("peak_host_rss_mib")),
            )
        )
    return "\n".join(lines)


def placement_label(options: dict[str, Any]) -> str:
    text_cpu = bool(options.get("keep_text_encoder_on_cpu"))
    offload = bool(options.get("offload_to_cpu"))
    if text_cpu and offload:
        return "text encoder CPU + parameter offload"
    if offload:
        return "parameter offload"
    if text_cpu:
        return "text encoder CPU"
    return "none"


def graph_budget_label(options: dict[str, Any]) -> str:
    value = options.get("max_vram_gib")
    if isinstance(value, (int, float)):
        return f"{float(value):.0f} GiB"
    return "unlimited"


def cuda_ablation_table(rows: list[dict[str, Any]]) -> str:
    source = [
        row
        for row in rows
        if row.get("suite")
        in {"cuda-optimization-ablation", "cuda-optimization-ablation-qwen"}
        and row.get("status") == "success"
    ]
    if not source:
        return ""
    lines = [
        "| Model | Build | cuDNN SDPA | CUDA Norm | CUDA RoPE | Fused Modulation | Median | Speedup | Peak VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baselines: dict[str, float] = {}
    for row in source:
        if row.get("scenario") in {"generic", "norm_rope"}:
            key = str(row.get("workload"))
            if key in baselines and row.get("scenario") != "generic":
                continue
            med = row.get("steady_state_median_ms")
            if isinstance(med, (int, float)) and med > 0:
                baselines[key] = float(med)
    for row in source:
        options = row.get("run_options", {})
        if not isinstance(options, dict):
            options = {}
        med = row.get("steady_state_median_ms")
        baseline = baselines.get(str(row.get("workload")))
        speedup = baseline / float(med) if isinstance(med, (int, float)) and baseline and med > 0 else None
        lines.append(
            "| {model} | {build} | {cudnn} | {norm} | {rope} | {mod} | {median} | {speedup} | {vram} |".format(
                model=row.get("model") or row.get("workload"),
                build=options.get("cuda_build_variant", row.get("scenario")),
                cudnn=check(options.get("cudnn_sdpa")),
                norm=check(options.get("cuda_norm")),
                rope=check(options.get("cuda_rope")),
                mod=check(options.get("cuda_modulation")),
                median=fmt_s(med),
                speedup=fmt_x(speedup),
                vram=fmt_mib(row.get("peak_vram_mib")),
            )
        )
    return "\n".join(lines)


def check(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    sections = []
    feature_tables = [
        ("## Overall Single-GPU Performance", overall_single_gpu_table(rows)),
        ("## Model and Task Coverage", task_coverage_table(rows)),
        ("## CFG Parallel Scaling", cfg_parallel_table(rows)),
        ("## Sequence Parallel Scaling", sequence_parallel_table(rows)),
        ("## Cache Speed-Quality Trade-off", cache_speed_quality_table(rows)),
        ("## Resource-Constrained Profiles", resource_profiles_table(rows)),
        ("## Quantization Trade-off", quantization_table(rows)),
        ("## VAE Tiling", vae_tiling_table(rows)),
        ("## CUDA Optimization Ablation", cuda_ablation_table(rows)),
    ]
    for title, table in feature_tables:
        if table:
            sections.extend([title, table])
    if not sections:
        sections = ["No benchmark rows found."]
    markdown = "\n\n".join(sections)
    markdown += "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
