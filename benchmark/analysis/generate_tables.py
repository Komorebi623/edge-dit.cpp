#!/usr/bin/env python3
"""Generate Markdown tables from benchmark summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt_ms(value: Any) -> str:
    return "Pending" if value is None else f"{float(value):.1f} ms"


def fmt_mib(value: Any) -> str:
    return "Pending" if value is None else f"{float(value):.0f} MiB"


def fmt_x(value: Any) -> str:
    return "Pending" if value is None else f"{float(value):.2f}x"


def fmt_pct(value: Any) -> str:
    return "Pending" if value is None else f"{float(value) * 100.0:.1f}%"


def single_gpu_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Scenario | System | Load | Median | P90 | Peak VRAM | Boundary | Status |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    single_rows = [
        row
        for row in rows
        if row.get("gpu_count", 1) == 1 and row.get("parallel_mode") is None
    ]
    if not single_rows:
        lines.append("| Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |")
    for row in single_rows:
        lines.append(
            "| {workload} | {scenario} | {system} | {load} | {median} | {p90} | {vram} | {boundary} | {status} |".format(
                workload=row.get("workload", "unknown"),
                scenario=row.get("scenario", "default"),
                system=row.get("system", "unknown"),
                load=fmt_ms(row.get("load_ms")),
                median=fmt_ms(row.get("steady_state_median_ms")),
                p90=fmt_ms(row.get("steady_state_p90_ms")),
                vram=fmt_mib(row.get("peak_vram_mib")),
                boundary=row.get("measurement_boundary") or "unknown",
                status=row.get("status", "unknown"),
            )
        )
    return "\n".join(lines)


def parallel_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Workload | Scenario | System | Mode | GPUs | Median | Speedup | Efficiency | Peak VRAM | Boundary | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    parallel_rows = [row for row in rows if row.get("parallel_mode") is not None]
    if not parallel_rows:
        lines.append("| Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |")
    for row in parallel_rows:
        lines.append(
            "| {workload} | {scenario} | {system} | {mode} | {gpus} | {median} | {speedup} | {efficiency} | {vram} | {boundary} | {status} |".format(
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
                status=row.get("status", "unknown"),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    markdown = "\n\n".join(
        [
            "## Single-GPU Inference",
            single_gpu_table(rows),
            "## Memory and Parallel Scaling",
            parallel_table(rows),
        ]
    )
    markdown += "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
