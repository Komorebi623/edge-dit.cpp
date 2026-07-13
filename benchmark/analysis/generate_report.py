#!/usr/bin/env python3
"""Generate a benchmark report from summary JSON and tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPORT_SECTIONS = [
    "Executive Summary",
    "Benchmark Methodology",
    "Workload Matrix",
    "Functional and Numerical Validation",
    "Single-GPU Performance",
    "Memory-Efficient Execution",
    "Graph and Operator Ablation",
    "Computation Reuse",
    "Parallel Scaling",
    "Stability and Reproducibility",
    "Known Limitations",
    "Reproduction",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    tables = args.tables.read_text(encoding="utf-8") if args.tables and args.tables.exists() else ""
    table_sections = split_table_sections(tables)
    lines = [
        "# Performance and Benchmarks",
        "",
        f"Suite: `{data.get('suite_id', 'unknown')}`",
        "",
        "> This report is generated from benchmark result summaries. Pending rows mean",
        "> the corresponding official benchmark has not been executed yet.",
        "",
    ]
    for section in REPORT_SECTIONS:
        lines.append(f"## {section}")
        lines.append("")
        if section == "Single-GPU Performance" and table_sections.get("Single-GPU Inference"):
            lines.append(table_sections["Single-GPU Inference"])
            lines.append("")
        elif section == "Parallel Scaling" and table_sections.get("Memory and Parallel Scaling"):
            lines.append(table_sections["Memory and Parallel Scaling"])
            lines.append("")
        else:
            lines.append("Pending official benchmark results.")
            lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


def split_table_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {
        title: "\n".join(lines).strip()
        for title, lines in sections.items()
        if "\n".join(lines).strip()
    }


if __name__ == "__main__":
    raise SystemExit(main())
