#!/usr/bin/env python3
"""Generate a benchmark report from summary JSON and tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--tables", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    tables = args.tables.read_text(encoding="utf-8") if args.tables and args.tables.exists() else ""
    table_sections = split_table_sections(tables)
    successful = [row for row in rows if row.get("status") == "success"]
    workloads = sorted({str(row.get("workload")) for row in rows if row.get("workload")})
    systems = sorted({str(row.get("system")) for row in rows if row.get("system")})
    lines = [
        "# Performance and Benchmarks",
        "",
        f"Suite: `{data.get('suite_id', 'unknown')}`",
        "",
        "This report is generated from benchmark result summaries.",
        "",
        "## Executive Summary",
        "",
        f"- Rows: {len(rows)} total, {len(successful)} successful.",
        f"- Workloads: {', '.join(workloads) if workloads else '-'}",
        f"- Systems: {', '.join(systems) if systems else '-'}",
        "",
    ]
    if table_sections:
        for title, body in table_sections.items():
            lines.append(f"## {title}")
            lines.append("")
            lines.append(body)
            lines.append("")
    else:
        lines.append("No benchmark rows found.")
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
