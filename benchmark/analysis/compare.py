#!/usr/bin/env python3
"""Compare benchmark summary rows by workload and GPU count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--primary", default="edge-dit.cpp")
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        key = (row["workload"], row.get("gpu_count", 1))
        by_key.setdefault(key, {})[row["system"]] = row

    comparisons = []
    for (workload, gpu_count), systems in sorted(by_key.items()):
        primary = systems.get(args.primary)
        if not primary:
            continue
        primary_latency = primary.get("steady_state_median_ms")
        for system, row in sorted(systems.items()):
            if system == args.primary:
                continue
            latency = row.get("steady_state_median_ms")
            speedup = None
            if primary_latency and latency:
                speedup = latency / primary_latency
            comparisons.append(
                {
                    "workload": workload,
                    "gpu_count": gpu_count,
                    "primary": args.primary,
                    "baseline": system,
                    "primary_median_ms": primary_latency,
                    "baseline_median_ms": latency,
                    "primary_speedup_vs_baseline": speedup,
                }
            )

    print(json.dumps({"comparisons": comparisons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
