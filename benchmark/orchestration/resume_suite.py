#!/usr/bin/env python3
"""Inspect completed and pending benchmark suite runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from .config import expand_runs, load_suite_graph, repo_root
    from .run_suite import filter_runs, find_completed_result
except ImportError:  # pragma: no cover - direct script execution
    from config import expand_runs, load_suite_graph, repo_root
    from run_suite import filter_runs, find_completed_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--systems", nargs="*", help="Optional system_id filter")
    parser.add_argument(
        "--fail-if-pending",
        action="store_true",
        help="Return a non-zero status when any suite entry is still pending",
    )
    args = parser.parse_args()

    graph = load_suite_graph(args.suite.resolve(), args.site.resolve())
    runs = filter_runs(expand_runs(graph), args.systems)
    completed = []
    pending = []

    for run in runs:
        result_dir = find_completed_result(args.results_root, graph["suite"]["suite_id"], run)
        item = {
            "system_id": run["system_id"],
            "workload_id": run["workload_id"],
            "gpu_count": run["gpu_count"],
            "parallel_mode": run["parallel_mode"],
            "scenario_id": run["scenario_id"],
        }
        if result_dir is None:
            pending.append(item)
        else:
            completed.append({**item, "output_dir": str(result_dir)})

    output = {
        "suite_id": graph["suite"]["suite_id"],
        "completed_count": len(completed),
        "pending_count": len(pending),
        "completed": completed,
        "pending": pending,
        "resume_command": (
            "python3 benchmark/orchestration/run_suite.py "
            f"--suite {args.suite} --site {args.site} --execute --resume"
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.fail_if_pending and pending:
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(repo_root()))
    raise SystemExit(main())
