#!/usr/bin/env python3
"""Wait until enough GPUs have sufficient free memory."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--min-free-mib", type=int, default=60000)
    parser.add_argument("--max-utilization", type=int, default=30)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument(
        "--prefer",
        help="Optional comma-separated GPU order or subset, for example 1,2,3,4.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        print("--count must be positive", file=sys.stderr)
        return 2
    if args.interval_sec <= 0:
        print("--interval-sec must be positive", file=sys.stderr)
        return 2

    preferred = parse_preferred(args.prefer)
    start = time.monotonic()
    while True:
        gpus = query_gpus()
        ordered = order_gpus(gpus, preferred)
        available = [
            gpu
            for gpu in ordered
            if gpu["free_mib"] >= args.min_free_mib
            and gpu["utilization"] <= args.max_utilization
        ]
        if len(available) >= args.count:
            print(",".join(str(gpu["index"]) for gpu in available[: args.count]))
            return 0

        status = " ".join(
            f"{gpu['index']}:free={gpu['free_mib']}MiB,util={gpu['utilization']}%"
            for gpu in ordered
        )
        print(
            "waiting for "
            f"{args.count} GPU(s) with >= {args.min_free_mib} MiB free and "
            f"<= {args.max_utilization}% utilization; current: {status}",
            file=sys.stderr,
            flush=True,
        )
        if args.timeout_sec > 0 and time.monotonic() - start >= args.timeout_sec:
            print("timed out waiting for available GPUs", file=sys.stderr)
            return 1
        time.sleep(args.interval_sec)


def parse_preferred(value: str | None) -> list[int] | None:
    if not value:
        return None
    preferred = []
    for item in value.split(","):
        item = item.strip()
        if item:
            preferred.append(int(item))
    return preferred or None


def query_gpus() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        raise SystemExit(completed.returncode)

    gpus = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        index, used, total, utilization = (int(field) for field in fields)
        gpus.append(
            {
                "index": index,
                "used_mib": used,
                "total_mib": total,
                "free_mib": total - used,
                "utilization": utilization,
            }
        )
    return gpus


def order_gpus(
    gpus: list[dict[str, int]],
    preferred: list[int] | None,
) -> list[dict[str, int]]:
    if preferred is None:
        return sorted(gpus, key=lambda gpu: gpu["index"])
    by_index = {gpu["index"]: gpu for gpu in gpus}
    return [by_index[index] for index in preferred if index in by_index]


if __name__ == "__main__":
    raise SystemExit(main())
