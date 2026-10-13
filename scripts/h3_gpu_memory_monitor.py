#!/usr/bin/env python3
"""Wrap a benchmark command with read-only, single-GPU memory sampling.

The child owns resource admission and timeouts (use h3_cuda_workflow.py). This
monitor never reserves a GPU, changes device state, or terminates a process.
Values are sampled whole-device usage, NOT an exact allocation/process peak.
"""
import argparse
import csv
import datetime
import json
import math
import re
import subprocess
import threading
import time
from pathlib import Path


def sample(uuid, invoke=subprocess.run):
    result = invoke(["nvidia-smi", "--id=" + uuid,
                     "--query-gpu=uuid,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=5, check=True)
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("expected exactly one GPU telemetry row")
    device, used, total, util = [v.strip() for v in rows[0]]
    values = [float(v) for v in (used, total, util)]
    if device != uuid or not all(math.isfinite(v) for v in values):
        raise ValueError("GPU identity mismatch or nonfinite telemetry")
    used, total, util = values
    if not 0 <= used <= total or total <= 0 or not 0 <= util <= 100:
        raise ValueError("out-of-range GPU telemetry")
    return {"gpu_uuid": device, "memory_used_mib": used,
            "memory_total_mib": total, "utilization_percent": util}


def summarize(samples, errors, interval_seconds):
    times = [s["elapsed_seconds"] for s in samples]
    return {
        "measurement": "sampled whole-device memory; not exact peak or process-only usage",
        "requested_interval_seconds": interval_seconds,
        "samples": len(samples), "telemetry_errors": len(errors),
        "telemetry_complete": bool(samples) and not errors,
        "sampled_peak_used_mib": max((s["memory_used_mib"] for s in samples), default=None),
        "max_sample_gap_seconds": max((b-a for a, b in zip(times, times[1:])), default=None),
    }


def monitor(command, uuid, output, interval=0.25):
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid):
        raise ValueError("explicit full GPU UUID required")
    if not command or not math.isfinite(interval) or not 0.1 <= interval <= 10:
        raise ValueError("command and sample interval 0.1..10 seconds required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    samples, errors = [], []
    stop = threading.Event()
    # Probe before launching: absent/unreadable telemetry must not launch a job.
    first = sample(uuid)
    first["elapsed_seconds"] = time.monotonic() - started
    samples.append(first)

    def collect():
        with (output / "samples.jsonl").open("x") as stream:
            stream.write(json.dumps(first) + "\n")
            stream.flush()
            while not stop.wait(interval):
                try:
                    item = sample(uuid)
                    item["elapsed_seconds"] = time.monotonic() - started
                    samples.append(item)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    item = {"elapsed_seconds": time.monotonic()-started, "error": str(exc)}
                    errors.append(item)
                stream.write(json.dumps(item) + "\n")
                stream.flush()

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    code, failure = None, None
    try:
        # Inherit logs; shell=False. No signal/kill/terminate calls here.
        code = subprocess.run(command, check=False).returncode
    except OSError as exc:
        failure = str(exc)
    finally:
        stop.set()
        thread.join(timeout=6)
        if thread.is_alive():
            errors.append({"error": "sampler did not finish within telemetry timeout"})
        report = {"gpu_uuid": uuid, "command": command, "returncode": code,
                  "launch_error": failure,
                  "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "wrapper_elapsed_seconds": time.monotonic()-started,
                  "summary": summarize(samples, errors, interval), "errors": errors,
                  "note": "wrapper elapsed includes preflight/hash/media checks; use workflow wall_ms for CLI latency"}
        (output / "memory-report.json").write_text(json.dumps(report, indent=2)+"\n")
    return code if code is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return monitor(command, args.gpu_uuid, args.output_dir, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
