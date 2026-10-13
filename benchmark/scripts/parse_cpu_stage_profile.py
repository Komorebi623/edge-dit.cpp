#!/usr/bin/env python3
"""Parse ggml CPU graph profiles; tracked so clean H3 checkouts work offline.

Accepts both stage-labelled historical logs and the pinned ggml's unlabelled
format. Times are milliseconds. This parser does not execute models or logs.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

HEADER = re.compile(r"=== edge-dit CPU op profile(?: \[stage=([^\]]+)\])? \(per graph_compute\) ===")

def parse(path: str) -> list[dict]:
    profiles: list[dict] = []
    current: dict | None = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        header = HEADER.search(line)
        if header:
            current = {"stage": header[1] or "unlabelled", "total": None, "ops": []}
            profiles.append(current)
            continue
        if current is None:
            continue
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        try:
            total_ms = float(fields[1])
        except ValueError:
            continue
        if not math.isfinite(total_ms) or total_ms < 0:
            continue
        if fields[0] == "TOTAL":
            current["total"] = total_ms
            current = None  # ignore custom sub-breakdowns and unrelated logs
        elif len(fields) == 7 and fields[2].isdigit():
            current["ops"].append({"op": fields[0], "total_ms": total_ms, "calls": int(fields[2])})
    return profiles
