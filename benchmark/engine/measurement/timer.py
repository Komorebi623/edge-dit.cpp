"""Timing helpers for benchmark runners."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import statistics
import time
from typing import Iterator


@dataclass
class TimedBlock:
    name: str
    elapsed_ms: float | None = None


@contextmanager
def timed_block(name: str) -> Iterator[TimedBlock]:
    block = TimedBlock(name=name)
    start = time.perf_counter()
    try:
        yield block
    finally:
        block.elapsed_ms = (time.perf_counter() - start) * 1000.0


def summarize_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median": None,
            "p90": None,
            "mean": None,
            "std": None,
            "coefficient_of_variation": None,
        }
    sorted_values = sorted(values)
    p90_index = min(len(sorted_values) - 1, int(0.9 * (len(sorted_values) - 1)))
    mean = statistics.mean(sorted_values)
    std = statistics.pstdev(sorted_values) if len(sorted_values) > 1 else 0.0
    return {
        "median": statistics.median(sorted_values),
        "p90": sorted_values[p90_index],
        "mean": mean,
        "std": std,
        "coefficient_of_variation": (std / mean) if mean else None,
    }
