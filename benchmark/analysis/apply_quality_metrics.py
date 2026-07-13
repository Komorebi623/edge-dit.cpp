#!/usr/bin/env python3
"""Apply evaluation summaries to benchmark result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "psnr": "psnr",
    "ssim": "ssim",
    "lpips": "lpips",
    "clip": "clip",
    "ir": "image_reward",
    "image_reward": "image_reward",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-summary",
        type=Path,
        required=True,
        help="summary.json produced by benchmark/evaluation/text_to_image/eval.py.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    result_path = result_dir / "result.json"
    metrics_path = result_dir / "metrics.json"
    if not result_path.exists():
        raise FileNotFoundError(f"missing result.json: {result_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")

    result = read_json(result_path)
    metrics = read_json(metrics_path)
    eval_summary = read_json(args.eval_summary)
    quality = extract_quality(eval_summary)

    result.setdefault("quality", {})
    for key, value in quality.items():
        result["quality"][key] = value

    metrics["quality_metrics_available"] = any(value is not None for value in quality.values())
    metrics["quality"] = quality
    metrics["quality_metric_source"] = str(args.eval_summary.resolve())

    write_json(result_path, result)
    write_json(metrics_path, metrics)
    return 0


def extract_quality(eval_summary: dict[str, Any]) -> dict[str, float | None]:
    raw_summary = eval_summary.get("summary", {})
    quality = {
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "clip": None,
        "image_reward": None,
    }
    if not isinstance(raw_summary, dict):
        return quality
    for source_name, target_name in METRIC_ALIASES.items():
        metric = raw_summary.get(source_name)
        if not isinstance(metric, dict):
            continue
        value = metric.get("mean")
        if isinstance(value, (int, float)):
            quality[target_name] = float(value)
    return quality


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
