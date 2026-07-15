#!/usr/bin/env python3
"""Evaluate cache outputs against a full-compute reference."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUN_ID_RE = re.compile(
    r"gpu(?P<gpus>\d+)-single-(?P<method>.+)_(?P<prompt>p\d+)_(?P<seed>s\d+)$"
)


@dataclass(frozen=True)
class CacheRun:
    method: str
    prompt: str
    seed: str
    result_dir: Path
    image: Path

    @property
    def key(self) -> tuple[str, str]:
        return (self.prompt, self.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="cache-quality benchmark result root.",
    )
    parser.add_argument(
        "--reference-method",
        default="full_compute",
        help="Method id used as quality reference.",
    )
    parser.add_argument(
        "--metrics",
        default="psnr,lpips",
        help="Comma-separated metrics passed to eval.py.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device used by LPIPS.",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Output root for paired image dirs and metric summaries.",
    )
    parser.add_argument(
        "--skip-if-json-exists",
        action="store_true",
        help="Reuse existing metric JSON files when present.",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Only compute metric summaries; do not write them into result dirs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.is_dir():
        raise NotADirectoryError(results_root)

    eval_root = (
        args.eval_root.resolve()
        if args.eval_root is not None
        else results_root / "eval_cache_quality"
    )
    eval_root.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(results_root)
    grouped = group_runs(runs)
    if args.reference_method not in grouped:
        raise ValueError(f"missing reference method: {args.reference_method}")

    reference = grouped[args.reference_method]
    expected_keys = set(reference)
    if not expected_keys:
        raise ValueError(f"reference method has no runs: {args.reference_method}")

    for method, method_runs in sorted(grouped.items()):
        keys = set(method_runs)
        if keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise ValueError(
                f"method {method} does not match reference keys; "
                f"missing={missing}, extra={extra}"
            )

    repo_root = Path(__file__).resolve().parents[2]
    eval_script = repo_root / "benchmark" / "evaluation" / "text_to_image" / "eval.py"
    apply_script = repo_root / "benchmark" / "analysis" / "apply_quality_metrics.py"

    for method, method_runs in sorted(grouped.items()):
        method_eval_root = eval_root / method
        ref_dir = method_eval_root / "ref"
        target_dir = method_eval_root / "target"
        summary_dir = method_eval_root / "summary"
        rebuild_pair_dirs(ref_dir, target_dir, reference, method_runs)
        run_eval(
            eval_script=eval_script,
            run_dir=target_dir,
            ref_dir=ref_dir,
            target_dir=target_dir,
            output_dir=summary_dir,
            metrics=args.metrics,
            device=args.device,
            skip_if_json_exists=args.skip_if_json_exists,
        )
        if not args.no_apply:
            apply_summary(apply_script, summary_dir / "summary.json", method_runs)

    return 0


def scan_runs(results_root: Path) -> list[CacheRun]:
    runs: list[CacheRun] = []
    for result_path in sorted(results_root.rglob("result.json")):
        result_dir = result_path.parent
        result = read_json(result_path)
        if result.get("status") != "success":
            continue
        run_id = str(result.get("run_id") or result_dir.name)
        match = RUN_ID_RE.search(run_id)
        if match is None:
            continue
        runner_metrics_path = result_dir / "runner_metrics.json"
        if not runner_metrics_path.exists():
            raise FileNotFoundError(runner_metrics_path)
        runner_metrics = read_json(runner_metrics_path)
        sample_output_dir = runner_metrics.get("sample_output_dir")
        if not isinstance(sample_output_dir, str):
            raise ValueError(f"missing sample_output_dir: {runner_metrics_path}")
        image_dir = Path(sample_output_dir) / "imgs"
        image = image_dir / "img_000000.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        runs.append(
            CacheRun(
                method=match.group("method"),
                prompt=match.group("prompt"),
                seed=match.group("seed"),
                result_dir=result_dir,
                image=image,
            )
        )
    if not runs:
        raise ValueError(f"no successful cache runs found under {results_root}")
    return runs


def group_runs(runs: list[CacheRun]) -> dict[str, dict[tuple[str, str], CacheRun]]:
    grouped: dict[str, dict[tuple[str, str], CacheRun]] = {}
    for run in runs:
        group = grouped.setdefault(run.method, {})
        if run.key in group:
            raise ValueError(f"duplicate cache run for {run.method} {run.key}")
        group[run.key] = run
    return grouped


def rebuild_pair_dirs(
    ref_dir: Path,
    target_dir: Path,
    reference: dict[tuple[str, str], CacheRun],
    target: dict[tuple[str, str], CacheRun],
) -> None:
    for path in (ref_dir, target_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    for index, key in enumerate(sorted(reference)):
        link_image(ref_dir / f"img_{index:06d}.png", reference[key].image)
        link_image(target_dir / f"img_{index:06d}.png", target[key].image)


def link_image(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel_src)


def run_eval(
    *,
    eval_script: Path,
    run_dir: Path,
    ref_dir: Path,
    target_dir: Path,
    output_dir: Path,
    metrics: str,
    device: str,
    skip_if_json_exists: bool,
) -> None:
    cmd = [
        sys.executable,
        str(eval_script),
        "--run_dir",
        str(run_dir),
        "--ref_dir",
        str(ref_dir),
        "--target_dir",
        str(target_dir),
        "--output_dir",
        str(output_dir),
        "--metrics",
        metrics,
        "--strict_missing",
        "--strict_size",
        "--device",
        device,
    ]
    if skip_if_json_exists:
        cmd.append("--skip_if_json_exists")
    subprocess.run(cmd, check=True)


def apply_summary(
    apply_script: Path,
    eval_summary: Path,
    runs: dict[tuple[str, str], CacheRun],
) -> None:
    for run in sorted(runs.values(), key=lambda item: item.key):
        subprocess.run(
            [
                sys.executable,
                str(apply_script),
                "--result-dir",
                str(run.result_dir),
                "--eval-summary",
                str(eval_summary),
            ],
            check=True,
        )


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
