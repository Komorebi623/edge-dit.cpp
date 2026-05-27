#!/usr/bin/env python3
"""
Run a Flux cache benchmark matrix and summarize speed / cache hits / image drift.

Example:
  python3 scripts/benchmark_flux_cache.py \
    --model /mnt/cfs/9n-das-admin/llm_models/flux-dev/ \
    --backend cuda \
    --width 1024 --height 1024 --steps 50 --seed 0 \
    --out-dir cache_test_flux

The script writes per-mode PNG/log files plus:
  - summary.csv
  - summary.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_PROMPT = "a cinematic photo of a glass teapot on a wooden table, soft morning light"


@dataclass
class CacheCase:
    name: str
    cache_mode: str | None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    name: str
    cache_mode: str
    command: list[str]
    image_path: Path
    log_path: Path
    returncode: int
    elapsed_s: float
    real_s: float | None
    skipped: int | None
    total_steps: int | None
    estimated_speedup: float | None
    image_md5: str | None
    width: int | None = None
    height: int | None = None
    mae: float | None = None
    rmse: float | None = None
    psnr: float | None = None
    max_abs: int | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_case(raw: str) -> CacheCase:
    parts = shlex.split(raw)
    if not parts:
        raise ValueError("--case cannot be empty")
    name = parts[0]
    if name in {"baseline", "none", "off", "disabled"}:
        return CacheCase(name="baseline", cache_mode=None, extra_args=parts[1:])
    return CacheCase(name=name, cache_mode=name, extra_args=parts[1:])


def default_cases() -> list[CacheCase]:
    return [
        CacheCase("baseline", None),
        CacheCase("easycache", "easycache"),
        CacheCase("ucache", "ucache"),
        CacheCase("dbcache", "dbcache"),
        CacheCase("taylorseer", "taylorseer"),
        CacheCase("cache-dit", "cache-dit"),
    ]


def shell_join(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(a)) for a in args)


def md5_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_log(log_path: Path) -> tuple[float | None, int | None, int | None, float | None]:
    if not log_path.is_file():
        return None, None, None, None
    text = log_path.read_text(errors="replace")

    real_s = None
    m = re.search(r"^real\s+([0-9]+(?:\.[0-9]+)?)$", text, re.MULTILINE)
    if m:
        real_s = float(m.group(1))

    skipped = None
    total = None
    speedup = None
    m = re.search(
        r"skipped\s+([0-9]+)/([0-9]+)\s+steps(?:\s+\(([0-9]+(?:\.[0-9]+)?)x estimated speedup\))?",
        text,
    )
    if m:
        skipped = int(m.group(1))
        total = int(m.group(2))
        speedup = float(m.group(3)) if m.group(3) is not None else None
    return real_s, skipped, total, speedup


def image_metrics(image_paths: list[Path], baseline_path: Path) -> dict[Path, dict[str, float | int]]:
    try:
        from PIL import Image
    except Exception:
        return {}

    if not baseline_path.is_file():
        return {}

    base = Image.open(baseline_path).convert("RGB")
    base_data = list(base.getdata())
    out: dict[Path, dict[str, float | int]] = {
        baseline_path: {
            "width": base.size[0],
            "height": base.size[1],
            "mae": 0.0,
            "rmse": 0.0,
            "psnr": float("inf"),
            "max_abs": 0,
        }
    }

    for path in image_paths:
        if not path.is_file() or path == baseline_path:
            continue
        img = Image.open(path).convert("RGB")
        if img.size != base.size:
            out[path] = {
                "width": img.size[0],
                "height": img.size[1],
                "mae": float("nan"),
                "rmse": float("nan"),
                "psnr": float("nan"),
                "max_abs": -1,
            }
            continue

        n = base.size[0] * base.size[1] * 3
        sad = 0
        sse = 0
        max_abs = 0
        for a, b in zip(base_data, img.getdata()):
            for x, y in zip(a, b):
                d = abs(x - y)
                sad += d
                sse += d * d
                if d > max_abs:
                    max_abs = d
        mae = sad / n
        rmse = math.sqrt(sse / n)
        psnr = float("inf") if rmse == 0.0 else 20.0 * math.log10(255.0 / rmse)
        out[path] = {
            "width": img.size[0],
            "height": img.size[1],
            "mae": mae,
            "rmse": rmse,
            "psnr": psnr,
            "max_abs": max_abs,
        }
    return out


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_summary(results: list[RunResult], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"

    fields = [
        "mode",
        "returncode",
        "real_s",
        "elapsed_s",
        "skipped",
        "total_steps",
        "estimated_speedup",
        "speedup_vs_baseline",
        "width",
        "height",
        "mae",
        "rmse",
        "psnr",
        "max_abs",
        "md5",
        "image",
        "log",
    ]

    baseline_real = next((r.real_s or r.elapsed_s for r in results if r.name == "baseline"), None)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            actual_time = r.real_s or r.elapsed_s
            writer.writerow(
                {
                    "mode": r.name,
                    "returncode": r.returncode,
                    "real_s": format_float(r.real_s, 2),
                    "elapsed_s": format_float(r.elapsed_s, 2),
                    "skipped": "" if r.skipped is None else r.skipped,
                    "total_steps": "" if r.total_steps is None else r.total_steps,
                    "estimated_speedup": format_float(r.estimated_speedup, 2),
                    "speedup_vs_baseline": format_float(
                        baseline_real / actual_time if baseline_real and actual_time else None,
                        2,
                    ),
                    "width": "" if r.width is None else r.width,
                    "height": "" if r.height is None else r.height,
                    "mae": format_float(r.mae, 4),
                    "rmse": format_float(r.rmse, 4),
                    "psnr": format_float(r.psnr, 2),
                    "max_abs": "" if r.max_abs is None else r.max_abs,
                    "md5": r.image_md5 or "",
                    "image": str(r.image_path),
                    "log": str(r.log_path),
                }
            )

    with md_path.open("w") as f:
        f.write("# Flux Cache Benchmark\n\n")
        f.write("| mode | real_s | skipped | speedup_vs_baseline | PSNR | MAE | RMSE | max_abs | image | log |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in results:
            actual_time = r.real_s or r.elapsed_s
            speedup_vs = baseline_real / actual_time if baseline_real and actual_time else None
            skipped = "" if r.skipped is None else f"{r.skipped}/{r.total_steps or ''}"
            f.write(
                "| "
                + " | ".join(
                    [
                        r.name,
                        format_float(r.real_s, 2),
                        skipped,
                        format_float(speedup_vs, 2),
                        format_float(r.psnr, 2),
                        format_float(r.mae, 4),
                        format_float(r.rmse, 4),
                        "" if r.max_abs is None else str(r.max_abs),
                        f"[png]({r.image_path.name})" if r.image_path.is_file() else "",
                        f"[log]({r.log_path.name})" if r.log_path.is_file() else "",
                    ]
                )
                + " |\n"
            )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


def run_case(args: argparse.Namespace, case: CacheCase, env: dict[str, str]) -> RunResult:
    image_path = args.out_dir / f"{args.prefix}_{case.name}.png"
    log_path = args.out_dir / f"{args.prefix}_{case.name}.log"

    cmd = [
        str(args.cli),
        "--backend",
        args.backend,
        "--model",
        args.model,
        "-p",
        args.prompt,
        "-W",
        str(args.width),
        "-H",
        str(args.height),
        "--steps",
        str(args.steps),
        "-s",
        str(args.seed),
        "--guidance",
        str(args.guidance),
        "-o",
        str(image_path),
    ]
    if args.threads is not None:
        cmd.extend(["-t", str(args.threads)])
    if args.flow_shift is not None:
        cmd.extend(["--flow-shift", str(args.flow_shift)])
    if args.cfg_scale is not None:
        cmd.extend(["--cfg-scale", str(args.cfg_scale)])
    if case.cache_mode is not None:
        cmd.extend(["--cache", case.cache_mode])
    cmd.extend(case.extra_args)
    cmd.extend(args.extra_cli_args)

    print("\n==", case.name, "==")
    print(shell_join(cmd))

    if args.dry_run:
        return RunResult(
            name=case.name,
            cache_mode=case.cache_mode or "off",
            command=cmd,
            image_path=image_path,
            log_path=log_path,
            returncode=0,
            elapsed_s=0.0,
            real_s=None,
            skipped=None,
            total_steps=None,
            estimated_speedup=None,
            image_md5=None,
        )

    start = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=args.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        returncode = proc.wait()

    elapsed_s = time.monotonic() - start
    real_s, skipped, total_steps, estimated_speedup = parse_log(log_path)
    return RunResult(
        name=case.name,
        cache_mode=case.cache_mode or "off",
        command=cmd,
        image_path=image_path,
        log_path=log_path,
        returncode=returncode,
        elapsed_s=elapsed_s,
        real_s=real_s,
        skipped=skipped,
        total_steps=total_steps,
        estimated_speedup=estimated_speedup,
        image_md5=md5_file(image_path),
    )


def main() -> int:
    root = repo_root()
    default_cli_cache = root / "build-cuda" / "bin" / "ld-cli-cache"
    default_cli = default_cli_cache if default_cli_cache.is_file() else root / "build-cuda" / "bin" / "ld-cli"

    parser = argparse.ArgumentParser(description="Benchmark Flux cache modes with ld-cli.")
    parser.add_argument("--cli", type=Path, default=default_cli, help="Path to ld-cli or ld-cli-cache")
    parser.add_argument("--model", required=True, help="Flux model or diffusers directory")
    parser.add_argument("--backend", default="cuda", help="Backend passed to ld-cli")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--flow-shift", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=root / "cache_test_flux")
    parser.add_argument("--prefix", default="flux")
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "Case spec. Examples: 'baseline', 'easycache --cache-threshold 0.2', "
            "'dbcache --cache-warmup-steps 4 --cache-residual-threshold 0.05'. "
            "If omitted, runs baseline/easycache/ucache/dbcache/taylorseer/cache-dit."
        ),
    )
    parser.add_argument(
        "--extra-cli-arg",
        action="append",
        default=[],
        help="Extra raw argument appended to every ld-cli call. Repeat for multiple args.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--cwd", type=Path, default=root, help="Working directory for subprocesses")
    args = parser.parse_args()

    args.cli = args.cli.resolve()
    args.out_dir = args.out_dir.resolve()
    args.cwd = args.cwd.resolve()
    args.extra_cli_args = args.extra_cli_arg

    if not args.cli.is_file():
        if args.dry_run:
            print(f"warning: CLI not found for dry-run: {args.cli}", file=sys.stderr)
        else:
            print(f"error: CLI not found: {args.cli}", file=sys.stderr)
            print("hint: build ld-cli first or pass --cli /path/to/ld-cli-cache", file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = [parse_case(c) for c in args.case] if args.case else default_cases()
    env = os.environ.copy()

    results: list[RunResult] = []
    for case in cases:
        result = run_case(args, case, env)
        results.append(result)
        if result.returncode != 0 and not args.continue_on_error:
            print(f"case {case.name} failed with return code {result.returncode}", file=sys.stderr)
            break

    baseline = next((r.image_path for r in results if r.name == "baseline"), None)
    if baseline is not None:
        metrics = image_metrics([r.image_path for r in results], baseline)
        for r in results:
            m = metrics.get(r.image_path)
            if not m:
                continue
            r.width = int(m["width"])
            r.height = int(m["height"])
            r.mae = float(m["mae"])
            r.rmse = float(m["rmse"])
            r.psnr = float(m["psnr"])
            r.max_abs = int(m["max_abs"])

    write_summary(results, args.out_dir)
    print("\nSummary:")
    md = args.out_dir / "summary.md"
    if md.is_file():
        print(md.read_text())

    return 0 if all(r.returncode == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
