"""Environment metadata collection for benchmark runs."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import importlib.metadata
import json
import subprocess
import sys
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def first_line(text: str | None) -> str | None:
    if not text:
        return None
    return text.splitlines()[0] if text.splitlines() else None


def nvcc_version() -> str | None:
    output = command_output(["nvcc", "--version"])
    if output is None:
        return None
    for line in output.splitlines():
        if "release" in line:
            return line.strip()
    return first_line(output)


def env_path_version(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def git_metadata(repo: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    status = run(["status", "--short"])
    return {
        "path": str(repo),
        "commit": run(["rev-parse", "HEAD"]) or "unknown",
        "branch": run(["branch", "--show-current"]),
        "dirty": bool(status),
    }


def nvidia_smi_gpus() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, name, memory_total, driver = [part.strip() for part in line.split(",", 3)]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(memory_total),
                "driver_version": driver,
            }
        )
    return gpus


def collect_environment(repositories: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "gpus": nvidia_smi_gpus(),
            "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
        },
        "software": {
            "cuda": nvcc_version(),
            "cuda_home": env_path_version("CUDA_HOME"),
            "cudnn": env_path_version("CUDNN_ROOT"),
            "nccl": env_path_version("NCCL_ROOT"),
            "mpi": first_line(command_output(["mpirun", "--version"]))
            or env_path_version("MPI_HOME"),
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "diffusers": package_version("diffusers"),
            "transformers": package_version("transformers"),
        },
        "repositories": {
            name: git_metadata(path)
            for name, path in repositories.items()
            if path is not None
        },
    }


def main() -> None:
    print(json.dumps(collect_environment({}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
