#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch edge-dit parallel collective tests.")
    parser.add_argument("--binary", required=True, help="Path to parallel-collective-test")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "nccl"], help="Communication backend")
    parser.add_argument("--world-size", type=int, default=2, help="Number of local processes")
    parser.add_argument("--store-path", default=None, help="Shared file store path")
    parser.add_argument("--keep-store", action="store_true", help="Do not delete the store path after the test")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        print(f"test binary does not exist: {binary}", file=sys.stderr)
        return 1

    if args.world_size <= 0:
        print("--world-size must be positive", file=sys.stderr)
        return 1

    store_path = Path(args.store_path) if args.store_path else Path(tempfile.mkdtemp(prefix="ed-parallel-"))
    store_path.mkdir(parents=True, exist_ok=True)

    procs = []
    try:
        for rank in range(args.world_size):
            cmd = [
                str(binary),
                "--backend",
                args.backend,
                "--rank",
                str(rank),
                "--world-size",
                str(args.world_size),
                "--local-rank",
                str(rank),
                "--device",
                str(rank),
                "--store-path",
                str(store_path / ("nccl_id.bin" if args.backend == "nccl" else "cpu_store")),
            ]
            env = os.environ.copy()
            env["ED_RANK"] = str(rank)
            env["ED_WORLD_SIZE"] = str(args.world_size)
            env["ED_LOCAL_RANK"] = str(rank)
            procs.append((rank, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)))

        failed = False
        for rank, proc in procs:
            stdout, stderr = proc.communicate()
            if stdout:
                sys.stdout.write(stdout)
            if stderr:
                sys.stderr.write(stderr)
            if proc.returncode != 0:
                print(f"rank {rank} exited with {proc.returncode}", file=sys.stderr)
                failed = True

        return 1 if failed else 0
    finally:
        for _, proc in procs:
            if proc.poll() is None:
                proc.kill()
        if not args.keep_store and args.store_path is None:
            shutil.rmtree(store_path, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
