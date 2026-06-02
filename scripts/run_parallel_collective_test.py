#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run edge-dit ProcessGroup collective tests")

    parser.add_argument("--binary", required=True, help="Path to parallel-collective-test")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "nccl"])
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--store-path", default=None)
    parser.add_argument("--keep-store", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def run_cpu(args, binary: Path) -> int:
    if args.store_path:
        store_root = Path(args.store_path).resolve()
        cleanup_store = False
    else:
        store_root = Path(tempfile.mkdtemp(prefix="edge_dit_pg_"))
        cleanup_store = True

    store_root.mkdir(parents=True, exist_ok=True)

    # 注意：CPU 后端所有 rank 必须拿到同一个 store 文件/目录路径
    # 这里统一给一个路径，不要每个 rank 一个。
    store_path = store_root / "cpu_store"

    print(f"[launcher] backend     = cpu")
    print(f"[launcher] world_size  = {args.world_size}")
    print(f"[launcher] binary      = {binary}")
    print(f"[launcher] store_path  = {store_path}")

    procs = []

    try:
        for rank in range(args.world_size):
            cmd = [
                str(binary),
                "--backend", "cpu",
                "--rank", str(rank),
                "--world-size", str(args.world_size),
                "--local-rank", str(rank),
                "--device", str(rank),
                "--store-path", str(store_path),
            ]

            env = os.environ.copy()

            # 同时设置多套环境变量，避免 C++ 侧到底读哪套不一致
            env["RANK"] = str(rank)
            env["WORLD_SIZE"] = str(args.world_size)
            env["LOCAL_RANK"] = str(rank)

            env["ED_RANK"] = str(rank)
            env["ED_WORLD_SIZE"] = str(args.world_size)
            env["ED_LOCAL_RANK"] = str(rank)

            env["STORE_PATH"] = str(store_path)
            env["ED_STORE_PATH"] = str(store_path)

            print(f"[launcher] rank {rank} cmd:")
            print("  " + " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            procs.append((rank, proc))

        failed = False

        for rank, proc in procs:
            stdout, stderr = proc.communicate()

            if stdout:
                print(f"\n===== rank {rank} stdout =====")
                print(stdout, end="")

            if stderr:
                print(f"\n===== rank {rank} stderr =====", file=sys.stderr)
                print(stderr, end="", file=sys.stderr)

            if proc.returncode != 0:
                print(f"[launcher] rank {rank} failed with code {proc.returncode}", file=sys.stderr)
                failed = True
            else:
                print(f"[launcher] rank {rank} passed")

        return 1 if failed else 0

    finally:
        for _, proc in procs:
            if proc.poll() is None:
                proc.kill()

        if cleanup_store and not args.keep_store:
            shutil.rmtree(store_root, ignore_errors=True)
        else:
            print(f"[launcher] keep store path: {store_root}")


def run_nccl(args, binary: Path) -> int:
    mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
    if mpirun is None:
        print("[launcher] NCCL test requires mpirun or mpiexec", file=sys.stderr)
        return 1

    cmd = [mpirun]

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        # OpenMPI root 环境需要这个；不是 OpenMPI 时一般会忽略或报错。
        # 如果你的 mpirun 不支持，删掉这两行。
        cmd.extend(["--allow-run-as-root"])

    cmd.extend([
        "-np", str(args.world_size),
        str(binary),
        "--backend", "nccl",
    ])

    print(f"[launcher] backend     = nccl")
    print(f"[launcher] world_size  = {args.world_size}")
    print(f"[launcher] binary      = {binary}")
    print("[launcher] cmd:")
    print("  " + " ".join(cmd))

    env = os.environ.copy()
    return subprocess.call(cmd, env=env)


def main() -> int:
    args = parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        print(f"[launcher] test binary does not exist: {binary}", file=sys.stderr)
        return 1

    if args.world_size <= 0:
        print("[launcher] --world-size must be positive", file=sys.stderr)
        return 1

    if args.backend == "cpu":
        return run_cpu(args, binary)

    if args.backend == "nccl":
        return run_nccl(args, binary)

    print(f"[launcher] unsupported backend: {args.backend}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())