#!/usr/bin/env python3
"""Micro-benchmark the exact Flux-dev DiT matmul shapes through torch (diffusers'
GEMM path: oneDNN/MKL bf16), wall-clock per call. Lets us compare torch's GEMM
throughput vs edge-dit's oneDNN shim on identical shapes, same machine/load.

Shapes are (M, N, K): out[M,N] = act[M,K] @ W[K,N]. bf16, CPU.
Run in container: python3 scripts/torch_gemm_bench.py
"""
import os, time, torch

torch.set_num_threads(int(os.environ.get("THREADS", "0")) or torch.get_num_threads())

# (M, N, K, label) — the real DiT shapes from edge-dit's ED_MM_TIMING trace.
SHAPES = [
    (1280, 21504, 3072, "single qkv+mlp_in"),
    (1280, 3072, 15360, "single proj+mlp_out"),
    (1024, 12288, 3072, "double img mlp_in"),
    (1024, 3072, 12288, "double img mlp_out"),
    (1024, 9216, 3072,  "double img qkv"),
    (1024, 3072, 3072,  "double img proj"),
    (256, 12288, 3072,  "double txt mlp_in"),
    (256, 3072, 12288,  "double txt mlp_out"),
    (256, 9216, 3072,   "double txt qkv"),
    (256, 3072, 3072,   "double txt proj"),
]

def bench(M, N, K, iters=50):
    a = torch.randn(M, K, dtype=torch.bfloat16)
    w = torch.randn(N, K, dtype=torch.bfloat16)   # torch.nn.Linear weight is [N,K]
    b = torch.randn(N, dtype=torch.bfloat16)
    # warmup
    for _ in range(5):
        _ = torch.nn.functional.linear(a, w, b)
    t = time.time()
    for _ in range(iters):
        o = torch.nn.functional.linear(a, w, b)
    dt = (time.time() - t) / iters
    gflops = 2.0 * M * N * K / dt / 1e9
    return dt * 1e6, gflops   # us, GFLOP/s

print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
print(f"{'shape (MxNxK)':<22}{'us/call':>10}{'GFLOP/s':>10}  label")
for M, N, K, lbl in SHAPES:
    us, gf = bench(M, N, K)
    print(f"{f'{M}x{N}x{K}':<22}{us:>10.0f}{gf:>10.0f}  {lbl}")
