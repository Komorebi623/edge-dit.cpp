#!/usr/bin/env python3
"""Cold-cache torch GEMM: cycle through N distinct weights per shape so each matmul
reads its weight cold from DRAM (mimics the real DiT flow: 57 blocks, each a
different weight, used once per step). Compares against hot-cache (reuse 1 weight)
to see if torch's high throughput was a cache artifact.
"""
import os, time, torch
torch.set_num_threads(int(os.environ.get("THREADS","0")) or torch.get_num_threads())

def bench(M,N,K,nweights,iters=57):
    a = torch.randn(M,K,dtype=torch.bfloat16)
    ws = [torch.randn(N,K,dtype=torch.bfloat16) for _ in range(nweights)]
    for i in range(5): torch.nn.functional.linear(a, ws[i%nweights])
    t=time.time()
    for i in range(iters): torch.nn.functional.linear(a, ws[i%nweights])
    dt=(time.time()-t)/iters
    return 2.0*M*N*K/dt/1e9

print(f"torch {torch.__version__}, threads={torch.get_num_threads()}")
print(f"{'shape':<20}{'hot(1w)':>10}{'cold(57w)':>11}")
for M,N,K,lbl in [(1024,3072,3072,'img proj'),(1024,12288,3072,'img mlp_in'),
                  (1280,21504,3072,'single qkv'),(256,3072,3072,'txt proj')]:
    hot = bench(M,N,K,1)
    cold = bench(M,N,K,57)
    print(f"{f'{M}x{N}x{K}':<20}{hot:>10.0f}{cold:>11.0f}")
