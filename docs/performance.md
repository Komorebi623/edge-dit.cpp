# Performance and Benchmarks

[Back to README](../README.md)

This page records the current README main-table benchmark snapshot. The table
is generated from local benchmark results produced on 2026-07-13 on a local
NVIDIA H200 node with the CUDA `performance` profile.

The root README intentionally shows only the main table. Smoke runs, cache
diagnostics, memory probes, ablations, and parallel debug artifacts remain
local benchmark evidence, but they are not public performance claims.

## Current Snapshot

Raw result root:
`benchmark/results/readme-main-table-20260713-clean`.

Suite config:
`benchmark/configs/suites/readme-main-table.yaml`.

All rows use the same text-to-image setting: 1024x1024, 50 denoising steps,
BF16, batch 1, seed 0, 2 warm-up runs, and 10 measured runs. `Load` is model
load time. `Median` and `P90` are load-once steady-state generation latency and
exclude output encoding.

| Model | System | Load (s) | Median (s) | P90 (s) | Peak VRAM (MiB) |
|---|---|---:|---:|---:|---:|
| FLUX.1-dev | edge-dit.cpp | 6.645 | 10.784 | 10.861 | 38341 |
| | Diffusers | 14.531 | 10.040 | 10.048 | 37711 |
| | stable-diffusion.cpp | 1.333 | 30.371 | 30.379 | 40331 |
| Stable Diffusion 3 Medium | edge-dit.cpp | 5.840 | 4.003 | 4.049 | 20833 |
| | Diffusers | 11.244 | 3.376 | 3.381 | 20283 |
| | stable-diffusion.cpp | 1.457 | 10.740 | 10.797 | 22997 |
| Qwen-Image | edge-dit.cpp | 11.621 | 10.697 | 10.736 | 59725 |
| | Diffusers | 25.220 | 9.558 | 9.565 | 60935 |
| | stable-diffusion.cpp | 1.782 | 62.671 | 62.728 | 61879 |

## Methodology

| Item | Value |
|---|---|
| Date | 2026-07-13 |
| Hardware | Local NVIDIA H200 node |
| GPU memory | 143771 MiB per GPU |
| Build profile | CUDA `performance` profile |
| Precision | BF16 |
| Batch size | 1 |
| Seed | 0 |
| Warm-up runs | 2 |
| Measured runs | 10 |
| Measurement boundary | `load_once_e2e_generation_no_output_encoding` |

The load-once boundary means each benchmark process loads the model once, runs
warm-up generations in the same process, then measures repeated generation
calls after the model is resident. Output encoding and file writes are outside
the core latency when the backend exposes that split.

Peak VRAM is measured externally. Environment metadata, commands, logs,
per-run `result.json` files, and runner metrics are stored under the raw result
root.

## Workloads

The README main table includes only workloads that have matching edge-dit.cpp,
Diffusers, and stable-diffusion.cpp results under the same 50-step text-to-image
contract.

| Workload | Model | Notes |
|---|---|---|
| `flux1-dev-t2i-1024-s50` | FLUX.1-dev | FLUX.1 text-to-image main-table workload |
| `sd3-medium-t2i-1024-s50` | Stable Diffusion 3 Medium | Local H200 checkpoint is SD3 Medium, not SD3.5 Large |
| `qwen-image-t2i-1024-s50` | Qwen-Image | Qwen-Image text-to-image main-table workload |

The local model tree currently does not contain a real SD3.5 Large checkpoint.
Do not report SD3 Medium results as SD3.5. If SD3.5 Large is added later, add a
real model path and rerun the same suite before updating public tables.

## Systems

| System | Main-table status |
|---|---|
| edge-dit.cpp | Native CUDA `performance` build through `ed-sample` |
| Diffusers | Python reference baseline through the configured Diffusers environment |
| stable-diffusion.cpp | Native baseline through the benchmark C API load-once wrapper |
| xDiT | Parallel baseline only; not part of the single-GPU README main table |

Diffusers metadata is collected from the configured Python executable in
`benchmark/configs/local/site-h200.yaml`. This is important because the shell
Python may differ from the benchmark Python environment.

## Excluded Results

The repository also contains local benchmark suites for smoke coverage, memory
targets, cache behavior, ablations, and parallel launch validation. Those suites
are useful engineering diagnostics, but they are intentionally excluded from the
README main table because they do not share the same workload contract or still
need quality/scaling validation.

| Suite | Purpose | Public-table treatment |
|---|---|---|
| `model-smoke` | One-run load/execute coverage for public model families | Smoke only |
| `memory` | Quantization, placement, tiling, and graph-cut probes | Diagnostic only |
| `ablation` | Operator, quantization, and cache ablation probes | Diagnostic only |
| `cache-matrix` | Repeated FLUX cache-mode latency probe | No cache acceleration claim without quality metrics |
| `parallel-smoke` / `cfg-smoke` | Launcher and schema validation | Not a scaling result |

## Reproduction

Validate benchmark configs:

```bash
python3 benchmark/orchestration/validate_config.py
```

Run the README main-table suite on an idle GPU:

```bash
BENCHMARK_CUDA_VISIBLE_DEVICES=3 \
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/readme-main-table.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/readme-main-table-YYYYMMDD
```

Aggregate a completed result root:

```bash
python3 benchmark/analysis/aggregate.py \
  --results-dir benchmark/results/readme-main-table-YYYYMMDD \
  --suite-id readme-main-table \
  --output benchmark/results/readme-main-table-YYYYMMDD/summary.json
```

Generate report artifacts if needed:

```bash
python3 benchmark/analysis/generate_tables.py \
  benchmark/results/readme-main-table-YYYYMMDD/summary.json \
  --output benchmark/results/readme-main-table-YYYYMMDD/tables.md

python3 benchmark/analysis/generate_report.py \
  benchmark/results/readme-main-table-YYYYMMDD/summary.json \
  --tables benchmark/results/readme-main-table-YYYYMMDD/tables.md \
  --output benchmark/results/readme-main-table-YYYYMMDD/report.md
```

## Next Steps

1. Add a real SD3.5 Large local path before publishing SD3.5 rows.
2. Add quality metrics for cache and quantization suites before making any
   acceleration or memory-quality tradeoff claims.
3. Promote parallel smoke into repeated official runs only after adding xDiT
   comparison, communication timing, and per-GPU memory reporting.
4. Keep README numbers sourced from a single frozen result root and the matching
   suite config.

## Related Documentation

- [Benchmark harness](../benchmark/README.md)
- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [Command line usage](cli.md)
- [Memory-efficient execution](optimization/memory-efficient-execution.md)
- [Graph and operator optimization](optimization/graph-and-operator-optimization.md)
- [Computation reuse](optimization/computation-reuse.md)
- [Parallel execution](optimization/parallel-execution.md)
