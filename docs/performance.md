# Performance and Benchmarks

[Back to README](../README.md)

This page is the public benchmark result page for the current H200 benchmark
snapshot. It is meant to answer the feature questions behind the README claims:
single-GPU runtime, task coverage, parallel execution, computation reuse,
low-memory execution, quantization, VAE tiling, and CUDA operator optimization.

The root README intentionally keeps only one main performance table. This page
keeps the supporting feature results and the reproducibility contract.

## Feature Results Summary

All rows below come from frozen local result roots under `benchmark/results/`.
Latency is reported in seconds. Lower latency is better. For "Relative to
Diffusers", values above 1.00x mean edge-dit.cpp is slower than Diffusers for
that workload; "Speedup" values above 1.00x mean the optimized configuration is
faster than its baseline.

| Capability | Representative workload | Compared configurations | Headline result |
|---|---|---|---|
| Native single-GPU runtime | FLUX.1-dev, SD3 Medium, Qwen-Image; 1024x1024, 50 steps | edge-dit.cpp vs Diffusers vs stable-diffusion.cpp | edge-dit.cpp is within 1.07x-1.19x of Diffusers and 2.68x-5.86x faster than stable-diffusion.cpp. |
| Model and task coverage | Text-to-image, image editing, text-to-video; 20 steps | edge-dit.cpp across six public-preview workloads | edge-dit.cpp completes all six task workloads, including FLUX-Kontext editing, Qwen-Image-Edit, and Wan text-to-video. |
| CFG parallelism | SD3 Medium, 1024x1024, 50 steps, CFG 4.5 | 1 GPU vs CFG-2 | CFG-2 reduces latency from 9.748 s to 6.341 s, a 1.54x speedup at 76.9% efficiency. |
| Sequence parallelism | FLUX 1024x1024, FLUX 2048x2048, Wan 832x480x81 | SP-1 vs SP-2 vs SP-4 | SP-4 reaches 2.33x on FLUX 1024, 3.27x on FLUX 2048, and 1.72x on Wan video. |
| Computation reuse | FLUX.1-dev, 1024x1024, 50 steps, 8 prompts x 3 seeds | Full compute vs EasyCache, CacheDiT, MagCache, DiCache, SenCache | MagCache is the fastest cache point at 2.69x; EasyCache reaches 2.09x; tuned SenCache now reaches 1.92x with 29-30/50 reused steps. |
| Weight quantization | FLUX.1-dev, 1024x1024 | BF16 vs Q8_0/Q6_K/Q4_K | Q4_K reduces peak VRAM from 40509 MiB to 17803 MiB, with 1.59x latency slowdown. |
| Low-memory execution | FLUX.1-dev, 1024x1024 | BF16 performance profile vs Q4_K + CPU placement/offload + graph budget + VAE tiling | The 8 GB experimental profile runs at 7591 MiB peak VRAM, with 6.01x latency slowdown. |
| VAE tiling | FLUX.1-dev, 2048x2048, 20 steps | Untiled vs 2x2 vs 4x4 tiling | 4x4 VAE tiling reduces peak VRAM by 40.1%, from 62029 MiB to 37185 MiB, with 1.01x latency slowdown. |
| CUDA operator optimization | FLUX.1-dev and Qwen-Image, 1024x1024, 50 steps | Generic ggml CUDA path vs optimized non-cuDNN CUDA builds | FLUX latency drops from 22.896 s to 18.835 s, a 1.22x speedup; Qwen gains 1.05x from fused modulation over the CUDA Norm + RoPE build. |

## Overall Performance

Reproduce the README main table with:

```bash
bash benchmark/scripts/run_readme_main_table.sh
```

The current frozen table expands to three real local text-to-image model
workloads. Add a fourth real 1024x1024, 50-step workload to the README
main-table suite before publishing a four-model table; the script checks the
expanded workload count before it runs.

Contract: local NVIDIA H200 node, CUDA `performance` profile, 1024x1024,
50 denoising steps, BF16, batch 1, seed 0, 2 warm-up runs, 10 measured runs,
load-once generation. Output encoding is outside `Median` and `P90`.

| Model | edge-dit.cpp Median | Relative to Diffusers | Speedup over sd.cpp | Peak VRAM |
|---|---:|---:|---:|---:|
| FLUX.1-dev | 10.784 s | 1.07x | 2.82x | 38341 MiB |
| Qwen-Image | 10.697 s | 1.12x | 5.86x | 59725 MiB |
| Stable Diffusion 3 Medium | 4.003 s | 1.19x | 2.68x | 20833 MiB |

The local model tree contains Stable Diffusion 3 Medium, not SD3.5 Large. These
results must not be reported as SD3.5. Adding SD3.5 requires adding a real local
checkpoint path and rerunning the same suite.

## Model and Task Coverage

Raw result root: `benchmark/results/perf-task-coverage-20260714`.

Suite config: `benchmark/configs/suites/task-coverage.yaml`.

Contract: local NVIDIA H200 node, BF16, batch 1, seed 0, 1 warm-up run,
5 measured runs. Text-to-image rows use load-once generation. Image editing and
video rows use the CLI single-run boundary, which includes load and output
encoding.

| Model | Task | Setting | Steps | edge-dit.cpp | Diffusers | Relative | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|
| FLUX.1-dev | text-to-image | 1024x1024 | 20 | 7.795 s | 4.070 s | 1.92x | 40509 MiB |
| Qwen-Image | text-to-image | 1024x1024 | 20 | 7.959 s | 3.852 s | 2.07x | 89297 MiB |
| Stable Diffusion 3 Medium | text-to-image | 1024x1024 | 20 | 4.275 s | 1.447 s | 2.95x | 23169 MiB |
| FLUX.1-Kontext-dev | image-editing | 1024x1024 | 20 | 29.544 s | - | - | 40577 MiB |
| Qwen-Image-Edit | image-editing | 1024x1024 | 20 | 39.204 s | - | - | 96449 MiB |
| Wan 2.x | text-to-video | 832x480x81 | 20 | 184.926 s | - | - | 76623 MiB |

Diffusers is used as the reference backend for text-to-image in this suite.
The current benchmark harness does not include matching load-once Diffusers
adapters for the image-editing and video rows, so those rows are presented as
edge-dit.cpp coverage results only.

## Parallel Execution

Parallel tables report latency and scaling only. Quality metrics are reserved
for the cache speed-quality suite.

### CFG Parallelism

Raw result root: `benchmark/results/perf-cfg-parallel-20260714`.

Suite config: `benchmark/configs/suites/cfg-parallel.yaml`.

Contract: Stable Diffusion 3 Medium, 1024x1024, 50 steps, CFG scale 4.5, BF16,
batch 1, seed 0, load-once generation.

| Mode | GPUs | Median | P90 | Speedup | Efficiency | Max VRAM / GPU |
|---|---:|---:|---:|---:|---:|---:|
| Single GPU | 1 | 9.748 s | 9.752 s | 1.00x | 100.0% | 23043 MiB |
| CFG parallel | 2 | 6.341 s | 6.380 s | 1.54x | 76.9% | 24323 MiB |

### Sequence Parallelism

Raw result root: `benchmark/results/perf-sp-parallel-20260714`.

Suite config: `benchmark/configs/suites/sp-parallel.yaml`.

Contract: local H200 node, BF16, batch 1. FLUX 1024 uses 50 steps. FLUX 2048
and Wan use 20 steps. The Wan row uses a CLI single-run boundary.

| Workload | System | Mode | GPUs | Median | Speedup | Efficiency | Max VRAM / GPU | Comm. | Segments / Step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| flux1-dev-t2i-1024-s50 | edge-dit.cpp | sequence | 1 | 18.719 s | 1.00x | 100.0% | 40387 MiB | - | - |
| flux1-dev-t2i-1024-s50 | edge-dit.cpp | sequence | 2 | 10.520 s | 1.78x | 89.0% | 41665 MiB | - | - |
| flux1-dev-t2i-1024-s50 | edge-dit.cpp | sequence | 4 | 8.034 s | 2.33x | 58.2% | 42485 MiB | - | - |
| flux1-dev-t2i-2048-s20 | edge-dit.cpp | sequence | 1 | 67.127 s | 1.00x | 100.0% | 61907 MiB | - | - |
| flux1-dev-t2i-2048-s20 | edge-dit.cpp | sequence | 2 | 35.772 s | 1.88x | 93.8% | 63185 MiB | - | - |
| flux1-dev-t2i-2048-s20 | edge-dit.cpp | sequence | 4 | 20.510 s | 3.27x | 81.8% | 64149 MiB | - | - |
| wan2-t2v-832x480-f81 | edge-dit.cpp | sequence | 1 | 106.410 s | 1.00x | 100.0% | 35973 MiB | - | - |
| wan2-t2v-832x480-f81 | edge-dit.cpp | sequence | 2 | 88.407 s | 1.20x | 60.2% | 36931 MiB | - | - |
| wan2-t2v-832x480-f81 | edge-dit.cpp | sequence | 4 | 61.879 s | 1.72x | 43.0% | 37665 MiB | - | - |

The current public SP result is edge-dit.cpp only. xDiT is not included because
the configured external checkout lacks the DistVAE dependency required for the
matching workloads, and the benchmark harness is configured not to mutate that
external repository.

## Computation Reuse

Reproduce this table with the cache-reuse table runner:

```bash
bash benchmark/scripts/run_cache_reuse_table.sh
```

The script runs the full-compute/cache matrix, the retuned MagCache and DiCache
rows, the 50-step SenCache calibration, the tuned SenCache row, and PSNR/LPIPS
evaluation against matched full-compute prompt/seed outputs.

Contract: FLUX.1-dev, 1024x1024, 50 steps, BF16, batch 1, 8 prompts x 3 seeds,
1 warm-up run, 5 measured runs, load-once generation with the build used by the
README main table. PSNR and LPIPS compare each method against the matching
`Full compute` prompt/seed output. CLIP is left out of the public table for
this snapshot.

Speedup below is computed against the matched full-compute subset for each row.
The MagCache and DiCache rows use their method-specific default thresholds. The
`0.08` residual threshold remains the DBCache/CacheDiT default and is only
applied to MagCache or DiCache when explicitly passed. The SenCache row uses a
50-step SenCache profile generated under this workload and
`cache_residual_threshold=0.60`.

| Method | Samples | Granularity | Median | Speedup vs Matched Full | Peak VRAM | Saved Steps | PSNR | LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Full compute | 24/24 | full | 10.765 s | 1.00x | 38341 MiB | - | 100.00 | 0.0000 |
| EasyCache | 24/24 | output | 5.154 s | 2.09x | 38341 MiB | 27/50 | 26.34 | 0.1016 |
| CacheDiT | 24/24 | block/output | 6.406 s | 1.68x | 38341 MiB | 21/50 | 29.03 | 0.0720 |
| MagCache method default | 24/24 | feature | 4.001 s | 2.69x | 38485 MiB | 35/50 | 23.30 | 0.1754 |
| DiCache method default | 24/24 | probe | 6.514 s | 1.65x | 39471 MiB | 30/50 | 26.89 | 0.0995 |
| SenCache tuned t=0.60 | 24/24 | feature | 5.613 s | 1.92x | 40923 MiB | 29/50 | 25.78 | 0.1231 |

## Resource-Constrained Execution

### Deployment Profiles

Raw result root: `benchmark/results/perf-resource-profiles-20260714`.

Suite config: `benchmark/configs/suites/resource-profiles.yaml`.

Contract: FLUX.1-dev, 1024x1024, BF16/Q8_0/Q4_K depending on profile, batch 1,
seed 0, load-once generation.

| Profile | Weight | CPU Placement / Offload | Graph Budget | VAE Tiling | Median | Slowdown | Peak VRAM | Host RAM |
|---|---|---|---:|---|---:|---:|---:|---:|
| Performance | BF16 | none | unlimited | off | 7.739 s | 1.00x | 40509 MiB | 15362 MiB |
| Memory-balanced | Q8_0 | none | unlimited | off | 11.795 s | 1.52x | 25331 MiB | 43784 MiB |
| 24 GB | Q4_K | text encoder CPU | unlimited | off | 18.685 s | 2.41x | 14623 MiB | 55078 MiB |
| 16 GB | Q4_K | text encoder CPU | unlimited | on | 18.162 s | 2.35x | 9785 MiB | 55904 MiB |
| 12 GB | Q4_K | text encoder CPU + parameter offload | 12 GiB | on | 32.082 s | 4.15x | 10513 MiB | 54734 MiB |
| 8 GB experimental | Q4_K | text encoder CPU + parameter offload | 8 GiB | on | 46.475 s | 6.01x | 7591 MiB | 53684 MiB |

The lowest-memory profile in this run reduces peak device memory from
40509 MiB to 7591 MiB. The cost is latency: 46.475 s vs 7.739 s for the BF16
performance profile.

### Quantization Trade-Off

Raw result root: `benchmark/results/perf-quantization-20260714`.

Suite config: `benchmark/configs/suites/quantization.yaml`.

Contract: FLUX.1-dev, 1024x1024, batch 1, seed 0, load-once generation.

| Weight Type | Load | Median | Slowdown | Peak VRAM | Host RAM | Policy |
|---|---:|---:|---:|---:|---:|---|
| BF16 | 6.371 s | 7.780 s | 1.00x | 40509 MiB | 17036 MiB | - |
| Q8_0 | 15.145 s | 11.715 s | 1.51x | 25341 MiB | 39382 MiB | - |
| Q6_K | 13.338 s | 13.913 s | 1.79x | 21871 MiB | 56722 MiB | - |
| Q4_K | 42.121 s | 12.370 s | 1.59x | 17803 MiB | 55895 MiB | - |
| Q4_K + precision rules | 40.733 s | 12.483 s | 1.60x | 17803 MiB | 55467 MiB | norm=f16,bias=f32 |

### VAE Tiling

Raw result root: `benchmark/results/perf-vae-tiling-20260714`.

Suite config: `benchmark/configs/suites/vae-tiling.yaml`.

Contract: FLUX.1-dev, 2048x2048, 20 steps, BF16, batch 1, seed 0,
load-once generation.

| VAE Mode | Tile Layout | Median | Slowdown | Peak VRAM | VRAM Reduction | Host RAM |
|---|---|---:|---:|---:|---:|---:|
| untiled | full image | 67.215 s | 1.00x | 62029 MiB | 0.0% | 16721 MiB |
| tiled | approximately 2x2 | 67.528 s | 1.00x | 42793 MiB | 31.0% | 16701 MiB |
| tiled | approximately 4x4 | 67.990 s | 1.01x | 37185 MiB | 40.1% | 16949 MiB |

VAE tiling is most visible at high resolution. In this 2048x2048 run, 4x4
tiling saves 24844 MiB of peak device memory with a 0.775 s median latency
increase.

## Graph and Operator Optimizations

Raw result roots:

- `benchmark/results/perf-cuda-optimization-ablation-20260714`
- `benchmark/results/perf-cuda-optimization-ablation-qwen-20260714`

Suite configs:

- `benchmark/configs/suites/cuda-optimization-ablation.yaml`
- `benchmark/configs/suites/cuda-optimization-ablation-qwen.yaml`

Contract: FLUX.1-dev and Qwen-Image, 1024x1024, 50 steps, BF16, batch 1,
seed 0, load-once generation.

| Model | Build | cuDNN SDPA | CUDA Norm | CUDA RoPE | Fused Modulation | Median | Speedup | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FLUX.1-dev | generic | no | no | no | no | 22.896 s | 1.00x | 40510 MiB |
| FLUX.1-dev | norm | no | yes | no | no | 22.345 s | 1.02x | 40510 MiB |
| FLUX.1-dev | norm_rope | no | yes | yes | no | 19.163 s | 1.19x | 40510 MiB |
| FLUX.1-dev | norm_rope_mod | no | yes | yes | yes | 18.835 s | 1.22x | 40510 MiB |
| Qwen-Image | norm_rope | no | yes | yes | no | 19.602 s | 1.00x | 61782 MiB |
| Qwen-Image | norm_rope_mod | no | yes | yes | yes | 18.687 s | 1.05x | 61782 MiB |

The public optimized build for this snapshot is the stable non-cuDNN CUDA build
with CUDA Norm, CUDA RoPE, and fused modulation. The cuDNN SDPA build was kept
out of public timing tables because this H200 environment hit a runtime symbol
loading issue around `cublasLtGetVersion` during cuDNN initialization.

## Recommended Runtime Profiles

These profiles summarize which benchmark result should guide a deployment
choice. They are recommendations from the measured configurations above, not a
separate benchmark suite.

| Profile | Suggested device | Configuration | Evidence |
|---|---|---|---|
| Maximum performance | H100/H200 or other large-memory CUDA GPU | BF16 + CUDA Norm/RoPE/modulation optimized build | Main table and resource `Performance` profile. |
| Balanced memory | 24 GB class GPU | Q8_0, or Q4_K with selected precision rules | Q8_0 uses 25341 MiB; Q4_K uses 17803 MiB in the quantization suite. |
| Fast approximate cache | Large-memory CUDA GPU | BF16 + EasyCache or MagCache method default | 2.09x and 2.69x speedup respectively in the matched cache table; MagCache is faster but more approximate. |
| Low memory | 12-16 GB class GPU | Q4_K + text encoder CPU + VAE tiling, optionally graph budget | 9785 MiB for the 16 GB profile; 10513 MiB for the 12 GB graph-budget profile. |
| Minimum memory | 8-12 GB class GPU | Q4_K + text encoder CPU + parameter offload + graph budget + VAE tiling | 7591 MiB in the 8 GB experimental profile, with high latency. |

## Reproducibility

The combined public summary and generated tables are stored at:

- `benchmark/results/performance-page-20260714/summary.json`
- `benchmark/results/performance-page-20260714/tables.md`

Validate benchmark configs:

```bash
python3 benchmark/orchestration/validate_config.py
```

Aggregate the frozen roots:

```bash
python3 benchmark/analysis/aggregate.py \
  --results-dir benchmark/results/readme-main-table-20260713-clean \
  --results-dir benchmark/results/perf-quantization-20260714 \
  --results-dir benchmark/results/perf-resource-profiles-20260714 \
  --results-dir benchmark/results/perf-vae-tiling-20260714 \
  --results-dir benchmark/results/perf-task-coverage-20260714 \
  --results-dir benchmark/results/perf-cuda-optimization-ablation-20260714 \
  --results-dir benchmark/results/perf-cuda-optimization-ablation-qwen-20260714 \
  --results-dir benchmark/results/perf-cfg-parallel-20260714 \
  --results-dir benchmark/results/perf-sp-parallel-20260714 \
  --results-dir benchmark/results/performance-page-readme-build-20260714/cache-quality \
  --results-dir benchmark/results/cache-retune-magcache-20260714/cache-quality-magcache-retuned \
  --results-dir benchmark/results/cache-retune-sencache-20260714/cache-quality-sencache-retuned \
  --suite-id performance-page \
  --output benchmark/results/performance-page-20260714/summary.json
```

Generate Markdown tables:

```bash
python3 benchmark/analysis/generate_tables.py \
  benchmark/results/performance-page-20260714/summary.json \
  --output benchmark/results/performance-page-20260714/tables.md
```

Cache quality metrics for the public cache table are produced by
`benchmark/scripts/run_cache_reuse_table.sh`, which scores PSNR/LPIPS for the
full cache matrix and the matched retuned cache rows after generation.

The retuned cache rows use matched reference/target image sets scored in:

- `benchmark/results/cache-retune-magcache-20260714/eval_magcache_default/summary/summary.json`
- `benchmark/results/cache-retune-magcache-20260714/eval_dicache_default/summary/summary.json`
- `benchmark/results/cache-retune-sencache-20260714/eval_sencache_t060/sencache_t060/summary/summary.json`

## Limitations

- Stable Diffusion results use Stable Diffusion 3 Medium because that is the
  checkpoint present in the local model tree.
- Diffusers task-coverage comparisons are available for text-to-image only in
  this snapshot because the benchmark harness does not yet expose matching
  load-once adapters for image editing and video.
- xDiT is kept out of the public SP table for this snapshot because the external
  checkout lacks the DistVAE dependency required by the selected workloads.
- SenCache is reported with a tuned 50-step profile and threshold 0.60 for this
  workload; additional thresholds need the same 24-sample speed/quality pass
  before they can replace the published row.
- The cuDNN SDPA build is excluded from public timing tables until the local
  cuDNN runtime symbol issue is resolved on this H200 environment.

## Related Documentation

- [Benchmark harness](../benchmark/README.md)
- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [Command line usage](cli.md)
- [Memory-efficient execution](optimization/memory-efficient-execution.md)
- [Graph and operator optimization](optimization/graph-and-operator-optimization.md)
- [Computation reuse](optimization/computation-reuse.md)
- [Parallel execution](optimization/parallel-execution.md)
