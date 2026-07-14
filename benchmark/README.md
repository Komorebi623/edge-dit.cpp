# edge-dit.cpp Benchmark

This directory defines the benchmark contract for edge-dit.cpp v0.1.0-alpha.
It is intentionally split into two layers:

- **Benchmark contract:** specifications, schemas, workload definitions, system
  definitions, hardware metadata, and prompts.
- **Benchmark execution:** runners, measurement utilities, orchestration,
  result checking, aggregation, and report generation.

The existing `benchmark/evaluation/` scripts remain quality-evaluation helpers.
They are reused by the harness for quality evaluation instead of replacing the
inference benchmark runner.

## v1 Goal

The v1 harness freezes and implements:

- what the v0.1.0-alpha benchmark is meant to prove;
- which workloads and systems are in scope;
- how latency, memory, quality, and parallel metrics are represented;
- which local paths are allowed only in local site overrides;
- which result directories must not be committed;
- how each run is expanded, executed, checked, aggregated, and reported.

Official numbers must be generated from result directories by the analysis
scripts. Do not hand-copy benchmark values into reports.

## Layout

```text
benchmark/
├── specs/       Benchmark policy and release-specific contract
├── configs/     Suite, workload, system, hardware, and local site configs
├── prompts/     Prompt sets used by workload configs
├── schemas/     Machine-readable result and environment schemas
├── runners/     System adapters for edge-dit.cpp, Diffusers, stable-diffusion.cpp, and xDiT
├── measurement/ Timing, environment, and resource measurement helpers
├── orchestration/ Suite validation, dry-run expansion, execution, resume, and result checks
├── analysis/    Result aggregation, table generation, and report generation
├── evaluation/  Existing quality-evaluation helpers
└── results/     Local benchmark outputs, ignored by Git
```

## Local Outputs

The following directories are local-only and ignored by Git:

```text
benchmark/results/
benchmark/cache/
benchmark/downloads/
benchmark/tmp/
```

Do not commit generated images, videos, raw logs, model weights, downloaded
models, or benchmark result directories.

## Validation

Validate the benchmark contract and configs:

```bash
python3 benchmark/orchestration/validate_config.py
```

Preview the FLUX pilot run matrix without executing models:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --dry-run
```

Build edge-dit.cpp before official runs. The CUDA build script defaults to the
performance profile:

```bash
bash scripts/build_cuda.sh
```

Run the README main-table suite. This is the source for the root README
Performance table and uses FLUX.1-dev, Stable Diffusion 3 Medium, and
Qwen-Image with the same 1024x1024, 50-step, BF16, batch-1 setting:

```bash
bash benchmark/scripts/run_readme_main_table.sh
```

Add a fourth real 1024x1024, 50-step text-to-image workload to the README
main-table suite before publishing a four-model table. The script checks the
expanded workload count before it runs.

Run the reproducible FLUX.1-dev single-GPU e2e suite:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/nightly-YYYYMMDD
```

The README main-table workloads are the `*-1024-s50` text-to-image configs:
1024x1024, 50 denoising steps, seed 0, batch 1, BF16. Shorter 20-step
workloads are smoke or diagnostic only and must not be aggregated into README
or main-table performance numbers. The local H200 config currently has Stable
Diffusion 3 Medium, not a real SD3.5 Large checkpoint.

## Full Performance Page Suites

`docs/performance.md` is backed by a wider feature-results matrix than the root
README. Run these suites into separate frozen result roots, then aggregate those
roots into a `performance-page` summary.

```bash
# Build the five CUDA variants used by cuda-optimization-ablation.
bash benchmark/scripts/build_cuda_ablation_matrix.sh

# Task coverage: T2I, image editing, and video.
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/task-coverage.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/perf-task-coverage-YYYYMMDD

# CFG scaling. This intentionally uses SD3 Medium, not FLUX distilled guidance.
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cfg-parallel.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/perf-cfg-parallel-YYYYMMDD

# Sequence-parallel scaling, including short and long sequence workloads.
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/sp-parallel.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --force-external-update \
  --output-root benchmark/results/perf-sp-parallel-YYYYMMDD

# Cache speed-quality. This is the only suite that gates public quality metrics.
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cache-quality.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/perf-cache-quality-YYYYMMDD

# Resource, quantization, VAE tiling, and CUDA optimization trade-offs.
for suite in resource-profiles quantization vae-tiling cuda-optimization-ablation; do
  python3 benchmark/orchestration/run_suite.py \
    --suite "benchmark/configs/suites/${suite}.yaml" \
    --site benchmark/configs/local/site-h200.yaml \
    --execute \
    --systems edge-dit.cpp \
    --output-root "benchmark/results/perf-${suite}-YYYYMMDD"
done
```

After cache quality evaluation, apply metric summaries back to the matching
target result directories:

```bash
python3 benchmark/analysis/apply_quality_metrics.py \
  --result-dir benchmark/results/perf-cache-quality-YYYYMMDD/.../target-run \
  --eval-summary benchmark/results/perf-cache-quality-YYYYMMDD/.../eval_summary/summary.json
```

Finally aggregate the frozen roots:

```bash
python3 benchmark/analysis/aggregate.py \
  --results-dir benchmark/results/perf-main-table-YYYYMMDD \
  --results-dir benchmark/results/perf-task-coverage-YYYYMMDD \
  --results-dir benchmark/results/perf-cfg-parallel-YYYYMMDD \
  --results-dir benchmark/results/perf-sp-parallel-YYYYMMDD \
  --results-dir benchmark/results/perf-cache-quality-YYYYMMDD \
  --results-dir benchmark/results/perf-resource-profiles-YYYYMMDD \
  --results-dir benchmark/results/perf-quantization-YYYYMMDD \
  --results-dir benchmark/results/perf-vae-tiling-YYYYMMDD \
  --results-dir benchmark/results/perf-cuda-optimization-ablation-YYYYMMDD \
  --suite-id performance-page \
  --output benchmark/results/performance-page-YYYYMMDD/summary.json
```

Run the public model smoke suite across all locally configured public-preview
model families:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/model-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/model-smoke-YYYYMMDD
```

Run memory-constrained and optimization probe suites:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/memory.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/memory-YYYYMMDD

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/ablation.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/ablation-YYYYMMDD
```

Generate the local SenCache profile and run the repeated cache-mode matrix:

```bash
mkdir -p benchmark/cache

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cache-calibration-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/cache-calibration-smoke-YYYYMMDD

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cache-matrix.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/cache-matrix-YYYYMMDD
```

Run quick edge-dit.cpp parallel smoke suites:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/parallel-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/parallel-smoke-YYYYMMDD

python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/cfg-smoke.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/cfg-smoke-YYYYMMDD
```

For unattended official FLUX runs, use the release script. It waits for enough
free GPU memory, runs the 50-step suite, and regenerates the benchmark report:

```bash
bash benchmark/scripts/run_flux_s50_release.sh single
```

For 1/2/4 GPU sequence-parallel and xDiT runs:

```bash
bash benchmark/scripts/run_flux_s50_release.sh parallel
```

Run the reproducible FLUX.1-dev parallel e2e suite for edge-dit.cpp:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/nightly-YYYYMMDD
```

Record xDiT baseline preflight or skipped status without mixing it into
edge-dit.cpp numbers:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems xdit \
  --force-external-update \
  --output-root benchmark/results/nightly-YYYYMMDD
```

For a tiny smoke run, override the run counts:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --warmup-runs 0 \
  --measured-runs 1
```

External baselines that are configured with `force_latest_origin_main` require
an explicit opt-in before execution:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems xdit \
  --force-external-update \
  --warmup-runs 0 \
  --measured-runs 1 \
  --max-runs 1
```

The execution harness records process-level wall time, external GPU memory
samples, host RSS samples, environment metadata, command lines, logs, and
machine-readable `result.json` files. Component-level timing fields remain
`null` until a system-specific runner can emit them reliably.

The official latency boundary is **load-once e2e generation**:

- model loading is timed and recorded as `latency_ms.load`;
- warmup generations run in the same process and are excluded from steady-state
  statistics;
- measured samples time one complete generation after the model is already
  loaded;
- output encoding and file writing are outside the core latency where the
  backend makes that split available.

System adapters that participate in official comparisons must write
`runner_metrics.json` with `load_ms`, `warmup_ms`, and `measured_ms`. The
orchestrator refuses official adapters that do not produce this file, so
process startup and model loading cannot be accidentally counted as steady-state
generation latency. edge-dit.cpp, Diffusers, and stable-diffusion.cpp have
load-once wrappers for the README text-to-image main table; xDiT still needs a
matching official wrapper before its numbers can be reported.

The executable adapters live in `benchmark/scripts/`:

```text
benchmark/scripts/run_edge_e2e.py       # wraps build-cuda/bin/ed-sample
benchmark/scripts/run_edge_cli_once.py  # wraps build-cuda/bin/ed-cli for non-T2I smoke
benchmark/scripts/run_diffusers_e2e.py  # runs Diffusers load-once loops
benchmark/scripts/sd_cpp_e2e.cpp        # stable-diffusion.cpp C API load-once wrapper
benchmark/scripts/build_sd_cpp_e2e.py   # builds the stable-diffusion.cpp wrapper
benchmark/scripts/prepare_sdcpp_sd3_transformer.py
```

Inspect completed and pending suite entries:

```bash
python3 benchmark/orchestration/resume_suite.py \
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml
```

Resume a suite while skipping matching successful runs:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --resume
```

Aggregate results:

```bash
python3 benchmark/analysis/aggregate.py \
  --results-dir benchmark/results/nightly-YYYYMMDD \
  --include-workload flux1-dev-t2i-1024-s50 \
  --suite-id nightly-YYYYMMDD \
  --output benchmark/reports/v0.1.0-alpha/summary.json
```

Generate Markdown tables and a report:

```bash
python3 benchmark/analysis/generate_tables.py \
  benchmark/reports/v0.1.0-alpha/summary.json \
  --output benchmark/reports/v0.1.0-alpha/tables.md

python3 benchmark/analysis/generate_report.py \
  benchmark/reports/v0.1.0-alpha/summary.json \
  --tables benchmark/reports/v0.1.0-alpha/tables.md \
  --output benchmark/reports/v0.1.0-alpha/report.md
```
