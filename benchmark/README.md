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

Run the reproducible FLUX.1-dev single-GPU e2e suite:

```bash
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --output-root benchmark/results/nightly-YYYYMMDD
```

The official FLUX main-table workload is `flux1-dev-t2i-1024-s50`:
1024x1024, 50 denoising steps, seed 0, batch 1, BF16. Shorter 20-step
workloads are diagnostic only and must not be aggregated into README or
main-table performance numbers.

For unattended official FLUX runs, use the release script. It waits for enough
free GPU memory, runs the 50-step suite, and regenerates the benchmark report:

```bash
WAIT_TIMEOUT_SEC=7200 \
RESULTS_ROOT=benchmark/results/release-v1-s50-YYYYMMDD \
bash benchmark/scripts/run_flux_s50_release.sh single
```

For 1/2/4 GPU sequence-parallel and xDiT runs:

```bash
WAIT_TIMEOUT_SEC=7200 \
RESULTS_ROOT=benchmark/results/release-v1-s50-YYYYMMDD \
bash benchmark/scripts/run_flux_s50_release.sh parallel
```

To avoid busy devices, provide an ordered device list. Each run consumes the
first `gpu_count` entries from the list:

```bash
BENCHMARK_CUDA_VISIBLE_DEVICES=1,2,3,4 \
python3 benchmark/orchestration/run_suite.py \
  --suite benchmark/configs/suites/flux-parallel-e2e.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems edge-dit.cpp \
  --output-root benchmark/results/nightly-YYYYMMDD
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
  --suite benchmark/configs/suites/pilot-flux.yaml \
  --site benchmark/configs/local/site-h200.yaml \
  --execute \
  --systems stable-diffusion.cpp \
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
generation latency. The stable-diffusion.cpp and xDiT adapters must gain their
own load-once e2e wrappers before their numbers can be reported.

The executable adapters live in `benchmark/scripts/`:

```text
benchmark/scripts/run_edge_e2e.py       # wraps build-cuda/bin/ed-sample
benchmark/scripts/run_diffusers_e2e.py  # runs Diffusers load-once loops
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
