# Benchmark Policy

[← Back to benchmark README](../README.md)

This policy defines how edge-dit.cpp benchmark results should be produced and
reported. It applies to the v0.1.0-alpha benchmark contract and future release
benchmarks unless a release-specific spec overrides it.

## Measurement Boundaries

Benchmark runs must separate:

- model loading;
- first generation;
- warmup runs;
- steady-state measured runs;
- output encoding and file writing.

Core pipeline latency must not include output file encoding. Encoding may be
recorded separately when useful.

## Fairness

Comparisons across systems must lock:

- checkpoint and model revision;
- tokenizer, text encoder, transformer, scheduler, and VAE;
- prompt, negative prompt, seed, resolution, frame count, and step count;
- guidance settings and batch size;
- precision and attention precision when supported by the compared systems.

Peak GPU memory should be measured externally, using NVML or `nvidia-smi`
polling, rather than framework-specific allocator counters.

## Reporting

Official results must report median and P90 latency. Mean, standard deviation,
and coefficient of variation should also be recorded in machine-readable output.

Unavailable metrics must be represented as `null` in JSON results. Do not write
`0` when a system cannot provide a metric.

## External Baselines

External baseline repositories may be updated by the benchmark runner only when
the suite explicitly requests it. For v0.1.0-alpha, xDiT is the only baseline
configured for force-synchronization to its latest `origin/main` during
benchmark execution. stable-diffusion.cpp is treated as a fixed local checkout.

The runner must record the resulting baseline commit and whether the checkout
was force-updated.
