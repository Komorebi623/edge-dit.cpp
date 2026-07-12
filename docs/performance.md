# Performance and Optimization

[← Back to README](../README.md)

This page is the public entry point for performance and optimization
documentation. It is intentionally kept as a TODO scaffold until the full CUDA
performance release gate is complete.

Do not add official benchmark numbers here until they have been reproduced on
the release validation environment.

## Performance Configuration

TODO:

- Document the official `performance` build profile.
- Document the optional `minimal` validation profile.
- Record required dependency versions for CUDA, cuDNN, NCCL, and MPI.
- Link to the final `build-config.txt` schema once it is frozen.

See [Build and installation](build.md) for current build commands.

## Model Representation and Precision

TODO:

- Document supported weight dtypes.
- Document mixed precision behavior.
- Document per-tensor dtype rule syntax and validation rules.
- Add model-specific precision recommendations after verification.

Tracking page:

- [Model representation and precision](optimization/model-representation-and-precision.md)

## Memory-Efficient Execution

TODO:

- Document CPU offload behavior.
- Document graph VRAM control.
- Document VAE tiling tradeoffs.
- Document component placement behavior.
- Add validated examples for each supported model family.

Tracking page:

- [Memory-efficient execution](optimization/memory-efficient-execution.md)

## Graph and Operator Optimization

TODO:

- Document cuDNN SDPA enablement and fallback behavior.
- Document DiT-specific CUDA operator coverage.
- Document tensor-layout optimization work after validation.
- Add operator-level correctness and performance validation notes.

Tracking page:

- [Graph and operator optimization](optimization/graph-and-operator-optimization.md)

## Computation Reuse

TODO:

- Document supported cache modes.
- Document policy categories and required calibration data.
- Document quality and latency validation methodology.
- Add model-specific cache recommendations only after smoke tests.

Tracking page:

- [Computation reuse](optimization/computation-reuse.md)

## Parallel Execution

TODO:

- Document CFG parallelism.
- Document sequence parallelism.
- Document NCCL/MPI multi-worker execution.
- Add validated launcher examples.
- Add workload-dependent limitations and release benchmark results.

Tracking page:

- [Parallel execution](optimization/parallel-execution.md)

## Profiling

TODO:

- Document public profiling flags.
- Separate stable user-facing profiling options from developer diagnostics.
- Define the metadata required for profiling reports.

## Benchmark Methodology

TODO:

- Define the official benchmark environment.
- Define model, resolution, step, seed, and repetition settings.
- Define required metadata for published results.
- Add comparison policy against Diffusers and other native runtimes.

## Known Limitations

TODO:

- List limitations confirmed by release validation.
- Keep workload-dependent optimization caveats here.
- Avoid unpublished or internal-only performance claims.

## Related Documentation

- [Build and installation](build.md)
- [Supported models and usage](models.md)
- [Command line usage](cli.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)
