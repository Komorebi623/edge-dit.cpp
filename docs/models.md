# Supported Models and Usage

[← Back to README](../README.md)

This document describes the public preview model scope, supported formats, and
model-specific limitations. For runnable commands, see
[Command line usage](cli.md). The source tree contains additional experimental
model scaffolding; only the families below are part of the current public
support commitment.

## Supported Models

| Model family | Task | Common format | Backend coverage | Status |
|---|---|---|---|---|
| SD3 / SD3.5 | Text-to-image | Diffusers-style directory or component weights | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| FLUX.1 | Text-to-image | Diffusers-style directory, top-level FLUX safetensors, or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| FLUX.1-Kontext | Image editing / reference-guided generation | Diffusers-style directory or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| Qwen-Image | Text-to-image | Diffusers-style directory or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| Qwen-Image-Edit | Image editing | Diffusers-style directory or components | CUDA first, CPU functional, Metal/Vulkan experimental | Public preview |
| Wan 2.x | Video generation | Diffusers-style directory or components | CUDA first, CPU functional for validation, Metal/Vulkan experimental | Public preview, still being optimized |

Backend availability means the runtime can be built for that backend. Model
quality, memory use, and speed are workload dependent and should be validated
for the exact model, resolution, and prompt set you plan to use.

## Model Formats

edge-dit.cpp can load:

- Diffusers-style directories.
- Standalone component weights for the diffusion model, VAE, text encoders,
  and model-specific vision/text components.
- `.safetensors` files.
- `.safetensors.index.json` shard indexes.
- GGUF files.

The simplest path is a model directory. Component loading is useful when
weights are stored separately. See [Command line usage](cli.md#model-loading)
for both forms.

### Step-distilled variants

Each supported family has step-distilled variants (FLUX.1-schnell, SD3.5-Turbo,
Qwen-Image-Lightning, Wan distill, …) that generate in 4–8 steps instead of 20+.
They load through the same pipeline as the base model; the runtime auto-detects
them and applies a few-step default when `--steps` is unset. Full-weight
distilled checkpoints load directly (`--model` or `--diffusion-model`);
LoRA-form distills must be merged into the base weights offline first. See
[Few-step distilled models](optimization/few-step-distilled-models.md).

## Text-to-Image

### FLUX.1

FLUX.1 text-to-image support uses Diffusers-style model directories,
standalone FLUX safetensors, or compatible component weights.

Command example: [FLUX.1-dev CLI](cli.md#flux1-dev).

### SD3 / SD3.5

SD3-family text-to-image support uses Diffusers-style directories or component
weights.

SD3 supports:

```bash
--no-t5
```

This reduces memory use and prompt adherence. The engine validates that
`--no-t5` is only used with SD3-family models.

Command example: [SD3 / SD3.5 CLI](cli.md#sd3-sd35).

### Qwen-Image

Qwen-Image text-to-image support uses Diffusers-style directories or component
weights.

Command example: [Qwen-Image CLI](cli.md#qwen-image).

## Image Editing

Image editing support depends on the model family and checkpoint format.

### FLUX.1-Kontext

FLUX.1-Kontext uses an input/reference image via `--image`.

Command example: [FLUX.1-Kontext CLI](cli.md#flux1-kontext).

### Qwen-Image-Edit

Qwen-Image-Edit uses an input/reference image via `--image`.

For Qwen-Image-Edit-2511, use:

```bash
--qwen-image-zero-cond-t
```

`--qwen-image-zero-cond-t` makes reference-image tokens use the `t = 0`
timestep condition while the image tokens being generated or edited still use
the current denoising timestep `t`.

It is an inference compatibility switch introduced for Qwen-Image-Edit-2511.
For other Qwen-Image-Edit models, ignore this option.

Command example: [Qwen-Image-Edit CLI](cli.md#qwen-image-edit).

## Video Generation

Wan video generation uses `--video`, `--frames`, and `--fps`.

Supported output formats are `auto`, `avi`, `mp4`, `mov`, `mkv`, and `webm`.
The CLI uses `ED_FFMPEG` when set and can also find imageio-ffmpeg binaries in
an active Python environment.

Wan 2.x remains an active optimization target. Validate memory use and output
quality for your exact resolution, frame count, and checkpoint.

Command example: [Wan video CLI](cli.md#video-generation).

## Quantization and Memory Options

The CLI supports on-load weight type selection:

```bash
--type f32|f16|bf16|q4_0|q4_1|q5_0|q5_1|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k
```

Per-tensor overrides are available with:

```bash
--tensor-type-rules "attn=q4_0,norm=f16"
```

Memory-oriented options:

```bash
--vae-tiling on|off|auto
--vae-tile-size <float>
--offload-to-cpu
--keep-text-encoder-on-cpu
--keep-vae-on-cpu
--auto-allocate
--max-vram <GB>
```

`--vae-tiling` takes an explicit `on|off|auto` value. Under `--auto-allocate`
with a `--max-vram` budget, the runtime decides per component (diffusion
transformer, text encoder, VAE) what stays resident on the GPU and what streams
from host memory, so large models can run within a fixed VRAM budget.

See [Command line usage](cli.md#quantization-and-memory) for runnable examples
and [Performance and optimization](performance.md) for cache, parallelism, and
profiling behavior.

## Model-Specific Limitations

- Public preview support is narrower than the internal enum list in the loader.
- Metal and Vulkan are experimental for DiT workloads and should be validated
  per model.
- Wan video support is available but still being optimized for memory and
  runtime behavior.
- Cache methods and sequence parallelism are workload dependent and may not be
  valid for every model or resolution.
- Model-specific options such as `--qwen-image-zero-cond-t`
  should only be used with the model families documented above.
- Component loading requires a complete, compatible set of text encoders, VAE,
  diffusion transformer, and optional vision components for the selected model.

## Related Documentation

- [Build and installation](build.md)
- [Command line usage](cli.md)
- [Performance and optimization](performance.md)
- [API and bindings](api.md)
- [Development and contributing](development.md)

[← Back to README](../README.md)
