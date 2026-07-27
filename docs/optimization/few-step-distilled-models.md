# Few-Step Distilled Models

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

## 1. Overview

A standard diffusion sampling loop runs the transformer many times — one full
forward per denoising step, commonly 20 or more. Step count is the single
largest linear factor in latency: halving the steps roughly halves sampling
time. *Step-distilled* checkpoints (FLUX.1-schnell, SD3.5-Turbo,
Qwen-Image-Lightning, Wan distill variants, and similar) are trained to
reproduce a full run in far fewer steps — typically 4 to 8 — so they cut
sampling time by 2.5–5x with little quality loss at their intended step count.

edge-dit.cpp treats the distilled checkpoint as an ordinary model of its family.
No separate pipeline or code path is required: the same FLUX / SD3 / Qwen /
Wan pipeline runs it. The only thing that must change is the default step count,
so the runtime auto-detects a distilled checkpoint and picks a sensible few-step
default when the user does not specify one.

This is a speed feature, not a quality one. A distilled model at its intended
low step count is close to its own many-step output, but distillation is itself
an approximation of the base model — validate output for the exact model and
step count you plan to run.

---

## 2. Automatic step selection

Step count is resolved per request in the following priority order:

1. **Explicit `--steps N` (N > 0)** always wins. The user's value is used as-is.
2. **Otherwise (`--steps` unset, i.e. the CLI passes `-1`)** the pipeline
   consults a distilled-detection signal and, if the model looks distilled,
   uses a few-step default (4 or 8). If not, it uses the family's base default
   (20 for most, 50 for Qwen-Image-Edit).

So existing commands that pass an explicit step count are unaffected; the
few-step default only fills in when the user asks the runtime to decide.

### Detection signals

Distilled checkpoints usually share the exact architecture and config of their
base model — the distillation changes only weight *values*, not structure — so
there is often no metadata field that distinguishes them. edge-dit.cpp uses two
signals, in order of reliability:

- **Weight-structure signal (FLUX only).** FLUX.1-schnell is a distinct
  guidance-distilled architecture: it lacks the `guidance_in` layer that
  FLUX.1-dev has. The FLUX pipeline detects this directly and defaults schnell
  to 4 steps, dev to 20.
- **Path-keyword signal (all families).** For same-architecture distills, the
  model or `--diffusion-model` path is scanned (case-insensitive) for
  `schnell`, `turbo`, `lightning`, `lightx2v`, `distill`, `hyper`, and
  `Nstep`/`Nsteps` markers. A `schnell`/`4step` hit defaults to 4 steps; other
  hits default to 8. No hit means the base default is used.

The keyword list is deliberately conservative. A miss is harmless — it just
falls back to the base step count (slower, not broken). A false positive would
run a base model at too few steps, so keep the list tight and prefer the
explicit `--steps` override when a distilled checkpoint has an unusual path.

### CFG and guidance

Distilled models are guidance-distilled: the classifier-free-guidance (CFG)
behavior is baked into the weights, so they run a single forward per step rather
than the two forwards CFG requires. edge-dit.cpp does not force this — the
default `cfg-scale` of `1.0` already yields a single forward, so distilled
models run single-forward out of the box. Passing `--cfg-scale > 1` on a
distilled model re-enables the two-forward path and is generally not what you
want.

---

## 3. Usage

Let the runtime pick the step count by leaving `--steps` unset (or `-1`):

```bash
# FLUX.1-schnell: weight-signal detected -> 4 steps automatically
ed-cli --backend cuda --type q8_0 --model models/flux.1-schnell \
  --steps -1 --auto-allocate --vae-tiling \
  --prompt "a glass teapot on a wooden table" -o out.png

# SD3.5-Turbo: path contains "turbo" -> 8 steps automatically
ed-cli --backend cuda --type q8_0 --model models/sd3.5-medium-turbo \
  --steps -1 --auto-allocate --vae-tiling --prompt "..." -o out.png
```

Distilled DiT weights shipped as a standalone file can be combined with a base
model's VAE and text encoders via `--diffusion-model` (the base supplies the
non-DiT components):

```bash
ed-cli --backend cuda --type q8_0 --model models/qwen-image \
  --diffusion-model models/qwen-image-lightning/transformer/diffusion_pytorch_model.safetensors.index.json \
  --steps -1 --auto-allocate --vae-tiling --prompt "..." -o out.png
```

Distilled variants distributed as LoRA adapters must be merged into the base
weights offline before use; the CLI loads full weights, not LoRA deltas.

See [Command line usage](../cli.md) for the full option reference and
[Supported models](../models.md) for per-family notes.
