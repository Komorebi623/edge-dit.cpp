# Test manifest (job manifest): full field docs + examples

`jobs/*.yaml` is the **single entry point** for writing a benchmark: declare "which models × which quantization/acceleration methods × what task", run one command to get tables. This file is the **authoritative reference + ready-made examples** for the manifest format — reading this one file is enough to write a job.

- To copy a ready-made manifest directly: see [**Examples**](#examples-walkthrough-example-yaml) at the end.
- To look up "which systems support a method": see [`../CAPABILITIES.md`](../CAPABILITIES.md).

---

## Structure: per-system sections

The manifest has a **per-system sectioned** structure:

- **Top level** holds fields shared across the three systems (`name` / `task` / `models` / `prompts` / `steps` / `metrics` / `device`);
- **Each system to test** opens its own section (section name = system alias), with that system's own quantization/acceleration tiers;
- **A system without a section = not tested**.

This way one job lets each system take what it needs (edge uses q8_0, diffusers uses fp8, sdcpp uses q4_k), running the entire cross-system matrix in one command.

A minimal manifest = 3 required top-level fields + one system section:

```yaml
name: my-first-job          # results land in results/my-first-job/
task: text-to-image         # task type, must match the model
models: [sd3-medium]        # model id (see ../models/)

edge-dit:                   # one system section = test that system
  quant: [fp16, q8, q4_k]   # quant is required inside the section
# everything else in the section uses defaults: offload=none, vae_tiling=auto, cache=none
# everything else at top level uses defaults: prompts=3, steps=default, all three metrics on
```

---

## Top-level shared fields

| field | required | default | notes |
|---|---|---|---|
| `name` | ✓ | — | results directory name `results/<name>/` (can be overridden by command-line `--output-root`) |
| `task` | ✓ | — | `text-to-image` / `image-editing` / `text-to-video`, must match each model yaml's `task` |
| `models` | ✓ | — | list of model ids, taken from `../models/` (14) |
| `prompts` | | `3` | take the first N prompts from that model's prompt set per config, generate each and average |
| `steps` | | `default` | sampling steps, **three forms** (see "Advanced 2") |
| `metrics` | | all three on | `quality` / `speed` / `vram` three toggles (see "Advanced 3") |
| `device` | | not locked | lock this job to a specific physical GPU (e.g. `device: 5`); if unset, use the default card. Command-line `--device N` overrides it. When running multiple jobs in parallel on multiple GPUs, assign each job a different card to avoid VRAM contention OOM |

## System section fields

The section name can be `edge-dit` (= `edge-dit.cpp`), `diffusers`, or `stable-diffusion.cpp` (= `sdcpp`).

| section field | required | default | notes |
|---|---|---|---|
| `quant` | ✓ | — | that system's quantization tier list; an item can be an **id string** or an **object** (see "Advanced 1"). edge/sdcpp accept `fp16`/`q8`/`q4_k`, diffusers accepts `bf16`/`fp8`/`w8a8` (see "Quantization tiers per-system" at the end) |
| `offload` | | `none` | `none` / `te-cpu` (text encoder only kept on CPU) / `full` (whole-model offload); **scalar single tier, a list sweeps this dimension** |
| `vae_tiling` | | `auto` | `auto` (engine decides by VRAM) / `yes` / `no` |
| `cache` | | `none` | cache method id; **scalar single tier, a list sweeps this dimension** |

## How the matrix expands

For each system section, expand the Cartesian product `models × quant × offload × cache × prompts`; summing across sections gives the total run count. Adding `--dry-run` prints each expanded run with its `run_options`, and prints the total run count at the start — **after writing the manifest, dry-run to verify before running for real**.

> Example: `edge-dit` section `models(1) × quant(3) × offload(1) × cache(1) × prompts(3) = 9`.

---

## Advanced 1 · per-quant object override (configure one tier individually)

Besides an id string, a `quant` list item can be an **object** `{type: <id>, offload:, vae_tiling:, cache:}` — the `offload` / `vae_tiling` / `cache` inside the object **override the section default**, applying only to that one tier. Used for "most tiers use the section default, individual tiers enable a specific switch".

```yaml
edge-dit:
  quant:
    - fp16                             # bare id: use section default (none / auto / none)
    - {type: q8, offload: full}        # only the q8 tier does whole-model offload
    - {type: q4_k, cache: taylorseer}  # only the q4_k tier adds taylorseer cache
  offload: none                        # section default (used by bare fp16; tiers overridden by an object are unaffected)
```

Expansion: `fp16` (none), `q8` (offload_to_cpu), `q4_k` (cache=taylorseer), each × prompts. Keys not written in the object fall back to the section default. Corresponding example `example-perquant.yaml`.

## Advanced 2 · per-model steps (individual step count for a model)

`steps` has three forms:

- `default` (default): each model uses `generation.steps` from its own model yaml — **distilled models are naturally few-step** (schnell / turbo / lightning / distill come with ~4 steps); leaving the default is recommended;
- **an integer**: globally force all models to use that step count (e.g. `steps: 20`);
- **a dict** `{model_id: N, default: N}`: per-model specification, giving a specific model its own step count; missing models fall back to `default`, and if that is also missing, fall back to that model's own step count.

```yaml
models: [sd3-medium, flux-dev, qwen-image]
steps:
  sd3-medium: 28     # SD3 runs a few more steps
  flux-dev: 25
  default: 20        # the rest (qwen-image) use 20
```

Corresponding example `example-per-model-steps.yaml`. (Note: `steps` lands in each run's workload `generation.steps`, not in `run_options`, so it does not appear in `--dry-run`'s `run_options=`, but the expanded run count and step count are correctly injected.)

## Advanced 3 · metrics three toggles (control evaluation and table output)

`metrics` has three boolean toggles, **all on by default**:

```yaml
metrics:
  quality: true      # quality: CLIP/aesthetic/IR + quantization loss PSNR/SSIM/LPIPS (vs same-system baseline)
  speed:   true      # speed: DiT sampling ms / end-to-end ms / TE_ms / VAE_ms
  vram:    true      # VRAM: peak + TE/DiT/VAE per stage
```

Semantic differences (important):

- `quality: false` → **actually skips quality eval** (does not run `scripts/eval_all.py`, saving a lot of time) + does not show quality columns in the tables;
- `speed: false` / `vram: false` → the data is still produced for free during generation; the toggle **only controls whether these columns are shown** in the tables (passing `--no-speed` / `--no-vram` respectively to the table script).

The specific quality metrics are routed automatically by `task`: t2i = CLIP/aesthetic/IR; image-editing = directional CLIP/preservation SSIM/preservation LPIPS/aesthetic/IR; text-to-video = per-frame CLIP/per-frame aesthetic/temporal consistency (temporal LPIPS/SSIM/flicker std). Quantization loss PSNR/SSIM/LPIPS is computed under all three tasks (vs same-system baseline).

Only care about quality, don't want to see speed/VRAM columns:

```yaml
metrics: {speed: false, vram: false}
```

Corresponding example `example-metrics.yaml`.

---

## cross_system capability filtering (no worry about misconfiguration)

If a section configures a method that system doesn't support (e.g. `quant:[q8]` in a `diffusers:` section, `quant:[fp8]` in an `edge-dit:` section, or a `cache:` item the system lacks), run.py **automatically skips that combination during expansion and prints** a line:

```
[run.py] skip → diffusers: quant 'q8' not supported (cross_system), skipped
```

Neither errors out nor wastes time running to failure. Which method is supported by which systems is in [`../CAPABILITIES.md`](../CAPABILITIES.md).

## Quantization tiers per-system

| system section | accepted quantization tiers | mechanism |
|---|---|---|
| `edge-dit` | `fp16` / `q8` (q8_0) / `q4_k` | weight-only, via `precision` |
| `stable-diffusion.cpp` | `fp16` / `q8` (q8_0) / `q4_k` | weight-only, via `precision` |
| `diffusers` | `bf16` (baseline) / `fp8` / `w8a8` | Optimum-Quanto, via `quant_weights` |

To compare quantization across systems: write each section's own tiers, and one job runs it all in one command (see Example 4 below). Quantization loss is only meaningful within the same system vs its own baseline (fp16 for edge/sdcpp, bf16 for diffusers), not comparable across systems; CLIP/aesthetic/IR absolute quality can be compared side by side.

## cache calibration note

`magcache` and `sencache` need per-model calibration to generate a profile, but both fail to calibrate on edge on-device (the profile cannot be generated, and forcing a run degrades quality), so **do not put either in the manifest**. Calibration-free and directly sweepable are `none` / `easycache` / `ucache` / `dbcache` / `taylorseer` / `cache-dit`; the edge-only calibration-free one is `dicache`. See [`../CAPABILITIES.md`](../CAPABILITIES.md) for details.

---

**How to run** (from the repo root):

```bash
python benchmark/run.py \
  --job  benchmark/jobs/<manifest>.yaml \
  --site benchmark/sites/site4090.yaml
# add --dry-run to only print the expanded run plan (no generation), for checking the manifest
# add --output-root <dir> to change the results directory (default results/<name>/)
# add --device N to lock this run to physical GPU N (overrides the device field in the job)
```

**Multi-GPU parallelism**: when the machine has multiple cards, assign each job a different `device` (or use `--device`) to run them simultaneously, each on its own card, without competing for VRAM:

```bash
# three jobs each lock one card, running in the background simultaneously (can omit --device when the job top level sets device: 0/1/2)
python benchmark/run.py --job benchmark/jobs/t2i.yaml   --site .../site4090.yaml --device 0 &
python benchmark/run.py --job benchmark/jobs/edit.yaml  --site .../site4090.yaml --device 1 &
python benchmark/run.py --job benchmark/jobs/video.yaml --site .../site4090.yaml --device 2 &
```

> Without locking cards, multiple jobs default to the same card (number 0), competing for VRAM and causing OOM — for multi-GPU parallelism, always use `device`.

---

---

## Examples (ready-made manifests, copy and edit directly)

Each corresponds to a same-named `example-*.yaml` under `jobs/`, with detailed comments in the file; the table below shows at a glance what capability each demonstrates:

| manifest | demonstrates | run count |
|---|---|:--:|
| `example-quant-sweep` | minimal start: single model, three quantization tiers + full quality | 9 |
| `example-speed-sweep` | multi-model × multi-quant, `metrics.quality:false` skips quality for speed | 18 |
| `example-cache-sweep` | sweep several cache methods (calibration-free) in a section's `cache` list | 9 |
| `example-xsys` | true cross-system: three sections each with their own quantization tiers, one command | 27 |
| `example-perquant` | per-quant object override (configure offload/cache for one tier individually) | 9 |
| `example-per-model-steps` | per-model steps (individual step count for a model) | 9 |
| `example-metrics` | metrics three toggles (turn off speed/VRAM columns) | 6 |

Three signature patterns (see the corresponding files for the rest):

```yaml
# cross-system in one command (example-xsys): each of three sections with its own accepted quantization tiers
models: [sd3-medium]
edge-dit:            { quant: [fp16, q8, q4_k] }
diffusers:           { quant: [bf16, fp8, w8a8] }
stable-diffusion.cpp: { quant: [fp16, q8, q4_k] }
```
```yaml
# per-quant object override (example-perquant): enable a switch for just one tier
edge-dit:
  quant: [fp16, {type: q8, offload: full}, {type: q4_k, cache: taylorseer}]
```
```yaml
# per-model steps (example-per-model-steps): different step counts per model
models: [sd3-medium, flux-dev, qwen-image]
steps:  {sd3-medium: 28, flux-dev: 25, default: 20}
```

Field semantics are above; method capability attribution is in [`../CAPABILITIES.md`](../CAPABILITIES.md).
