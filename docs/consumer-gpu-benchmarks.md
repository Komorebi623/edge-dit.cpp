# Consumer-GPU Budget Benchmarks (RTX 4090)

[Back to performance](performance.md) | [Back to README](../README.md)

This page reports edge-dit.cpp results on a single consumer RTX 4090 (24 GiB),
focused on running the supported model families under a fixed VRAM budget with
`--auto-allocate`, and on step-distilled variants. It is a separate snapshot
from the [H200 benchmark page](performance.md) — different GPU, precision, and
step count — so numbers are not comparable across the two pages.

## Setup

- GPU: 1x RTX 4090 (24 GiB), serial runs.
- Precision: `--type q8_0` (weight-only 8-bit GGUF).
- Placement: `--auto-allocate` at three budgets — 24g (no cap), 16g
  (`--max-vram 16`), 8g (`--max-vram 8`) — with `--vae-tiling`.
- Workload: 1024x1024, 20 steps, seed 0 (Wan is 832x480, 9 frames).
- Peak VRAM sampled with `nvidia-smi` and cross-checked against the sampler.
- Forward count: FLUX / Kontext use distilled guidance (single forward); SD3 /
  Wan use CFG at scale 5.0 (two forwards); Qwen-Image / Qwen-Image-Edit run a
  single forward (no negative prompt supplied).

`net inference` is the generate time excluding one-time model load; `end-to-end`
includes load (weights are read and quantized to q8_0 on every run since these
are not pre-quantized GGUF).

## Full-precision families under a VRAM budget

The `offload` column is the placement decision for DiT / TE / VAE, where
R = resident on GPU and O = offloaded (streamed from host per step).

| Model | Budget | Status | offload | Forwards | TE (ms) | DiT (ms) | VAE (ms) | Net inf (ms) | E2E (ms) | Peak (MiB) | In budget |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| FLUX.1-dev | 24g | OK | R/R/R | 1 | 150 | 10930 | 530 | 11710 | 27436 | 19530 | yes |
| FLUX.1-dev | 16g | OK | R/O/O | 1 | 500 | 11110 | 570 | 12320 | 34773 | 14480 | yes |
| FLUX.1-dev | 8g | OK | O/O/R | 1 | 430 | 26180 | 540 | 27300 | 58564 | 6092 | yes |
| SD3 Medium | 24g | OK | R/R/R | 2 | 210 | 3490 | 480 | 4360 | 10876 | 10338 | yes |
| SD3 Medium | 16g | OK | R/R/R | 2 | 230 | 3540 | 470 | 4420 | 11329 | 10302 | yes |
| SD3 Medium | 8g | OK | R/O/R | 2 | 620 | 3460 | 470 | 5170 | 15156 | 6564 | yes |
| FLUX.1-Kontext | 24g | OK | R/R/R | 1 | 44 | 25000 | 470 | 26400 | 41447 | 19830 | yes |
| FLUX.1-Kontext | 16g | OK | R/O/O | 1 | 400 | 24890 | 560 | 30000 | 47678 | 14816 | yes |
| FLUX.1-Kontext | 8g | OK | O/O/R | 1 | 345 | 40020 | 530 | 45480 | 73211 | 5958 | yes |
| Wan 2.1 T2V 1.3B | 24g | OK | R/R/R | 2 | 141 | 6320 | 1190 | 8238 | 25625 | 12062 | yes |
| Wan 2.1 T2V 1.3B | 16g | OK | R/R/R | 2 | 132 | 6130 | 1110 | 7870 | 24522 | 12062 | yes |
| Wan 2.1 T2V 1.3B | 8g | OK | R/O/R | 2 | 481 | 6280 | 1140 | 8798 | 27749 | 6356 | yes |
| Qwen-Image | 24g | OK | O/R/R | 1 | 180 | 40030 | 500 | 40870 | 95296 | 19296 | yes |
| Qwen-Image | 16g | OK | O/R/R | 1 | 210 | 41330 | 450 | 42100 | 86568 | 13002 | yes |
| Qwen-Image | 8g | OK | O/O/R | 1 | 680 | 39580 | 450 | 40930 | 99324 | 6428 | yes |
| Qwen-Image-Edit | 24g | OK | O/R/R | 1 | 630 | 60850 | 510 | 63160 | 114270 | 19834 | yes |
| Qwen-Image-Edit | 16g | OK | O/R/R | 1 | 640 | 64420 | 610 | 66890 | 125437 | 13666 | yes |
| Qwen-Image-Edit | 8g | OK | O/O/R | 1 | 1090 | 62220 | 500 | 64890 | 119511 | 6302 | yes |

All 18 configurations complete, produce valid output, and stay within budget
(8g ≤ 8192 MiB, 16g ≤ 16384 MiB). Observations:

- **DiT stays resident when it fits.** FLUX/Kontext DiT (11.8 GiB) is resident
  at 24g/16g and only offloads at 8g; SD3 (2.1 GiB) and Wan (1.4 GiB) DiT are
  resident at every budget. Only Qwen (20.2 GiB DiT) exceeds 24 GiB and always
  offloads. Offload is a fallback for weights that do not fit, not the default.
- **Budget scaling costs latency only when DiT must offload.** SD3/Wan DiT time
  is flat across budgets (DiT never moves); FLUX 8g DiT jumps (11.8 GiB streamed
  per step); Qwen is offloaded at all budgets so its DiT time is roughly flat.
- The 8g profile brings every family down to ~6 GiB peak while still producing
  valid output — a memory floor the model would not otherwise reach on 24 GiB.

## Step-distilled variants (few-step)

Same q8_0 / seed 0 / 24g setup, using `--steps -1` so the runtime auto-selects
the few-step count. Speedup is DiT sampling versus the same family's 20-step
base run above.

| Model | Distilled variant | Steps | DiT (s) | Speedup vs 20-step | Peak (MiB) |
|---|---|---:|---:|---:|---:|
| FLUX | schnell | 4 | 2.34 | 4.7x | 19506 |
| SD3 | 3.5-medium-turbo | 8 | 0.97 | 3.6x | 11222 |
| Qwen-Image | Lightning | 8 | 4.93 | 8.1x | 22408 |
| Qwen-Image-Edit | Edit-Lightning | 8 | 23.78 | 2.6x | 19834 |
| Wan 2.1 T2V 1.3B | Distill | 8 | 1.45 | 4.4x | 12078 |

- **2.6x–8x faster** by cutting steps from 20 to 4/8, with output validated by
  eye at each variant's intended step count.
- Qwen-Image-Edit stays slower per run because its edit sequence includes the
  reference-image tokens (larger attention), not because few-step regressed.
- Distilled models are the intended low-step operating point. Forcing a
  distilled checkpoint to a high step count can over-cook the image; the
  few-step output is the correct one.

See [Few-step distilled models](optimization/few-step-distilled-models.md) for
how detection and step selection work, and [Command line usage](cli.md) for the
exact commands.
