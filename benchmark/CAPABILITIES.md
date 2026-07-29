# Capability support table: which systems support which methods

Quick lookup for "I want to use a method, which systems can test it". The data source is the `cross_system` field of each `methods/*/*.yaml` and the `capabilities` in `systems/*.yaml`, consistent with `run.py`'s capability-filtering logic.

- ✓ = supported; — = not supported. **If a section configures a method marked —, run.py automatically skips it during expansion and prints `[run.py] skip → ...`** (no error, does not run to failure).
- **System aliases**: `edge-dit` (= edge-dit.cpp), `diffusers`, `stable-diffusion.cpp` (= `sdcpp`, abbreviated sd.cpp in the tables below).
- **kind**: `runtime` = just change parameters and test, same binary; `build-variant` = needs a separately compiled dedicated binary.

---

## Quantization (quant)

Placed in a system section's `quant:` list. edge/sd.cpp are weight-only (via `precision`), diffusers uses Optimum-Quanto (`quant_weights`).

| method id | kind | edge-dit | diffusers | sd.cpp | notes |
|---|---|:--:|:--:|:--:|---|
| `fp16` | runtime | ✓ | ✓ | ✓ | f16 shared by all three systems; the same-system quality baseline for edge/sd.cpp |
| `q8` | runtime | ✓ | — | ✓ | q8_0 weight-only int8. diffusers's q8=W8A8 has different semantics (crashes on SD3), so not attributed to diffusers |
| `q4_k` | runtime | ✓ | — | ✓ | 4-bit K-quant, extreme VRAM saving; diffusers has no q4 |
| `bf16` | runtime | — | ✓ | — | diffusers's unquantized baseline (its same-system quality baseline); edge/sd.cpp use fp16 |
| `fp8` | runtime | — | ✓ | — | Optimum-Quanto qfloat8 (weights + activations) |
| `w8a8` | runtime | — | ✓ | — | Optimum-Quanto qint8 (weights + activations); crashes on SD3 |

**To compare quantization across systems**: write `[fp16, q8, q4_k]` in the edge/sd.cpp sections and `[bf16, fp8, w8a8]` in the diffusers section, and one job runs all three sections together (see `jobs/example-xsys.yaml`). Quantization loss is only comparable within the same system vs its own baseline, not across systems; CLIP/aesthetic/IR absolute quality can be compared side by side.

---

## Cache (cache)

Placed in a system section's `cache:` list (scalar single tier / list to sweep).

| method id | kind | needs calibration | edge-dit | diffusers | sd.cpp | notes |
|---|---|---|:--:|:--:|:--:|---|
| `easycache` | runtime | no | ✓ | — | ✓ | output change-rate gating |
| `ucache` | runtime | no | ✓ | — | ✓ | EasyCache + adaptive threshold |
| `dbcache` | runtime | no | ✓ | — | ✓ | residual-difference gating (dual-block) |
| `taylorseer` | runtime | no | ✓ | — | ✓ | residual Taylor extrapolation |
| `cache-dit` | runtime | no | ✓ | — | ✓ | DBCache gating + TaylorSeer extrapolation |
| `dicache` | runtime | no | ✓ | — | — | shallow-probe trajectory alignment; **edge-only** |
| `magcache` | runtime | **needs calibration, but on-device calibration fails (profile cannot be generated, do not put in the manifest)** | ✓ | — | — | magnitude-ratio table step-skipping; **edge-only** |
| `sencache` | runtime | **needs calibration, and not yet implemented on-device (disabled in the engine), do not put in the manifest** | ✓ | — | — | sensitivity Jacobian bound; **edge-only** |

**diffusers has no cache methods at all**. Calibration-free and directly sweepable: `none / easycache / ucache / dbcache / taylorseer / cache-dit` (shared by edge and sd.cpp). `magcache` and `sencache` both fail to calibrate on edge on-device, so do not use them for now; when you need an edge-only cache, use the calibration-free `dicache`.

---

## Attention (attention) / Memory (memory) / Parallelism (parallel)

| category | method id | kind | edge-dit | diffusers | sd.cpp | job-orchestrable | notes |
|---|---|---|:--:|:--:|:--:|:--:|---|
| attention | `flash` | runtime | ✓ | ✓ | ✓ | indirect | on by default (included in baseline); turn it off to compare |
| attention | `cudnn-sdpa` | build-variant | ✓ | ✓ | — | — | auto-triggered at L≥4096, from the performance build |
| attention | `sage` | build-variant | ✓ | — | — | — | SageAttention2; needs `-DED_ENABLE_CUDA_SAGE_ATTN=ON` build + `ED_SAGE_ATTN=1`; **edge-only** |
| memory | `offload-te` | runtime | ✓ | — | — | ✓ (`offload: te-cpu` in section) | text encoder only kept on CPU; **edge-only** — diffusers/sd.cpp runners have no separate TE offload, only whole-model offload |
| memory | `offload-full` | runtime | ✓ | ✓ | ✓ | ✓ (`offload: full` in section) | whole-model CPU offload |
| memory | `vae-tiling` | runtime | ✓ | — | ✓ | ✓ (`vae_tiling: yes` in section) | high-resolution VAE tiling; diffusers does not list this memory_mode |
| parallel | `cfg-parallel` | runtime (multi-GPU) | ✓ | ✓ | — | — | CFG parallelism, ~1.77× @ 2 GPUs |
| parallel | `sequence-parallel` | runtime (multi-GPU) | ✓ | — | — | — | Ulysses sequence parallelism, up to 2.59× measured on FLUX; **edge-only** |

**"job-orchestrable"**: currently `run.py` executes single-card, so the only dimensions a job section can sweep directly are **quant / cache / offload / vae_tiling**. `flash` is the on-by-default baseline (marked "indirect": to compare, turn it off via an engine-side switch); `cudnn-sdpa`/`sage` need build variants; `cfg-parallel`/`sequence-parallel` need a multi-GPU path — these three categories have no dedicated job field yet and must go through engine-side switches or the corresponding binary.

---

## System capabilities (systems/*.yaml)

| capability | edge-dit.cpp | diffusers | stable-diffusion.cpp |
|---|---|---|---|
| role | primary (main subject) | python_reference (reference) | native_baseline (native baseline) |
| tasks | t2i / editing / t2v | t2i / editing / t2v | t2i / editing / t2v (**Wan: see note**) |
| backends | cuda / cpu / metal / vulkan | **cuda only** | cuda / cpu / metal / vulkan |
| parallel | cfg / sequence | — | — |
| memory_modes | quantization / cpu_offload / component_placement (TE-offload) / vae_tiling / graph_cut | torch_dtype / cpu_offload (whole) / attention_backend | quantization / offload (whole) / vae_tiling / graph_cut |

All three systems cover all three tasks (text-to-image / image-editing / text-to-video). edge-dit.cpp is the most capable tier (four backends + multi-GPU parallelism + the most memory modes); diffusers, as the Python reference, is CUDA-only with quantization via torch_dtype/Quanto; sd.cpp is the native baseline, four backends + quantization/offload/VAE tiling, but no multi-GPU parallelism and no exclusive cache. **Note on Wan: stable-diffusion.cpp DOES support Wan, but it needs component-separated loading (--diffusion-model + --vae + --t5xxl, with -M vid_gen). This benchmark runner currently passes a single --model directory, which Wan does not accept ("get sd version from file failed"). So Wan is not benchmarkable on the stable-diffusion.cpp side until the runner implements component-separated loading; do not put Wan in a stable-diffusion.cpp section for now.**

> A method's `kind`/`needs calibration`/description is authoritative in `methods/<category>/<id>.yaml`; cross-system attribution is authoritative in its `cross_system` field (this table is aggregated from it). After adding a method/system, please sync this table.
