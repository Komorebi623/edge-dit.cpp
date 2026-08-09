# MiniMax-H3 Edge Generation Performance Baseline — 2026-08-09

This document preserves the baseline and optimization context for the MiniMax-H3 Edge generation performance work planned for the evening of 2026-08-09.

## Primary objective

Reduce Edge's resident/warm video generation time while preserving the current output quality. The primary comparison target is the diffusers TorchAO INT4 generation path, not cold-start wall time.

## Benchmark task

- Workflow: I2VA
- Content: adult woman raises a selfie stick, waves, makes a V-sign, then triggers a synchronized flash and shutter sound
- Resolution: 864x480
- Frames: 124
- FPS: 24
- Output duration: 5.1667 seconds
- Inference steps: 20, resulting in 19 transformer calls
- Seed: 314159
- CFG scale: 1 for Edge and sd.cpp
- Hardware: one NVIDIA H200
- Input: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/inputs/first.png`
- Prompt: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/prompt.txt`

## Weight and runtime configurations

### Edge

- DiT: `minimax_h3_fl2va-Q4_K_M.gguf`
- Text encoder: `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`
- Video VAE: FP16 safetensors
- Audio VAE: FP32 safetensors
- Flash attention enabled
- Forced MiniMax-H3 VAE tiling remains active despite the CLI `--vae-tiling off` setting

### sd.cpp

- Uses the same GGUF DiT, GGUF text encoder, video VAE, and audio VAE files as Edge
- Flash attention and VAE tiling enabled
- CPU RNG
- Unmodified downloaded sd.cpp build

### diffusers

- Official MiniMax-H3 diffusers checkpoint
- Transformer and text encoder use TorchAO weight-only INT4, group size 128
- BF16 activations/compute where applicable
- Components remain resident on CUDA during generation

The Edge/sd.cpp and diffusers weights and quantization formats are not numerically identical. Performance comparisons therefore measure practical framework configurations rather than identical kernels over identical quantized tensors.

## Timing results

| Framework | Cold script wall | Framework pipeline/generate | Denoising/transformer | Decode |
|---|---:|---:|---:|---:|
| Edge Q4_K_M | 117.140 s | 107.921 s generation | 94.445 s | 11.914 s |
| sd.cpp Q4_K_M | 274.124 s | 271.120 s generation | 239.050 s sampling | included in remaining time |
| diffusers TorchAO INT4 | 118.724 s | 59.397 s CUDA generation | 53.621 s transformer | 3.731 s video VAE + 0.129 s audio VAE |

Additional cold-start data:

- Edge model load: 6.572 s
- Edge output save: 1.425 s
- diffusers pipeline initialization, checkpoint loading, quantization, and finalization: 46.492 s
- diffusers mux: 0.640 s
- diffusers internal end-to-end time: 106.530 s

## Important interpretation

- Edge and diffusers have similar cold script wall time only because diffusers spends about 46.5 seconds loading and quantizing components.
- The relevant steady-state comparison is Edge generation `107.921 s` versus diffusers CUDA generation `59.397 s`.
- Edge generation is currently about **1.82x slower** than diffusers after components are ready.
- Edge DiT diffusion is about **1.76x slower** than diffusers transformer forwards: `94.445 / 53.621`.
- Edge decode is about **3.09x slower** than the measured diffusers video and audio VAE decode total: `11.914 / 3.859`.
- Edge is about **2.34x faster** than sd.cpp in cold script wall time for this task.

## Edge profile breakdown

```text
total=107921 ms
conditioning=1047 ms
diffusion=94445 ms (19 steps, 19 calls)
decode=11914 ms
noise=258 ms

conditioning:
  context=647 ms
  vision image prepare=130 ms
  vision encode=208 ms
  text tokenize=20 ms
  text encode=289 ms
  keyframe VAE=400 ms

decode:
  video VAE=10831 ms
  video copy=109 ms
  audio VAE=969 ms
  audio copy=5 ms
```

## diffusers profile breakdown

```text
generate_cuda=59.397 s
transformer_forward_cuda=53.621 s (19 calls)
text_encoder=0.477 s
video_vae_encode=0.156 s
video_vae_decode=3.731 s
audio_vae_decode=0.129 s
```

## Quality checkpoint

All three frameworks generated a usable video with:

- a selfie stick and attached phone;
- a friendly wave;
- a V-sign pose;
- a late flash near frames 118-120;
- a synchronized late audio transient near 4.91-4.99 seconds.

No obvious small pixel-block artifact was observed in this task. Edge produced the strongest visible flash. sd.cpp remained slightly softer. The optimization must preserve this Edge quality baseline.

## Optimization priority

1. Profile each Edge DiT block and operator over the 19 calls; diffusion represents 87.5% of Edge generation time.
2. Compare Edge attention, linear/Q4_K, modulation, RoPE, and graph-launch timings with the diffusers transformer profile.
3. Verify CUDA graph coverage, graph breaks, temporary allocations, synchronization, and host-device traffic during each DiT call.
4. Investigate the video VAE decode path, especially forced spatial tiling, repeated tile graph execution, overlap blending, and per-tile launch overhead.
5. Re-run this exact benchmark after each change and report warm generation time separately from load/save time.
6. Reject optimizations that regress selfie-stick geometry, hands, flash timing, audio transient, or introduce block artifacts.

## Reproduction and artifacts

- Edge command: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge/run.sh`
- Edge log: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge/run.log`
- sd.cpp command: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/sdcpp/run.sh`
- sd.cpp log: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/sdcpp/run.log`
- diffusers command: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/diffusers/run.sh`
- diffusers profile: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/diffusers/final.json`
- Consolidated timing: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/review/summary.json`
- Visual montage: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/review/montage.jpg`
- Audio transient analysis: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/review/audio-transients.json`
- Edge video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge/final.mp4`
- sd.cpp video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/sdcpp/final.mp4`
- diffusers video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/diffusers/final.mp4`

## 2026-08-09 SM90 Q4_K result

A full clean CUDA rebuild confirmed that the original selfie baseline used the
quality-safe GGUF Q4_K MMQ route. The earlier 124-frame result that was close to
Diffusers used the existing Hopper-specific dequantize + cuBLAS route via
`GGML_CUDA_SM90_Q4K_CUBLAS=1`; omitting that environment switch explains the
large apparent regression. This is an execution-kernel choice, not a reduction
in frames, steps, resolution, conditioning, or VAE quality settings.

The final same-task run used the SM90 Q4_K route plus the previously validated
equivalent decode/backend switches. Request parameters, model files, seed,
scheduler calls, flash attention, and forced 16x16 latent VAE tiling remained
unchanged.

| Metric | Original Edge | Optimized Edge | Diffusers | Edge improvement |
|---|---:|---:|---:|---:|
| Generation | `107.921s` | `73.114s` | `59.397s` | `32.3%` |
| DiT diffusion / transformer | `94.445s` | `61.769s` | `53.621s` | `34.6%` |
| Decode | `11.914s` | `9.768s` | `3.859s` | `18.0%` |
| Script wall | `117.140s` | `82.076s` | `118.724s` cold | `29.9%` |

The optimized Edge DiT is now `1.15x` the Diffusers transformer time rather
than `1.76x`. The remaining generation gap is about `13.7s`, split between an
approximately `8.1s` DiT gap and an approximately `5.9s` non-DiT gap.

Quality gates:

- Output has 124 frames at 864x480 with a 5.17-second audio stream.
- The person identity, selfie stick, wave, V-sign, and late camera flash remain.
- The optimized video has no visible black-frame failure or obvious micro-block artifact.
- Mean block-boundary metrics changed only slightly: b4 `0.01085 -> 0.01099`,
  b8 `0.01128 -> 0.01152`, b16 `0.01169 -> 0.01200`.
- The late audio transient remains at `4.998s` versus `4.997s` in the original.
- The optimized trajectory is not numerically equivalent to MMQ (`22.18dB`
  average encoded-video PSNR), so the SM90 route remains an explicit validated
  performance profile rather than an unconditional global default.

Artifacts:

- Reproduction: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-sm90-safe-full/run.sh`
- Optimized log: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-sm90-safe-full/run.log`
- Optimized video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-sm90-safe-full/final.mp4`
- Visual comparison: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/review-sm90/base-vs-sm90.jpg`
- Block metrics: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/review-sm90/block-metrics.json`
- Audio transients: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/review-sm90/audio-transients.json`

## Layered DiT and VAE profiling

A same-shape `864x480`, 124-frame, 3-step probe (two DiT forwards) was used
before repeating the full 20-step task. Existing runner timing and CUDA-event
operator profiling were enabled together. Nsight Compute was attempted, but
this node denies hardware performance counters (`ERR_NVGPUCTRPERM`).

The first DiT call took `5.374s` (`5.369s` GPU compute), while the steady call
took `3.122s` (`3.105s` GPU compute). Build, allocation, and copy were only
about `17ms` in the steady call, ruling them out as the remaining DiT gap.

CUDA-event profiling covered `2.903s` of a `3.170s` profiled steady call:

| Operator | Device time | Share |
|---|---:|---:|
| `MUL_MAT` | `1.549s` | `53.4%` |
| `FLASH_ATTN_EXT` | `0.710s` | `24.5%` |
| `SCALE` | `0.255s` | `8.8%` |
| `CONCAT` | `0.124s` | `4.3%` |
| `CONT` | `0.074s` | `2.5%` |
| `CPY` | `0.056s` | `1.9%` |
| fused RoPE custom op | `0.053s` | `1.8%` |
| `GLU` | `0.049s` | `1.7%` |

The matmul internal profile identifies the backend-format bottleneck: 150
large Q4_K MLP projections per forward use quality-safe FP32/TF32 SGEMM,
while 50 QKV projections use FP16 Tensor Core GEMM. FC1 SGEMM costs `637.3ms`,
the two FC2 groups cost `319.4ms + 161.1ms`, QKV GEMM costs `270.2ms`, and QKV
output conversion costs `70.1ms` per forward. The remaining quality-safe gap
is primarily GGUF Q4_K layout and repeated dequantization versus Diffusers'
resident packed INT4 Hopper kernel, not graph/framework overhead.

The video VAE executes seven temporal chunks with ten spatial tiles each. A
representative call-chain split was:

| Stage | Time |
|---|---:|
| 70 tile runner calls | `5.806s` |
| tile split | `0.041s` |
| spatial tile merge | `0.436s` |
| temporal output slicing | `0.141s` |
| temporal overlap blend | `1.427s` |
| final temporal collection | `0.758s` |

The VAE device operator profile totals about `4.27s`; `CONT` (`1.60s`) and
flash attention (`1.20s`) are the largest categories. Three explicit,
mathematically equivalent copy-path experiments were added:

- `ED_VAE_PLANE_PARALLEL_TILE_COPY=1`
- `ED_MINIMAX_H3_MOVE_TEMPORAL_BLEND=1`
- `ED_MINIMAX_H3_PARALLEL_TEMPORAL_COPY=1`

On a same-binary short A/B, the combined path reduced video VAE
`9.269s -> 8.208s` (`11.4%`). Input VAE latent SHA256 matched and decoded video
PSNR was `inf`. On the full same-binary task, control and combined video VAE
were `8.810s` and `8.744s`; full decoded-video PSNR was also `inf`, but the
`66ms` timing difference is within run variance. Keep these switches explicit
until repeated full runs demonstrate stable benefit.

Artifacts:

- Operator profile: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/profile-2026-08-09/op-profile-3step.log`
- Parsed operators: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/profile-2026-08-09/op-profile-summary.json`
- Temporal profile: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/profile-2026-08-09/vae-temporal-breakdown-3step.log`
- Full control: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-current-control-full/run.log`
- Full combined VAE path: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-vae-all-copy-full/run.log`
- Full quality check: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/profile-2026-08-09/full-control-vs-vae-all-copy-psnr.log`

## 2026-08-09 clean-build and three-framework recheck

The current Edge worktree was rebuilt from an empty `build-cuda-docker`
directory at commit `2f74ed70638518094860f5fe6f6f14885ca6ceef`. The build used
Release mode, CUDA architecture 90, cuDNN SDPA, CUDA norm, CUDA RoPE, CUDA
modulation, and benchmark markers. Because the clean tree did not contain a
vendored `cudnn-frontend`, CMake fetched the project's pinned `07b368a`
revision through `ED_FETCH_CUDNN_FRONTEND=ON`; no backend capability was
removed to make the clean build pass.

All three frameworks were then run sequentially on GPU 0 with the same input,
prompt, seed `314159`, `864x480`, 124 requested frames, 24 fps, CFG 1, and 20
requested steps. All resulting MP4 files contain 124 frames at `864x480` and a
5.175-second audio stream.

| Framework | Generation / sampling | DiT / transformer | Video VAE | Script wall |
|---|---:|---:|---:|---:|
| Edge clean rebuild | `72.819s` | `61.710s` | `8.449s` | `81.784s` |
| Diffusers rerun | `59.430s` CUDA | `53.461s` | `3.760s` | `112.073s` cold |
| unmodified sd.cpp rerun | `271.27s` generate | `239.17s` sampling | `17.20s` video + `9.29s` audio | `274.462s` |

The reruns are consistent with the previous measurements. Edge's previous
same-path DiT result was `61.749s`, and Diffusers' previous transformer result
was `53.621s`. The clean binary therefore does not remove the remaining gap:
Edge DiT is `8.249s`, or `15.4%`, slower than the practical Diffusers TorchAO
INT4 configuration. Edge video VAE is `4.689s`, or `2.25x`, slower.

The earlier apparent `94.445s` versus `53.621s` DiT regression came from an
Edge run that omitted `GGML_CUDA_SM90_Q4K_CUBLAS=1`. Restoring that existing
Hopper Q4_K execution path reduced Edge DiT to about `61.7s` without changing
resolution, frame count, steps, prompt, conditioning, seed, or model files.
This is the large non-parameter-tuning acceleration that should be preserved.
It changes floating-point execution order and is not bit-exact with the MMQ
trajectory, so it remains an explicit quality-validated profile rather than a
global unconditional default. The clean-rebuild MP4 is bit-identical after
decode to the previous same-path control (`PSNR inf`).

The request-level benchmark parameters are aligned, but the three numerical
pipelines are not identical:

- Edge and sd.cpp load the same Q4_K_M GGUF DiT and Q4_K_M GGUF conditioner.
- Diffusers loads the official BF16 checkpoint and quantizes selected linear
  layers to resident TorchAO group-size-128 INT4 at startup.
- Edge and sd.cpp therefore do not execute the same quantized tensors or INT4
  kernel as Diffusers; performance and generated trajectories may differ.
- Edge reports forced 16x16 latent spatial tiling even when the CLI requests
  `--vae-tiling off`, executing seven temporal chunks and ten spatial tiles per
  chunk. The official Diffusers decoder also enables spatial tiling by default;
  its single profiled Python decode call contains the internal tiled decode and
  blending work. The remaining VAE gap is therefore an implementation and data
  residency difference, not simply "Edge tiles while Diffusers does not".
- sd.cpp explicitly enables VAE tiling and uses `--rng cpu`. It is the
  downloaded, unmodified commit `c6beeef35526c6dc94b74a7fb69f9d2e6a2a7a12`
  with binary SHA256
  `3e840a252f480116d5fc8fc58236ee95b8f6cc50b1738c91afa1464638a6fcea`.

The steady Edge runner profile still attributes only about `17ms` per DiT
forward to graph build/allocation/copy. The remaining DiT gap is therefore not
explained by stale binaries or generic framework overhead. Operator profiling
continues to point to repeated GGUF Q4_K dequantization plus FP32/TF32 SGEMM,
especially the MLP projections, versus Diffusers' resident packed INT4 Hopper
kernel. The next quality-preserving acceleration target should retain exact
Q4_K weights in a resident prepacked representation or provide a dedicated
SM90 kernel, rather than reducing generation parameters or widening selected
layers to a trajectory-changing FP16 path.

Recheck artifacts:

- Edge command and manifest: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/rebuild-2f74ed7-full/`
- Diffusers command, manifest, and JSON: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/recheck-diffusers-2026-08-09/`
- sd.cpp command and binary manifest: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/recheck-sdcpp-2026-08-09/`
- Clean-build equivalence check: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/recheck-diffusers-2026-08-09/edge-old-vs-clean-rebuild-psnr.log`

## 2026-08-09 independent clean-build optimized recheck

To rule out stale incremental objects again after adding the QKV, FC1, and VAE
blend experiments, a second independent build was configured from an empty
`build-cuda-clean-20260809` directory. It uses the same Release, SM90, CUDA
graph, cuDNN SDPA, CUDA norm, CUDA RoPE, and CUDA modulation options as the
active build. Only the pinned `cudnn-frontend` source checkout at revision
`07b368a` was reused; no object files or binaries were reused. The first clean
configure attempt also confirmed that the project-level switch must be
`ED_GGML_CUDA=ON`, not the lower-level `GGML_CUDA=ON`, because the former adds
the Edge CUDA extension sources, definitions, and include paths.

The full 124-frame benchmark was rerun from that independent binary with the
same prompt, input, model files, seed, dimensions, frame count, CFG, and step
count. No resolution, frame, step, scheduler, or cache tuning was used. The
following explicit experimental switches were enabled:

- `ED_MINIMAX_H3_QKV_F32_OUTPUT=1`
- `ED_MINIMAX_H3_FC1_F32_OUTPUT=1`
- `ED_MINIMAX_H3_FAST_TEMPORAL_BLEND=1`

The CLI still says `--vae-tiling off`, but the H3 decoder overrides the generic
flag and logs ten 16x16-latent spatial tiles for every temporal chunk. Thus the
actual H3 VAE remains tiled, matching the established Edge benchmark behavior.

| Path | Generation | DiT | Video VAE | Audio VAE | Script wall |
|---|---:|---:|---:|---:|---:|
| Edge independent clean build | `63.914s` | `53.836s` | `7.441s` | `0.974s` | `72.821s` |
| Diffusers TorchAO INT4 | `59.430s` | `53.461s` | `3.760s` | `0.125s` | `112.073s` cold |

The optimized Edge DiT is only `0.375s` (`0.7%`) slower than Diffusers on this
task. This closes the practical DiT timing gap, but it does not yet establish
an exact or generally quality-safe replacement: the FC1 route converts Q4_K
weights and FP32 activations to FP16 Tensor Core inputs with FP32 accumulation
and output. It changes numerical precision and remains behind an explicit
switch. A second task with different content must pass a full visual quality
check before considering it for a default performance profile. The QKV route
has the same precision caveat, although its smaller scope makes it the safer
candidate.

The temporal blend optimization is mathematically equivalent and preserves the
original operation order per output pixel. Its short same-binary A/B produced
decoded-video `PSNR=inf`, reducing temporal overlap blend from about `1.44s` to
`0.37s`. In the independent full run it took `0.205s`; retain this optimization
while pursuing the remaining VAE gap.

The clean-build video contains 124 frames at 864x480 and preserves the person,
selfie stick, wave, V-sign, and final flash. Fixed-frame inspection found no
new micro-block or seam artifact. Mean boundary differences were b4 `0.01189`,
b8 `0.01265`, and b16 `0.01321`, compared with Diffusers b4 `0.01368`, b8
`0.01450`, and b16 `0.01581`. These metrics do not prove quality, but they do
not indicate a new periodic block-boundary regression.

The clean-build trajectory differs from the previous FC1 run (`33.73dB` decoded
video PSNR), confirming that this Tensor Core precision path is not bit-exact
and can vary with GEMM algorithm/build details. It must not be described as a
lossless optimization. The visible output remains usable, so it is retained as
an experimental high-performance path rather than enabled unconditionally.

Artifacts:

- Build and run manifest: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/build-manifest.txt`
- Reproduction command: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/run.sh`
- Full timing log: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/run.log`
- Output video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/final.mp4`
- Fixed-frame comparison: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/review/three-way-frames.jpg`
- Boundary metrics: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/review/block-metrics.json`
- Previous-FC1 PSNR check: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-build-qkv-fc1-fast-vae-full/review/psnr.txt`

## 2026-08-09 forced clean rebuild and benchmark-accounting audit

The current working tree at Edge commit `2f74ed70638518094860f5fe6f6f14885ca6ceef`
was reconfigured, cleaned with the CMake `clean` target, and rebuilt through all
CPU, CUDA, Edge extension, library, and CLI targets. The resulting binary was
linked at `2026-08-09 04:35:32 +0800`. Its cache explicitly records Release,
SM90, `ED_GGML_CUDA=ON`, CUDA graphs, CUDA modulation/norm/RoPE, cuDNN SDPA,
and benchmark markers. This rules out both the earlier `ED_GGML_CUDA=OFF`
configuration mistake and stale incremental objects for the result below.

The full benchmark retained `864x480`, 124 frames at 24 FPS, 20 requested
steps/19 transformer forwards, seed 314159, CFG 1, the same input and prompt,
the same Edge model files, and the forced H3 16x16-latent VAE tiling. No
resolution, frame-count, step-count, scheduler, cache, or quality setting was
reduced.

| Path | Generation | DiT | Video VAE | Audio VAE | Script wall |
|---|---:|---:|---:|---:|---:|
| Edge forced clean rebuild | `62.615s` | `53.905s` | `5.978s` | `0.992s` | `72.413s` |
| Previous optimized Edge run | `62.194s` | `53.820s` | `5.605s` | `0.977s` | `71.109s` |
| Diffusers TorchAO INT4 | `59.397s` | `53.621s` | `3.731s` | `0.129s` | `106.530s` cold |

The clean Edge DiT is only `0.284s` (`0.5%`) slower than Diffusers. Therefore,
the previously observed large DiT gap is not present in the current build and
was not caused by this node's current source revision. Run-to-run VAE variance
accounts for most of the `0.421s` generation change versus the immediately
preceding optimized run. The remaining generation gap is now concentrated in
video VAE (`2.247s`) and audio VAE (`0.863s`), not DiT.

The benchmark is aligned for workload and timing comparison, but it is not a
strict same-weight numerical A/B. Edge loads a Q4_K_M GGUF transformer and
Q4_K_M GGUF conditioner. Diffusers loads the original Diffusers safetensors
and applies TorchAO 4-bit weight-only quantization at runtime, with selected
input/output, embedding, normalization, token-refiner, and vision modules left
unquantized. These representations can quantize the same source checkpoint
differently, so different denoising trajectories and visible motion are
expected even with the same seed. This does not invalidate the performance
comparison, but it prevents attributing cross-framework visual differences to
the pipeline alone.

The rebuilt video is exactly equal to the previous optimized Edge video after
decode (`PSNR=inf`). An eight-frame visual review preserves the person,
selfie stick, wave, V-sign, and final flash and shows no new micro-block or
tile-seam artifact. Its fixed-frame boundary means are b4 `0.01256`, b8
`0.01331`, and b16 `0.01391`. These metrics are only regression indicators;
the fixed-frame and full-video visual checks remain authoritative.

Future acceleration must retain the fixed workload. Material speedups should
remain available behind explicit switches until a second, different-content
full task passes visual review. Mathematically exact copy/blend reductions may
be retained directly. The next optimization target is VAE execution and data
movement, especially the `5.004s` temporal tile decode plus `0.971s` measured
output slicing, blending, and collection, rather than reducing generation
parameters.

Artifacts:

- Build manifest: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-rebuild-verified-2026-08-09/build-manifest.txt`
- Full timing log: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-rebuild-verified-2026-08-09/run.log`
- Output video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-rebuild-verified-2026-08-09/final.mp4`
- Fixed-frame comparison: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-rebuild-verified-2026-08-09/review/three-way-frames.jpg`
- Exact-output check: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-rebuild-verified-2026-08-09/review/psnr-vs-prior.log`
- Boundary metrics: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-rebuild-verified-2026-08-09/review/block-metrics.json`

## 2026-08-09 VAE and audio acceleration update

The latest full run keeps the same benchmark contract: `864x480`, 124 frames at
24 FPS, 20 requested steps/19 DiT forwards, seed `314159`, CFG 1, the same
prompt/input, the same Edge model files, and forced MiniMax-H3 VAE tiling. No
resolution, frame count, diffusion step count, scheduler, or quality knob was
reduced.

| Path | Generation | DiT | Video VAE | Audio VAE | Script wall |
|---|---:|---:|---:|---:|---:|
| Original Edge baseline | `107.921s` | `94.445s` | about `10.831s` | not isolated | not comparable |
| Edge forced clean rebuild | `62.615s` | `53.905s` | `5.978s` | `0.992s` | `72.413s` |
| Latest optimized Edge | `60.889s` | `53.842s` | `5.181s` | `0.161s` | `70.030s` |
| Diffusers TorchAO INT4 | `59.397s` | `53.621s` | `3.731s` | `0.129s` | `106.530s` cold |

From the original Edge baseline to the latest optimized Edge, generation time
dropped by `47.032s`, or `43.6%`. Relative to the verified clean rebuild, the
latest VAE/audio work saves another `1.726s` generation time and `2.383s` script
wall time. Edge is now only `1.492s` slower than Diffusers for measured CUDA
generation; DiT itself differs by only `0.221s`, so the remaining gap is almost
entirely video VAE (`1.450s`) plus a small audio VAE delta (`0.032s`).

Latest retained switches for this run:

- `ED_MINIMAX_H3_FUSED_TEMPORAL_ASSEMBLY=1` writes temporal chunks directly into
  the final video buffer and blends overlap frames in place, removing legacy
  slice/concat/full-video copy work.
- `ED_VAE_PLANE_PARALLEL_TILE_COPY=1`, `ED_VAE_PARALLEL_TILE_COPY=1`, and
  `ED_VAE_PARALLEL_TILE_COPY_THREADS=8` parallelize spatial tile merging; the
  merge path now precomputes destination ranges and smootherstep weights once
  per tile instead of once per plane.
- `ED_MINIMAX_H3_AUDIO_DIRECT_DEPTHWISE=1` maps the BigVGAN depthwise Conv1D and
  depthwise transpose/downsample stages to the direct depthwise CUDA path instead
  of expanding them through IM2COL plus MUL_MAT.
- Existing high-performance DiT/VAE switches remain enabled for this artifact:
  `ED_MINIMAX_H3_QKV_F32_OUTPUT=1`, `ED_MINIMAX_H3_FC1_F32_OUTPUT=1`,
  `ED_MINIMAX_H3_FAST_TEMPORAL_BLEND=1`, `ED_CUDNN_SDPA_SHORT_F16_SELF_ATTN=1`,
  and `ED_MINIMAX_H3_VAE_FUSED_SWIGLU=1`.

Validation performed:

- Fused temporal assembly was checked inside one candidate run against a legacy
  reconstruction from the exact same decoded chunks; raw tensor `max_abs=0`, so
  the assembly/blend math and memory layout are exact for identical inputs.
- Spatial tile merge precompute was checked against the previous implementation;
  decoded-video PSNR was `inf` and fixed frames matched.
- Audio direct depthwise was checked after MP4 s16 packaging against the control
  path; audio PSNR was about `171dB`, consistent with negligible floating-point
  differences rather than an audible change.
- Full-video fixed-frame review on the person/selfie-stick prompt showed normal
  identity, selfie stick, hand gesture, and final flash, with no new visible
  micro-block or tile seam artifact. The last video-path review reported b4
  `0.0125115`, b8 `0.0134006`, and b16 `0.0142055` boundary means.
- CUDA/operator profiling was used to attribute remaining video VAE time. With
  fused SwiGLU, the 70-tile video VAE profile still spends roughly `1.400s` in
  layout operations, `0.473s` in attention, `0.371s` in elementwise work, and
  `0.269s` in norm. The top remaining removable cost appears to be repeated
  `CONT` layout copies around MiniMax-H3 VAE decoder attention Q/K/V and RoPE
  packing, not DiT denoising.

Rejected or not-retained experiments:

- Cross-temporal-chunk CUDA graph reuse was tested and was not faster
  (`5.203s` video VAE with reuse versus `5.042s` without in the A/B run), while
  preserving `PSNR=inf`; it remains disabled.
- General Conv1D cuDNN reshaping, FP16 ordinary Conv1D weights, and two-channel
  batched BigVGAN experiments did not beat the conservative direct-depthwise
  audio path and were removed from the code path.

Artifacts:

- Latest full run script: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-final-vae-audio-optimized-full/run.sh`
- Latest full timing log: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-final-vae-audio-optimized-full/run.log`
- Latest output video: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-final-vae-audio-optimized-full/final.mp4`
- Previous visual review frame sheet: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/edge-clean-qkv-fc1-fast-vae-fused-assembly-plane-merge-full/review/comparison.jpg`
- VAE CONT node profile: `outputs/minimax-h3/person-selfie-three-frameworks-5s-2026-08-09/profile-2026-08-09/vae-swiglu-node-top400/run.log`
