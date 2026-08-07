# MiniMax-H3 optimization and Diffusers handoff (2026-08-07)

## Scope and constraints

- Repository: `/export/home/wangtianyang.21/code1/edge-dit.cpp`
- Current branch: `main`; last committed H3 baseline: `5472c67 Align MiniMax H3 conditioning with sd.cpp`.
- Use **one CUDA GPU only** for inference/performance work. Do not run CPU inference.
- This is a shared machine. Do not kill processes that are not created by this task.
- Do not commit or push without an explicit user request.
- The user wants eventual quality and performance comparison against official Hugging Face Diffusers, not just sd.cpp.

## Current working tree

Intentional, uncommitted source changes that predate this handoff:

- `src/dit_models/pipelines/minimax_h3_pipeline.cpp`
- `src/dit_models/pipelines/minimax_h3_pipeline.hpp`

These add `ED_MINIMAX_H3_PROFILE=1` phase timing and make H3 honor explicit VAE tiling disablement. Keep them unless separately reviewed.

New, uncommitted Diffusers benchmark launcher files:

- `scripts/diffusers/run_minimax_h3_fl2va.py`
- `scripts/diffusers/run_minimax_h3_fl2va.sh`

Other untracked paths shown by `git status` (assets, benchmarks, `.codex-tmp`, `third_party/onednn`, and so on) were already present or are unrelated; do not clean them blindly.

## Current edge-dit H3 implementation

The project supports both MiniMax-H3 checkpoint families through the existing H3 pipeline:

- FL2VA: text-to-audio-video and first/last-keyframe-to-audio-video.
- Ref2VA: ordered image/video/audio reference conditioning.

The C++ model baseline is quantized GGUF:

- `/models/MiniMax-H3-GGUF/minimax_h3_fl2va-Q4_K_M.gguf`
- `/models/MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`
- `/models/MiniMax-H3-GGUF/minimax_h3_video_vae_fp16.safetensors`
- `/models/MiniMax-H3-GGUF/minimax_h3_audio_vae_fp32.safetensors`

The main H3 implementation is in:

- `src/dit_models/models/minimax_h3_full.hpp`
- `src/dit_models/pipelines/minimax_h3_pipeline.cpp`
- `src/dit_models/pipelines/minimax_h3_pipeline.hpp`

sd.cpp reference source is read-only at `../stable-diffusion.cpp`.

## CUDA build and runtime state

The active container is `wty-edgedit-dev`; workspace inside it is `/workspace/edge-dit.cpp`.

The active CUDA build is `build-cuda-docker`, built successfully with:

```bash
docker exec wty-edgedit-dev bash -lc \
  'cd /workspace/edge-dit.cpp && cmake --build build-cuda-docker -j64'
```

Relevant active CMake options:

```text
ED_ENABLE_CUDA_MODULATION=ON
ED_ENABLE_CUDA_NORM=ON
ED_ENABLE_CUDA_ROPE=ON
GGML_CUDA_FA=ON
GGML_CUDA_FORCE_MMQ=OFF
GGML_CUDA_FORCE_CUBLAS=OFF
GGML_CUDA_GRAPHS=OFF
```

`GGML_CUDA_GRAPHS=ON` was tested with an isolated rebuild. It produced identical first-frame output but did not improve H3: 5-step diffusion changed from `25.593s` to `25.619s`. It is intentionally restored to `OFF`.

## Performance baseline and operator profile

Primary small regression case (FL2VA):

```text
Resolution: 864x480
Frames: 56 at 24 fps
Steps: 5
Seed: 42
CFG: 2
VAE tiling: explicitly off
```

Baseline script and output:

- `outputs/minimax-h3/perf-fl2va-regression-2026-08-07/run.sh`
- `outputs/minimax-h3/perf-fl2va-regression-2026-08-07/final.mp4`
- `outputs/minimax-h3/perf-fl2va-regression-2026-08-07/run.log`

Baseline phase timing:

```text
generation total: 34.221s
diffusion:        25.593s (10 forwards: conditional + unconditional per step)
conditioning:      0.916s
decode:            7.524s
end-to-end wall:  42.811s
```

For the larger mixed Ref2VA baseline (`864x480`, 56 frames, 20 steps, CFG 2):

```text
total:                     260.523s
diffusion:                 222.113s (85.3%)
reference video VAE encode: 24.498s
```

CUDA op profiling for the FL2VA 5-step probe:

```text
one DiT CUDA graph:       about 2.46s
Q4_K x FP32 MMQ/GEMM:     1.61-1.64s (about 66%)
attention:                0.30-0.32s (about 12-13%)
layout/materialization:   about 0.44s (about 18%)
elementwise/norm/RoPE:    about 0.23s (about 9%)
```

Percentages are profiler buckets and overlap slightly; do not force them to sum exactly to 100%.

Heavy MMQ shapes observed:

```text
m=28672 n=7919 k=5376, 50 calls: about 642ms
m=21504 n=7919 k=5376, 50 calls: about 480ms
m=5376  n=7919 k=14336, 50 calls: about 329ms
m=5376  n=7919 k=7168, 50 calls: about 166ms
```

Relevant artifacts:

- `outputs/minimax-h3/perf-baseline-mixed-2026-08-07/baseline.mp4`
- `outputs/minimax-h3/perf-runner-probe-2026-08-07/run.log`
- `outputs/minimax-h3/perf-cuda-op-profile-fl2va-2026-08-07/run.log`
- `outputs/minimax-h3/perf-cudnn-sdpa-fl2va-2026-08-07/run.log`

## MMQ, Tensor Cores, and cuDNN

The main Q4_K DiT linear layers use GGML CUDA MMQ. On the H200 (SM90), this takes GGML's Turing-or-newer MMA path, so it does use Tensor Cores. The activation is dynamically quantized to `Q8_1` and multiplied with `Q4_K` weights; it is not a normal BF16 GEMM dispatched through cuBLAS/cuDNN.

This is not unique to H3: the CUDA backend dispatches every quantized matrix multiplication meeting the same conditions (quantized weights, F32 activation/output, compatible layout and sufficiently large batch) through the common MMQ route. Different models and quantization types select different MMQ specializations; unsupported layouts or small/vector cases use MMVQ or fall back to cuBLAS after conversion. H3's main Q4_K projections meet the MMQ conditions.

cuDNN still helps a separate part of the model: H3 long self-attention successfully hits cuDNN SDPA (`b=1`, `h=56`, `d=128`, sequence length around `7919`). Some padded/unsupported attention shapes fall back. MMQ is the primary throughput limit and is not replaced by cuDNN SDPA.

Existing Flux/Qwen fused modulation or QKV paths do not automatically match H3. H3 has multi-span AdaLN modulation and concatenation in `minimax_h3_full.hpp`; a specialized implementation is needed rather than reusing those model-specific fusions blindly.

## Attempted H3 modulation fusion — rejected

An experiment temporarily connected H3 spans in `minimax_h3_full.hpp` to the existing generic CUDA custom ops:

- `edgedit::ggml_ext::fused_modulate_custom`
- `edgedit::ggml_ext::fused_residual_gate_custom`

The experiment was fully reverted. There is no remaining H3 fusion change in the source tree.

Why it was rejected:

- It made the 5-step diffusion timer look better: `25.593s -> 25.174s` (about 1.6%).
- Although mathematically equivalent, the fused kernel's floating-point instruction path differs from the original graph's `mul -> add -> add` scheduling.
- Diffusion amplifies tiny differences. The resulting 56-frame video diverged visibly: middle frames were around `23dB` PSNR against the baseline, so the output is not quality-equivalent.

Experimental artifacts (use only for diagnosis, not as accepted outputs):

- `outputs/minimax-h3/perf-fused-modulation-fl2va-2026-08-07/final.mp4`
- `outputs/minimax-h3/perf-fused-modulation-fl2va-2026-08-07/run.log`

Any future H3 fusion must use a strict quality gate:

1. Preserve the original operation/rounding order where possible; do not use algebraic rewrites or FMA that change it.
2. Run the same deterministic input and seed.
3. Compare all decoded frames and audio, not just MP4 hashes or the first frame.
4. Only retain an optimization after proving quality is acceptable and timing improvement survives repeated trials.

The likely next low-risk target is a H3-specific, rounding-preserving span implementation that reduces temporary layout/copy work without changing `mul -> add -> add` semantics. It should be profiled before implementation; it can only address the layout/elementwise portion, not the approximately 66% MMQ limit.

## Official Diffusers baseline

Official model URL: `https://huggingface.co/MiniMaxAI/MiniMax-H3`.

Official Diffusers documentation: `https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3`.

The installed release `diffusers==0.39.0` did not contain MiniMax-H3. Diffusers main was cloned to:

```text
/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers/code/diffusers-main
revision 53e5b00
```

Inside the container, use it without editable installation because `/models` is read-only:

```bash
export PYTHONPATH=/models/MiniMax-H3-Diffusers/code/diffusers-main/src${PYTHONPATH:+:$PYTHONPATH}
```

Runtime dependency state inside `wty-edgedit-dev`:

```text
torch 2.7.1+cu128
transformers 5.12.0
accelerate 1.14.0
torchao 0.15.0
fbgemm-gpu-genai 1.2.0
PyAV installed
imageio-ffmpeg installed
sentencepiece installed
```

Do not upgrade `torchao` blindly: `torchao 0.18.0` was incompatible with the container's PyTorch 2.7.1. `torchao==0.15.0` plus `fbgemm-gpu-genai==1.2.0` is the currently verified combination for TorchAO int4 smoke tests and the MiniMax-H3 FL2VA int4 run. `fbgemm-gpu-genai==1.8.0` was rejected because it tried to load CUDA 13 runtime (`libcudart.so.13`) on this CUDA 12.8 container.

### Important comparison caveat

The edge benchmark uses quantized GGUF and CFG 2 with two DiT forwards per step. Official Diffusers uses original BF16 weights and documents CFG-distilled behavior: no negative prompt or guidance scale, one model forward per step. These are not yet an apples-to-apples timing configuration. The official run is necessary to establish its own quality/timing baseline; do not claim direct speed parity until equivalent workflow parameters are analyzed.

The edge H3 pipeline already has the matching fast path: `cfg_scale == 1.0f` skips unconditional context construction and the unconditional DiT call. The existing performance probe deliberately used `--cfg-scale 2`, hence its 5 steps produce 10 DiT calls. Switching to CFG 1 changes the sampling configuration/output and is not a lossless optimization of the CFG 2 baseline; measure it separately as the official distilled-model comparison configuration.

Measured on the 864x480, 56-frame, 5-step FL2VA probe on one H200:

```text
CFG 2: 10 DiT calls, diffusion 25.519s
CFG 1:  5 DiT calls, diffusion 12.798s
```

### MMQ versus forced cuBLAS experiment — rejected

`GGML_CUDA_FORCE_CUBLAS=ON` was compiled and tested on the same CFG 1 probe. It made DiT much faster (`12.798s -> 8.314s` for five forwards), but it did not preserve the quantized execution's output: decoded frame comparison gave minimum PSNR `28.86dB`, average PSNR `35.15dB`, and maximum RGB difference `207`. This is a genuine deterministic difference because two default-MMQ runs produced byte-identical MP4 files and all 56 decoded frames were pixel-identical.

The CMake build was restored to both `GGML_CUDA_FORCE_CUBLAS=OFF` and `GGML_CUDA_FORCE_MMQ=OFF`. Do not enable forced cuBLAS as an accepted H3 optimization without an explicit quality/compatibility decision. The likely reason is that the fallback uses a different quantized-weight conversion/accumulation path than MMQ.

### Download and automatic benchmark state

Writable host/CFS target:

```text
/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers
```

Inside container the same target is `/models/MiniMax-H3-Diffusers` and is read-only.

The correct Diffusers components are at repository root (`transformer/`, `text_encoder/`, `vae/`, `audio_vae/`, scheduler/tokenizer/processor), not in `FL2VA/`. `FL2VA/` is original-format deployment material and was not retained. The active download selects exactly the official Diffusers FL2VA components (61 files, about 134.16 GiB).

Do not kill these two task-owned background processes; both are detached with `setsid` and resumable curl:

```text
PID stored in .../logs/download-diffusers-fl2va.pid
script: /mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers/download_diffusers_fl2va.sh

PID stored in .../logs/run-when-ready.pid
script: /mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers/run_when_diffusers_fl2va_ready.sh
```

The watcher performs exact file-size completeness checks, waits for GPU 0 to be under 1GiB used, then runs:

```bash
docker exec wty-edgedit-dev bash -lc \
  'cd /workspace/edge-dit.cpp && CUDA_VISIBLE_DEVICES=0 scripts/diffusers/run_minimax_h3_fl2va.sh'
```

It writes:

```text
/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers/logs/diffusers-fl2va-benchmark.log
/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers/logs/diffusers-fl2va-benchmark-status.log
outputs/minimax-h3/diffusers-fl2va-768p/final.mp4
outputs/minimax-h3/diffusers-fl2va-768p/final.json
```

The generated Python runner records component loading, CUDA generation, mux, and end-to-end time. It uses official `ComponentsManager.enable_auto_cpu_offload(device="cuda:0")`: model execution occurs on CUDA, while components not active in the current stage reside in host memory. This is the official single-GPU recipe for this combined model size.

### Diffusers TorchAO quantized comparison

Artifacts and scripts added for Diffusers quantized comparison:

- `scripts/diffusers/run_minimax_h3_fl2va_quantized.py`
- `outputs/minimax-h3/diffusers-fl2va-480p124-int8-gpu3/final.mp4`
- `outputs/minimax-h3/diffusers-fl2va-480p124-int4-gpu0/final.mp4`
- `outputs/minimax-h3/diffusers-fl2va-480p124-int4-resident-gpu0/final.mp4`
- `outputs/minimax-h3/compare-diffusers-fl2va-480p124-q4-cfg1-noneg-2026-08-07/final.mp4`

Common benchmark configuration:

```text
Workflow: FL2VA
Prompt: Create a smooth cinematic transition between the supplied first and last frame, preserving the subject, lighting, and composition with natural coherent motion and synchronized ambient audio.
Negative prompt: none
Inputs: outputs/minimax-h3/fl2va/edge/frames/first.png and outputs/minimax-h3/fl2va/edge/frames/last.png
Resolution: 864x480
Frames: 124 at 24 fps, duration 5.175s
Steps: 20
Seed: 42
Guidance: edge uses --cfg-scale 1, giving one DiT forward per step; Diffusers MiniMax-H3 uses the official CFG-distilled one-forward path.
```

Important quantization scope caveat: neither implementation is fully 4-bit end to end. Edge uses GGUF `Q4_K_M` mixed quantization for the DiT and Qwen3VL conditioner, but the video VAE is FP16 and the audio VAE is FP32. Diffusers TorchAO int4/int8 uses weight-only quantization for the `transformer` and `text_encoder` large linear weights only; `vae`, `audio_vae`, schedulers, tokenizer, and processor are not quantized. Some transformer and text encoder modules are intentionally excluded from quantization (`proj_in`, `audio_proj_in`, time/context embeddings, output projections, norms/rope, Qwen visual tower, embeddings, and `lm_head`).

MiniMax-H3 GGUF component sizes inside the container:

```text
/models/MiniMax-H3-GGUF/minimax_h3_fl2va-Q4_K_M.gguf                    18G, 18779848448 bytes
/models/MiniMax-H3-GGUF/minimax_h3_ref2va_pruned-Q4_K_M.gguf            11G, 11420663904 bytes
/models/MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf              17G, 18218065024 bytes
/models/MiniMax-H3-GGUF/minimax_h3_video_vae_fp16.safetensors          4.9G, 5207808496 bytes
/models/MiniMax-H3-GGUF/minimax_h3_audio_vae_fp32.safetensors          578M, 605254808 bytes
```

Observed edge weight stats for the comparable FL2VA Q4 run:

```text
overall:          f32:930 | f16:562 | q4_0:50 | q4_K:458 | q6_K:100 | bf16:816
conditioner:      q4_K:250 | q6_K:100 | bf16:552
diffusion model:  f32:13 | q4_0:50 | q4_K:208 | bf16:264
VAE:              f32:917 | f16:562
```

Measured timings:

| Runner | Quantization | Residency | Generation timing | End-to-end |
| --- | --- | --- | ---: | ---: |
| edge.cpp | GGUF Q4_K_M mixed Q4/Q6/BF16/F32 | GPU resident | `108.619s` diffusion, `131.870s` generation | `141.249s` wall |
| Diffusers | TorchAO int4 weight-only | GPU resident | `62.511s` `generate_cuda` | `105.194s` |
| Diffusers | TorchAO int4 weight-only | CPU/GPU group offload | `144.141s` `generate_cuda` | `181.148s` |
| Diffusers | TorchAO int8 weight-only | CPU/GPU group offload | `256.031s` `generate_cuda` | `271.528s` |

The resident Diffusers int4 path is the closest current timing comparison to edge's GPU-resident Q4 run. It is faster for this task, but it is still not identical quantization: GGUF `Q4_K_M` and TorchAO int4 weight-only use different layouts, skipped-module sets, activation handling, kernels, and numerical behavior.

File identity and quality sanity checks:

```text
edge Q4 resident:         d63a9dad18f65001674a763d89630f308a181dc5b04c9fbafbcb8dc4a11c930a
Diffusers int8 offload:   f6cddc3d6f466a3627c83c22c2b2410558b7b582dedb2e950dea267932414d86
Diffusers int4 offload:   78bee7947ffb0cd0ae09e660183de3a3b5f2e5f2ad73885915e7d6f5059d577f
Diffusers int4 resident:  38eb05e9eb8b69cfe570c6ba1bf3267fd4dad4b537f3c89f9678a9312053e351
```

The edge Q4 and Diffusers int8 videos can look similar at the endpoints because FL2VA strongly constrains the first and last frames, but they are not the same file or the same generated video. Sample decoded-frame PSNR for edge Q4 versus Diffusers int8 was about `35.70dB` on the first sampled frame, `10.50dB` on the middle sampled frame, and `35.95dB` on the last sampled frame.

Dependency notes for reproducing Diffusers int4:

```bash
docker exec wty-edgedit-dev bash -lc \
  'python3 -m pip install --user --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple torchao==0.15.0 fbgemm-gpu-genai==1.2.0'
```

`fbgemm-gpu-genai==1.8.0` should not be used on this container because it requires CUDA 13 runtime. The verified int4 smoke test is a CUDA BF16 `torch.nn.Linear` quantized with `Int4WeightOnlyConfig(group_size=128)`.

To inspect progress safely:

```bash
ROOT=/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-Diffusers
cat "$ROOT/logs/diffusers-fl2va-benchmark-status.log"
tail -100 "$ROOT/logs/diffusers-fl2va-benchmark.log"
ps -p "$(cat "$ROOT/logs/download-diffusers-fl2va.pid")" -o pid,stat,etimes,cmd
```

## Suggested continuation order

1. Let the official download and watcher complete; inspect the automatic Diffusers benchmark output/logs.
2. Verify official run success, MP4 streams, video dimensions/frame count, audio, and JSON phase times.
3. Make a documented comparison: original BF16 Distilled Diffusers vs Q4_K edge-dit. Separate model-format/config differences from framework overhead.
4. If optimizing edge H3 further, profile a stable repeated baseline first, then target layout/materialization with a strict output-quality gate.
5. Consider MMQ specialization only after the above: it is the principal bottleneck and requires SM90/H3-shape-specific kernel work plus robust correctness testing.

## 2026-08-07 DiT performance follow-up

Strict timing clarification for the 864x480, 124-frame, steps=20, seed=42 FL2VA keyframe benchmark:

- Diffusers TorchAO int4 resident with transformer forward hooks measured `generate_cuda=62.748s`, `transformer_forward_cuda=56.724s`, `transformer_forward_calls=19`.
- Edge before scheduler alignment had `diffusion=110.552s`, `diffusion_calls=20` in the profiled run, so `108.6s vs 62.5s` was not DiT-only vs DiT-only and edge also ran one extra transformer call.
- MiniMax-H3 official scheduler treats `num_inference_steps` as sigma grid points including the terminal clean point. Therefore `steps=20` runs 19 model evaluations.
- Edge was fixed to loop while `step + 1 < steps`, matching Diffusers. Correct FL2VA keyframe run now reports `diffusion=103.062s`, `diffusion_calls=19`, total generation `125.974s`.
- The per-forward gap remains large: edge is about `5.42s/forward`; Diffusers transformer-only is about `2.99s/forward`.

Validation artifacts:

- Diffusers transformer-only profile: `outputs/minimax-h3/diffusers-fl2va-480p124-int4-resident-profile3-gpu0/final.json`
- Edge corrected full run: `outputs/minimax-h3/perf-fl2va-keyframes-480p124-cfg1-steps20-scheduler19-2026-08-07/run.log`
- Edge 56-frame smoke baseline: `outputs/minimax-h3/perf-fl2va-keyframes-480p56-cfg1-steps5-default-mmq-2026-08-07/run.log`

Experiments rejected or left off by default:

- `GGML_CUDA_FORCE_MMQ=ON`, which permits larger MMQ tiles on SM90, did not improve the 56-frame/5-step smoke case: `diffusion=10.308s` vs default `10.321s` for four calls, within noise.
- H3 compiled-graph reuse was prototyped behind `ED_MINIMAX_H3_REUSE_GRAPHS=1` and left disabled by default. The current small benchmark showed no measurable DiT improvement (`10.360s` for four calls). Do not enable it by default without further investigation.
- The previous `ED_CUDA_DISABLE_MMQ_STREAM_K` experiment was reverted; stream-k disabling was much slower.

Current bottleneck after the scheduler alignment is still GGML Q4_K MMQ for H3's large projection shapes. Diffusers TorchAO/fbgemm int4 resident is likely using a Hopper-friendlier weight-only int4 layout/kernel. A real next performance step requires either a better SM90 Q4_K kernel/layout, or an optional prepacked resident int4/fp16-weight path for H3's DiT projections. Layout/materialization remains a secondary target, but it cannot close the main 5.42s vs 2.99s per-forward gap alone.

## 2026-08-07 SM90 Q4_K DiT optimization

The first practical DiT speedup came from revisiting the cuBLAS fallback instead of treating GGML MMQ as always best on Hopper.

Change made:

- `third_party/ggml/src/ggml-cuda/mmq.cu` now sends NVIDIA SM90+ `Q4_K` matrix multiplications with `ne11 >= 512` through the existing dequantize + cuBLAS path instead of MMQ.
- Small-column `Q4_K` calls still use MMQ, preserving the fast small-batch path used by text/auxiliary projections.
- The rule is disabled by compiling with `GGML_CUDA_FORCE_MMQ`; it can also be disabled at runtime with `GGML_CUDA_SM90_Q4K_CUBLAS=0`.

Measured impact on the H200 FL2VA benchmark (`864x480`, `124` frames, `24fps`, `steps=20`, `cfg=1`, seed `42`, one DiT forward per scheduler step):

| Build | Total generation | DiT diffusion | DiT calls | Decode | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline scheduler-aligned edge | `125.974s` | `103.062s` | `19` | `21.573s` | `outputs/minimax-h3/perf-fl2va-keyframes-480p124-cfg1-steps20-scheduler19-2026-08-07/run.log` |
| SM90 Q4_K cuBLAS rule | `92.378s` | `67.220s` | `19` | `23.586s` | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-480p124-cfg1-steps20-2026-08-07/run.log` |

The DiT part improved by about `34.8%` (`103.062s -> 67.220s`). This moves edge much closer to the Diffusers TorchAO int4 resident reference (`transformer_forward_cuda=56.724s`, `19` calls), though Diffusers is still faster.

Small validation task (`864x480`, `56` frames, `steps=5`, effective `4` DiT calls):

| Build | Total generation | DiT diffusion | Artifact |
| --- | ---: | ---: | --- |
| Baseline MMQ | `20.912s` | `10.845s` | `outputs/minimax-h3/perf-dit-default-56f5s-2026-08-07/run.log` |
| SM90 Q4_K cuBLAS rule | `16.926s` | `7.239s` | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-56f5s-2026-08-07/run.log` |

Kernel-path evidence from the 56-frame run:

- Baseline large H3 projections used `path=mmq` for `Q4_K` shapes such as `m=28672 n=7919 k=5376`, totaling about `2.559s` across the profiled calls for that shape.
- Optimized run used `path=op_cublas` for the same large shapes, totaling about `1.242s` for `m=28672 n=7919 k=5376`.
- Small `n=38` `Q4_K` calls remained on `path=mmq` in the 124-frame run, confirming the threshold did not force every quantized GEMM through cuBLAS.

Quality/numerics note:

- This is not bit-identical to MMQ. The cuBLAS route dequantizes and accumulates through a different path, so diffusion can diverge slightly even with the same seed.
- On the first eight decoded frames of the 56-frame smoke output, baseline-vs-optimized frame PSNR was roughly `44.3dB` on frame 1 and `33.5-40.7dB` on frames 2-8. The output remains visually close in this smoke check, but quality should be spot-checked on representative prompts before release.
- The change is framework-level for SM90+ large `Q4_K` GEMMs, not MiniMax-H3-specific. It may also benefit other large-Q4_K video DiTs on H100/H200, but it should be validated per model because numerical differences are possible.

Next bottlenecks after this change:

1. Diffusion still trails Diffusers int4 resident by about `10.5s` transformer time on the 124-frame benchmark (`67.2s` vs `56.7s`). TorchAO/fbgemm likely benefits from a prepacked TensorCoreTiled int4 layout, while GGUF `Q4_K` still pays dequantization/materialization costs in the cuBLAS path.
2. Layout/materialization remains around the next largest category in the old profile. Re-run `ED_PROFILE_CUDA_OPS=1` with the new rule and target the top layout nodes next.
3. A true longer-term fix is an SM90-native prepacked int4/Q4_K GEMM kernel or conversion path for resident H3 projections, with strict image/video quality gates.

## 2026-08-07 MLP SwiGLU materialization optimization

After the SM90 Q4_K cuBLAS rule, the next DiT profile showed layout/materialization as the second-largest category. In the 56-frame/5-step DiT-only graph subset:

```text
GEMM        2.873s
LAYOUT      1.771s
ELEMENTWISE 0.927s
FLASH_ATTN  0.831s
```

The top layout nodes were `CONT` materializations of the two strided halves of the MLP `fc1` output:

```text
CONT [14336,7919] from chunk/view: ~0.478s over 4 DiT calls
CONT [7168,7919]  from chunk/view: ~0.362s over 4 DiT calls
```

Change made:

- `src/dit_models/models/minimax_h3_full.hpp` now uses `ggml_swiglu(ctx, fc1(x))` for H3 MLP instead of `chunk(fc1(x), 2) -> silu(first) * second`.
- This reuses ggml's existing CUDA `SWIGLU` unary-gated kernel and removes the largest `[14336, seq]` materialization path.
- Semantics match the previous expression because `ggml_swiglu` treats the first half as the activated value and the second half as the gate, which is exactly `silu(uv[0]) * uv[1]`.

Measured impact on the same H200 FL2VA benchmark:

| Build | Total generation | DiT diffusion | DiT calls | Artifact |
| --- | ---: | ---: | ---: | --- |
| SM90 Q4_K cuBLAS rule | `92.378s` | `67.220s` | `19` | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-480p124-cfg1-steps20-2026-08-07/run.log` |
| SM90 Q4_K cuBLAS + MLP SwiGLU | `86.592s` | `62.061s` | `19` | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-swiglu-480p124-cfg1-steps20-2026-08-07/run.log` |

This second step improves DiT by another `7.7%` (`67.220s -> 62.061s`). Combined with scheduler alignment and SM90 Q4_K cuBLAS, edge's comparable 124-frame DiT time moved from `103.062s` to `62.061s`, now close to Diffusers TorchAO int4 resident transformer-only time (`56.724s`).

56-frame op-profile validation:

| Build | DiT diffusion | DiT layout category | Artifact |
| --- | ---: | ---: | --- |
| SM90 Q4_K cuBLAS rule | `7.276s` | `1.771s` | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-opprofile-56f5s-2026-08-07/run.log` |
| + MLP SwiGLU | `6.730s` | `1.311s` | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-swiglu-56f5s-2026-08-07/run.log` |

Remaining DiT gap vs Diffusers:

- Diffusers TorchAO/fbgemm int4 resident still has the better int4 GEMM layout: `TensorCoreTiledLayout(inner_k_tiles=8)` + TinyGEMM, packed as `[n/8, k/(inner_k_tiles*16), 32, inner_k_tiles/2]`, called through `aten._weight_int4pack_mm`.
- Edge still uses GGUF `Q4_K` weights. On SM90 the large projections now dequantize and use cuBLAS, which is faster than GGML MMQ for H3's large-column shapes, but still not as specialized as a resident prepacked int4 Tensor Core layout.
- Remaining layout hotspots are mostly attention layout/rope materialization (`[96,56,seq]`, `[128,seq,56]`, `[32,seq,56]`) and the second MLP half materialization (`[7168,seq]`). A larger next step would be an H3-specific fused QKV/RoPE/attention pack path, similar in spirit to the existing MMDiT attention pair-pack helpers.

### 2026-08-07 DiT layout optimization checkpoints and quality guardrails

Current H3 DiT optimization changes are intentionally split into independently-disableable checkpoints so any quality regression can be bisected quickly:

| # | Area | Code path | Default | Disable switch | Quality risk |
|---|---|---|---|---|---|
| 1 | SM90 large `Q4_K` GEMM routing | GGML CUDA `Q4_K` uses dequantize + cuBLAS for large H3 matrices instead of MMQ | Off; experimental opt-in | enable with `GGML_CUDA_SM90_Q4K_CUBLAS=1` | High: caused all-black 124-frame H3 output on the FL2VA quality probe |
| 2 | H3 MLP SwiGLU materialization | `fc2(ggml_swiglu(fc1(x)))` instead of chunk + silu + mul | On | `ED_MINIMAX_H3_DISABLE_SWIGLU_FUSION=1` | Low/medium: same formula, but fused kernel/order can differ slightly |
| 3 | H3 QKV split materialization | Q/K/V are `ggml_view_4d` views into `qkv_proj` output instead of three `ggml_cont` chunks | On | `ED_MINIMAX_H3_DISABLE_QKV_VIEW=1` | Low: layout-only view rewrite, intended mathematically identical |
| 4 | H3 partial-RoPE slice materialization | RoPE rotated prefix and unrotated tail use non-contiguous views instead of eager slice copies | On | `ED_MINIMAX_H3_DISABLE_ROPE_SLICE_VIEW=1` | Low: layout-only view rewrite; custom RoPE kernel reads strides |
| 5 | H3 modulation/segment materialization | segment/modulation slices and chunks use views instead of eager copies | On | `ED_MINIMAX_H3_DISABLE_SEGMENT_VIEW=1` | Low: layout-only view rewrite |
| 6 | H3 MLP Tensor Core experiment | allow MLP `fc1/fc2` to avoid `force_prec_f32` and use fp16 Tensor Core cuBLAS | Off | enable with `ED_MINIMAX_H3_MLP_FP16_CUBLAS=1` | High/experimental: faster but PSNR smoke checks dropped to `36-37dB` |

Performance checkpoints on the 56-frame / 5-step FL2VA smoke case (`864x480`, `cfg=1`, `steps=5`, 4 actual DiT calls):

| Checkpoint | Diffusion time | DiT graph layout time | Notes |
|---|---:|---:|---|
| SM90 cuBLAS + SwiGLU baseline | `6.730s` | `1.311s` total previously; DiT graph layout around `330ms`/call group in older op profile | Before H3-specific view rewrites |
| + QKV view | `6.017s` unprofiled / `6.402s` profiled | `236.151ms` | Removes the former `~91ms` qkv chunk copy group |
| + RoPE slice view | `6.184s` profiled | `175.416ms` | Removes `~46ms` rotated-prefix slice copy group |
| + Segment view | `5.777s` profiled | `128.197ms` | Removes many small segment/modulation copies and elementwise materializations |
| Current default with bisection switches | `5.821s` unprofiled | Not re-profiled in this run | Build after adding switches; profile-free timing is comparable to `+ Segment view` |

Quality guardrails:

- The low-risk view rewrites should preserve mathematical values, but CUDA graph execution and video/audio encode paths are not bit-exact across repeated runs; do not rely on byte equality.
- A quick PSNR check between `+ Segment view` and the current default rerun produced `average=39.775dB`, `min=33.590dB`, `max=50.368dB` for 56 frames, so this smoke check did not show a catastrophic visual drift, but it is not a substitute for visual inspection.
- If a quality regression appears, first ensure `GGML_CUDA_SM90_Q4K_CUBLAS` is not set to `1`, then try `ED_MINIMAX_H3_DISABLE_SWIGLU_FUSION=1`, then the three low-risk view switches one by one.
- For release-quality validation, run at least one 124-frame task against the pre-optimization baseline and Diffusers/TorchAO with the same prompt, seed, frames, scheduler semantics, `cfg=1`, VAE tiling off, and one DiT forward per scheduler step.

### 2026-08-07 Q4_K GEMM and attention follow-up

Internal profiling of the rejected SM90-cuBLAS path showed the post-layout bottleneck was no longer GGML MMQ. For the 56-frame / 5-step FL2VA smoke case, that experimental DiT graph time was dominated by cuBLAS-backed large `Q4_K` GEMMs plus cuDNN SDPA:

| Run | Diffusion | DiT graph GEMM | DiT graph attention | DiT graph layout | Artifact |
|---|---:|---:|---:|---:|---|
| Current default + internal profile | `5.825s` | `793.597ms` | `394.973ms` | `127.410ms` | `outputs/minimax-h3/perf-dit-current-internal-56f5s-2026-08-07/run.log` |
| MLP Tensor Core opt-in | `5.184s` | `638.170ms` | `407.586ms` | `122.620ms` | `outputs/minimax-h3/perf-dit-mlp-fp16-cublas-56f5s-2026-08-07/run.log` |
| MLP Tensor Core + forced f32 dst | `4.859s` | large `Q4_K` GEMMs down to `191.689/145.587/109.936/56.921ms` groups | not separately op-profiled | not separately op-profiled | `outputs/minimax-h3/perf-dit-mlp-fp16-cublas-f32dst-56f5s-2026-08-07/run.log` |

Key finding for the rejected experimental path:

- The old statement "Q4_K MMQ is the main bottleneck" was outdated only while the SM90 routing experiment was enabled. Because that path caused black 124-frame output, current quality-safe default is back to MMQ for large `Q4_K`; MMQ is again the principal safe-path DiT bottleneck.
- The largest MLP matrices were intentionally constructed with `force_prec_f32=true`, so even with quantized weights they take the f32 SGEMM branch. Internal profile examples: `m=28672,n=7919,k=5376` spent `297.116ms` in `f32/sgemm`, and `m=5376,n=7919,k=14336` spent `146.245ms` in `f32/sgemm` over 50 calls.
- Enabling the experimental `ED_MINIMAX_H3_MLP_FP16_CUBLAS=1` switches those MLP projections to fp16 Tensor Core cuBLAS. It improves smoke diffusion from `5.825s` to `5.184s`, but PSNR vs current default is only `average=37.029dB`, `min=30.218dB`; with `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`, speed improves further to `4.859s`, but PSNR drops to `average=36.214dB`, `min=30.106dB`.
- Because of that quality risk, `ED_MINIMAX_H3_MLP_FP16_CUBLAS=1` remains opt-in and should not be enabled by default without visual/full-length validation against Diffusers and the previous edge baseline.

Attention finding:

| Run | Diffusion | DiT graph attention | Artifact |
|---|---:|---:|---|
| cuDNN SDPA enabled | `5.783s` | `395.983ms` | `outputs/minimax-h3/perf-dit-attn-default-56f5s-2026-08-07/run.log` |
| cuDNN SDPA disabled | `10.922s` | `1442.747ms` | `outputs/minimax-h3/perf-dit-attn-nocudnn-56f5s-2026-08-07/run.log` |

- H3 is already using the cuDNN SDPA path for long attention. Disabling it roughly doubles diffusion time on the smoke case, so keeping cuDNN SDPA enabled is mandatory for H200/Hopper performance.
- The remaining attention-adjacent layout hotspot is mostly output layout materialization before `out_proj` (`CONT [128,7919,56,1]`, about `30-33ms` per profiled DiT graph group). Removing it safely likely needs a fused attention-output-projection layout or an out-projection matmul variant that consumes the attention output layout directly.

TorchAO/fbgemm comparison update:

- Diffusers TorchAO `Int4WeightOnlyConfig` keeps weights in a resident Tensor Core tiled layout and calls `aten._weight_int4pack_mm`; edge GGUF `Q4_K` remains a GGUF block layout and must either use MMQ or dequantize/convert into cuBLAS-compatible fp16/f32 buffers.
- The current fastest edge opt-in path still pays `src0_to_f16`, `src1_to_f16`, and sometimes `dst_to_f32`, while TorchAO/fbgemm avoids repeated GGUF block unpacking by construction. Matching that class of performance likely requires a resident prepacked H3 weight layout or a dedicated Q4_K Tensor Core kernel, not just more graph-level layout cleanup.

### 2026-08-07 124-frame full baseline after view optimizations

A full 124-frame FL2VA run was executed after the low-risk view optimizations, with MLP Tensor Core experiment **disabled**. This run used the now-rejected SM90 large-`Q4_K` cuBLAS route and produced an all-black MP4, so the timing is **not quality-valid**:

| Framework/path | End-to-end / total | DiT / transformer | Calls | Notes | Artifact |
|---|---:|---:|---:|---|---|
| Edge rejected SM90 cuBLAS default | `79.780s` profiled total, `87.878s` wall | `54.912s` | `19` | all-black output; invalid quality benchmark | `outputs/minimax-h3/perf-dit-current-default-480p124-cfg1-steps20-2026-08-07/run.log` |
| Edge previous SM90 cuBLAS + SwiGLU | `86.592s` total | `62.061s` | `19` | all-black output; invalid quality benchmark | `outputs/minimax-h3/perf-dit-sm90-q4-cublas-swiglu-480p124-cfg1-steps20-2026-08-07/run.log` |
| Diffusers TorchAO int4 resident | `104.725s` end-to-end, `62.748s` generation CUDA | `56.724s` | `19` | TorchAO 4-bit transformer + text encoder, VAE/audio VAE not quantized | `outputs/minimax-h3/diffusers-fl2va-480p124-int4-resident-profile3-gpu0/final.json` |

The rejected SM90-cuBLAS edge DiT timing (`54.912s`) must not be compared as a valid result against Diffusers because the output is black. The comparison remains useful only as a performance diagnostic:

- Diffusers timing is official PyTorch/TorchAO path with different kernel/layout/numerics and includes its own surrounding scheduler/pipeline behavior.
- Edge output MP4 in this run is not valid for visual quality because decoded sample frames had mean/std `0/0`.
- The remaining performance work should focus on either preserving quality while reducing MLP cuBLAS precision overhead, or implementing a resident prepacked int4/Q4 layout to avoid repeated GGUF block conversion. The fp16 MLP Tensor Core experiment is fast but currently too risky for default quality.

### 2026-08-07 MLP Tensor Core quality check on real-world FL2VA inputs

The 124-frame FL2VA keyframe benchmark was visually uninformative: decoded frames were identical and near-static, so it could not validate MLP Tensor Core quality. A second 56-frame FL2VA check used high-variance real-world frames from `outputs/minimax-h3/ref2va-realworld-full-benchmark-2026-08-07/inputs/video-frames/001.png` and `056.png`.

| Path | Diffusion | Total | Artifact |
|---|---:|---:|---|
| Current default | `5.832s` | `14.499s` | `outputs/minimax-h3/perf-quality-fl2va-realworld-current-vs-mlp-56f5s-2026-08-07/current/final.mp4` |
| `ED_MINIMAX_H3_MLP_FP16_CUBLAS=1` | `5.226s` | `14.040s` | `outputs/minimax-h3/perf-quality-fl2va-realworld-current-vs-mlp-56f5s-2026-08-07/mlp/final.mp4` |

Quality metric: `ffmpeg` PSNR current-vs-MLP is `average=11.195dB`, `min=10.582dB`, `max=12.735dB`, with visual contact sheet at `outputs/minimax-h3/perf-quality-fl2va-realworld-current-vs-mlp-56f5s-2026-08-07/summary/contact_current_vs_mlp.jpg`.

Conclusion: keep `ED_MINIMAX_H3_MLP_FP16_CUBLAS=1` strictly opt-in. It is a useful performance experiment, but it changes generation materially on realistic inputs and should not be enabled by default.

### 2026-08-07 black-frame regression fix

The user-reported all-black output was reproduced on the 124-frame FL2VA keyframe task. The root cause was the experimental SM90 large-`Q4_K` cuBLAS routing being enabled by default.

Repro and bisection artifacts:

| Path | MP4 size | Sample frame brightness | Artifact |
|---|---:|---:|---|
| Rejected default with SM90 cuBLAS | `90,514` bytes | all sampled frames mean/std `0.0/0.0` | `outputs/minimax-h3/black-regression-124-bisect-2026-08-07/default/final.mp4` |
| Same build with `GGML_CUDA_SM90_Q4K_CUBLAS=0` | `4,199,541` bytes | means `115.12`, `119.23`, `135.72`, `154.29`, `166.02`; std around `52-55` | `outputs/minimax-h3/black-regression-124-bisect-2026-08-07/no_sm90_cublas/final.mp4` |
| Fixed current default after making SM90 cuBLAS opt-in | `4,199,541` bytes | same non-black sample stats as the disabled run | `outputs/minimax-h3/quality-fix-sm90-q4k-default-mmq-124f-2026-08-07/final.mp4` |

Current code status:

- `third_party/ggml/src/ggml-cuda/mmq.cu` keeps MMQ as the default for SM90 `Q4_K`; the cuBLAS route now requires explicit `GGML_CUDA_SM90_Q4K_CUBLAS=1`.
- The opt-in path is retained only for future diagnosis/performance work and must not be used for quality or benchmark claims until the black-frame cause is fixed.
- Contact sheet for the fixed default: `outputs/minimax-h3/quality-fix-sm90-q4k-default-mmq-124f-2026-08-07/contact_sheet.jpg`.

### 2026-08-07 safe SM90 Q4_K cuBLAS partial re-enable

The all-or-nothing SM90 `Q4_K` cuBLAS route was too broad: it made the 124-frame FL2VA output black. A shape bisection showed the long-sequence H3 `fc2` projection is the unsafe part, while the long-sequence QKV and `fc1` projections remain visually non-black on the same probe.

Current switch behavior:

- `GGML_CUDA_SM90_Q4K_CUBLAS=1` enables only the validated H3 shapes: mid-sequence `n=7919` QKV/FC1 shapes plus long-sequence `n>=16000` QKV and FC1 shapes.
- Long-sequence FC2 (`m=5376`, `k=14336` or `7168`, `n>=16000`) remains on MMQ by default because it reproduced the all-black output.
- `GGML_CUDA_SM90_Q4K_CUBLAS_LONG_FC2=1` exists only as an explicit unsafe diagnostic switch; do not use it for quality benchmarks or user-facing runs.

124-frame, 5-step FL2VA quality/speed bisection (`864x480`, `cfg=1`, seed `42`, 4 DiT calls):

| Variant | Diffusion | Output | Artifact |
|---|---:|---|---|
| MMQ quality-safe baseline | `21.799s` | non-black | `outputs/minimax-h3/quality-speed-q4k-long-shape-bisect-124f5s-2026-08-07/mid_only/final.mp4` |
| Long QKV only | `19.069s` | non-black | `outputs/minimax-h3/quality-speed-q4k-long-shape-bisect-124f5s-2026-08-07/long_qkv/final.mp4` |
| Long FC1 only | `19.444s` | non-black | `outputs/minimax-h3/quality-speed-q4k-long-shape-bisect-124f5s-2026-08-07/long_fc1/final.mp4` |
| Long FC2 only | `19.019s` | all-black | `outputs/minimax-h3/quality-speed-q4k-long-shape-bisect-124f5s-2026-08-07/long_fc2/final.mp4` |
| Safe QKV+FC1 | `16.893s` | non-black | `outputs/minimax-h3/quality-speed-q4k-safe-default-124f5s-2026-08-07/final.mp4` |

The safe partial route recovers most of the speedup without reproducing black frames on this probe: `21.799s -> 16.893s` diffusion, about `22.5%` faster for the 124-frame/5-step task. The fixed contact sheet is at `outputs/minimax-h3/quality-speed-q4k-safe-default-124f5s-2026-08-07/contact_sheet.jpg`.

Important quality caveat: this is a non-black and visually sane validation, not a proof of exact equivalence. Before making this path a default user-facing optimization, run the full 20-step task and compare against the MMQ baseline and Diffusers with the same seed/prompt/input.

### 2026-08-07 FC2 cuBLAS quality fix

FC2 meaning: in MiniMax-H3 each Transformer MLP uses `fc1 -> SwiGLU -> fc2`. `fc1` expands hidden features from `5376` to the gated FFN width (`28672`, two `14336` halves), and `fc2` projects the SwiGLU result back from FFN width to hidden size (`5376`). It is a major DiT GEMM hotspot.

Root cause of the FC2 black-frame regression:

- Routing long-sequence H3 FC2 `Q4_K` weights through cuBLAS with the default Hopper fp16 accumulation/output path caused all-black 124-frame output.
- Running the same FC2 cuBLAS route with `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` fixed the black output, proving the problem is the fp16 cuBLAS compute/output mode for this FC2 shape, not cuBLAS routing in general.
- The code now forces f32 cuBLAS compute/destination only for the H3 SM90 long-sequence FC2 shape when `GGML_CUDA_SM90_Q4K_CUBLAS=1` is enabled. This keeps the QKV/FC1 fast path and makes FC2 usable without requiring the global `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` environment variable.

Current switch behavior after the fix:

- `GGML_CUDA_SM90_Q4K_CUBLAS=1` enables validated H3 QKV, FC1, and FC2 `Q4_K` cuBLAS routes on SM90/Hopper.
- Long FC2 automatically uses f32 cuBLAS compute/output to preserve quality.
- `GGML_CUDA_SM90_Q4K_CUBLAS_DISABLE_LONG_FC2=1` disables only the long FC2 cuBLAS route for rollback/bisection.

124-frame, 5-step FL2VA timing summary (`864x480`, `cfg=1`, seed `42`, 4 DiT calls):

| Variant | Diffusion | Output | Artifact |
|---|---:|---|---|
| MMQ quality-safe baseline | `21.799s` | non-black | `outputs/minimax-h3/quality-speed-q4k-long-shape-bisect-124f5s-2026-08-07/mid_only/final.mp4` |
| Safe QKV+FC1, FC2 on MMQ | `16.893s` | non-black | `outputs/minimax-h3/quality-speed-q4k-safe-default-124f5s-2026-08-07/final.mp4` |
| QKV+FC1+FC2 with global f32 cuBLAS compute | `13.848s` | non-black | `outputs/minimax-h3/quality-speed-q4k-long-fc2-f32compute-124f5s-2026-08-07/final.mp4` |
| QKV+FC1+FC2 with code-level FC2 f32 fix | `13.185s` to `15.095s` observed | non-black | `outputs/minimax-h3/quality-speed-q4k-fc2-f32fix-124f5s-2026-08-07/final.mp4`, `outputs/minimax-h3/quality-speed-q4k-fc2-default-f32fix-124f5s-2026-08-07/final.mp4` |

Net effect: compared with the MMQ baseline, using FC2 after the f32 compute fix improves this short 124-frame DiT probe by roughly `31-40%` (`21.799s -> 15.095s` best conservative repeat, or `13.185s` in the faster repeat) while avoiding the previous all-black failure.

Quality caveat: current validation proves the FC2 path no longer collapses to black on the 124-frame/5-step probe. It still needs a full 20-step visual/metric comparison before treating it as release-quality default.

### 2026-08-07 full 20-step FC2 f32-fix validation

A full 124-frame, 20-step FL2VA task was rerun with the current FC2 f32 cuBLAS fix enabled through `GGML_CUDA_SM90_Q4K_CUBLAS=1`.

Configuration:

- `--video-frames 124`, `--steps 20`, `--cfg-scale 1`, `--seed 42`
- `864x480`, `24fps`, VAE tiling off, `--diffusion-fa`
- First/end frame FL2VA prompt: `Create a smooth cinematic transition between the supplied first and last frame, preserving the subject, lighting, and composition with natural coherent motion and synchronized ambient audio.`

Result:

| Metric | Value |
|---|---:|
| Total generation | `85.129s` |
| DiT diffusion | `62.367s` |
| DiT calls | `19` |
| Decode | `21.451s` |
| Wall time | `94.077s` |

Quality sanity check:

- Output MP4: `outputs/minimax-h3/quality-speed-q4k-fc2-f32fix-124f20s-2026-08-07/final.mp4`
- Contact sheet: `outputs/minimax-h3/quality-speed-q4k-fc2-f32fix-124f20s-2026-08-07/contact_sheet.jpg`
- Sampled frame means: `116.31`, `115.57`, `134.97`, `155.41`, `166.60`
- Sampled frame stddevs: `54.66`, `56.61`, `55.93`, `57.80`, `52.85`

Conclusion: the full 20-step run is not black and visually sane on the contact sheet. The FC2 f32 cuBLAS fix keeps the long-sequence FC2 speed path usable for this probe. A broader quality comparison against MMQ baseline and Diffusers is still needed before treating this as final release-quality parity.

### 2026-08-07 committed state and Diffusers gap

Committed locally after the FC2 f32-fix validation:

- Main repository commit: `a761c7c Optimize MiniMax H3 CUDA inference`
- GGML submodule commit: `b9882160 Optimize H3 Q4_K CUDA routing`

Current closest same-task timing comparison on the 124-frame FL2VA benchmark (`864x480`, `steps=20`, actual 19 DiT/transformer forwards, `cfg=1`, seed `42`):

| Framework/path | Transformer/DiT time | Calls | Per-call | Notes |
|---|---:|---:|---:|---|
| Diffusers TorchAO int4 resident | `56.724s` | `19` | `2.985s` | `outputs/minimax-h3/diffusers-fl2va-480p124-int4-resident-profile3-gpu0/final.json` |
| edge.cpp Q4_K cuBLAS + FC2 f32-fix | `62.367s` | `19` | `3.282s` | `outputs/minimax-h3/quality-speed-q4k-fc2-f32fix-124f20s-2026-08-07/run.log` |

Current gap: edge DiT is `5.643s` slower over 19 calls, or about `9.95%` slower than Diffusers transformer time. Per model call the gap is about `0.297s`.

Important apples-to-apples caveat: the runs match the high-level task, resolution, frames, steps, seed, and one-forward CFG-distilled setting. They still do not have identical quantization kernels/layouts: edge uses GGUF `Q4_K_M` with selective cuBLAS fallback and FC2 f32 compute; Diffusers uses TorchAO/fbgemm int4 resident packed TensorCore layout. Therefore the remaining gap is best interpreted as backend/kernel/layout overhead, not model semantics alone.

Current short op-profile after FC2 f32-fix (`56` frames, `5` steps, 4 DiT calls):

- Artifact: `outputs/minimax-h3/perf-op-profile-q4k-fc2-f32fix-56f5s-2026-08-07/run.log`
- Diffusion time: `6.666s` for 4 calls.
- Main DiT graphs show GEMM still dominant: around `990-1003ms` GEMM per graph group, followed by attention (`155-362ms`), elementwise (`~152ms`), and layout (`~122ms`) on the main transformer graphs.
- Full decode graph remains a separate non-DiT cost and should not be mixed into DiT-vs-Diffusers transformer comparisons.

Next optimization candidates from this profile:

1. Reduce remaining H3 DiT elementwise/layout overhead without changing math order: view/materialization cleanup and possibly fusing strictly layout-only paths.
2. Inspect remaining GEMM internals, especially repeated activation conversion (`src1_to_f16`) and fp16 destination conversion (`dst_to_f32`) in cuBLAS fallback. TorchAO's advantage is still its resident packed layout avoiding repeated GGUF dequant/activation conversion in hot paths.
3. Avoid globally forcing f32 compute except for FC2 long-sequence shape; the quality fix is intentionally shape-scoped.

### 2026-08-07 FC1 fp16-only experiment

A new split switch was added for diagnosis:

- `ED_MINIMAX_H3_FC1_FP16_CUBLAS=1` enables fp16 Tensor Core cuBLAS for H3 MLP `fc1` only.
- `ED_MINIMAX_H3_FC2_FP16_CUBLAS=1` enables the analogous path for `fc2` only.
- `ED_MINIMAX_H3_MLP_FP16_CUBLAS=1` still enables both and remains experimental.

56-frame, 5-step FL2VA probe with `GGML_CUDA_SM90_Q4K_CUBLAS=1`:

| Variant | Diffusion | Output | Artifact |
|---|---:|---|---|
| Baseline FC1 f32 path | `6.853s` | non-black | `outputs/minimax-h3/perf-quality-fc1-fp16-only-56f5s-2026-08-07/baseline/final.mp4` |
| `ED_MINIMAX_H3_FC1_FP16_CUBLAS=1` | `6.693s` | non-black but diverges | `outputs/minimax-h3/perf-quality-fc1-fp16-only-56f5s-2026-08-07/fc1_fp16/final.mp4` |

Frame PSNR baseline-vs-FC1-fp16 over 56 frames: min `25.034dB`, average `33.162dB`, max `44.824dB`.

Conclusion: FC1 fp16-only is not suitable as a default quality-preserving optimization. It saves only about `2.3%` diffusion time on this probe while causing visible numerical divergence. Keep it opt-in only.

### 2026-08-07 FC1 fp16 with f32 compute experiment

To test whether FC1 quality loss came only from fp16 accumulation/output, FC1 was rerun with `ED_MINIMAX_H3_FC1_FP16_CUBLAS=1` plus global `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` on the same 56-frame, 5-step FL2VA probe.

| Variant | Diffusion | Artifact |
|---|---:|---|
| Baseline FC1 f32 path | `6.821s` | `outputs/minimax-h3/perf-quality-fc1-fp16-f32compute-56f5s-2026-08-07/baseline/final.mp4` |
| FC1 fp16 inputs + f32 cuBLAS compute | `6.356s` | `outputs/minimax-h3/perf-quality-fc1-fp16-f32compute-56f5s-2026-08-07/fc1_f32compute/final.mp4` |

Frame PSNR baseline-vs-FC1-fp16-f32compute over 56 frames: min `24.403dB`, average `31.772dB`, max `44.441dB`.

Conclusion: FC1 divergence is not fixed by f32 accumulation alone. The lossy part is dequantizing the `Q4_K` FC1 weights and/or activations into fp16 for Tensor Core GEMM. FC1 fp16 should remain diagnostic-only; keep the quality-safe FC1 path on f32 SGEMM unless a resident int4 kernel/layout is implemented.

### 2026-08-07 RoPE tail-view experiment

A local H3 partial-RoPE experiment added `ED_MINIMAX_H3_ROPE_TAIL_VIEW=1`, which keeps the unrotated tail dimensions as a stride view instead of materializing the `[32, seq, heads]` tail before concatenating it with the rotated prefix.

56-frame, 5-step FL2VA probe with `GGML_CUDA_SM90_Q4K_CUBLAS=1`:

| Variant | Diffusion | Main graph tail-cont cost | Artifact |
|---|---:|---:|---|
| Baseline | `6.748s` | `62.469ms` over 4 DiT calls | `outputs/minimax-h3/perf-quality-rope-tail-view-56f5s-2026-08-07/baseline/final.mp4` |
| `ED_MINIMAX_H3_ROPE_TAIL_VIEW=1` | `6.666s` | eliminated from top nodes | `outputs/minimax-h3/perf-quality-rope-tail-view-56f5s-2026-08-07/tail_view/final.mp4` |

Frame PSNR baseline-vs-tail-view over 56 frames: min `31.263dB`, average `36.053dB`, max `45.292dB`.

Conclusion: the layout optimization does remove the targeted tail materialization, but the end-to-end diffusion improvement is small on this probe and the output is not pixel-equivalent. Keep `ED_MINIMAX_H3_ROPE_TAIL_VIEW=1` opt-in only; it is not enabled by default.
