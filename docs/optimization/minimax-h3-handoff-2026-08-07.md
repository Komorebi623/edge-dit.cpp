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

### 2026-08-07 SageAttention CUDA dispatch validation

The first SageAttention comparison was not a valid CUDA-kernel measurement: the build cache had `ED_ENABLE_CUDA_SAGE_ATTN=OFF`, so `ED_SAGE_ATTN=1` only changed graph construction and the CUDA backend fell through to the normal attention path. The CUDA build was reconfigured with `-DED_ENABLE_CUDA_SAGE_ATTN=ON`, then rerun with the same one-GPU FL2VA probe (`864x480`, `56` frames, `5` steps, `cfg=1`, seed `42`, VAE tiling off, and `GGML_CUDA_SM90_Q4K_CUBLAS=1`).

| Variant | DiT diffusion | Result | Artifact |
|---|---:|---|---|
| Normal attention | `6.829s` | baseline | `outputs/minimax-h3/sage-real-kernel-baseline-56f5s-2026-08-07/final.mp4` |
| `ED_SAGE_ATTN=1` | `6.811s` | real fused CUDA dispatch, rejected | `outputs/minimax-h3/sage-real-kernel-56f5s-2026-08-07/final.mp4` |

The Sage log contains `1080` `dispatch: running fused INT8-QK+F16-PV kernel` lines, proving that the custom CUDA kernel ran. Its target shape was `d_head=64`, `L_q=L_k=1797`, and `HN=32`; this is a VAE attention shape, not the H3 DiT self-attention shape. The measured DiT difference is only `18ms` over four calls (within normal run variation), while the generated-video comparison gives min/average/max PSNR `24.906/31.876/44.398dB`. Sampled optimized-frame mean/std values remain non-black (`114.95/54.04` through `166.02/52.02`), but the numerical divergence is too large for a default quality-safe optimization.

Conclusion: keep `ED_SAGE_ATTN` runtime-disabled by default. Do not treat SageAttention as a MiniMax-H3 DiT optimization unless it is redesigned for the DiT's `d=128`, `h=56`, `seq=7919` attention shape and passes the same quality gate. Contact sheet and machine-readable quality result: `outputs/minimax-h3/sage-real-kernel-56f5s-2026-08-07/contact_sheet.jpg` and `outputs/minimax-h3/sage-real-kernel-56f5s-2026-08-07/quality.txt`.

### 2026-08-07 cuDNN SDPA H3 dispatch confirmation

With `ED_PROFILE_CUDNN_SDPA=1`, a short one-call H3 probe confirms that the existing cuDNN SDPA backend already handles the DiT's long self-attention:

```text
b=1, h=56, sq=7919, sk=7919, d=128, scale=11.3137, dst_type=f32
```

At 16 accumulated calls the profile reported `46.847ms` execution, `1.698ms` Q conversion, and `2.286ms` output conversion, or roughly `2.93ms` execution per attention call. The much more numerous `d=64`, `sq=1797`, padded `sk=2048` VAE attention calls are intentionally rejected by the current padding-mask gate. Artifact: `outputs/minimax-h3/cudnn-sdpa-dispatch-probe-56f2s-2026-08-07/run.log`.

Conclusion: H3 DiT already uses cuDNN SDPA; attention is not the remaining dominant gap to Diffusers. Further H3 work should prioritize Q4_K FC1 kernel/layout efficiency and only pursue attention changes after a profile shows a material regression there.

### 2026-08-07 TF32 status and FC1 bottleneck check

The dominant quality-safe FC1 route is still `Q4_K -> FP32` dequantization followed by FP32 `cublasSgemm` (`m=28672`, `n=7919`, `k=5376`). Its 56-frame operator profile gives about `5.83ms` per FC1 projection; dequantization is only about `0.37ms` per projection. Caching dequantized FP32 FC1 weights would consume substantial persistent VRAM while having an upper-bound benefit of only about 6% of FC1 time (and roughly 1% of whole DiT time), so it is not the next worthwhile change.

An isolated one-call GPU diagnostic compared the default cuBLAS configuration with `NVIDIA_TF32_OVERRIDE=0`:

| Variant | DiT diffusion | Interpretation |
|---|---:|---|
| Default | `2.391s` | Hopper TF32 Tensor Core path active |
| TF32 explicitly disabled | `4.300s` | FP32 CUDA-core fallback, `~80%` slower |

The CUDA backend sets `CUBLAS_TF32_TENSOR_OP_MATH` when it creates the cuBLAS handle, so the current quality-safe FC1 SGEMM is already using Tensor Cores. A cuBLASLt rewrite would need cached descriptors/algorithms and a demonstrated advantage over this already-TF32-accelerated route before it is worth adding another runtime switch.

### 2026-08-07 cuBLASLt FC1 algorithm scan

A standalone H200 GPU microbenchmark measured the exact FC1 SGEMM shape (`m=28672`, `n=7919`, `k=5376`, FP32 inputs/output, TF32 compute) without changing repository code. Current `cublasSgemm` measured `5.299ms`. A `128MiB`-workspace cuBLASLt scan returned eight algorithms; the fastest measured `5.294ms`, while the others ranged from `5.383ms` to `6.625ms`.

Conclusion: there is no material cuBLASLt algorithm advantage for the current quality-safe FP32/TF32 FC1 path on H200. Do not add a descriptor cache or a runtime switch for this equivalent path. The remaining meaningful route to close the roughly 10% Diffusers gap is a resident packed int4 kernel/layout compatible with `Q4_K`, avoiding the FP32 SGEMM path itself; this is a larger backend project and must be separately quality-gated.

### 2026-08-07 Torch int4 packed-kernel feasibility check

The CUDA environment includes TorchAO/fbgemm SM90 int4 support. Its `aten._weight_int4pack_mm` accepts group size `32`, so the group granularity can in principle represent the GGUF Q4_K block's 32-value scale/min pairs. A real H3 FC1 shape probe (`M=7919`, `K=5376`, `N=28672`) completed successfully with BF16 activations and group size `32`, but measured `83.97ms` per matrix multiplication.

That is far slower than edge.cpp's current quality-safe Q4_K-to-FP32 TF32 SGEMM path (`~5.8ms` FC1 projection). The generic Aten packed kernel therefore is not the kernel/layout used by the faster Diffusers H3 benchmark, or is not intended for this extreme H3 matrix aspect ratio. It must not be integrated into edge.cpp.

The installed TorchAO CUTLASS int4 interfaces exposed in this environment are rowwise-scaled `s8s4/s4s4` paths rather than a direct reusable C++ library API for GGUF's asymmetric Q4_K blocks. A future high-value path needs either the exact TorchAO weight-only kernel/layout used by the Diffusers run or a purpose-built Hopper kernel; it cannot be obtained by calling the generic Aten int4-pack op.

### 2026-08-07 TorchAO v2 rowwise kernel follow-up

The preceding generic `aten._weight_int4pack_mm` result is not the operator used by the current Diffusers MiniMax-H3 TorchAO run. `Int4WeightOnlyConfig(group_size=128)` dispatches `fbgemm::bf16i4bf16_rowwise`, a CUTLASS SM90 Hopper Tensor Core kernel. A direct H200 probe at the exact H3 FC1 shape (`M=7919`, `K=5376`, `N=28672`) measured the following steady-state forward times:

| Group size | Scale/zero dtype | Result | Minimum time |
|---:|---|---|---:|
| `32` | BF16 | CUTLASS `can_implement` rejects it | n/a |
| `32` | FP32 | CUTLASS `can_implement` rejects it | n/a |
| `64` | BF16 | supported | `3.396ms` |
| `64` | FP32 | supported | `4.352ms` |
| `128` | BF16 | supported | `3.756ms` |
| `128` | FP32 | supported | `4.341ms` |

This corrects the earlier interpretation: the official TorchAO path is genuinely much faster than the current Edge quality-safe FC1 (`Q4_K -> FP32 -> TF32 SGEMM`, approximately `5.3-5.8ms` per FC1), and it is the right performance comparison. It is not directly reusable for the existing GGUF weights. Q4_K has eight independent 32-value groups in every 256-value block, with exact decode `q * (d * scale) - (dmin * min)`. Since fbgemm stores signed INT4, its exact rowwise representation requires `group_size=32`, `q_signed=q-8`, `scale=d*scale`, and `zero_point=8*scale-dmin*min`; the stock Hopper kernel rejects that group size even when its scale/zero inputs remain FP32.

Therefore the remaining roughly 10% end-to-end DiT gap is now attributable to a concrete format/kernel incompatibility, not to attention, cuBLAS algorithm selection, or a missed Tensor Core setting. A quality-safe improvement requires a dedicated SM90 mixed-dtype Q4_K kernel (or a modified CUTLASS/fbgemm mainloop that supports exact 32-value groups) plus resident packed Q4_K weights. Re-quantizing 32-value groups into the supported 64/128 format would change weights and is not an acceptable default optimization without a separate full-generation quality evaluation.

### 2026-08-08 SM90 32-group source-level feasibility follow-up

The `group_size=32` rejection is now traced to a specific fbgemm/CUTLASS mainloop restriction, not an undocumented Hopper limitation. In the installed `bf16i4bf16_rowwise` source, `TileShapeK` is `64` for BF16 mixed input. CUTLASS checks `group_size == K || group_size % TileShapeK == 0`; consequently stock `64/128` groups pass while exact Q4_K `32` groups fail.

A throwaway, out-of-tree prototype changed only `TileShapeK` from `64` to `32`. It was compiled for `sm_90a` after first confirming that a plain `sm_90` target cannot issue Hopper GMMA instructions. The valid-target compilation then failed in the CUTLASS mixed-input mainloop with the deliberate assertion `K_BLOCK_MAX >= 4`. This is expected: a 32-value threadblock K tile contains only two BF16 GMMA K-blocks (`16` elements each), while the pipelined mainloop requires at least four for its prefetch/commit schedule.

More importantly, the existing mainloop assumes only one scale/zero row per threadblock K tile. It calculates `reload_factor = ceil(group_size / TileShapeK)`, TMA-loads one scale/zero row for each tile, and `copy_tensors_MK` copies that metadata only when `k_block == 0`. Simply removing the divisibility assertion would therefore silently apply one 32-value Q4_K scale/min to the other 32 values in a 64-value tile; the source explicitly warns that relaxing the related scale-layout assertion without changing transaction accounting can hang.

The correct implementation scope is now concrete:

1. Preserve a practical K tile (`64` or preferably `128`) for Hopper GMMA occupancy.
2. Extend the mixed-input scale shared-memory layout, TMA transaction bytes, and producer reload logic to load two/four scale-plus-zero rows per K tile.
3. Select the correct 32-value row inside `dequantize_A_kblock` for each GMMA K-block, while retaining FP32 Q4_K-derived scale/zero values where numerical equivalence requires it.
4. Add a resident Q4_K-to-rowwise packed-weight cache, including Q4_K's interleaved nibble reorder, so this conversion is not performed per DiT call.

This is a real kernel/mainloop extension rather than a safe one-line fbgemm configuration change. No repository inference code was changed by the prototype. The prototype source/build output lives only under container `/tmp/q4k_g32_proto`; its failed build and the earlier `sm_90` invalid execution are diagnostic only and must not be used as performance or quality data.

### 2026-08-08 Hopper W4A8 group-32 feasibility result

CUTLASS has an upstream mixed-input W4A8 route that is materially closer to GGML MMQ than the Diffusers BF16×W4 route. The official CUTLASS `v3.5.0` Hopper mixed-dtype example was built out of tree for `sm_90a`, then changed exactly as documented by CUTLASS maintainers in issue `#1332`: INT8 activation, INT4 weight, `TileShapeK=32`, and INT32 accumulation/output. It passed the example reference check both on a small shape and on the exact H3 FC1 shape with `group_size=32`:

| Shape (`M x N x K`) | Group size | Result | Average time |
|---|---:|---|---:|
| `128 x 512 x 1024` | `32` | passed | `0.0173ms` |
| `7919 x 28672 x 5376` | `32` | passed | `3.388ms` |

This is important evidence that H200 Hopper hardware and CUTLASS's W4A8 GMMA path can efficiently execute a 32-value group. The stock BF16×W4 fbgemm dispatch rejects 32 only because its BF16 tile-K is 64; its 32-tile experiment is not evidence against the W4A8 route.

It still cannot be substituted for GGML Q4_K MMQ unchanged. The mixed-input collective accepts scale/zero metadata for only the narrow INT4 operand. GGML MMQ has two independently quantized operands for Q4_K projections:

```text
weight[g, n, k]     = q4[g, n, k] * weight_scale[n, k_group]
                    - weight_min[n, k_group]
activation[m, k]    = q8[m, k] * activation_scale[m, k_group]
```

The product needs both per-output-channel Q4_K scale/min and per-token Q8_1 scale. The existing W4A8 collective can apply the former before INT8 MMA, but has no second activation-side 32-value scale input. Pre-multiplying them is impossible because the required scale product varies across both `m` and `n`; ignoring the activation scale would change the model result. Therefore this benchmark proves a promising kernel foundation and performance target, not a quality-safe drop-in replacement.

The next exact route is a dual-quantized W4A8 Hopper kernel: retain CUTLASS's group-32 W4A8 GMMA pipeline, add an activation `Q8_1` scale stream and accumulate the Q4_K min correction using the activation block sum (which GGML already stores). This directly mirrors GGML's Q4_K×Q8_1 algebra and avoids BF16 re-quantization. It needs an independent CUDA prototype plus numerical comparison against the existing MMQ output before any edge.cpp integration.

### 2026-08-08 W4A8 group-32 semantic caveat

The W4A8 `3.388ms` result is a hardware-throughput feasibility result only, not yet a numerically compatible Q4_K implementation. Reading the v3.5 collective shows that its scale/zero path upcasts narrow INT4 values to the scale type, applies `q * scale + zero`, **then converts the result to the MMA type**. In the maintained W4A8 configuration the MMA type is INT8, so group-scale values are rounded into INT8 before GMMA. Its reference path uses the same conversion and therefore correctly validates that specific integer GEMM, but does not validate the floating-point Q4_K decode used by Edge.

For Q4_K, `d * scale_group` and `dmin * min_group` are FP16-derived floating values and the existing GGML MMQ accumulates their contribution in floating point. Feeding them through the stock W4A8 converter would change the decoded weight before the dot product. This is not acceptable for a default generation path, even though the raw `group=32` GMMA timing is attractive.

The exact custom-kernel requirement is consequently narrower and stricter: use the W4A8 GMMA instruction only for the integer dot product, retain Q4_K and Q8_1 scales/min/sums in FP16/FP32 registers, and apply

```text
output += activation_scale * (weight_scale * dot_int8_int4 - weight_min * sum_int8)
```

in floating point for every 32-value group. That is closer to GGML's existing `vec_dot_q8_1_q8_1_mma` algebra than to stock CUTLASS's scale-and-convert collective. No quality claim or speed projection from the stock W4A8 scale path should be used until this exact post-MMA scaling prototype is measured.

### 2026-08-08 actual H3 Q4_K repacking error check

A read-only mmap sample of the actual `blocks.0.mlp.fc1.weight` H3 GGUF tensor (`256 x 5376` values) was decoded with the same Q4_K scale/min and nibble rules as GGML. It was then independently affine-requantized into ordinary unsigned int4 groups. This evaluates the *additional* distortion after the existing Q4_K quantization, not the original BF16-model quantization error.

| Repacked affine group size | Additional weight RMSE | Maximum absolute error | Weight SNR |
|---:|---:|---:|---:|
| `32` | `0.00554` | `0.04049` | `27.60dB` |
| `64` | `0.01098` | `0.04496` | `21.66dB` |
| `128` | `0.01330` | `0.05068` | `19.99dB` |

The actual sampled Q4_K decoded weights have RMS `0.13290` and range `[-1.0334, 0.9273]`. Even a fresh 32-value ordinary affine representation is not lossless because Q4_K's stored 6-bit superblock scale/min construction is different; the 64/128 representations accepted by stock fbgemm add materially more error.

Conclusion: converting current H3 Q4_K weights into a TorchAO/fbgemm-compatible 64/128-group format is not a responsible default performance optimization. It can only be considered as an explicitly experimental, full-video quality-gated alternative. The quality-safe route remains an exact Q4_K×Q8_1 Hopper kernel with FP16/FP32 post-MMA scale/min handling.

### 2026-08-08 signed-int4 mapping correction and TF32×INT4 check

The fbgemm/CUTLASS `int4b_t` storage was probed directly. It is signed two's-complement INT4: nibble `0..7` decodes to `0..7`, while `8..15` decodes to `-8..-1`. Its affine operation is `signed_q * scale + zero_point` (the zero-point is added, not subtracted).

The exact static Q4_K weight mapping is therefore:

```text
q_signed   = q4_unsigned - 8
scale       = d * scale_group
zero_point  = 8 * scale - dmin * min_group
weight      = q_signed * scale + zero_point
```

A GPU probe using the supported fbgemm group-64 path confirmed this mapping; against a BF16-activation FP32 reference, the remaining output RMSE was `0.00858` and is attributable to fbgemm's BF16 output storage, rather than an affine-mapping error. The prior text that stated `zero=dmin*min` was incomplete because it omitted this signed-nibble offset; it is superseded by the formula above.

A second out-of-tree prototype replaced fbgemm's BF16 activation/output types with F32 to look for a fused `TF32×INT4` analogue of the current quality-safe `Q4_K -> F32 -> cublasSgemm` path. It failed at CUTLASS mixed-input layout instantiation before launch. Searching SM90 GMMA definitions shows native integer S8/U8×S8/U8 operations and FP16/BF16/TF32 homogeneous operations, but no Hopper TF32×INT4 GMMA operation. This is a hardware instruction-set boundary rather than a missed cuBLAS setting.

Consequences:

- Stock BF16×INT4 fused kernels can represent Q4_K weights after signed-nibble packing, but necessarily convert the activation/weight computation to BF16 and produce BF16 output; this must remain opt-in until full video quality validation.
- The current F32/TF32 cuBLAS route remains the quality-safe implementation because Hopper has no direct TF32×INT4 MMA to fuse exact F32 Q4_K decoding into that GEMM.
- An exact fast path must instead retain the existing Q8_1 integer activation representation and implement Q4_K/Q8_1 scale/min/sum correction around Hopper integer GMMA, as described in the dual-quantized route.

### 2026-08-08 TorchAO W4A8 and Marlin reuse audit

Two additional TorchAO CUDA paths were audited to avoid overlooking an existing dual-scale implementation:

1. `rowwise_scaled_linear_cutlass_s8s4` has an INT32 accumulator and a CUTLASS epilogue visitor that multiplies a per-token FP32 input scale and per-output-channel FP32 weight scale. Its public schema is `input_scale[M]`, `weight_scale[N]`. It has no K-group scale/min dimension, so it cannot express Q4_K's `weight_scale[K_group,N]` or `weight_min[K_group,N]`. The installed `torchao==0.15.0` wheel also omitted its CUDA implementation; its source is useful only as an epilogue-visitor reference.

2. `marlin_qqq_gemm` accepts `s_tok[M]`, `s_ch[N]`, and `s_group[K/group,N]`, which initially appears closer. Its actual checks accept only `group_size=128`; its group dequantization is symmetric INT4 (`q-8`) with a scale only, no per-group zero/min. It uses Ampere-style `mma.m16n8k16` INT8 Tensor Core instructions, not Hopper GMMA. Like the CUTLASS W4A8 group-scale path, it folds the group scale into an integer operand before MMA, so it cannot preserve Q4_K's FP scale/min arithmetic. It is not a quality-safe reuse target.

The exact Q4_K×Q8_1 path therefore has no available stock TorchAO/fbgemm/Marlin operator. It requires a dedicated kernel that uses S8×S8 GMMA to obtain a separate INT32 partial for each 32-value group, then applies the FP correction `activation_scale * (weight_scale * dot - weight_min * sum_q8)` before adding to the FP output accumulator. This is the remaining kernel-design task, not a dispatcher or layout flag.

### 2026-08-08 mid-sequence FC2 TF32 routing experiment

The existing SM90 Q4_K cuBLAS route already uses FP32 decode plus TF32 Tensor Cores for the H3 mid-sequence QKV and FC1 projections. The `n=7919` FC2 projections were intentionally left on exact GGML MMQ because the prior long-sequence FC2 route was quality-sensitive. A new **default-off** diagnostic switch was added:

```text
GGML_CUDA_SM90_Q4K_CUBLAS_MID_FC2=1
```

It routes only these two H3 Q4_K shapes through the existing FP32/TF32 cuBLAS path:

```text
m=5376, n=7919, k=14336
m=5376, n=7919, k=7168
```

On the fixed-seed FL2VA probe (`864x480`, 56 frames, 5 steps, cfg=1, four DiT calls), the FC2 matmul timings and end-to-end DiT time improved materially:

| Metric | Current quality-safe route | Mid-FC2 experiment |
|---|---:|---:|
| FC2 `k=14336` | MMQ `6.65ms` | FP32/TF32 cuBLAS `3.11ms` |
| FC2 `k=7168` | MMQ `3.33ms` | FP32/TF32 cuBLAS `1.59ms` |
| Four-call DiT diffusion | `6.666s` | `5.844s` |

The experimental video is non-black and structurally coherent, but it does **not** satisfy the default quality gate. Decoded fixed-seed frames against the current MMQ-FC2 baseline had mean PSNR `32.70dB`, minimum `25.19dB` (first/middle/last: `44.74/33.72/40.37dB`). The contact comparison is `outputs/minimax-h3/exp-mid-fc2-f32-tf32-56f5s-2026-08-08/contact_sheet_compare.jpg`.

Conclusion: retain the switch solely as a performance diagnostic. Diffusion amplifies the numerical difference between MMQ's exact Q4_K×Q8_1 accumulation and Q4_K-to-FP32 TF32 SGEMM even though both outputs look plausible. Do not enable it by default; the exact integer-GMMA Q4_K route remains the quality-preserving performance path.

### 2026-08-08 mid-sequence FC2 full-task validation

The mid-sequence FC2 experiment was validated on the full FL2VA comparison task, not only the 56-frame smoke task: `864x480`, `124` frames, `20` requested steps / `19` actual forwards, `cfg=1`, seed `42`, VAE tiling off, identical prompt and first/last frames.

The matching Diffusers run was recovered from the **local** source tree shipped alongside the model, rather than downloading source code:

```text
PYTHONPATH=/models/MiniMax-H3-Diffusers/code/diffusers-main/src
diffusers source revision: 53e5b00
```

This is the H3-capable `diffusers 0.40.0.dev0` source. The container-global PyPI package is `diffusers==0.39.0`, which does not contain `MiniMaxH3Transformer3DModel`; no model weight download is required to restore the benchmark environment.

| Path | DiT / transformer time | Calls | Per-call | End-to-end / generation metric |
|---|---:|---:|---:|---:|
| Edge Q4_K quality-safe baseline | `62.367s` | `19` | `3.282s` | `85.129s` generation |
| Edge Q4_K mid-FC2 TF32 experiment | `60.766s` | `19` | `3.198s` | `87.663s` generation |
| Diffusers TorchAO int4 resident | `55.680s` | `19` | `2.930s` | `61.646s` generate CUDA / `102.121s` process end-to-end |

The experiment improves Edge DiT by `1.601s` (`2.57%`) versus the quality-safe baseline, but remains `5.086s` (`9.14%`) slower than the restored Diffusers resident transformer benchmark. Its full-task decoded-frame comparison against the Edge MMQ-FC2 baseline measured mean PSNR `42.38dB`, minimum `40.69dB`; frames are visually coherent but not numerically equivalent. Artifacts:

- `outputs/minimax-h3/exp-mid-fc2-f32-tf32-124f20s-2026-08-08/final.mp4`
- `outputs/minimax-h3/exp-mid-fc2-f32-tf32-124f20s-2026-08-08/contact_sheet_compare.jpg`
- `outputs/minimax-h3/diffusers-fl2va-480p124-int4-resident-profile4-gpu1-2026-08-08/final.json`

Conclusion remains unchanged: the switch is a diagnostic only and must not become the default Q4_K route. It quantifies the maximum gain available from switching the remaining FC2 projections to dequantize-plus-TF32 SGEMM, leaving an exact Q4_K packed kernel as the only route to close the remaining quality-safe gap.

### 2026-08-08 cuDNN audio VAE transposed-convolution path

The MiniMax-H3 audio VAE's BigVGAN decoder uses seven repeated F32 `ConvTranspose1D` upsampling layers per audio stream. The previous CUDA implementation used one output element per thread and iterated over the input sequence and channels directly. This made audio decode a major end-to-end cost despite the DiT being relatively close to Diffusers.

A guarded cuDNN backward-data-convolution route now maps the supported 1D transposed convolution form to NCHW 2D (`H=1`) and keeps the original CUDA kernel as the fallback. It only accepts contiguous F32 tensors with one group, zero padding, dilation one, and stride greater than one—the exact form used by H3's BigVGAN decoder.

- Build integration: enabled whenever the existing `ED_ENABLE_CUDNN_SDPA` build option is enabled.
- Runtime switch: `ED_CUDNN_CONV_TRANSPOSE_1D=1`.
- Default: disabled; unset or `0` always uses the original CUDA kernel.
- No new third-party dependency: uses the already-linked cuDNN runtime.

Full fixed-seed FL2VA validation used `864x480`, `124` frames, `24fps`, `20` requested steps (`19` DiT forwards), `cfg=1`, seed `42`, VAE tiling off, `--diffusion-fa`, and the first/last images `assets/minimax-h3-ref2va/video-frames/001.png` and `056.png`. Prompt:

```text
Create a smooth cinematic transition between the supplied first and last frame, preserving the subject, lighting, and composition with natural coherent motion and synchronized ambient audio.
```

| Variant | Generation | Wall total | Video comparison | Audio comparison |
|---|---:|---:|---|---|
| Original CUDA transposed convolution | `85.974s` | `94.626s` | baseline | baseline |
| cuDNN path (`ED_CUDNN_CONV_TRANSPOSE_1D=1`) | `78.063s` | `86.443s` | PSNR `inf` (pixel-identical decoded frames) | same `331,776` samples; PSNR `48.109dB`, correlation `0.999796` |

This saves `7.911s` generation time (`9.20%`) without changing the generated video frames. The retained artifacts are:

- `outputs/minimax-h3/audio-convt1d-baseline-124f20s-2026-08-08/final.mp4`
- `outputs/minimax-h3/audio-convt1d-cudnn-124f20s-2026-08-08/final.mp4`
- `outputs/minimax-h3/audio-convt1d-cudnn-124f20s-2026-08-08/quality/audio-comparison.json`
- `outputs/minimax-h3/audio-convt1d-cudnn-124f20s-2026-08-08/quality/video-psnr-summary.log`

### 2026-08-08 opt-in CUDA Graph capture

The CUDA build previously compiled without CUDA Graph support, so repeated H3 video-VAE tile graphs (each `2491` graph nodes) were directly launched for every tile. CUDA Graph support is now compiled by default when `ED_ENABLE_CUDA_GRAPHS=ON`, but remains runtime-disabled unless explicitly requested:

```bash
ED_CUDA_GRAPHS=1 ed-cli ...
```

`GGML_CUDA_DISABLE_GRAPHS` remains an emergency override. The runtime gate is global because the backend captures compatible graph executions, not a model-specific arithmetic path. It does not alter weights, operators, numerical precision, or sampling.

A full FL2VA comparison used the same configuration as the cuDNN audio-VAE validation: `864x480`, 124 frames, 24fps, 20 requested steps / 19 DiT forwards, cfg `1`, seed `42`, VAE tiling on, `--diffusion-fa`, and `ED_CUDNN_CONV_TRANSPOSE_1D=1` in both variants.

| Variant | Generation | Wall total | Video | Audio |
|---|---:|---:|---|---|
| CUDA Graph runtime disabled | `78.063s` | `86.443s` | baseline | baseline |
| `ED_CUDA_GRAPHS=1` | `77.881s` | `86.504s` | PSNR `inf` | exact waveform (PSNR `inf`) |

The path is numerically exact on this task, but the end-to-end win is only `0.182s` (`0.23%`). It is retained as an independent opt-in scheduling control rather than enabled by default. The repeated VAE tile graph did capture and replay successfully; the limited end-to-end result means the primary remaining gap is compute, not launch overhead.

Artifacts:

- `outputs/minimax-h3/cuda-graphs-124f20s-2026-08-08/final.mp4`
- `outputs/minimax-h3/cuda-graphs-124f20s-2026-08-08/quality/video-psnr-summary.log`
- `outputs/minimax-h3/cuda-graphs-124f20s-2026-08-08/quality/audio-comparison.json`

### 2026-08-08 opt-in H3 video output postprocess fast path

MiniMax-H3 video decode returns a contiguous host `Tensor<float>` in `[width, height, frames, channels, batch]` storage order. The original final conversion loop used `Tensor::index(...)` for every output channel, constructing and bounds-checking an index vector per value. This is outside the VAE math but was a measurable end-to-end CPU postprocess cost at 480p.

The guarded fast path reads the equivalent contiguous offset directly while retaining the existing float clamp and `round(... * 255)` conversion. It is disabled by default:

```bash
ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS=1 ed-cli ...
```

For debugging the direct offset can be checked against every original `Tensor::index(...)` lookup during a run:

```bash
ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS=1 \
ED_MINIMAX_H3_VERIFY_FAST_VIDEO_POSTPROCESS=1 ed-cli ...
```

The verification run completed without an assertion, establishing that the two paths read identical source floats. The full FL2VA validation used the same fixed task as the current performance baseline: `864x480`, `124` frames, `24fps`, `20` requested steps (`19` DiT forwards), cfg `1`, seed `42`, first/last frames `001.png`/`056.png`, `--diffusion-fa`, `GGML_CUDA_SM90_Q4K_CUBLAS=1`, and `ED_CUDNN_CONV_TRANSPOSE_1D=1`.

| Variant | Generation | DiT | Video VAE | Video conversion | Video quality | Audio quality |
|---|---:|---:|---:|---:|---|---|
| Original conversion | `77.629s` | `62.454s` | `10.886s` | `1.960s` | baseline | baseline |
| Fast conversion | `78.417s` | `62.307s` | `12.683s` | `1.048s` | PSNR `inf` | PCM exact, PSNR `inf` |

The VAE phase has normal run-to-run variation (the second run was about `1.8s` slower in VAE compute), so the isolated conversion measurement is the relevant result: `0.912s` saved for this full task (`46.5%` of conversion time). The decoded 124-frame videos are pixel-identical, and decoded audio contains `331,776` identical samples. This does not affect GPU memory or model arithmetic.

Artifacts are intentionally local and ignored by Git:

- `outputs/minimax-h3/fast-video-postprocess-full-124f20s-2026-08-08/baseline/final.mp4`
- `outputs/minimax-h3/fast-video-postprocess-full-124f20s-2026-08-08/fast/final.mp4`
- `outputs/minimax-h3/fast-video-postprocess-full-124f20s-2026-08-08/quality/video-psnr-summary.log`
- `outputs/minimax-h3/fast-video-postprocess-full-124f20s-2026-08-08/quality/audio-comparison.json`

### 2026-08-08 opt-in parallel H3 video output conversion

The direct H3 output conversion path above can now parallelize independent decoded frames. It is only activated after the direct-offset path is explicitly enabled, and is otherwise absent from the normal execution path:

```bash
ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS=1 \
ED_MINIMAX_H3_FAST_VIDEO_POSTPROCESS_THREADS=8 ed-cli ...
```

The thread count is bounded by the number of output frames; unset, invalid, or `1` leaves the single-threaded direct-offset path in use. Worker threads only read the decoded host float tensor and write their own `ed_image_t` frame allocation, so it neither changes model arithmetic nor allocates GPU memory.

The full fixed-seed FL2VA validation used the same `864x480`, 124-frame, 20-step/cfg-1 task and runtime options as the direct-offset validation. The comparison uses the single-thread direct-offset run as the reference:

| Variant | Generation | DiT | Video VAE | Video conversion | Video quality | Audio quality |
|---|---:|---:|---:|---:|---|---|
| Single-thread direct conversion | `78.417s` | `62.307s` | `12.683s` | `1.048s` | baseline | baseline |
| Parallel direct conversion (`8` threads) | `75.420s` | `62.173s` | `10.813s` | `0.119s` | PSNR `inf` | PCM exact |

The conversion phase improves by `0.929s` (`88.6%`) compared to the already optimized single-thread path and by `1.841s` compared to the original indexed path. The complete run was additionally faster by `2.997s`, though only the conversion saving should be attributed to this change because video-VAE timing varies modestly between cold model loads. Decoded video has infinite PSNR for all 124 frames and the `331,776` decoded audio samples match byte-for-byte.

Artifacts:

- `outputs/minimax-h3/parallel-video-postprocess-full-124f20s-2026-08-08/parallel/final.mp4`
- `outputs/minimax-h3/parallel-video-postprocess-full-124f20s-2026-08-08/quality/video-psnr-summary.log`
- `outputs/minimax-h3/parallel-video-postprocess-full-124f20s-2026-08-08/quality/serial.f32le`
- `outputs/minimax-h3/parallel-video-postprocess-full-124f20s-2026-08-08/quality/parallel.f32le`
