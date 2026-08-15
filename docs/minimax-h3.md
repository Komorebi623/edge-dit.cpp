# MiniMax-H3

edge-dit.cpp supports MiniMax-H3 video-and-audio generation through standalone
components: one FL2VA or Ref2VA diffusion model, one Qwen3-VL text/vision
encoder, the video VAE, and the optional audio VAE. CUDA is the validated
backend for this model family.

## Checkpoints and inputs

FL2VA and Ref2VA share Qwen3-VL and both VAEs, but require different diffusion
checkpoints.

| Checkpoint | Supported conditioning |
|---|---|
| FL2VA | Text; first frame; last frame; first and last frames |
| Ref2VA | Repeatable reference images, videos, paired video audio, and additional audio |

FL2VA uses `--image`/`--init-img` for the first frame and `--end-img` for the
last frame. Ref2VA uses the following options:

| Input | CLI option | Behavior |
|---|---|---|
| Image | `--ref-image <path>` | Repeatable; presented as `<Picture N>` |
| Image sizing | `--ref-image-size match|max` | `match` only downsizes to the output pixel area; `max` uses the 2048px-short-edge path (default) |
| Video | `--ref-video <path>` | Repeatable frame directory or `mp4`/`mov`/`mkv`/`webm`/`avi`; media files require `ffmpeg` |
| Paired audio | `--ref-video-audio <wav>` | The Nth WAV is paired with the Nth video and overrides embedded audio |
| Additional audio | `--ref-audio <wav>` | Repeatable; requires at least one image or video reference |

When `--ref-video` points to a media file, the CLI decodes it at 24 fps and
automatically extracts an embedded audio track. Explicit paired WAV files map
positionally to videos. Additional audio is numbered after paired or embedded
video audio. Ref2VA references cannot be combined with FL2VA first/last-frame
options, and audio-only Ref2VA requests are rejected.

## Model files

### Downloadable weights

The complete, unpruned checkpoints are recommended for the best quality.
edge-dit.cpp also supports pruned DiT weights in BF16 safetensors format. Both
full and pruned BF16 DiTs can be converted to Q8_0 GGUF with `ed-convert`, and
the resulting Q8_0 DiTs can be loaded directly. Performance and quality results
from pruned and full DiTs are not directly comparable.

| Precision | Component | File | Repository |
|---|---|---|---|
| Q4_K_M | FL2VA DiT | `minimax_h3_fl2va-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Q4_K_M | Ref2VA DiT | `minimax_h3_ref2va-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Q4_K_M | Qwen3-VL | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| BF16 | FL2VA DiT | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| BF16 | Ref2VA DiT | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| BF16, pruned | FL2VA DiT | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` | [`Comfy-Org/MiniMax-H3 diffusion models`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| BF16, pruned | Ref2VA DiT | `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` | [`Comfy-Org/MiniMax-H3 diffusion models`](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| BF16 | Qwen3-VL | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| FP16 | Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| FP32 | Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |

The official [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
Diffusers shard indexes are also accepted for BF16 transformer loading. Merged
Comfy-Org files are usually more convenient for standalone component commands.

Example downloads with the Hugging Face CLI:

```bash
hf download leejet/MiniMax-H3-GGUF \
  minimax_h3_fl2va-Q4_K_M.gguf minimax_h3_ref2va-Q4_K_M.gguf \
  qwen3vl_32b_minimax_h3-Q4_K_M.gguf --local-dir models/minimax-h3-q4

hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir models/minimax-h3
```

### Persistent Q8_0 GGUF

Q8_0 benchmark files are offline conversions of the full BF16 DiTs and Qwen3-VL,
not Comfy-Org INT8 ConvRot weights. A pruned BF16 DiT can be converted with the
same command when lower storage and memory usage are preferred. Convert once
with `ed-convert` instead of quantizing during every model load:

```bash
ed-convert --model models/minimax-h3/diffusion_models/minimax_h3_fl2va_bf16.safetensors \
  --type q8_0 --output models/minimax-h3-q8/minimax_h3_fl2va-Q8_0.gguf
ed-convert --model models/minimax-h3/diffusion_models/minimax_h3_ref2va_bf16.safetensors \
  --type q8_0 --output models/minimax-h3-q8/minimax_h3_ref2va-Q8_0.gguf
ed-convert --model models/minimax-h3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --type q8_0 --output models/minimax-h3-q8/qwen3vl_32b_minimax_h3-Q8_0.gguf
```

The same converter accepts an official transformer
`model.safetensors.index.json`; the resulting persistent GGUF is equivalent at
the selected quantization type and avoids repeated online conversion.

## Duration and frame count

MiniMax-H3 always generates at 24 fps. Its frame count must satisfy `17k + 5`,
for example `5`, `22`, `39`, `56`, `73`, `90`, `107`, or `124`.

Use `--video-duration <seconds>` for the convenient interface. The CLI converts
the requested duration at 24 fps and selects the nearest legal frame count. For
example, `--video-duration 5` resolves to 124 frames, or approximately 5.17
seconds. The resolved value is printed before generation. Use
`--video-frames <count>` when an exact legal frame count is required. The two
options are mutually exclusive.

## Usage

Set component paths for the desired precision. The video and audio VAE files are
shared by every precision and checkpoint.

```bash
# Q4_K_M example. Replace the DiT with the Ref2VA file for reference workflows.
DIT=models/minimax-h3-q4/minimax_h3_fl2va-Q4_K_M.gguf
LLM=models/minimax-h3-q4/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
VIDEO_VAE=models/minimax-h3/vae/minimax_h3_video_vae_fp16.safetensors
AUDIO_VAE=models/minimax-h3/vae/minimax_h3_audio_vae_fp32.safetensors
```

For BF16, point `DIT` and `LLM` to the downloaded BF16 safetensors. For Q8_0,
point them to the converted GGUF files.

### FL2VA

```bash
# Text to video and audio
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --prompt "A cinematic sunset over layered mountain ridges with quiet natural ambience." \
  --video-format mp4 --output t2va.mp4

# First frame
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" --image first.png \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --prompt "Starting from <Picture 1>, preserve the scene and add subtle natural motion." \
  --video-format mp4 --output i2va.mp4

# Last frame: use --end-img last.png
# First and last frames: use --image first.png --end-img last.png
```

### Ref2VA

Use the Ref2VA DiT for all commands in this section.

The ComfyUI MiniMax-H3 template uses the `res_multistep` sampler with the
`simple` sigma schedule. Add `--sampler res_multistep --scheduler simple` when
reproducing that workflow. The default remains `euler` with `discrete` for
compatibility with existing edge-dit.cpp commands.

```bash
# Image reference
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-image reference.png --video-duration 5 -W 864 -H 480 --steps 20 \
  --cfg-scale 1 --prompt "Use <Picture 1> as the strict visual reference." \
  --video-format mp4 --output ref-image.mp4

# MP4 reference; embedded audio is paired automatically when present
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-video reference.mp4 --video-duration 5 -W 864 -H 480 --steps 20 \
  --cfg-scale 1 --prompt "Preserve <Video 1> and its <Audio 1> throughout the result." \
  --video-format mp4 --output ref-video.mp4

# Frame directory with explicit paired audio
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-video reference-frames --ref-video-audio soundtrack.wav \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --prompt "Preserve <Video 1> and synchronize it with <Audio 1>." \
  --video-format mp4 --output ref-video-audio.mp4

# Mixed image, video, and additional audio
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --ref-image reference.png --ref-video reference.mp4 --ref-audio music.wav \
  --video-duration 5 -W 864 -H 480 --steps 20 --cfg-scale 1 \
  --prompt "Use <Video 1>, <Picture 1>, and every supplied audio reference in a natural transition." \
  --video-format mp4 --output ref-mixed.mp4
```

## Memory placement

The explicit component controls are `--dit-offload`,
`--text-encoder-offload`, and `--vae-offload`. For automatic placement, use
`--auto-fit --max-vram <GB>`. MiniMax-H3 independently places the DiT,
Qwen3-VL, video VAE, and audio VAE while retaining CUDA compute.

```bash
ed-cli --video --diffusion-model "$DIT" --llm "$LLM" \
  --vae "$VIDEO_VAE" --audio-vae "$AUDIO_VAE" \
  --auto-fit --max-vram 40 --video-duration 5 -W 864 -H 480 --steps 20 \
  --cfg-scale 1 --prompt "A cinematic sunset over layered mountain ridges." \
  --video-format mp4 --output auto-fit.mp4
```

For long Ref2VA jobs on GPUs that can hold one major component at a time, add
`--minimax-h3-stage-lifecycle`. Qwen and the conditioning VAEs are released
after context/reference encoding, the DiT remains resident across all denoise
steps, and the decode VAEs are staged only after the DiT is released. The
encoded context remains on the GPU and DiT throughput is unchanged apart from
normal run-to-run variation.

`--max-vram` is a placement and graph-planning budget, not a hard CUDA process
limit. Backend workspaces and a single graph segment's activations can exceed
it; setting `--max-vram 24` on a larger GPU does not by itself prove that the
same workload fits a physical 24 GiB GPU.

MiniMax-H3 always uses its fixed `16x16` video-VAE tiling path. Generic
`--vae-tiling` and `--vae-tile-size` values do not replace this model-specific
layout.

## Ref2VA 15-second demo

The [Edge-DiT.cpp / ComfyUI / Diffusers comparison](optimization/minimax-h3-ref2va-edge-comfyui-diffusers-15s-demo-2026-08-16.md)
contains two vertically stacked 15-second videos, aligned inputs and sampling
parameters, stage timings, peak VRAM, lifecycle validation, a 24 GiB planning
budget experiment, and reproducible command/workflow entry points.

## H200 BF16 comparison

The tables below use one H200, resident components, `864x480`, 124 frames at
24 fps, 20 steps, CFG 1, and seed `424242`. edge-dit.cpp uses the complete
Comfy-Org BF16 DiT/Qwen files, FP16 video VAE, and FP32 audio VAE. Diffusers uses the
official complete BF16 DiT shards, BF16 Qwen3-VL, and its FP32 VAEs. “Generate”
excludes model loading, output muxing, and process cleanup. Peak VRAM is sampled
over the complete process. Values are `edge-dit.cpp / Diffusers`.

FL2VA does not use Ref2VA resize preprocessing:

| Task | Generate | Peak VRAM |
|---|---:|---:|
| Text | 51.396s / 53.986s | 125,051 / 128,801 MiB |
| First frame | 54.810s / 57.817s | 125,353 / 129,957 MiB |
| Last frame | 55.323s / 57.793s | 125,351 / 129,957 MiB |
| First + last frames | 58.747s / 61.405s | 125,573 / 130,587 MiB |

### Ref2VA

Current preprocessing follows Diffusers geometry: image short edge 2048;
video short edge 768 with a pre-rounding `768x1344` area cap; preserved aspect
ratio; dimensions rounded to multiples of 32; Lanczos resize. edge-dit.cpp is
faster than Diffusers in all four measured Ref2VA generation paths:

| Current task | Generate | edge-dit.cpp speedup | Peak VRAM |
|---|---:|---:|---:|
| Image | 126.915s / 136.656s | 1.08x | 130,445 / 139,253 MiB |
| MP4 video / video frames † | 182.649s / 183.921s | 1.01x | 132,661 / 137,161 MiB |
| Video frames + paired audio † | 182.667s / 185.035s | 1.01x | 132,663 / 137,183 MiB |
| Mixed references † | 301.435s / 319.435s | 1.06x | 138,113 / 141,915 MiB |

The image row is a strict same-image, same-prompt comparison. The
mixed-reference path has a clear 1.06x lead. The two video rows retain smaller
1.01x measured leads; they are marked † because the MP4 run lets edge-dit.cpp
extract embedded audio while the Diffusers run uses decoded video frames, and the
paired-audio prompts differ slightly. A locked-command rerun is required before
treating the approximately 1% margins as statistically significant.

## Limitations

- Reference adherence is prompt dependent. Name `<Picture N>`, `<Video N>`, and
  `<Audio N>` explicitly when their roles matter.
- Matching seeds across frameworks does not guarantee identical videos because
  numerical kernels and weight formats differ.
- Media-file reference decoding and embedded-audio extraction require `ffmpeg`.
- Without `--audio-vae`, video generation works but generated audio is not
  decoded or muxed.
