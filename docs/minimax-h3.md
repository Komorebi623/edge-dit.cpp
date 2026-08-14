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

The complete, unpruned checkpoints are recommended. The smaller files whose
names contain `pruned` are loadable but are not directly comparable with the
official full DiTs.

| Precision | Component | File | Repository |
|---|---|---|---|
| Q4_K_M | FL2VA DiT | `minimax_h3_fl2va-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Q4_K_M | Ref2VA DiT | `minimax_h3_ref2va-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Q4_K_M | Qwen3-VL | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | [`leejet/MiniMax-H3-GGUF`](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| BF16 | FL2VA DiT | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| BF16 | Ref2VA DiT | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3) |
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
not Comfy-Org INT8 ConvRot weights. Convert once with `ed-convert` instead of
quantizing during every model load:

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

Flash attention is enabled by default; use `--no-flash-attention` only for
diagnosis. The legacy `--rng` option is accepted for command compatibility but
does not select a different RNG, so it is intentionally omitted below.

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

MiniMax-H3 always uses its fixed `16x16` video-VAE tiling path. Generic
`--vae-tiling` and `--vae-tile-size` values do not replace this model-specific
layout.

## H200 BF16 comparison

The tables below use one H200, resident components, `864x480`, 124 frames at
24 fps, 20 steps, CFG 1, and seed `424242`. Edge uses the complete Comfy-Org
BF16 DiT/Qwen files, FP16 video VAE, and FP32 audio VAE. Diffusers uses the
official complete BF16 DiT shards, BF16 Qwen3-VL, and its FP32 VAEs. “Generate”
excludes model loading, output muxing, and process cleanup. Peak VRAM is sampled
over the complete process. Values are `Edge / Diffusers`.

FL2VA does not use Ref2VA resize preprocessing:

| Task | Generate | Peak VRAM |
|---|---:|---:|
| Text | 51.396s / 53.986s | 125,051 / 128,801 MiB |
| First frame | 54.810s / 57.817s | 125,353 / 129,957 MiB |
| Last frame | 55.323s / 57.793s | 125,351 / 129,957 MiB |
| First + last frames | 58.747s / 61.405s | 125,573 / 130,587 MiB |

### Ref2VA resize correction

The legacy Edge snapshot below predates Diffusers-compatible reference resize.
It encoded smaller reference images/videos, so its apparent speed and memory
advantage does not represent equal conditioning geometry. Diffusers values are
shown only as the comparison baseline.

| Legacy task | Generate | Peak VRAM |
|---|---:|---:|
| Image | 54.858s / 136.656s | 125,245 / 139,253 MiB |
| Video frames | 114.843s / 183.921s | 130,029 / 137,161 MiB |
| Video frames + paired audio | 115.488s / 185.035s | 130,029 / 137,183 MiB |
| Mixed references | 121.558s / 319.435s | 130,029 / 141,915 MiB |

Current preprocessing follows Diffusers geometry: image short edge 2048;
video short edge 768 with a pre-rounding `768x1344` area cap; preserved aspect
ratio; dimensions rounded to multiples of 32; Lanczos resize. Repeating the same
BF16 tasks with aligned conditioning gives:

| Aligned task | Generate | Peak VRAM |
|---|---:|---:|
| Image | 140.827s / 136.656s | 130,379 / 139,253 MiB |
| Video frames | 232.816s / 183.921s | 131,225 / 137,161 MiB |
| Video frames + paired audio | 232.327s / 185.035s | 131,539 / 137,183 MiB |
| Mixed references | 386.005s / 319.435s | 137,191 / 141,915 MiB |

The two Ref2VA tables differ only in Edge reference preprocessing. They must not
be combined into one speed trend: the first is historical and geometrically
under-sized; the second is the cross-framework conditioning comparison.

## Limitations

- Reference adherence is prompt dependent. Name `<Picture N>`, `<Video N>`, and
  `<Audio N>` explicitly when their roles matter.
- Matching seeds across frameworks does not guarantee identical videos because
  numerical kernels and weight formats differ.
- Media-file reference decoding and embedded-audio extraction require `ffmpeg`.
- Without `--audio-vae`, video generation works but generated audio is not
  decoded or muxed.
