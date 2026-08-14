# MiniMax-H3

edge-dit.cpp supports MiniMax-H3 video + audio generation through the standalone
component-loading path. The implementation follows the MiniMax-H3 interface used
by `stable-diffusion.cpp`: one diffusion checkpoint, Qwen3-VL text/vision encoder,
video VAE, and optional audio VAE.

## Model files

MiniMax-H3 runs with four model components. The diffusion checkpoint changes by
mode; the encoder and VAEs are shared.

| Component | File name | Source |
|---|---|---|
| Qwen3-VL text/vision encoder | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |

### FL2VA checkpoint

| Component | File name | Source |
|---|---|---|
| Diffusion model | `minimax_h3_fl2va-Q4_K_M.gguf` | [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) |

Supported modes:

| Mode | Inputs | Output |
|---|---|---|
| T2VA | Text prompt | Video + audio |
| I2VA | Text prompt + first frame image | Video + audio |
| L2VA | Text prompt + last frame image | Video + audio |
| FL2VA | Text prompt + first frame image + last frame image | Video + audio |

### Ref2VA checkpoint

| Component | File name | Source |
|---|---|---|
| Diffusion model | `minimax_h3_ref2va-Q4_K_M.gguf` | [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) |

Use the complete `minimax_h3_ref2va-Q4_K_M.gguf` for quality comparisons. The
smaller `minimax_h3_ref2va-Q4_K_M.gguf` is loadable, but it is a pruned
checkpoint and is not directly comparable with the official full Ref2VA DiT.

Supported reference inputs:

| Input | CLI flag | Notes |
|---|---|---|
| Reference image | `--ref-image <image>` | Repeatable. Presented as `<Picture N>` to Qwen3-VL. |
| Reference video | `--ref-video <path>` | Repeatable. Accepts a frame directory or `mp4`/`mov`/`mkv`/`webm`/`avi`; media files are decoded at 24 fps with `ffmpeg`. |
| Paired video audio | `--ref-video-audio <wav>` | Repeatable. The Nth WAV replaces/sets audio for the Nth `--ref-video`. Without it, an input media file's embedded audio is extracted automatically. |
| Additional audio | `--ref-audio <wav>` | Repeatable, but must be combined with at least one reference image or video. Audio-only Ref2VA requests are rejected. |

Reference images, videos, paired video audio, and additional audio can be mixed.
Flags preserve command-line order. `--ref-video-audio` entries map positionally
to the leading `--ref-video` entries, so keep each video and its explicit WAV in
the same order. Ref2VA references cannot be combined with `--init-img`/`--image`
or `--end-img` in the same request.

MiniMax-H3 control frames are not supported for either checkpoint.

## Common parameters

MiniMax-H3 is a 24 fps audio-video model. The usual full-resolution setting is:

```bash
-W 864 -H 480 --fps 24 --video-frames 56 --cfg-scale 1.0 --diffusion-fa --rng cpu
```

The frame count must satisfy `17k + 5` (for example `5`, `22`, `39`, `56`). The
examples below use 56 frames and 20 steps.

`ed-cli` writes MiniMax-H3 output at 24 fps even if another `--fps` value is
provided, matching sd.cpp and preserving the model's audio-video timing.

Use `--video-format mp4` for H.264/AAC MP4 output when `ffmpeg` is available.
Without `--audio-vae`, the model can still generate video, but no generated audio
is decoded or muxed.

## FL2VA usage

### Text to video + audio

```bash
ed-cli --video \
  --diffusion-model minimax_h3_fl2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "A cat surfs on an ocean wave with upbeat surf-rock music." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_t2va.mp4
```

### First-frame image to video + audio

```bash
ed-cli --video \
  --diffusion-model minimax_h3_fl2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --init-img first_frame.png \
  -p "Animate the provided first frame into a cinematic shot with matching sound." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_i2va.mp4
```

### First + last frame to video + audio

```bash
ed-cli --video \
  --diffusion-model minimax_h3_fl2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --image first_frame.png --end-img last_frame.png \
  -p "Create a smooth transition from the first frame to the last frame with natural audio." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_fl2va.mp4
```

## Ref2VA usage

When using references, write the prompt so it explicitly mentions the reference
slots, such as `<Picture 1>`, `<Video 1>`, and `<Audio 1>`. This makes reference
adherence easier to evaluate and avoids text-only prompts dominating the output.

### Reference image

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ref-image reference.png \
  -p "Use the landscape and lighting from <Picture 1> to create a cinematic video." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_ref_image.mp4
```

### Reference video

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ref-video reference_frames \
  -p "Follow the camera motion and scene layout from <Video 1>." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_ref_video.mp4
```

### Reference video with paired audio

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ref-video reference_frames \
  --ref-video-audio reference_soundtrack.wav \
  -p "Follow the motion from <Video 1> and use its paired soundtrack as audio guidance." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_ref_video_audio.mp4
```

### Mixed references

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ref-image reference.png \
  --ref-video reference_frames \
  --ref-video-audio reference_soundtrack.wav \
  --ref-audio extra_audio.wav \
  -p "Use the style from <Picture 1>, motion from <Video 1>, the paired video soundtrack, and additional ambience from <Audio 2>." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_ref_mixed.mp4
```

In a mixed prompt, paired video audio is presented before the video as `<Audio 1>`.
Standalone `--ref-audio` entries are numbered after video-paired audio, so the
first standalone audio in the example is `<Audio 2>`.

## Memory placement

MiniMax-H3 supports the regular component offload flags:

```text
--dit-offload
--text-encoder-offload
--vae-offload
--vae-tiling on|off|auto
```

For automatic placement, pass `--auto-fit --max-vram <GB>`. The runtime measures
a conservative Ref2VA mixed-conditioning graph at the requested width, height,
and frame count, then independently places the DiT, Qwen3-VL, video VAE, and
audio VAE. `--max-vram` is a hard placement budget under `--auto-fit`; components
that do not fit are segmented and staged from host memory while CUDA remains the
compute backend.

```bash
ed-cli --video \
  --diffusion-model minimax_h3_fl2va-diffusers-Q8_0.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q8_0.gguf \
  --auto-fit --max-vram 40 --vae-tiling auto \
  -W 864 -H 480 --video-frames 124 --fps 24 --steps 20 \
  --cfg-scale 1 --diffusion-fa --rng cpu \
  -p "A cinematic sunset over layered mountain ridges." \
  --video-format mp4 -o minimax_h3_autofit.mp4
```

The explicit offload flags remain useful when placement must be fixed manually.
Do not combine them merely to emulate `--auto-fit`; automatic placement normally
chooses a safer component combination for the requested budget.

MiniMax-H3's video VAE always uses its model-specific fixed `16x16` spatial
tiling path. This intentionally overrides the generic `--vae-tiling` state and
`--vae-tile-size`; those flags continue to control other VAE implementations.
The internal override can be disabled only with the diagnostic environment
variable `ED_MINIMAX_H3_DISABLE_FORCED_VAE_TILING=1`, which is not recommended
for normal generation.

## H200 performance

The following Edge measurements use CUDA on one H200, `864x480`, 124 frames,
24 fps, 20 steps, CFG 1, CPU RNG, seed `424242`, resident components, and warmed
model files. Wall time includes load, generation, mux/save, and cleanup.

| Precision | T2VA | I2VA | L2VA | FL2VA |
|---|---:|---:|---:|---:|
| BF16 wall / peak VRAM | 71s / 125,051 MiB | 73s / 125,251 MiB | 81s / 125,247 MiB | 76s / 125,471 MiB |
| Q8_0 wall / peak VRAM | 81.1s / 71,393 MiB | 84.5s / 71,585 MiB | 84.6s / 71,583 MiB | 90.0s / 71,805 MiB |
| Q4_K_M wall / peak VRAM | 73.6s / 47,459 MiB | 73.1s / 47,609 MiB | 73.0s / 47,607 MiB | 76.7s / 47,785 MiB |

Use persistent GGUF files for Q8/Q4 production runs. Loading official BF16
shards with `--tensor-type-rules ...=q8_0` performs quantization on every start;
in the measured FL2VA run that increased load from about 9.3s to about 45s.
On SM90, the optimized MiniMax-H3 route applies only to named main-block QKV/FC1
projections; it does not alter unrelated models.

`--auto-fit` was also validated with the same Q8_0 components at two sampling
steps. Every observed peak stayed below its requested cap:

| `--max-vram` | DiT | Qwen3-VL | Video/Audio VAE | Peak VRAM | Wall |
|---:|---|---|---|---:|---:|
| 120 GB | resident | resident | resident / resident | 71,397 MiB | 28.7s |
| 60 GB | resident | offloaded | offloaded / offloaded | 45,129 MiB | 39.8s |
| 40 GB | offloaded | offloaded | resident / resident | 31,699 MiB | 46.5s |
| 24 GB | offloaded | offloaded | offloaded / offloaded | 18,337 MiB | 50.6s |

The 40 GB configuration additionally passed FL2VA first+last-frame conditioning
at 31,865 MiB. A Ref2VA mixed image+video+paired-audio+additional-audio run passed
at a 60 GB budget with a 49,361 MiB peak. Full framework comparisons are in the
[BF16 benchmark](optimization/minimax-h3-fl2va-ref2va-edge-sdcpp-diffusers-h200-bf16-full-ref-2026-08-13.md),
[framework-native 8-bit benchmark](optimization/minimax-h3-fl2va-ref2va-edge-sdcpp-diffusers-h200-8bit-full-ref-2026-08-13.md),
and [Q4 benchmark](optimization/minimax-h3-fl2va-ref2va-edge-sdcpp-diffusers-h200-q4-full-ref-2026-08-14.md).

## Current limitations

- Audio-only Ref2VA is intentionally rejected; `--ref-audio` needs at least one
  `--ref-image` or `--ref-video`.
- Matching seeds across Edge, sd.cpp, and Diffusers do not guarantee identical
  videos because their quantization and numerical kernels differ.
- Reference adherence remains prompt dependent. Mention `<Picture N>`,
  `<Video N>`, and `<Audio N>` explicitly and visually inspect the result.
- Media-file `--ref-video` decoding and embedded-audio extraction require
  `ffmpeg`; use a frame directory plus `--ref-video-audio` when it is unavailable.
- Audio reference effects are easiest to judge with listening tests; contact
  sheets only validate the video stream.
