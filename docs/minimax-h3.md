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
| FL2VA | Text prompt + first frame image + last frame image | Video + audio |

### Ref2VA checkpoint

| Component | File name | Source |
|---|---|---|
| Diffusion model | `minimax_h3_ref2va_pruned-Q4_K_M.gguf` | [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) |

Supported reference inputs:

| Input | CLI flag | Notes |
|---|---|---|
| Reference image | `--ref-image <image>` | Repeatable. Presented as `<Picture N>` to Qwen3-VL. |
| Reference video | `--ref-video <frame-dir>` | Repeatable. Directory of image frames sorted lexicographically; treated as 24 fps. |
| Paired video audio | `--ref-video-audio <wav>` | Repeatable. The Nth WAV is paired with the Nth `--ref-video`. |
| Standalone audio | `--ref-audio <wav>` | Repeatable. Independent audio reference, not attached to a video. |

The reference inputs can be combined freely within Ref2VA. Ref2VA cannot be used
with `--image` or `--end-img` in the same request.

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
  --image first_frame.png \
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
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
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
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
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
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
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

### Standalone reference audio

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --ref-audio reference_audio.wav \
  -p "Generate a video with audio inspired by <Audio 1>." \
  --cfg-scale 1 -W 864 -H 480 --fps 24 --video-frames 56 --steps 20 \
  --diffusion-fa --rng cpu --video-format mp4 \
  -o minimax_h3_ref_audio.mp4
```

### Mixed references

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va_pruned-Q4_K_M.gguf \
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

## Current limitations

- Ref2VA behavior is functional but still under quality alignment with
  `stable-diffusion.cpp`; matching seeds do not yet imply matching generated
  videos.
- Video reference conditioning can be strong. Use explicit prompts and visually
  inspect whether the output is following the intended reference semantics.
- Audio reference effects are easiest to judge with listening tests; contact
  sheets only validate the video stream.
