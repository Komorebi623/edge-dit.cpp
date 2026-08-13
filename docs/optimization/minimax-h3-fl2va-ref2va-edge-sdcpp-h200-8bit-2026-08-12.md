# MiniMax-H3 FL2VA/Ref2VA Edge vs sd.cpp H200 8-bit Benchmark（2026-08-12）

## 结论

- 本文主表现已覆盖为 2026-08-13 最终口径：Edge 使用已完成的内部 profile 八项；sd.cpp 使用官方 Comfy INT8 ConvRot DiT，重新以 `--eager-load` 全组件常驻方式串行运行八项。所有任务均为 `rc=0`。
- Edge 在 8 个任务上的外部 wall 均更短，相对 sd.cpp eager 加速为 `1.56x–2.23x`；最大加速来自 Ref2VA 混合参考。
- 全程最大单卡显存峰值为 `72,490 MiB`，来自 sd.cpp eager 的 FL2VA 尾帧任务。Edge 全部任务最大值为 `70,673 MiB`。
- sd.cpp eager 与旧 lazy 结果的 wall 基本相同，差异约在 `-5s–+2s`；常驻并没有带来可辨认的速度收益。它主要改变显存：例如 Ref2VA 视频帧参考从旧 lazy 的 `56,245 MiB` 上升至 eager 的 `62,595 MiB`。
- 两框架不是相同量化算法：Edge DiT 使用 GGML `Q8_0`，sd.cpp DiT 使用 Comfy `INT8 ConvRot`；Qwen、Video VAE、Audio VAE 相同。因此本表是“各框架可运行的 8-bit 路径”对比，不能把质量差异只归因于框架。
- Edge FL2VA 本轮仍为官方 Diffusers BF16 shards 加载时动态转 `Q8_0`，不是随后下载的 Comfy FL2VA BF16；对应结果不能标记为“Comfy BF16 转持久化 Q8_0”。
- 噪声根因不是 Q8 精度、Qwen 或 VAE：官方 Diffusers DiT 的 52 个 SwiGLU 输入投影半区顺序与原生/Comfy 权重相反，且 shards 省略了可推导的 `rope.inv_freq`。Edge 现已在量化前归一化 SwiGLU 布局，并在缺失时合成 16 个 FP32 RoPE 频率。
- Ref2VA 视频任务实际输入为同一帧目录，不是 MP4。最终视频帧与帧序列+配对音频任务使用字节完全一致的干净视觉 Prompt，唯一输入差异是后者增加 `--ref-video-audio`。
- 使用不提 Audio、不列举负面实体的同一 Prompt 完成 Edge 单变量 A/B 后，无音频和配对音频两组都严格保持公园帧内容，对输入前 56 帧的 SSIM 分别为 `0.314700` 和 `0.314327`，未观察到参考音频降低画面参考能力。
- 2026-08-13 使用 `ED_MINIMAX_H3_PROFILE=1` 按相同 H200 参数串行重跑 Edge 八项，补齐 Qwen context、关键帧/参考 VAE、noise、DiT、视频/音频解码等内部阶段时间；八项均为 `rc=0`，wall 与原结果基本一致。

## 固定口径

| 项目 | 口径 |
|---|---|
| GPU | NVIDIA H200，串行执行，`CUDA_VISIBLE_DEVICES=0` |
| Edge binary | [`build-cuda-refvideo-align-20260810/bin/ed-cli`](../../build-cuda-refvideo-align-20260810/bin/ed-cli) |
| sd.cpp binary | `/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-8bit-Benchmark/builds/stable-diffusion.cpp-cuda12.8-h200-bcc7e295/bin/sd-cli` |
| 分辨率与长度 | `864x480`、`124 frames`、`24 fps` |
| 采样参数 | `20 steps`、Euler、CFG `1`、seed `424242`、CPU RNG |
| 执行方式 | Edge 默认全组件常驻；sd.cpp `--eager-load` 全参数常驻；均无 CPU offload、启用 `--diffusion-fa`、GPU0 串行执行 |
| Profile | Edge `ED_MINIMAX_H3_PROFILE=1`；sd.cpp `-v` 原生阶段日志 |
| Edge FL2VA DiT | 官方 Diffusers BF16 shards，`--tensor-type-rules model.diffusion_model.=q8_0` 动态转 GGML Q8_0 |
| Edge Ref2VA DiT | 本地持久化 `minimax_h3_ref2va_pruned-Q8_0.gguf`；Comfy BF16 转 GGML Q8_0 |
| sd.cpp FL2VA DiT | Comfy `minimax_h3_fl2va_int8_convrot.safetensors`，直接加载 INT8 ConvRot |
| sd.cpp Ref2VA DiT | Comfy `minimax_h3_ref2va_pruned_int8_convrot.safetensors`，直接加载 INT8 ConvRot |
| Shared Qwen3-VL | Comfy BF16 转本地 `qwen3vl_32b_minimax_h3-Q8_0.gguf`，两框架使用同一文件 |
| Shared Video VAE | Comfy `minimax_h3_video_vae_fp16.safetensors`，FP16，不量化 |
| Shared Audio VAE | Comfy `minimax_h3_audio_vae_fp32.safetensors`，FP32，不量化 |
| Edge manifest/status | [`status.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/status.log) |
| sd.cpp eager manifest | [`manifest.txt`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/manifest.txt) |
| 正式全量 benchmark | `/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-8bit-Benchmark/run_h200_8bit_quality_benchmark.sh` |
| 本轮 sd.cpp-only benchmark | `/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-8bit-Benchmark/run_h200_sdcpp_eager_int8_profile_20260813.sh` |

## 输入与 Prompt 核对

最终主表中，每个任务的 Edge 与 sd.cpp `prompt.txt` 均通过 `cmp` 字节一致，参考输入路径、顺序、尺寸、seed 和采样参数一致；差异仅为框架与 DiT 8-bit 表示。视频帧与帧序列+配对音频还共同使用同一份视觉 Prompt，以隔离配对音频变量。

| 任务 | 输入 | Prompt | 口径核对 |
|---|---|---|---|
| 文生视频 | 无参考输入 | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-t2va/prompt.txt) | Edge/sd.cpp 字节一致 |
| 首帧生视频 | [`first.png`](../../outputs/minimax-h3/fl2va-checkpoint-four-tasks-2026-08-10/inputs/first.png) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-i2va/prompt.txt) | Edge/sd.cpp 字节一致；`<Picture 1>` 对应首帧 |
| 尾帧生视频 | [`last.png`](../../outputs/minimax-h3/fl2va-checkpoint-four-tasks-2026-08-10/inputs/last.png) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-l2va/prompt.txt) | Edge/sd.cpp 字节一致；`<Picture 1>` 对应尾帧 |
| 首尾帧生视频 | [`first.png`](../../outputs/minimax-h3/fl2va-checkpoint-four-tasks-2026-08-10/inputs/first.png) + [`last.png`](../../outputs/minimax-h3/fl2va-checkpoint-four-tasks-2026-08-10/inputs/last.png) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-fl2va/prompt.txt) | Edge/sd.cpp 字节一致；标签顺序为首帧 `<Picture 1>`、尾帧 `<Picture 2>` |
| 图片参考 | [`landscape-reference.png`](../../assets/minimax-h3-ref2va/images/landscape-reference.png) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-image/prompt.txt) | Edge/sd.cpp 字节一致 |
| 视频帧参考 | [`video-frames`](../../assets/minimax-h3-ref2va/video-frames) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-video-frames/prompt.txt) | Edge/sd.cpp 字节一致；只引用存在的 `<Video 1>` |
| 帧序列+配对音频 | [`video-frames`](../../assets/minimax-h3-ref2va/video-frames) + [`video-audio.wav`](../../assets/minimax-h3-ref2va/video-audio.wav) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-frame-audio/prompt.txt) | 与视频帧任务 Prompt 也字节一致；唯一新增输入为配对音频 |
| 混合参考 | [`landscape-reference.png`](../../assets/minimax-h3-ref2va/images/landscape-reference.png) + [`video-frames`](../../assets/minimax-h3-ref2va/video-frames) + [`video-audio.wav`](../../assets/minimax-h3-ref2va/video-audio.wav) + [`standalone-audio.wav`](../../assets/minimax-h3-ref2va/standalone-audio.wav) | [`prompt.txt`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-mixed/prompt.txt) | Edge/sd.cpp 字节一致；沿用已完成 Edge run 的 Prompt |

混合任务的已完成 Edge Prompt 将 `<Audio 1>` 描述为 standalone guidance；按参考注册顺序，`<Audio 1>` 更可能是视频配对音频、独立音频是 `<Audio 2>`。为遵守“不重跑 Edge、两框架 Prompt 完全相同”的约束，本轮 sd.cpp 原样复用该 Prompt。因此混合任务的 wall/显存对比公平，画面要求也一致，但不能用本轮结果判断独立音频标签是否被正确遵循。正式全量 benchmark 脚本已将后续运行改为 `<Audio 1>` 配对音频、`<Audio 2>` 独立音频。

## FL2VA Checkpoint 对比

| 任务 | Edge 输出 | sd.cpp 输出 | Edge wall | sd.cpp wall | Edge 加速 | Edge 显存 | sd.cpp 显存 | Edge 阶段耗时 | sd.cpp 阶段耗时 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 文生视频 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-t2va/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-t2va/final.mp4) | 140s | 219s | 1.56x | 70,333 MiB | 71,107 MiB | load 44.716s / Qwen 0.141s / noise 0.482s / DiT 87.719s / video 4.895s / audio 0.259s / save 1.143s | load 8.31s / cond 0.14s / DiT 175.62s / video 19.58s / audio 9.10s |
| 首帧生视频 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-i2va/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-i2va/final.mp4) | 146s | 236s | 1.62x | 70,493 MiB | 71,435 MiB | load 44.608s / Qwen 0.505s / keyframe VAE 0.435s / noise 0.482s / DiT 93.043s / video 4.898s / audio 0.215s / save 1.121s | load 8.48s / keyframe VAE 1.41s / cond 0.44s / DiT 190.93s / video 20.11s / audio 9.09s |
| 尾帧生视频 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-l2va/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-l2va/final.mp4) | 147s | 235s | 1.60x | 70,491 MiB | 72,490 MiB | load 45.108s / Qwen 0.562s / keyframe VAE 0.435s / noise 0.493s / DiT 92.976s / video 4.816s / audio 0.206s / save 1.411s | load 8.36s / keyframe VAE 1.40s / cond 0.43s / DiT 190.47s / video 19.68s / audio 9.08s |
| 首尾帧生视频 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-fl2va/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-fl2va/final.mp4) | 153s | 257s | 1.68x | 70,673 MiB | 71,463 MiB | load 45.569s / Qwen 0.836s / keyframe VAE 0.794s / noise 0.480s / DiT 98.761s / video 4.844s / audio 0.177s / save 1.199s | load 8.39s / keyframe VAE 2.53s / cond 0.77s / DiT 209.14s / video 20.89s / audio 9.08s |

## Ref2VA Checkpoint 对比

| 任务 | Edge 输出 | sd.cpp 输出 | Edge wall | sd.cpp wall | Edge 加速 | Edge 显存 | sd.cpp 显存 | Edge 阶段耗时 | sd.cpp 阶段耗时 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 图片参考 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-image/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-image/final.mp4) | 110s | 228s | 2.07x | 57,339 MiB | 59,081 MiB | load 7.778s / Qwen 0.607s / ref VAE 0.443s / noise 0.491s / DiT 93.242s / video 4.783s / audio 0.168s / save 1.106s | load 6.77s / ref VAE 1.27s / cond 0.50s / DiT 184.82s / video 19.80s / audio 9.10s |
| 视频帧参考 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-video-frames/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-video-frames/final.mp4) | 192s | 417s | 2.17x | 62,931 MiB | 62,595 MiB | load 7.508s / Qwen 1.716s / ref VAE 24.293s / noise 0.475s / DiT 145.866s / video 4.848s / audio 0.173s / save 1.843s | load 6.86s / ref VAE 32.71s / cond 1.51s / DiT 334.36s / video 19.73s / audio 9.08s |
| 帧序列+配对音频 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-frame-audio/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-frame-audio/final.mp4) | 192s | 424s | 2.21x | 62,931 MiB | 62,595 MiB | load 7.655s / Qwen 1.689s / ref VAE 24.403s / audio VAE 0.022s / noise 0.478s / DiT 146.078s / video 4.844s / audio 0.171s / save 1.779s | load 6.79s / ref VAE 33.29s / audio VAE 0.07s / cond 1.41s / DiT 338.90s / video 20.14s / audio 9.07s |
| 混合参考 | [`final.mp4`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-mixed/final.mp4) | [`final.mp4`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-mixed/final.mp4) | 204s | 454s | 2.23x | 62,931 MiB | 62,595 MiB | load 7.656s / Qwen 2.289s / ref VAE 24.673s / audio VAE 0.044s / noise 0.477s / DiT 156.592s / video 4.826s / audio 0.222s / save 1.828s | load 6.77s / ref VAE 34.25s / audio VAE 0.09s / cond 2.03s / DiT 368.01s / video 19.83s / audio 9.08s |

## Ref2VA 视频音频单变量复核（2026-08-13）

本节只验证 Edge 中“给相同视频帧增加配对音频是否破坏画面参考”。两组使用同一 Ref2VA Q8 GGUF、Qwen、VAE、56 张输入帧、`864x480`、124 帧、24 fps、20 steps、CFG `1`、seed `424242` 和字节完全一致的 Prompt。为缩短等待时间，两组分别在空闲 H200 GPU0/GPU1 并行执行，因此 wall 只用于确认运行成本接近，不与上方串行框架对比表混用。

| 条件 | 输出 | 命令 | Wall | 峰值显存 | 输入前 56 帧 SSIM | 观察 |
|---|---|---|---:|---:|---:|---|
| 仅视频帧 | [`final.mp4`](../../outputs/minimax-h3/h200-ref2va-audio-ablation-20260813/video-only/final.mp4) | [`cmd.txt`](../../outputs/minimax-h3/h200-ref2va-audio-ablation-20260813/video-only/cmd.txt) | 193s | 62,931 MiB | 0.314700 | 保持公园、树林、长椅、白狗和参考运动 |
| 同视频帧+配对音频 | [`final.mp4`](../../outputs/minimax-h3/h200-ref2va-audio-ablation-20260813/video-with-paired-audio/final.mp4) | [`cmd.txt`](../../outputs/minimax-h3/h200-ref2va-audio-ablation-20260813/video-with-paired-audio/cmd.txt) | 195s | 62,931 MiB | 0.314327 | 与无音频组内容和参考程度接近 |

两组生成视频之间的全 124 帧 SSIM 为 `0.823399`。加入配对音频会改变联合 DiT 条件和逐像素结果，这是 MiniMax-H3 联合音视频 self-attention 的预期行为；但本次受控结果没有显示画面参考质量下降。正式 benchmark 已改为运行时生成一份公共 Prompt 文件，并让 `ref2va-video-frames` 与 `ref2va-frame-audio` 共同读取该文件，二者唯一输入差异是后者增加 `--ref-video-audio`。

## Edge 各阶段耗时（2026-08-13 Profile 重跑）

本表使用 `ED_MINIMAX_H3_PROFILE=1` 在 H200 GPU0 串行重跑。FL2VA 四项使用修复后的官方 Diffusers BF16 在线 Q8 路径；Ref2VA 图片和混合任务沿用原输入与 Prompt，视频帧两项使用修正后的公共 Prompt。完整状态见 [`status.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/status.log)。

| 任务 | Profile 日志 | Wall | 峰值显存 | Load | Qwen context | 关键帧 VAE | 参考视觉 VAE | 参考音频准备+VAE | Noise | DiT | 视频解码+复制 | 音频解码+复制 | Save |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 文生视频 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-t2va/run.log) | 140s | 70,333 MiB | 44.716s | 0.141s | 0.000s | 0.000s | 0.000s | 0.482s | 87.719s | 4.895s | 0.259s | 1.143s |
| 首帧生视频 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-i2va/run.log) | 146s | 70,493 MiB | 44.608s | 0.505s | 0.435s | 0.000s | 0.000s | 0.482s | 93.043s | 4.898s | 0.215s | 1.121s |
| 尾帧生视频 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-l2va/run.log) | 147s | 70,491 MiB | 45.108s | 0.562s | 0.435s | 0.000s | 0.000s | 0.493s | 92.976s | 4.816s | 0.206s | 1.411s |
| 首尾帧生视频 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-fl2va/run.log) | 153s | 70,673 MiB | 45.569s | 0.836s | 0.794s | 0.000s | 0.000s | 0.480s | 98.761s | 4.844s | 0.177s | 1.199s |
| 图片参考 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-image/run.log) | 110s | 57,339 MiB | 7.778s | 0.607s | 0.000s | 0.443s | 0.000s | 0.491s | 93.242s | 4.783s | 0.168s | 1.106s |
| 视频帧参考 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-video-frames/run.log) | 192s | 62,931 MiB | 7.508s | 1.716s | 0.000s | 24.293s | 0.000s | 0.475s | 145.866s | 4.848s | 0.173s | 1.843s |
| 帧序列+配对音频 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-frame-audio/run.log) | 192s | 62,931 MiB | 7.655s | 1.689s | 0.000s | 24.403s | 0.022s | 0.478s | 146.078s | 4.844s | 0.171s | 1.779s |
| 混合参考 | [`run.log`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-mixed/run.log) | 204s | 62,931 MiB | 7.656s | 2.289s | 0.000s | 24.673s | 0.044s | 0.477s | 156.592s | 4.826s | 0.222s | 1.828s |

计时口径：

- `Load`、`Save` 来自进程外层 `[ED_WALL]`；其余列来自 pipeline 内部 `minimax-h3 profile`。
- `Qwen context` 是完整条件上下文阶段，内部包含视觉输入准备、Qwen3-VL vision encode、tokenize 和 text encode；不与关键帧/参考 VAE 重叠。
- `参考视觉 VAE` 同时承载 Ref2VA 图片和视频参考编码，因此图片任务的 `0.443s` 也列在该列。
- `DiT` 为 20 次 conditional diffusion call；本轮 CFG 为 `1`，没有 unconditional call 和 CFG combine。
- `视频解码+复制`、`音频解码+复制` 分别合并 VAE decode 与输出 postprocess/copy；各主阶段可以相加解释内部 generation，但不包含进程启动、模型加载和容器封装。

## sd.cpp 各阶段耗时（2026-08-13 Eager 重跑）

本表使用官方 Comfy INT8 ConvRot DiT 和 `--eager-load` 在 H200 GPU0 串行运行。`run.log` 中 FL2VA/Ref2VA 分别记录 `64,256.98 MiB`/`51,867.89 MiB` 的 prepared params backend buffer，确认模型参数整体常驻 GPU，而非按层 lazy 搬运。

| 任务 | 日志 | Wall | 峰值显存 | Load | 视觉 VAE encode | 音频 VAE encode | Condition | DiT | 视频解码 | 音频解码 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 文生视频 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-t2va/run.log) | 219s | 71,107 MiB | 8.31s | 0.00s | 0.00s | 0.14s | 175.62s | 19.58s | 9.10s |
| 首帧生视频 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-i2va/run.log) | 236s | 71,435 MiB | 8.48s | 1.41s | 0.00s | 0.44s | 190.93s | 20.11s | 9.09s |
| 尾帧生视频 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-l2va/run.log) | 235s | 72,490 MiB | 8.36s | 1.40s | 0.00s | 0.43s | 190.47s | 19.68s | 9.08s |
| 首尾帧生视频 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-fl2va/run.log) | 257s | 71,463 MiB | 8.39s | 2.53s | 0.00s | 0.77s | 209.14s | 20.89s | 9.08s |
| 图片参考 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-image/run.log) | 228s | 59,081 MiB | 6.77s | 1.27s | 0.00s | 0.50s | 184.82s | 19.80s | 9.10s |
| 视频帧参考 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-video-frames/run.log) | 417s | 62,595 MiB | 6.86s | 32.71s | 0.00s | 1.51s | 334.36s | 19.73s | 9.08s |
| 帧序列+配对音频 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-frame-audio/run.log) | 424s | 62,595 MiB | 6.79s | 33.29s | 0.07s | 1.41s | 338.90s | 20.14s | 9.07s |
| 混合参考 | [`run.log`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-mixed/run.log) | 454s | 62,595 MiB | 6.77s | 34.25s | 0.09s | 2.03s | 368.01s | 19.83s | 9.08s |

## 阶段耗时说明

- Edge 阶段来自 `ED_MINIMAX_H3_PROFILE=1` 内部 profile；sd.cpp 阶段来自 `-v` 的原生 timing 日志。两套阶段名和边界并不完全相同，适合定位各自热点，不应逐列视为严格同一计时区间。
- sd.cpp `Load` 是各次 `loading tensors completed` 的累计值；`Condition` 对应 `get_learned_condition`；DiT 对应 sampling；视觉/音频 VAE encode 与 decode 使用日志中各阶段完成时间。
- 两框架外部 wall 都包含进程启动、模型加载、生成、框架内部保存和正常退出。sd.cpp wall 在外部 FFmpeg 将 `final.webm` 转为 `final.mp4` 前结束，因此不包含文档封装转码时间。
- 峰值显存均从完整 `gpu_mem.csv` 的 GPU0 `memory.used` 数值列取最大值；修正后的 parser 使用数值转换，避免旧脚本按字符串比较导致错误峰值。

## sd.cpp Eager 与旧 Lazy 对比

| 任务 | Lazy wall | Eager wall | Lazy 峰值显存 | Eager 峰值显存 |
|---|---:|---:|---:|---:|
| 文生视频 | 219s | 219s | 65,951 MiB | 71,107 MiB |
| 图片参考 | 228s | 228s | 54,749 MiB | 59,081 MiB |
| 视频帧参考 | 422s | 417s | 56,245 MiB | 62,595 MiB |
| 帧序列+配对音频 | 422s | 424s | 57,389 MiB | 62,595 MiB |
| 混合参考 | 455s | 454s | 57,433 MiB | 62,595 MiB |

`--eager-load` 对速度没有可辨认的稳定收益：八项 wall 差异为 `-5s–+2s`，属于运行波动；但参数常驻会明显提高峰值显存。本文最终主表采用 eager 数据，是为了让 Edge 与 sd.cpp 都按组件常驻口径比较显存，而不是为了宣称 eager 更快。

## 文件索引

每个任务目录均包含 `cmd.txt`、`prompt.txt`、`run.log`、`gpu_mem.csv`、`metrics.json` 和 `final.mp4`；sd.cpp 目录还包含 `inputs.txt` 并保留原始 `final.webm`。

| 任务 | Edge 目录 | sd.cpp 目录 |
|---|---|---|
| 文生视频 | [`fl2va-t2va`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-t2va) | [`sdcpp-fl2va-t2va`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-t2va) |
| 首帧生视频 | [`fl2va-i2va`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-i2va) | [`sdcpp-fl2va-i2va`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-i2va) |
| 尾帧生视频 | [`fl2va-l2va`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-l2va) | [`sdcpp-fl2va-l2va`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-l2va) |
| 首尾帧生视频 | [`fl2va-fl2va`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/fl2va-fl2va) | [`sdcpp-fl2va-fl2va`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-fl2va-fl2va) |
| 图片参考 | [`ref2va-image`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-image) | [`sdcpp-ref2va-image`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-image) |
| 视频帧参考 | [`ref2va-video-frames`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-video-frames) | [`sdcpp-ref2va-video-frames`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-video-frames) |
| 帧序列+配对音频 | [`ref2va-frame-audio`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-frame-audio) | [`sdcpp-ref2va-frame-audio`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-frame-audio) |
| 混合参考 | [`ref2va-mixed`](../../outputs/minimax-h3/h200-edge-profile-8tasks-20260813/ref2va-mixed) | [`sdcpp-ref2va-mixed`](../../outputs/minimax-h3/h200-sdcpp-eager-int8-profile-20260813/sdcpp-ref2va-mixed) |
