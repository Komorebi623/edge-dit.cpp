# MiniMax-H3 Ref2VA 15 秒三框架 Demo（2026-08-16）

本文对比 Edge-DiT.cpp、ComfyUI 和 Diffusers 在两个 MiniMax-H3 Ref2VA 15 秒任务上的表现，并记录阶段时间、显存、复现入口、阶段式生命周期和 24 GiB 预算验证结果。

## 结论

- 两项任务使用相同 prompt、原始参考图片、输出分辨率、362 帧、24 FPS、20 steps 参数和 seed；但参考图 resize **没有对齐**，Diffusers 实际执行的 Transformer forward 也是 19 次而非 20 次。
- Edge-DiT.cpp 和 ComfyUI 显式使用 `match`，保留这些小参考图的原始量级；官方 Diffusers 强制将每张图的短边上采样到 2048。Diffusers 因而在每个 DiT step 中处理多得多的参考 latent token，本报告中的三框架总耗时不能作为同计算量的框架速度排名。
- 当前数据中 Edge-DiT.cpp 的 DiT 用时更短，但由于参考序列长度和权重口径不同，不能据此计算纯框架加速比；输出 Video VAE decode 与参考 resize 无关，Edge 约 43 秒，仍约为 ComfyUI/Diffusers 的 20 秒两倍。
- MiniMax-H3 阶段式生命周期将 Edge 双角色任务峰值从 77,377 MiB 降至 52,777 MiB，生成阶段速度基本不变。
- `--max-vram 24 --auto-allocate` 不是进程级硬上限：该长序列完整 20-step 任务实测峰值仍为 39,505 MiB；把预算降至 8 GiB 后仍为 33,921 MiB，因此当前版本不能在 24 GiB GPU 上按此分辨率和时长运行。

## Demo

三联视频从上到下依次为 Edge-DiT.cpp、ComfyUI、Diffusers。为避免三路声音叠加，三联视频保留 Edge-DiT.cpp 音轨。

| 任务 | 三联视频 |
|---|---|
| 四图森林战斗，1280×736 | [Edge-DiT.cpp / ComfyUI / Diffusers](../assets/minimax-h3-ref2va-four-image-edge-comfyui-diffusers-demo.mp4) |
| 双角色都市短剧，736×1280 | [Edge-DiT.cpp / ComfyUI / Diffusers](../assets/minimax-h3-ref2va-two-character-edge-comfyui-diffusers-demo.mp4) |

## 请求参数

| 项目 | 四图森林战斗 | 双角色都市短剧 |
|---|---:|---:|
| 分辨率 | 1280×736 | 736×1280 |
| 输出帧数 | 362 | 362 |
| FPS / 时长 | 24 / 15.083 秒 | 24 / 15.083 秒 |
| Steps | 20 | 20 |
| Seed | 157368968253448 | 157368968253448 |
| CFG | 1.0 | 1.0 |
| Prompt SHA-256 | `801dca98b90fad33b43df8a5dec86f70d792fbf09cc9f75a4f31be3907ef996a` | `6987c6c54c3824edcd62e260bd073f3c2117e032561786b879c1ee864a008ce6` |

三个框架的 prompt 哈希完全一致。输入图片也使用同一文件或字节完全相同的副本：

- 四图输入：[图片 1](../assets/minimax-h3-ref2va-demo-four-image-1.png)、[图片 2](../assets/minimax-h3-ref2va-demo-four-image-2.png)、[图片 3](../assets/minimax-h3-ref2va-demo-four-image-3.png)、[图片 4](../assets/minimax-h3-ref2va-demo-four-image-4.png)
- 双角色输入：[顾清岚](../assets/minimax-h3-ref2va-demo-two-character-gu-qinglan.png)、[周叙白](../assets/minimax-h3-ref2va-demo-two-character-zhou-xubai.png)

Diffusers 官方 pipeline 在传入 `num_inference_steps=20` 时实际记录到 19 次 Transformer forward；Edge-DiT.cpp 和 ComfyUI 均为 20 次。这是官方 Diffusers scheduler/pipeline 的实际执行行为，命令参数没有被改成 19 steps。

### 参考图 resize 未对齐

这组 Demo 的原始输入一致，但进入 Video VAE、Qwen 和 DiT 的参考图尺寸不同：

| 任务 | 原图尺寸 | Edge-DiT.cpp / ComfyUI `match` | 官方 Diffusers | Diffusers 单图面积倍率 |
|---|---|---|---|---:|
| 四图森林战斗 | 502×319 / 567×319 / 518×291 / 519×291 | 512×320 / 576×320 / 512×288 / 512×288 | 3232×2048 / 3648×2048 / 3648×2048 / 3648×2048 | 40.40× / 40.53× / 50.67× / 50.67× |
| 双角色都市短剧 | 403×237 / 417×237 | 416×224 / 416×224 | 3488×2048 / 3616×2048 | 76.66× / 79.47× |

- Edge-DiT.cpp 和本次 ComfyUI 工作流的 `match` 都按输出像素面积等比例缩小、禁止放大，再把宽高分别对齐到 32；本次输入本来就小于输出面积，因此只发生 32 对齐。
- 官方 Diffusers Ref2VA 不提供 `match` 请求参数。它固定把每张图片短边缩放到 2048，**包括上采样**，不设面积上限，再对齐到 32。
- 参考图不仅影响一次性 VAE encode；其 latent 被拼入每次 Transformer forward。因此 40–79 倍的参考像素面积会显著增加每一步 DiT 的序列长度和计算量，尤其影响四图任务。
- Edge-DiT.cpp 的默认 `--ref-image-size max` 使用 Diffusers 的 2048 短边几何，包括小图上采样；所以 Edge 可以对齐 Diffusers。当前 Demo 为了与 ComfyUI 工作流一致，显式用了 `--ref-image-size match`。
- 当前 ComfyUI 的 `max` 实现为 `min(1, 2048 / short_edge)`，只会缩小大图而不会把小图放大，因此小图场景下仍不等同于官方 Diffusers。

所以本报告保留视频供主观效果参考，也保留各框架实测数据，但不再把当前端到端差值解释为纯框架性能差异。要做严格速度对比，应让 Edge 使用 `max`，并让 ComfyUI 增加一个允许上采样到 2048 的 Diffusers-compatible 模式后重跑。

## 权重口径

| 组件 | Edge-DiT.cpp | ComfyUI | Diffusers |
|---|---|---|---|
| Ref2VA DiT | 裁剪版 50 层 Q8_0 GGUF | 裁剪版 50 层 INT8 ConvRot safetensors | 官方完整 Diffusers 权重，运行时 TorchAO INT8 |
| Qwen3-VL | 50 层 Q4_K_M GGUF | 50 层 NVFP4 AWQ safetensors | 官方完整 Diffusers 权重，运行时 TorchAO INT8 |
| Video VAE | FP16 safetensors | FP16 safetensors | 官方 FP16/BF16 pipeline 组件，不量化 |
| Audio VAE | FP32 safetensors | FP32 safetensors | 官方 FP32 pipeline 组件，不量化 |
| 常驻策略 | 基准任务全部常驻 | 动态阶段加载/卸载 | 全部组件常驻 |

因此三框架的原始输入、prompt 和输出采样参数一致，但参考图预处理、量化格式及 Diffusers 的完整/裁剪权重口径并不相同。报告不把速度或质量差异简单归因于框架本身。

## 性能数据

> **注意：** 下表是已生成 Demo 的实测记录，不是严格同计算量 benchmark。Edge-DiT.cpp / ComfyUI 使用 `match`，Diffusers 使用 2048 短边参考图；Diffusers 还使用完整权重并实际执行 19 次 Transformer forward。只有相同输出尺寸下的 decode 阶段较少受到参考图 resize 影响。

### 四图森林战斗

| 指标 | Edge-DiT.cpp | ComfyUI | Diffusers |
|---|---:|---:|---:|
| 端到端 | **943.575 s** | 1070.928 s | 2343.937 s |
| Conditioning / context | **2.582 s** | 未单独记录 | 9.070 s text + 5.854 s VAE encode |
| DiT / sampler | **872.089 s**（20 calls） | 约 998.8 s（20 calls） | 2263.720 s（19 calls） |
| Video VAE decode | 43.506 s | 未单独记录 | **20.679 s** |
| Audio VAE decode | 0.383 s | 未单独记录 | **0.145 s** |
| 保存 / mux | 5.307 s | 包含在剩余时间中 | 4.699 s |
| GPU 峰值 | 77,143 MiB | 未采集 | 111,565 MiB |

### 双角色都市短剧

| 指标 | Edge-DiT.cpp | ComfyUI | Diffusers |
|---|---:|---:|---:|
| 端到端 | **878.903 s** | 1025.864 s | 1625.178 s |
| Conditioning / context | **1.155 s** | 33.140 s | 4.194 s text + 2.982 s VAE encode |
| DiT / sampler | **811.664 s**（20 calls） | 965.504 s（20 calls） | 1553.626 s（19 calls） |
| Video VAE decode | 43.101 s | **19.947 s** | 20.567 s |
| Audio VAE decode | 0.348 s | 0.501 s | **0.144 s** |
| 保存 / mux | **3.440 s** | 5.927 s | 3.481 s |
| GPU 峰值 | 77,377 MiB | **49,159 MiB** | 109,345 MiB |

## Video VAE 差距

三个实现都采用 MiniMax-H3 的 temporal chunk 语义。Edge-DiT.cpp 日志显示 362 帧输出被拆成 21 个 temporal decode graph，每个约 1.89–1.92 秒，总计约 43 秒；ComfyUI 和 Diffusers 完整 decode 均约 20.0–20.7 秒。

当前最明确的实现差异是：

- Edge-DiT.cpp 的 VAE 图以 F32 activation 为主，Conv3D 使用 FP16 权重与 F32 输入/输出的混合 cuDNN 路径。
- ComfyUI 日志明确使用 `torch.float16` Video VAE，并由 PyTorch/cuDNN 执行 FP16 activation 路径。
- 两边都使用约 256px 的空间 tile，因此差距不是“Edge 多切了 temporal chunk”导致的。

下一步最有价值的优化是为 MiniMax-H3 Video VAE 增加 FP16 activation/typed Conv3D 路径，并用逐帧 PSNR、音频哈希和主观视频做质量门禁。当前报告只定位差异，没有用降低精度的未验证改动替换基准结果。

## 阶段式生命周期

新增 `--minimax-h3-stage-lifecycle`：

1. Conditioning 阶段加载 Qwen 和所需 VAE。
2. Context/参考 latent 完成后释放 Qwen/VAE，只保留 conditioning tensor。
3. DiT 一次加载并跨全部 steps 常驻。
4. DiT 完成后释放 DiT，再加载 Video/Audio VAE decode。

双角色 20-step 完整验证：

| 指标 | 默认常驻 | 阶段式生命周期 | 变化 |
|---|---:|---:|---:|
| GPU 峰值 | 77,377 MiB | **52,777 MiB** | **-24,600 MiB** |
| DiT | 811.664 s | **810.259 s** | -0.17% |
| Generation | 866.212 s | 866.826 s | +0.07% |
| 端到端 | **878.903 s** | 892.219 s | +1.51% |
| Video PSNR | - | 61.637 dB | 几乎一致 |
| Audio PCM | - | SHA-256 完全一致 | 一致 |

端到端增加主要来自 CPU pinned 权重初始化及阶段搬运；DiT 每步速度没有受到实质影响。

生命周期版本只在基准命令上增加 `--minimax-h3-stage-lifecycle`，不改变采样参数。

## 24 GiB 验证

同一个 736×1280、362 帧任务先用 1 step 验证内存路径：

| 配置 | DiT | Video VAE decode | GPU 峰值 | 结果 |
|---|---:|---:|---:|---|
| 默认常驻 | 40.868 s | 44.364 s | 77,377 MiB | 成功 |
| 阶段式生命周期 | 41.020 s | 43.362 s | 52,777 MiB | 成功 |
| `--max-vram 24 --auto-allocate` | 43.361 s | 45.284 s | **39,503 MiB** | 成功，但超过 24 GiB |
| `--max-vram 8 --auto-allocate` | 44.759 s | 159.939 s | **33,921 MiB** | 成功，仍超过 24 GiB且 VAE 显著变慢 |

24 GiB 规划版本与同 seed、同输入、同 prompt 的常驻 1-step 输出相比，视频平均 PSNR 为 **48.769 dB**；解码音频波形相关系数为 **0.98347**、SNR 为 **14.835 dB**。图切分没有造成明显画面退化，但音频存在可测的数值变化；由于实测峰值仍超过 24 GiB，这不是物理 24 GiB GPU 上的质量保证。

随后用完全相同的双角色任务完成 20-step 质量和速度门禁：

| 指标 | 常驻基准 | `--max-vram 24 --auto-allocate` | 变化 |
|---|---:|---:|---:|
| Conditioning | **1.155 s** | 1.954 s | +69.18%（绝对增加 0.799 s） |
| DiT | **811.664 s** | 854.183 s | +5.24% |
| Video VAE decode | **43.101 s** | 44.896 s | +4.16% |
| Generation | **866.212 s** | 911.731 s | +5.25% |
| 端到端 | **878.903 s** | 935.137 s | +6.40% |
| GPU 峰值 | 77,377 MiB | **39,505 MiB** | -48.94% |

[常驻 / 24 GiB 规划左右对比视频](../assets/minimax-h3-ref2va-two-character-resident-max24-comparison.mp4)显示两者均保持角色、雨夜酒会场景和镜头质量，没有明显质量崩坏。扩散轨迹经过 20 steps 后并非像素一致，平均 PSNR 为 **20.536 dB**；预算版本音频与常驻版的波形相关系数为 **0.98089**、SNR 为 **14.157 dB**。该结果证明当前 graph-cut 路径能够完成高质量生成，但不能改变其 39.5 GiB 实测峰值仍不适用于物理 24 GiB GPU 的结论。

首次 24 GiB 分段测试还暴露了超长 RMSNorm CUDA grid 维度限制；修复为展平 outer grid 后，分段 DiT 可以正确完成。图切分 profile 显示 51 个 DiT segment、单段 cache 约 2079.9 MiB，但单个长序列 block 的 activation/backend workspace 仍使进程峰值高于规划预算。

因此当前结论是：

- `--max-vram` 控制组件 placement 和 graph-cut 预算，不是 CUDA 进程峰值硬限制。
- H200 上设置 24 GiB 只能模拟预算逻辑，不能模拟 RTX 4090 算力或证明任务能在 24 GiB 物理显存上运行。
- 该 15 秒长序列任务当前不能作为 4090 可运行 Demo；需要进一步降低单 block activation/workspace，或支持 block 内切分/CPU activation staging。

1-step 和 20-step 两次命令均在常驻基准上增加 `--max-vram 24 --auto-allocate`，未叠加阶段式生命周期。

## 回归验证

| 范围 | 验证 |
|---|---|
| Ref2VA | Q4 图片参考，39 帧，默认与生命周期均成功；视频 PSNR 49.329 dB |
| FL2VA | Q4 文生视频，39 帧，默认与生命周期均成功；视频 PSNR 55.119 dB |
| 其他模型 | Wan2.1-T2V-1.3B Diffusers 目录，17 帧，携带生命周期 flag 仍成功；flag 未影响非 MiniMax pipeline |
| 构建 | CUDA 全量目标 `ed-cli` 编译成功 |

## 复现入口

### Edge-DiT.cpp

核心参数：

```bash
ed-cli --video \
  --diffusion-model minimax_h3_ref2va_pruned-Q8_0.gguf \
  --llm qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --vae minimax_h3_video_vae_fp16.safetensors \
  --audio-vae minimax_h3_audio_vae_fp32.safetensors \
  --cfg-scale 1 -W <width> -H <height> --fps 24 \
  --video-frames 362 --steps 20 --seed 157368968253448 \
  --sampler res_multistep --scheduler simple --diffusion-fa \
  --ref-image <picture-1> [--ref-image <picture-N>] \
  --ref-image-size match --video-format mp4 -p "<prompt>" -o final.mp4
```

### ComfyUI

两项任务均使用 `ResolutionSelector` 的 0.9 MP、32 对齐，`res_multistep` sampler、20 steps；四图选择 16:9，双角色选择 9:16。模型节点使用 Ref2VA INT8 ConvRot DiT、NVFP4 AWQ Qwen3-VL、FP16 Video VAE 和 FP32 Audio VAE。

### Diffusers

- Runner：[`run_minimax_h3_ref2va_quantized_official_video.py`](../../scripts/diffusers/run_minimax_h3_ref2va_quantized_official_video.py)

```bash
python3 scripts/diffusers/run_minimax_h3_ref2va_quantized_official_video.py \
  --model /path/to/MiniMax-H3-Diffusers \
  --output final.mp4 --metrics profile.json --prompt "<prompt>" \
  --seed 157368968253448 --width <width> --height <height> \
  --num-frames 362 --fps 24 --steps 20 --bits 8 --resident \
  --profile-transformer --profile-components \
  --ref-image <picture-1> [--ref-image <picture-N>]
```
