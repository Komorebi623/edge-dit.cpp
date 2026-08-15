# MiniMax-H3 Ref2VA 15 秒三框架 Demo（2026-08-16）

本文对比 Edge-DiT.cpp、ComfyUI 和 Diffusers 在两个 MiniMax-H3 Ref2VA 15 秒任务上的表现，并记录阶段时间、显存、复现入口、阶段式生命周期和 24 GiB 预算验证结果。

## 结论

- 两项任务使用相同 prompt、相同参考图片、相同分辨率、362 帧、24 FPS、20 steps 和 seed。
- Edge-DiT.cpp 两项任务的端到端速度均快于 ComfyUI 和 Diffusers。
- Edge-DiT.cpp 的 DiT 明显快于另外两个框架，但 Video VAE decode 约 43 秒，约为 ComfyUI/Diffusers 的 20 秒两倍。
- MiniMax-H3 阶段式生命周期将 Edge 双角色任务峰值从 77,377 MiB 降至 52,777 MiB，生成阶段速度基本不变。
- `--max-vram 24 --auto-allocate` 不是进程级硬上限：该长序列任务实测峰值仍为 39,503 MiB；把预算降至 8 GiB 后仍为 33,921 MiB，因此当前版本不能在 24 GiB GPU 上按此分辨率和时长运行。

## Demo

三联视频从上到下依次为 Edge-DiT.cpp、ComfyUI、Diffusers。为避免三路声音叠加，三联视频保留 Edge-DiT.cpp 音轨。

| 任务 | 三联视频 |
|---|---|
| 四图森林战斗，1280×736 | [Edge-DiT.cpp / ComfyUI / Diffusers](../assets/minimax-h3-ref2va-four-image-edge-comfyui-diffusers-demo.mp4) |
| 双角色都市短剧，736×1280 | [Edge-DiT.cpp / ComfyUI / Diffusers](../assets/minimax-h3-ref2va-two-character-edge-comfyui-diffusers-demo.mp4) |

## 对齐口径

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

## 权重口径

| 组件 | Edge-DiT.cpp | ComfyUI | Diffusers |
|---|---|---|---|
| Ref2VA DiT | 裁剪版 50 层 Q8_0 GGUF | 裁剪版 50 层 INT8 ConvRot safetensors | 官方完整 Diffusers 权重，运行时 TorchAO INT8 |
| Qwen3-VL | 50 层 Q4_K_M GGUF | 50 层 NVFP4 AWQ safetensors | 官方完整 Diffusers 权重，运行时 TorchAO INT8 |
| Video VAE | FP16 safetensors | FP16 safetensors | 官方 FP16/BF16 pipeline 组件，不量化 |
| Audio VAE | FP32 safetensors | FP32 safetensors | 官方 FP32 pipeline 组件，不量化 |
| 常驻策略 | 基准任务全部常驻 | 动态阶段加载/卸载 | 全部组件常驻 |

因此三框架输入、prompt 和采样参数一致，但底层量化格式及 Diffusers 的完整/裁剪权重口径并不完全相同。报告不把质量差异简单归因于框架本身。

## 性能数据

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

[生命周期 metrics](../assets/minimax-h3-ref2va-demo-edge-lifecycle-metrics.json)记录了完整阶段数据。生命周期版本在基准命令上增加 `--minimax-h3-stage-lifecycle`，不改变采样参数。

## 24 GiB 验证

同一个 736×1280、362 帧任务先用 1 step 验证内存路径：

| 配置 | DiT | Video VAE decode | GPU 峰值 | 结果 |
|---|---:|---:|---:|---|
| 默认常驻 | 40.868 s | 44.364 s | 77,377 MiB | 成功 |
| 阶段式生命周期 | 41.020 s | 43.362 s | 52,777 MiB | 成功 |
| `--max-vram 24 --auto-allocate` | 43.361 s | 45.284 s | **39,503 MiB** | 成功，但超过 24 GiB |
| `--max-vram 8 --auto-allocate` | 44.759 s | 159.939 s | **33,921 MiB** | 成功，仍超过 24 GiB且 VAE 显著变慢 |

首次 24 GiB 分段测试还暴露了超长 RMSNorm CUDA grid 维度限制；修复为展平 outer grid 后，分段 DiT 可以正确完成。图切分 profile 显示 51 个 DiT segment、单段 cache 约 2079.9 MiB，但单个长序列 block 的 activation/backend workspace 仍使进程峰值高于规划预算。

因此当前结论是：

- `--max-vram` 控制组件 placement 和 graph-cut 预算，不是 CUDA 进程峰值硬限制。
- H200 上设置 24 GiB 只能模拟预算逻辑，不能模拟 RTX 4090 算力或证明任务能在 24 GiB 物理显存上运行。
- 该 15 秒长序列任务当前不能作为 4090 可运行 Demo；需要进一步降低单 block activation/workspace，或支持 block 内切分/CPU activation staging。

[24 GiB profile metrics](../assets/minimax-h3-ref2va-demo-edge-max24-metrics.json)记录了该次运行的完整阶段数据；命令在常驻基准上增加 `--max-vram 24 --auto-allocate` 并将 steps 设为 1，未叠加阶段式生命周期。

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

- 四图提交任务：[ComfyUI API JSON](../assets/minimax-h3-ref2va-demo-four-image-comfyui-api.json)
- 双角色提交任务：[ComfyUI API JSON](../assets/minimax-h3-ref2va-demo-two-character-comfyui-api.json)

两项任务均使用 `ResolutionSelector` 的 0.9 MP、32 对齐，`res_multistep` sampler、20 steps；四图选择 16:9，双角色选择 9:16。模型节点使用 Ref2VA INT8 ConvRot DiT、NVFP4 AWQ Qwen3-VL、FP16 Video VAE 和 FP32 Audio VAE。

### Diffusers

- Runner：[`run_minimax_h3_ref2va_quantized_official_video.py`](../../scripts/diffusers/run_minimax_h3_ref2va_quantized_official_video.py)
- 四图 profile：[profile.json](../assets/minimax-h3-ref2va-demo-four-image-diffusers-profile.json)
- 双角色 profile：[profile.json](../assets/minimax-h3-ref2va-demo-two-character-diffusers-profile.json)

```bash
python3 scripts/diffusers/run_minimax_h3_ref2va_quantized_official_video.py \
  --model /path/to/MiniMax-H3-Diffusers \
  --output final.mp4 --metrics profile.json --prompt "<prompt>" \
  --seed 157368968253448 --width <width> --height <height> \
  --num-frames 362 --fps 24 --steps 20 --bits 8 --resident \
  --profile-transformer --profile-components \
  --ref-image <picture-1> [--ref-image <picture-N>]
```
