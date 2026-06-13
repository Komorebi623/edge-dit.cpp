# edge-dit.cpp

`edge-dit.cpp` 是一个基于 C/C++ 的 DiT 推理项目，提供 `ed-cli` 命令行工具运行 SD3、Flux、Wan、Qwen-Image 等模型。

## 编译

进入项目目录：

```bash
cd /export/home/liuyiming54/edge-dit.cpp
```

编译：

```bash
bash ./scripts/build_cuda.sh
bash ./scripts/build_cpu.sh
```

编译完成后，命令行程序位于：

```bash
./build-cuda/bin/ed-cli
```

## 基本用法

使用 diffusers 目录加载模型：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/diffusers-model-dir \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  -o output.png
```

使用 CFG 并行在两张 GPU 上拆分 conditional / unconditional 分支：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/diffusers-model-dir \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --cfg-scale 5.0 \
  --devices 0,1 \
  --cfg-parallel-size 2 \
  -o output.png
```

使用 Sequence Parallel 在多张 GPU 上按 token/sequence 维切分 DiT 主干：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/diffusers-model-dir \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --devices 0,1 \
  --sp-size 2 \
  -o output.png
```

使用组件式路径加载模型：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --diffusion-model /path/to/transformer.safetensors \
  --clip_l /path/to/clip_l.safetensors \
  --clip_g /path/to/clip_g.safetensors \
  --t5xxl /path/to/t5xxl.safetensors.index.json \
  --vae /path/to/vae.safetensors \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  -o output.png
```

## 运行示例

### SD3 diffusers 目录

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers \
  -p "a cute cat holding a white sign with the exact text 'sd3.cpp' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 3.0 \
  -o sd3.png
```

### SD3 组件式加载

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --diffusion-model /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers/transformer/diffusion_pytorch_model.safetensors \
  --clip_l /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers/text_encoder/model.safetensors \
  --clip_g /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers/text_encoder_2/model.safetensors \
  --t5xxl /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers/text_encoder_3/model.safetensors \
  --vae /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers/vae/diffusion_pytorch_model.safetensors \
  -p "a cute cat holding a white sign with the exact text 'sd3.cpp' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 3.0 \
  -o sd3.png
```

### Flux diffusers 目录

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /mnt/cfs/9n-das-admin/llm_models/flux-dev/ \
  -p "a cinematic photo of a glass teapot on a wooden table, soft morning light" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --guidance 3.5 \
  -o flux_edgedit_test.png
```

### Flux 组件式加载

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --diffusion-model /mnt/cfs/9n-das-admin/llm_models/flux-dev/transformer/diffusion_pytorch_model.safetensors.index.json \
  --clip_l /mnt/cfs/9n-das-admin/llm_models/flux-dev/text_encoder/model.safetensors \
  --t5xxl /mnt/cfs/9n-das-admin/llm_models/flux-dev/text_encoder_2/model.safetensors.index.json \
  --vae /mnt/cfs/9n-das-admin/llm_models/flux-dev/ae.safetensors \
  -p "a cinematic photo of a glass teapot on a wooden table, soft morning light" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --guidance 3.5 \
  -o flux.png
```

### Wan 文生视频 diffusers 目录

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --video \
  --model /export/home/liuyiming54/models/Wan2.1-T2V-1.3B-Diffusers \
  -p "a small robot walking through a rainy neon street, cinematic lighting" \
  -W 832 -H 480 --frames 81 --fps 16 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 5.0 \
  -o wan.mp4
```

### Qwen-Image

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /export/home/liuyiming54/models/Qwen-Image \
  -p "a cute cat holding a white sign with the exact text 'qwen image' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  -o qwen_image.png
```

## CFG 并行推理

`edge-dit.cpp` 支持最轻量的一种多卡并行：CFG parallel。它把 CFG 的 unconditional 和 conditional 两个 diffusion forward 分到两个 GPU worker 上执行，然后通过 NCCL all-gather 收集结果并在 root rank 上完成 CFG 合成：

```text
rank 0 -> unconditional branch
rank 1 -> conditional branch
root  -> uncond + cfg_scale * (cond - uncond)
```

用户侧只需要指定并行算法规模和 GPU 列表：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/model \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --cfg-scale 5.0 \
  --devices 0,1 \
  --cfg-size 2 \
  -o output.png
```

参数语义：

```text
--cfg-parallel-size 2   开启 CFG 并行，当前支持 1 或 2
--cfg-size 2            --cfg-parallel-size 的短别名
--devices 0,1           选择参与并行的 GPU worker，数量需要和 cfg parallel size 一致
```

`--devices` 只表示使用哪些 GPU，不单独表示开启哪种并行算法。并行算法由 `--cfg-parallel-size`、未来的 `--tp-size`、`--sp-size` 等参数控制。

CFG parallel 可以和已有 cache 参数一起使用；cond/uncond 仍然按原有 `CacheBranch::Cond` 和 `CacheBranch::Uncond` 分支维护 cache。cache 命中只跳过本地 diffusion forward，每一步仍会参与 all-gather，以保持多卡通信同步。

## SP 并行推理

`edge-dit.cpp` 也支持 Sequence Parallel。SP 会把 diffusion transformer 的 sequence/token 维拆到多个 GPU worker 上，attention 前后通过 Ulysses 风格的 all-to-all 在 sequence/head 布局之间转换：

```text
rank i -> local sequence shard
q/k/v  -> seq-to-head all-to-all
attn   -> local head shard over gathered sequence layout
out    -> head-to-seq all-to-all, then restore sequence shard/full output
```

用户侧通过 `--sp-size` 开启 SP，通过 `--devices` 指定参与 worker。`--devices` 的数量必须和 `--sp-size` 一致：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers \
  -p "a cute cat holding a white sign with the exact text 'sd3.cpp' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 3.0 \
  --devices 3,5 \
  --sp-size 2 \
  -o sd3_sp.png
```

Flux 示例：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /mnt/cfs/9n-das-admin/llm_models/flux-dev/ \
  -p "a cinematic photo of a glass teapot on a wooden table, soft morning light" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --guidance 3.5 \
  --devices 3,5 \
  --sp-size 2 \
  -o flux_sp.png
```

Qwen-Image 示例：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /export/home/liuyiming54/models/Qwen-Image \
  -p "a cute cat holding a white sign with the exact text 'qwen image' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --devices 3,5 \
  --sp-size 2 \
  -o qwen_image_sp.png
```

Wan 示例：

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --video \
  --model /export/home/liuyiming54/models/Wan2.1-T2V-1.3B-Diffusers \
  -p "a small robot walking through a rainy neon street, cinematic lighting" \
  -W 832 -H 480 --frames 40 --fps 16 --steps 10 -s 0 \
  --cfg-scale 5.0 --flow-shift 5.0 \
  --devices 3,5 \
  --sp-size 2 \
  -o wan_sp.mp4
```

参数语义：

```text
--sp-size 2            开启 2-way sequence parallel
--sp-size 4            开启 4-way sequence parallel
--devices 3,5          选择参与 SP 的 GPU worker，数量需要和 sp size 一致
```

注意事项：

```text
1. SP 主要加速 diffusion transformer sampling hot path；模型加载、文本编码、VAE decode、图片/视频保存仍会影响端到端 wall time。
2. 小分辨率/短序列时，通信和 layout 转换开销可能抵消收益；分辨率越高、视频序列越长，SP 收益通常越明显。
3. Wan 的分辨率需要满足模型 latent/patch 对齐要求；报告中使用 416x240、640x384、832x480。
```

统一测试脚本：

```bash
python3 scripts/benchmark_sp_matrix.py \
  --models sd3,flux,qwen,wan \
  --image-resolutions 512x512,1024x1024,2048x2048 \
  --video-resolutions 416x240,640x384,832x480 \
  --gpu-groups '1:3;2:3,5;4:3,4,5,6' \
  --image-steps 2 \
  --video-steps 2 \
  --video-frames 40 \
  --out-dir /tmp/edge_dit_sp_benchmark \
  --report docs/sp_benchmark_report.md
```

当前状态：

```text
CFG parallel: 已支持 SD3 / Flux / Wan / Qwen-Image
TP parallel: 预留 CLI 参数，尚未实现
SP parallel: 已支持 SD3 / Flux / Wan / Qwen-Image
```

更多性能结果见 `docs/sp_benchmark_report.md`。在已测试的 SD3、Flux、Wan、Qwen-Image 上，SP 对高分辨率图片和长序列视频的 sampling hot path 有明显加速。

## 常用参数

```text
--backend cuda          使用 CUDA 后端
--model                 加载 diffusers 模型目录
--diffusion-model       单独指定 DiT/transformer 权重
--clip_l                单独指定 CLIP-L 权重
--clip_g                单独指定 CLIP-G 权重
--t5xxl                 单独指定 T5XXL 权重
--vae                   单独指定 VAE 权重
-p, --prompt            文本提示词
-W, --width             输出宽度
-H, --height            输出高度
--steps                 采样步数
-s, --seed              随机种子
--cfg-scale             CFG scale
--guidance              Flux guidance
--flow-shift            Flow scheduler shift
--devices               GPU worker 列表，例如 0,1
--cfg-parallel-size     CFG 并行规模，当前支持 1 或 2
--cfg-size              --cfg-parallel-size 的短别名
--tp-size               Tensor parallel 规模，预留参数，当前必须为 1
--sp-size               Sequence parallel 规模，支持 1 / 2 / 4 等与 --devices 数量一致的配置
--profile-graph-cuts    打印 graph-cut compute/communication timing summary
--video                 生成视频
--frames                视频帧数
--fps                   视频帧率
-o, --output            输出文件
```
