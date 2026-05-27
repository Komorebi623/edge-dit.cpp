# light-dit.cpp

`light-dit.cpp` 是一个基于 C/C++ 的 DiT 推理项目，提供 `ld-cli` 命令行工具运行 SD3、Flux、Wan、Qwen-Image 等模型。

## 编译

进入项目目录：

```bash
cd /export/home/liuyiming54/light-dit.cpp
```

编译：

```bash
bash ./scripts/build_cuda.sh
bash ./scripts/build_cpu.sh
```

编译完成后，命令行程序位于：

```bash
./build-cuda/bin/ld-cli
```

## 基本用法

使用 diffusers 目录加载模型：

```bash
./build-cuda/bin/ld-cli --backend cuda \
  --model /path/to/diffusers-model-dir \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  -o output.png
```

使用组件式路径加载模型：

```bash
./build-cuda/bin/ld-cli --backend cuda \
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
./build-cuda/bin/ld-cli --backend cuda \
  --model /export/home/liuyiming54/models/stable-diffusion-3-medium-diffusers \
  -p "a cute cat holding a white sign with the exact text 'sd3.cpp' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 3.0 \
  -o sd3.png
```

### SD3 组件式加载

```bash
./build-cuda/bin/ld-cli --backend cuda \
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
./build-cuda/bin/ld-cli --backend cuda \
  --model /mnt/cfs/9n-das-admin/llm_models/flux-dev/ \
  -p "a cinematic photo of a glass teapot on a wooden table, soft morning light" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  --guidance 3.5 \
  -o flux_lightdit_test.png
```

### Flux 组件式加载

```bash
./build-cuda/bin/ld-cli --backend cuda \
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
./build-cuda/bin/ld-cli --backend cuda \
  --video \
  --model /export/home/liuyiming54/models/Wan2.1-T2V-1.3B-Diffusers \
  -p "a small robot walking through a rainy neon street, cinematic lighting" \
  -W 832 -H 480 --frames 81 --fps 16 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 5.0 \
  -o wan.mp4
```

### Qwen-Image

```bash
./build-cuda/bin/ld-cli --backend cuda \
  --model /export/home/liuyiming54/models/Qwen-Image \
  -p "a cute cat holding a white sign with the exact text 'qwen image' written clearly on it" \
  -W 1024 -H 1024 --steps 50 -s 0 \
  -o qwen_image.png
```

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
--video                 生成视频
--frames                视频帧数
--fps                   视频帧率
-o, --output            输出文件
```
