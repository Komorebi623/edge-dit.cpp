# MiniMax-H3 Ref2VA 四图参考：ComfyUI 官方模板与 edge-dit.cpp

日期：2026-08-15
设备：同节点 NVIDIA H200；ComfyUI 使用 GPU 0，edge-dit.cpp 使用独立空闲 GPU 2。
执行方式：先完成并固化 ComfyUI 产物，再启动 edge-dit.cpp；两个框架不共享 GPU，不发生显存竞争。

## 结论

错误输出不是 Prompt 写法、四张图片顺序或 Ref2VA 权重切换造成的。修复前后使用完全相同的 Prompt、图片像素、seed、分辨率、帧数和模型文件；问题来自 edge-dit.cpp 的条件编码与采样实现。

修复前输出持续显示白底角色设定板并提前复制反派；修复后进入森林与基地场景，女主被藤蔓攻击、点火突破，最后才在基地门口与唯一反派交锋，整体语义与 ComfyUI 一致。

## 根因与修复

| 根因 | 修复前证据 | 修复与数值门禁 |
|---|---|---|
| 中文 tokenizer 丢失标点 token | 标点分支局部变量遮蔽；对 `char32_t` 错用 C `isspace` | 修复变量遮蔽并使用 Unicode 空白判断；完整纯文本 Prompt 与 HF tokenizer 均为 2,446 IDs，逐项一致 |
| Qwen3-VL 长序列 masked Flash Attention 错误 | 同一 BF16 Qwen、纯文本长 Prompt 的最终 context RMS 为 5.7087，而 ComfyUI 为 7.6243 | 禁用该错误的 masked Flash 路径后 Edge RMS 为 7.5851，对 ComfyUI RMSE 为 0.0762；四图任务由白底设定板恢复为真实场景 |
| Qwen3-VL Vision patch Conv3D 漏加 bias | patch projection 对 ComfyUI RMSE 为 0.32591、cos 为 0.71232 | linear 与 cuDNN direct 两条路径均传递 bias；cuDNN 补齐 BF16/F16/F32 同 dtype 输出的 bias，RMSE 降至 0.003783、cos 升至 0.999967 |
| Qwen3-VL Vision Attention 错设 `bias=false` | Block 0 QKV RMSE 为 0.86078，几乎等于遗漏 bias 的 RMS 0.86076 | QKV 与输出投影启用 bias；Block 0 cos 为 0.999898，27 层 merged vision embedding cos 为 0.999085 |
| Edge 固定 Euler/discrete，与官方工作流不一致 | ComfyUI 使用 `res_multistep/simple`，Edge CLI 参数此前未进入 MiniMax-H3 pipeline | MiniMax-H3 支持并显式使用 `--sampler res_multistep --scheduler simple`；21 个 sigma 与 ComfyUI 最大误差为 0 |
| RES 音频时间映射缺少初始 carry scale | 每步和收尾采用 AV carry 公式，但初始 audio latent 未乘 `video_shift/audio_shift` | 采样前补齐 `12/3=4` 的 audio scale，结束后恢复原尺度 |

关键的图片与 presentation 校验也已通过：四张原图与 ComfyUI input 逐像素相同；两侧 presentation 均为 3,106 tokens，其中 636 个视觉标签；`<Picture 1>`～`<Picture 4>` 的顺序、尺寸与插入位置一致。

## 任务与产物

- Prompt：[完整文本](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/config/prompt.txt)
- ComfyUI 实际提交配置：[JSON](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/config/comfyui-submitted-task.json)
- 输入图：[Picture 1](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/inputs/image1.png) / [Picture 2](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/inputs/image2.png) / [Picture 3](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/inputs/image3.png) / [Picture 4](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/inputs/image4.png)
- ComfyUI 产物：[final.mp4](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/comfyui/final.mp4)
- edge-dit.cpp 产物：[final.mp4](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/edge/final.mp4)
- Edge 接触图：[contact.jpg](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/edge/contact.jpg)
- 修复前错误接触图：[白底设定板](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/diagnostics/original-bad-q8dit-q4qwen/contact.jpg)
- ComfyUI 接触图：[contact.jpg](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/diagnostics/comfyui-contact.jpg)
- Edge 命令：[cmd.txt](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/edge/cmd.txt)
- Edge 完整日志：[run.log](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/edge/run.log)
- Edge 显存采样：[gpu_mem.csv](../../outputs/minimax-h3/comfyui-edge-ref2va-four-image-official-20260815/edge/gpu_mem.csv)

## 对齐口径

| 参数 | ComfyUI | edge-dit.cpp |
|---|---|---|
| Checkpoint | Ref2VA pruned | Ref2VA pruned |
| Prompt | 同一文本，SHA-256 `27f03c90d09cbdecd3cba22a366057049b76014d53638a410c623d97de9224bc` | 相同 |
| 参考图 | 4张，`match`，`<Picture 1>`～`<Picture 4>` | 同一像素内容、同一顺序；按输出尺寸匹配 |
| 分辨率 | 1280×736 | 1280×736 |
| 帧数 / FPS | 362 / 24 | 362 / 24 |
| Steps / CFG | 20 / 1.0 | 20 / 1.0 |
| Seed | 157368968253448 | 157368968253448 |
| DiT | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `minimax_h3_ref2va_pruned-Q8_0.gguf` |
| Qwen3-VL | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` |
| Video / Audio VAE | FP16 / FP32，同一文件 | FP16 / FP32，同一文件 |
| GPU / 组件驻留 | GPU 0；ComfyUI NORMAL_VRAM 动态管理 | GPU 2；全组件 GPU 常驻，无 CPU offload |

重要限制：ComfyUI 官方模板的 INT8 ConvRot 与 NVFP4 AWQ 依赖 Comfy Kitchen 专用算子，edge-dit.cpp 不能直接加载这两个文件。因此这里按组件位宽对齐：DiT 为8-bit 对8-bit，Qwen为4-bit 对4-bit；两侧 checkpoint、剪枝层数、输入和生成参数一致，但量化算法不是逐字节相同。两侧均使用 `res_multistep/simple`。

## 输出核对

| 框架 | 尺寸 | 帧数 | FPS | 视频时长 | 端到端墙钟 |
|---|---:|---:|---:|---:|---:|
| ComfyUI | 1280×736 | 362 | 24/1 | 15.083s | 1070.928s |
| edge-dit.cpp | 1280×736 | 362 | 24/1 | 15.083s | 942.954s |

## Edge 分阶段性能

| 阶段 | 用时（秒） | 阶段采样峰值显存（MiB） |
|---|---:|---:|
| 模型加载 | 8.481 | 45,513 |
| 条件上下文（4图+文本） | 2.582 | 48,529 |
| 噪声初始化 | 3.237 | 46,093 |
| DiT 20步 | 872.089 | 76,915 |
| Video/Audio VAE 解码 | 44.269 | 77,143 |
| 封装保存 | 5.307 | 77,111 |
| 端到端 | 942.954 | 77,143 |

全程采样峰值显存：**77,143 MiB**；运行前 GPU 基线：**0 MiB**。

阶段显存以 250ms 周期采样，并按照 Edge 内部 profile 的阶段耗时边界切分；短阶段可能存在不超过一个采样周期的误差。ComfyUI 本次任务启动前未启用持续显存采样，因此不伪造其阶段峰值数据。

## Edge 条件与解码细分

| 子阶段 | 用时（秒） |
|---|---:|
| Context 总计 | 2.269 |
| 视觉图片预处理 | 0.095 |
| 视觉图片编码 | 0.380 |
| 文本 tokenize | 0.059 |
| 文本编码 | 1.735 |
| 参考图 Video VAE encode | 0.313 |
| Video VAE decode | 43.506 |
| Audio VAE decode | 0.383 |
